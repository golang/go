// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This wrapper uses syscall.Flock to prevent concurrent adb commands,
// so for now it only builds on platforms that support that system call.
// TODO(#33974): use a more portable library for file locking.

//go:build darwin || dragonfly || freebsd || illumos || linux || netbsd || openbsd

// This program can be used as go_android_GOARCH_exec by the Go tool.
// It executes binaries on an android device using adb.
package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
)

func adbRun(args string) (int, error) {
	// The exit code of adb is often wrong. In theory it was fixed in 2016
	// (https://code.google.com/p/android/issues/detail?id=3254), but it's
	// still broken on our builders in 2023. Instead, append the exitcode to
	// the output and parse it from there.
	filter, exitStr := newExitCodeFilter(os.Stdout)
	args += "; echo -n " + exitStr + "$?"

	cmd := adbCmd("exec-out", args)
	cmd.Stdout = filter
	// If the adb subprocess somehow hangs, go test will kill this wrapper
	// and wait for our os.Stderr (and os.Stdout) to close as a result.
	// However, if the os.Stderr (or os.Stdout) file descriptors are
	// passed on, the hanging adb subprocess will hold them open and
	// go test will hang forever.
	//
	// Avoid that by wrapping stderr, breaking the short circuit and
	// forcing cmd.Run to use another pipe and goroutine to pass
	// along stderr from adb.
	cmd.Stderr = struct{ io.Writer }{os.Stderr}
	err := cmd.Run()

	// Before we process err, flush any further output and get the exit code.
	exitCode, err2 := filter.Finish()

	if err != nil {
		return 0, fmt.Errorf("adb exec-out %s: %v", args, err)
	}
	return exitCode, err2
}

func adb(args ...string) error {
	if out, err := adbCmd(args...).CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "adb %s\n%s", strings.Join(args, " "), out)
		return err
	}
	return nil
}

func adbCmd(args ...string) *exec.Cmd {
	if flags := os.Getenv("GOANDROID_ADB_FLAGS"); flags != "" {
		args = append(strings.Split(flags, " "), args...)
	}
	return exec.Command("adb", args...)
}

const (
	deviceRoot   = "/data/local/tmp/go_android_exec"
	deviceGoroot = deviceRoot + "/goroot"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("go_android_exec: ")
	exitCode, err := runMain()
	if err != nil {
		log.Fatal(err)
	}
	os.Exit(exitCode)
}

func runMain() (int, error) {
	// Concurrent use of adb is flaky, so serialize adb commands.
	// See https://github.com/golang/go/issues/23795 or
	// https://issuetracker.google.com/issues/73230216.
	lockPath := filepath.Join(os.TempDir(), "go_android_exec-adb-lock")
	lock, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0666)
	if err != nil {
		return 0, err
	}
	defer lock.Close()
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX); err != nil {
		return 0, err
	}

	// In case we're booting a device or emulator alongside all.bash, wait for
	// it to be ready. adb wait-for-device is not enough, we have to
	// wait for sys.boot_completed.
	if err := adb("wait-for-device", "exec-out", "while [[ -z $(getprop sys.boot_completed) ]]; do sleep 1; done;"); err != nil {
		return 0, err
	}

	// Done once per make.bash.
	if err := adbCopyGoroot(); err != nil {
		return 0, err
	}

	// Prepare a temporary directory that will be cleaned up at the end.
	// Binary names can conflict.
	// E.g. template.test from the {html,text}/template packages.
	binName := filepath.Base(os.Args[1])
	deviceGotmp := fmt.Sprintf(deviceRoot+"/%s-%d", binName, os.Getpid())
	deviceGopath := deviceGotmp + "/gopath"
	defer adb("exec-out", "rm", "-rf", deviceGotmp) // Clean up.

	// Determine the package by examining the current working
	// directory, which will look something like
	// "$GOROOT/src/mime/multipart" or "$GOPATH/src/golang.org/x/mobile".
	// We extract everything after the $GOROOT or $GOPATH to run on the
	// same relative directory on the target device.
	importPath, isStd, modPath, modDir, err := pkgPath()
	if err != nil {
		return 0, err
	}
	var deviceCwd string
	if isStd {
		// Note that we use path.Join here instead of filepath.Join:
		// The device paths should be slash-separated even if the go_android_exec
		// wrapper itself is compiled for Windows.
		deviceCwd = path.Join(deviceGoroot, "src", importPath)
	} else {
		deviceCwd = path.Join(deviceGopath, "src", importPath)
		if modDir != "" {
			// In module mode, the user may reasonably expect the entire module
			// to be present. Copy it over.
			deviceModDir := path.Join(deviceGopath, "src", modPath)
			if err := adb("exec-out", "mkdir", "-p", path.Dir(deviceModDir)); err != nil {
				return 0, err
			}
			// We use a single recursive 'adb push' of the module root instead of
			// walking the tree and copying it piecewise. If the directory tree
			// contains nested modules this could push a lot of unnecessary contents,
			// but for the golang.org/x repos it seems to be significantly (~2x)
			// faster than copying one file at a time (via filepath.WalkDir),
			// apparently due to high latency in 'adb' commands.
			if err := adb("push", modDir, deviceModDir); err != nil {
				return 0, err
			}
		} else {
			if err := adb("exec-out", "mkdir", "-p", deviceCwd); err != nil {
				return 0, err
			}
			if err := adbCopyTree(deviceCwd, importPath); err != nil {
				return 0, err
			}

			// Copy .go files from the package.
			goFiles, err := filepath.Glob("*.go")
			if err != nil {
				return 0, err
			}
			if len(goFiles) > 0 {
				args := append(append([]string{"push"}, goFiles...), deviceCwd)
				if err := adb(args...); err != nil {
					return 0, err
				}
			}
		}
	}

	deviceBin := fmt.Sprintf("%s/%s", deviceGotmp, binName)
	if err := adb("push", os.Args[1], deviceBin); err != nil {
		return 0, err
	}

	// Forward SIGQUIT from the go command to show backtraces from
	// the binary instead of from this wrapper.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGQUIT)
	go func() {
		for range quit {
			// We don't have the PID of the running process; use the
			// binary name instead.
			adb("exec-out", "killall -QUIT "+binName)
		}
	}()
	cmd := `export TMPDIR="` + deviceGotmp + `"` +
		`; export GOROOT="` + deviceGoroot + `"` +
		`; export GOPATH="` + deviceGopath + `"` +
		`; export CGO_ENABLED=0` +
		`; export GOPROXY=` + os.Getenv("GOPROXY") +
		`; export GOCACHE="` + deviceRoot + `/gocache"` +
		`; export PATH="` + deviceGoroot + `/bin":$PATH` +
		`; export HOME="` + deviceRoot + `/home"` +
		`; cd "` + deviceCwd + `"` +
		"; '" + deviceBin + "' " + strings.Join(os.Args[2:], " ")
	code, err := adbRun(cmd)
	signal.Reset(syscall.SIGQUIT)
	close(quit)
	return code, err
}

type exitCodeFilter struct {
	w      io.Writer // Pass through to w
	exitRe *regexp.Regexp
	buf    bytes.Buffer
}

func newExitCodeFilter(w io.Writer) (*exitCodeFilter, string) {
	const exitStr = "exitcode="

	// Build a regexp that matches any prefix of the exit string at the end of
	// the input. We do it this way to avoid assuming anything about the
	// subcommand output (e.g., it might not be \n-terminated).
	var exitReStr strings.Builder
	for i := 1; i <= len(exitStr); i++ {
		fmt.Fprintf(&exitReStr, "%s$|", exitStr[:i])
	}
	// Finally, match the exit string along with an exit code.
	// This is the only case we use a group, and we'll use this
	// group to extract the numeric code.
	fmt.Fprintf(&exitReStr, "%s([0-9]+)$", exitStr)
	exitRe := regexp.MustCompile(exitReStr.String())

	return &exitCodeFilter{w: w, exitRe: exitRe}, exitStr
}

func (f *exitCodeFilter) Write(data []byte) (int, error) {
	n := len(data)
	f.buf.Write(data)
	// Flush to w until a potential match of exitRe
	b := f.buf.Bytes()
	match := f.exitRe.FindIndex(b)
	if match == nil {
		// Flush all of the buffer.
		_, err := f.w.Write(b)
		f.buf.Reset()
		if err != nil {
			return n, err
		}
	} else {
		// Flush up to the beginning of the (potential) match.
		_, err := f.w.Write(b[:match[0]])
		f.buf.Next(match[0])
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

func (f *exitCodeFilter) Finish() (int, error) {
	// f.buf could be empty, contain a partial match of exitRe, or
	// contain a full match.
	b := f.buf.Bytes()
	defer f.buf.Reset()
	match := f.exitRe.FindSubmatch(b)
	if len(match) < 2 || match[1] == nil {
		// Not a full match. Flush.
		if _, err := f.w.Write(b); err != nil {
			return 0, err
		}
		return 0, fmt.Errorf("no exit code (in %q)", string(b))
	}

	// Parse the exit code.
	code, err := strconv.Atoi(string(match[1]))
	if err != nil {
		// Something is malformed. Flush.
		if _, err := f.w.Write(b); err != nil {
			return 0, err
		}
		return 0, fmt.Errorf("bad exit code: %v (in %q)", err, string(b))
	}
	return code, nil
}

// pkgPath determines the package import path of the current working directory,
// and indicates whether it is
// and returns the path to the package source relative to $GOROOT (or $GOPATH).
func pkgPath() (importPath string, isStd bool, modPath, modDir string, err error) {
	errorf := func(format string, args ...any) (string, bool, string, string, error) {
		return "", false, "", "", fmt.Errorf(format, args...)
	}
	goTool, err := goTool()
	if err != nil {
		return errorf("%w", err)
	}
	cmd := exec.Command(goTool, "list", "-e", "-f", "{{.ImportPath}}:{{.Standard}}{{with .Module}}:{{.Path}}:{{.Dir}}{{end}}", ".")
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
			return errorf("%v: %s", cmd, ee.Stderr)
		}
		return errorf("%v: %w", cmd, err)
	}

	parts := strings.SplitN(string(bytes.TrimSpace(out)), ":", 4)
	if len(parts) < 2 {
		return errorf("%v: missing ':' in output: %q", cmd, out)
	}
	importPath = parts[0]
	if importPath == "" || importPath == "." {
		return errorf("current directory does not have a Go import path")
	}
	isStd, err = strconv.ParseBool(parts[1])
	if err != nil {
		return errorf("%v: non-boolean .Standard in output: %q", cmd, out)
	}
	if len(parts) >= 4 {
		modPath = parts[2]
		modDir = parts[3]
	}

	return importPath, isStd, modPath, modDir, nil
}

// adbCopyTree copies testdata, go.mod, go.sum files from subdir
// and from parent directories all the way up to the root of subdir.
// go.mod and go.sum files are needed for the go tool modules queries,
// and the testdata directories for tests.  It is common for tests to
// reach out into testdata from parent packages.
func adbCopyTree(deviceCwd, subdir string) error {
	dir := ""
	for {
		for _, name := range []string{"testdata", "go.mod", "go.sum"} {
			hostPath := filepath.Join(dir, name)
			if _, err := os.Stat(hostPath); err != nil {
				continue
			}
			devicePath := path.Join(deviceCwd, dir)
			if err := adb("exec-out", "mkdir", "-p", devicePath); err != nil {
				return err
			}
			if err := adb("push", hostPath, devicePath); err != nil {
				return err
			}
		}
		if subdir == "." {
			break
		}
		subdir = filepath.Dir(subdir)
		dir = path.Join(dir, "..")
	}
	return nil
}

// adbCopyGoroot clears deviceRoot for previous versions of GOROOT, GOPATH
// and temporary data. Then, it copies relevant parts of GOROOT to the device,
// including the go tool built for android.
// A lock file ensures this only happens once, even with concurrent exec
// wrappers.
func adbCopyGoroot() error {
	goTool, err := goTool()
	if err != nil {
		return err
	}
	cmd := exec.Command(goTool, "version")
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("%v: %w", cmd, err)
	}
	goVersion := string(out)

	// Also known by cmd/dist. The bootstrap command deletes the file.
	statPath := filepath.Join(os.TempDir(), "go_android_exec-adb-sync-status")
	stat, err := os.OpenFile(statPath, os.O_CREATE|os.O_RDWR, 0666)
	if err != nil {
		return err
	}
	defer stat.Close()
	// Serialize check and copying.
	if err := syscall.Flock(int(stat.Fd()), syscall.LOCK_EX); err != nil {
		return err
	}
	s, err := io.ReadAll(stat)
	if err != nil {
		return err
	}
	if string(s) == goVersion {
		return nil
	}

	goroot, err := findGoroot()
	if err != nil {
		return err
	}

	// Delete the device's GOROOT, GOPATH and any leftover test data,
	// and recreate GOROOT.
	if err := adb("exec-out", "rm", "-rf", deviceRoot); err != nil {
		return err
	}

	// Build Go for Android.
	cmd = exec.Command(goTool, "install", "cmd")
	out, err = cmd.CombinedOutput()
	if err != nil {
		if len(bytes.TrimSpace(out)) > 0 {
			log.Printf("\n%s", out)
		}
		return fmt.Errorf("%v: %w", cmd, err)
	}
	if err := adb("exec-out", "mkdir", "-p", deviceGoroot); err != nil {
		return err
	}

	// Copy the Android tools from the relevant bin subdirectory to GOROOT/bin.
	cmd = exec.Command(goTool, "list", "-f", "{{.Target}}", "cmd/go")
	cmd.Stderr = os.Stderr
	out, err = cmd.Output()
	if err != nil {
		return fmt.Errorf("%v: %w", cmd, err)
	}
	platformBin := filepath.Dir(string(bytes.TrimSpace(out)))
	if platformBin == "." {
		return errors.New("failed to locate cmd/go for target platform")
	}
	if err := adb("push", platformBin, path.Join(deviceGoroot, "bin")); err != nil {
		return err
	}

	// Copy only the relevant subdirectories from pkg: pkg/include and the
	// platform-native binaries in pkg/tool.
	if err := adb("exec-out", "mkdir", "-p", path.Join(deviceGoroot, "pkg", "tool")); err != nil {
		return err
	}
	if err := adb("push", filepath.Join(goroot, "pkg", "include"), path.Join(deviceGoroot, "pkg", "include")); err != nil {
		return err
	}

	cmd = exec.Command(goTool, "list", "-f", "{{.Target}}", "cmd/compile")
	cmd.Stderr = os.Stderr
	out, err = cmd.Output()
	if err != nil {
		return fmt.Errorf("%v: %w", cmd, err)
	}
	platformToolDir := filepath.Dir(string(bytes.TrimSpace(out)))
	if platformToolDir == "." {
		return errors.New("failed to locate cmd/compile for target platform")
	}
	relToolDir, err := filepath.Rel(filepath.Join(goroot), platformToolDir)
	if err != nil {
		return err
	}
	if err := adb("push", platformToolDir, path.Join(deviceGoroot, relToolDir)); err != nil {
		return err
	}

	// Copy all other files from GOROOT.
	dirents, err := os.ReadDir(goroot)
	if err != nil {
		return err
	}
	for _, de := range dirents {
		switch de.Name() {
		case "bin", "pkg":
			// We already created GOROOT/bin and GOROOT/pkg above; skip those.
			continue
		}
		if err := adb("push", filepath.Join(goroot, de.Name()), path.Join(deviceGoroot, de.Name())); err != nil {
			return err
		}
	}

	if _, err := stat.WriteString(goVersion); err != nil {
		return err
	}
	return nil
}

func findGoroot() (string, error) {
	gorootOnce.Do(func() {
		// If runtime.GOROOT reports a non-empty path, assume that it is valid.
		// (It may be empty if this binary was built with -trimpath.)
		gorootPath = runtime.GOROOT()
		if gorootPath != "" {
			return
		}

		// runtime.GOROOT is empty — perhaps go_android_exec was built with
		// -trimpath and GOROOT is unset. Try 'go env GOROOT' as a fallback,
		// assuming that the 'go' command in $PATH is the correct one.

		cmd := exec.Command("go", "env", "GOROOT")
		cmd.Stderr = os.Stderr
		out, err := cmd.Output()
		if err != nil {
			gorootErr = fmt.Errorf("%v: %w", cmd, err)
		}

		gorootPath = string(bytes.TrimSpace(out))
		if gorootPath == "" {
			gorootErr = errors.New("GOROOT not found")
		}
	})

	return gorootPath, gorootErr
}

func goTool() (string, error) {
	goroot, err := findGoroot()
	if err != nil {
		return "", err
	}
	return filepath.Join(goroot, "bin", "go"), nil
}

var (
	gorootOnce sync.Once
	gorootPath string
	gorootErr  error
)
