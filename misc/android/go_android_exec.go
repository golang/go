// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// This program can be used as go_android_GOARCH_exec by the Go tool.
// It executes binaries on an android device using adb.
package main

import (
	"bytes"
	"errors"
	"fmt"
	"go/build"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
)

func run(args ...string) (string, error) {
	cmd := adbCmd(args...)
	buf := new(bytes.Buffer)
	cmd.Stdout = io.MultiWriter(os.Stdout, buf)
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
	if err != nil {
		return "", fmt.Errorf("adb %s: %v", strings.Join(args, " "), err)
	}
	return buf.String(), nil
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
	subdir, inGoRoot, err := subdir()
	if err != nil {
		return 0, err
	}
	deviceCwd := filepath.Join(deviceGopath, subdir)
	if inGoRoot {
		deviceCwd = filepath.Join(deviceGoroot, subdir)
	} else {
		if err := adb("exec-out", "mkdir", "-p", deviceCwd); err != nil {
			return 0, err
		}
		if err := adbCopyTree(deviceCwd, subdir); err != nil {
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
	// In light of
	// https://code.google.com/p/android/issues/detail?id=3254
	// dont trust the exitcode of adb. Instead, append the exitcode to
	// the output and parse it from there.
	const exitstr = "exitcode="
	cmd := `export TMPDIR="` + deviceGotmp + `"` +
		`; export GOROOT="` + deviceGoroot + `"` +
		`; export GOPATH="` + deviceGopath + `"` +
		`; export CGO_ENABLED=0` +
		`; export GOPROXY=` + os.Getenv("GOPROXY") +
		`; export GOCACHE="` + deviceRoot + `/gocache"` +
		`; export PATH=$PATH:"` + deviceGoroot + `/bin"` +
		`; cd "` + deviceCwd + `"` +
		"; '" + deviceBin + "' " + strings.Join(os.Args[2:], " ") +
		"; echo -n " + exitstr + "$?"
	output, err := run("exec-out", cmd)
	signal.Reset(syscall.SIGQUIT)
	close(quit)
	if err != nil {
		return 0, err
	}

	exitIdx := strings.LastIndex(output, exitstr)
	if exitIdx == -1 {
		return 0, fmt.Errorf("no exit code: %q", output)
	}
	code, err := strconv.Atoi(output[exitIdx+len(exitstr):])
	if err != nil {
		return 0, fmt.Errorf("bad exit code: %v", err)
	}
	return code, nil
}

// subdir determines the package based on the current working directory,
// and returns the path to the package source relative to $GOROOT (or $GOPATH).
func subdir() (pkgpath string, underGoRoot bool, err error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", false, err
	}
	cwd, err = filepath.EvalSymlinks(cwd)
	if err != nil {
		return "", false, err
	}
	goroot, err := filepath.EvalSymlinks(runtime.GOROOT())
	if err != nil {
		return "", false, err
	}
	if subdir, err := filepath.Rel(goroot, cwd); err == nil {
		if !strings.Contains(subdir, "..") {
			return subdir, true, nil
		}
	}

	for _, p := range filepath.SplitList(build.Default.GOPATH) {
		pabs, err := filepath.EvalSymlinks(p)
		if err != nil {
			return "", false, err
		}
		if subdir, err := filepath.Rel(pabs, cwd); err == nil {
			if !strings.Contains(subdir, "..") {
				return subdir, false, nil
			}
		}
	}
	return "", false, fmt.Errorf("the current path %q is not in either GOROOT(%q) or GOPATH(%q)",
		cwd, runtime.GOROOT(), build.Default.GOPATH)
}

// adbCopyTree copies testdata, go.mod, go.sum files from subdir
// and from parent directories all the way up to the root of subdir.
// go.mod and go.sum files are needed for the go tool modules queries,
// and the testdata directories for tests.  It is common for tests to
// reach out into testdata from parent packages.
func adbCopyTree(deviceCwd, subdir string) error {
	dir := ""
	for {
		for _, path := range []string{"testdata", "go.mod", "go.sum"} {
			path := filepath.Join(dir, path)
			if _, err := os.Stat(path); err != nil {
				continue
			}
			devicePath := filepath.Join(deviceCwd, dir)
			if err := adb("exec-out", "mkdir", "-p", devicePath); err != nil {
				return err
			}
			if err := adb("push", path, devicePath); err != nil {
				return err
			}
		}
		if subdir == "." {
			break
		}
		subdir = filepath.Dir(subdir)
		dir = filepath.Join(dir, "..")
	}
	return nil
}

// adbCopyGoroot clears deviceRoot for previous versions of GOROOT, GOPATH
// and temporary data. Then, it copies relevant parts of GOROOT to the device,
// including the go tool built for android.
// A lock file ensures this only happens once, even with concurrent exec
// wrappers.
func adbCopyGoroot() error {
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
	s, err := ioutil.ReadAll(stat)
	if err != nil {
		return err
	}
	if string(s) == "done" {
		return nil
	}
	// Delete GOROOT, GOPATH and any leftover test data.
	if err := adb("exec-out", "rm", "-rf", deviceRoot); err != nil {
		return err
	}
	deviceBin := filepath.Join(deviceGoroot, "bin")
	if err := adb("exec-out", "mkdir", "-p", deviceBin); err != nil {
		return err
	}
	goroot := runtime.GOROOT()
	// Build go for android.
	goCmd := filepath.Join(goroot, "bin", "go")
	tmpGo, err := ioutil.TempFile("", "go_android_exec-cmd-go-*")
	if err != nil {
		return err
	}
	tmpGo.Close()
	defer os.Remove(tmpGo.Name())

	if out, err := exec.Command(goCmd, "build", "-o", tmpGo.Name(), "cmd/go").CombinedOutput(); err != nil {
		return fmt.Errorf("failed to build go tool for device: %s\n%v", out, err)
	}
	deviceGo := filepath.Join(deviceBin, "go")
	if err := adb("push", tmpGo.Name(), deviceGo); err != nil {
		return err
	}
	for _, dir := range []string{"src", "test", "lib", "api"} {
		if err := adb("push", filepath.Join(goroot, dir), filepath.Join(deviceGoroot)); err != nil {
			return err
		}
	}

	// Copy only the relevant from pkg.
	if err := adb("exec-out", "mkdir", "-p", filepath.Join(deviceGoroot, "pkg", "tool")); err != nil {
		return err
	}
	if err := adb("push", filepath.Join(goroot, "pkg", "include"), filepath.Join(deviceGoroot, "pkg")); err != nil {
		return err
	}
	runtimea, err := exec.Command(goCmd, "list", "-f", "{{.Target}}", "runtime").Output()
	pkgdir := filepath.Dir(string(runtimea))
	if pkgdir == "" {
		return errors.New("could not find android pkg dir")
	}
	if err := adb("push", pkgdir, filepath.Join(deviceGoroot, "pkg")); err != nil {
		return err
	}
	tooldir := filepath.Join(goroot, "pkg", "tool", filepath.Base(pkgdir))
	if err := adb("push", tooldir, filepath.Join(deviceGoroot, "pkg", "tool")); err != nil {
		return err
	}

	if _, err := stat.Write([]byte("done")); err != nil {
		return err
	}
	return nil
}
