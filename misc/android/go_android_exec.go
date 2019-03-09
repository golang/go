// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// This program can be used as go_android_GOARCH_exec by the Go tool.
// It executes binaries on an android device using adb.
package main

import (
	"bytes"
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

func run(args ...string) string {
	if flags := os.Getenv("GOANDROID_ADB_FLAGS"); flags != "" {
		args = append(strings.Split(flags, " "), args...)
	}
	buf := new(bytes.Buffer)
	cmd := exec.Command("adb", args...)
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
	log.Printf("adb %s", strings.Join(args, " "))
	err := cmd.Run()
	if err != nil {
		log.Fatalf("adb %s: %v", strings.Join(args, " "), err)
	}
	return buf.String()
}

const (
	deviceRoot   = "/data/local/tmp/go_exec_android"
	deviceGoroot = deviceRoot + "/goroot"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("go_android_exec: ")

	// Concurrent use of adb is flaky, so serialize adb commands.
	// See https://github.com/golang/go/issues/23795 or
	// https://issuetracker.google.com/issues/73230216.
	lockPath := filepath.Join(os.TempDir(), "go_android_exec-adb-lock")
	lock, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0666)
	if err != nil {
		log.Fatal(err)
	}
	defer lock.Close()
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX); err != nil {
		log.Fatal(err)
	}

	// In case we're booting a device or emulator alongside all.bash, wait for
	// it to be ready. adb wait-for-device is not enough, we have to
	// wait for sys.boot_completed.
	run("wait-for-device", "exec-out", "while [[ -z $(getprop sys.boot_completed) ]]; do sleep 1; done;")

	// Done once per make.bash.
	adbCopyGoroot()

	// Prepare a temporary directory that will be cleaned up at the end.
	// Binary names can conflict.
	// E.g. template.test from the {html,text}/template packages.
	binName := filepath.Base(os.Args[1])
	deviceGotmp := fmt.Sprintf(deviceRoot+"/%s-%d", binName, os.Getpid())
	deviceGopath := deviceGotmp + "/gopath"
	defer run("exec-out", "rm", "-rf", deviceGotmp) // Clean up.

	// Determine the package by examining the current working
	// directory, which will look something like
	// "$GOROOT/src/mime/multipart" or "$GOPATH/src/golang.org/x/mobile".
	// We extract everything after the $GOROOT or $GOPATH to run on the
	// same relative directory on the target device.
	subdir, inGoRoot := subdir()
	deviceCwd := filepath.Join(deviceGopath, subdir)
	if inGoRoot {
		deviceCwd = filepath.Join(deviceGoroot, subdir)
	} else {
		run("exec-out", "mkdir", "-p", deviceCwd)
		adbCopyTestdata(deviceCwd, subdir)

		// Copy .go files from the package.
		goFiles, err := filepath.Glob("*.go")
		if err != nil {
			log.Fatal(err)
		}
		if len(goFiles) > 0 {
			args := append(append([]string{"push"}, goFiles...), deviceCwd)
			run(args...)
		}
	}

	deviceBin := fmt.Sprintf("%s/%s", deviceGotmp, binName)
	run("push", os.Args[1], deviceBin)

	// Forward SIGQUIT from the go command to show backtraces from
	// the binary instead of from this wrapper.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGQUIT)
	go func() {
		for range quit {
			// We don't have the PID of the running process; use the
			// binary name instead.
			run("exec-out", "killall -QUIT "+binName)
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
		`; export GOCACHE="` + deviceRoot + `/gocache"` +
		`; export PATH=$PATH:"` + deviceGoroot + `/bin"` +
		`; cd "` + deviceCwd + `"` +
		"; '" + deviceBin + "' " + strings.Join(os.Args[2:], " ") +
		"; echo -n " + exitstr + "$?"
	output := run("exec-out", cmd)
	signal.Reset(syscall.SIGQUIT)
	close(quit)

	exitIdx := strings.LastIndex(output, exitstr)
	if exitIdx == -1 {
		log.Fatalf("no exit code: %q", output)
	}
	code, err := strconv.Atoi(output[exitIdx+len(exitstr):])
	if err != nil {
		log.Fatalf("bad exit code: %v", err)
	}
	os.Exit(code)
}

// subdir determines the package based on the current working directory,
// and returns the path to the package source relative to $GOROOT (or $GOPATH).
func subdir() (pkgpath string, underGoRoot bool) {
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	cwd, err = filepath.EvalSymlinks(cwd)
	if err != nil {
		log.Fatal(err)
	}
	goroot, err := filepath.EvalSymlinks(runtime.GOROOT())
	if err != nil {
		log.Fatal(err)
	}
	if strings.HasPrefix(cwd, goroot) {
		subdir, err := filepath.Rel(goroot, cwd)
		if err != nil {
			log.Fatal(err)
		}
		return subdir, true
	}

	for _, p := range filepath.SplitList(build.Default.GOPATH) {
		pabs, err := filepath.EvalSymlinks(p)
		if err != nil {
			log.Fatal(err)
		}
		if !strings.HasPrefix(cwd, pabs) {
			continue
		}
		subdir, err := filepath.Rel(pabs, cwd)
		if err == nil {
			return subdir, false
		}
	}
	log.Fatalf("the current path %q is not in either GOROOT(%q) or GOPATH(%q)",
		cwd, runtime.GOROOT(), build.Default.GOPATH)
	return "", false
}

// adbCopyTestdata copies testdata directories from subdir to deviceCwd
// on the device.
// It is common for tests to reach out into testdata from parent
// packages, so copy testdata directories all the way up to the root
// of subdir.
func adbCopyTestdata(deviceCwd, subdir string) {
	dir := ""
	for {
		testdata := filepath.Join(dir, "testdata")
		if _, err := os.Stat(testdata); err == nil {
			devicePath := filepath.Join(deviceCwd, dir)
			run("exec-out", "mkdir", "-p", devicePath)
			run("push", testdata, devicePath)
		}
		if subdir == "." {
			break
		}
		subdir = filepath.Dir(subdir)
		dir = filepath.Join(dir, "..")
	}
}

// adbCopyGoroot clears deviceRoot for previous versions of GOROOT, GOPATH
// and temporary data. Then, it copies relevant parts of GOROOT to the device,
// including the go tool built for android.
// A lock file ensures this only happens once, even with concurrent exec
// wrappers.
func adbCopyGoroot() {
	// Also known by cmd/dist. The bootstrap command deletes the file.
	statPath := filepath.Join(os.TempDir(), "go_android_exec-adb-sync-status")
	stat, err := os.OpenFile(statPath, os.O_CREATE|os.O_RDWR, 0666)
	if err != nil {
		log.Fatal(err)
	}
	defer stat.Close()
	// Serialize check and copying.
	if err := syscall.Flock(int(stat.Fd()), syscall.LOCK_EX); err != nil {
		log.Fatal(err)
	}
	s, err := ioutil.ReadAll(stat)
	if err != nil {
		log.Fatal(err)
	}
	if string(s) == "done" {
		return
	}
	// Delete GOROOT, GOPATH and any leftover test data.
	run("exec-out", "rm", "-rf", deviceRoot)
	deviceBin := filepath.Join(deviceGoroot, "bin")
	run("exec-out", "mkdir", "-p", deviceBin)
	goroot := runtime.GOROOT()
	// Build go for android.
	goCmd := filepath.Join(goroot, "bin", "go")
	tmpGo, err := ioutil.TempFile("", "go_android_exec-cmd-go-*")
	if err != nil {
		log.Fatal(err)
	}
	tmpGo.Close()
	defer os.Remove(tmpGo.Name())

	if out, err := exec.Command(goCmd, "build", "-o", tmpGo.Name(), "cmd/go").CombinedOutput(); err != nil {
		log.Fatalf("failed to build go tool for device: %s\n%v", out, err)
	}
	deviceGo := filepath.Join(deviceBin, "go")
	run("push", tmpGo.Name(), deviceGo)
	for _, dir := range []string{"pkg", "src", "test", "lib", "api"} {
		run("push", filepath.Join(goroot, dir), filepath.Join(deviceGoroot))
	}

	if _, err := stat.Write([]byte("done")); err != nil {
		log.Fatal(err)
	}
}
