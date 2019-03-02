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
	deviceGoroot = "/data/local/tmp/goroot"
	deviceGopath = "/data/local/tmp/gopath"
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

	// Prepare a temporary directory that will be cleaned up at the end.
	deviceGotmp := fmt.Sprintf("/data/local/tmp/%s-%d",
		filepath.Base(os.Args[1]), os.Getpid())
	run("exec-out", "mkdir", "-p", deviceGotmp)

	// Determine the package by examining the current working
	// directory, which will look something like
	// "$GOROOT/src/mime/multipart" or "$GOPATH/src/golang.org/x/mobile".
	// We extract everything after the $GOROOT or $GOPATH to run on the
	// same relative directory on the target device.
	subdir, inGoRoot := subdir()
	deviceCwd := filepath.Join(deviceGoroot, subdir)
	if !inGoRoot {
		deviceCwd = filepath.Join(deviceGopath, subdir)
	} else {
		adbSyncGoroot()
	}
	run("exec-out", "mkdir", "-p", deviceCwd)

	// Binary names can conflict.
	// E.g. template.test from the {html,text}/template packages.
	binName := fmt.Sprintf("%s-%d", filepath.Base(os.Args[1]), os.Getpid())
	deviceBin := fmt.Sprintf("%s/%s", deviceGotmp, binName)
	run("push", os.Args[1], deviceBin)

	if _, err := os.Stat("testdata"); err == nil {
		run("push", "--sync", "testdata", deviceCwd)
	}

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
		`; cd "` + deviceCwd + `"` +
		"; '" + deviceBin + "' " + strings.Join(os.Args[2:], " ") +
		"; echo -n " + exitstr + "$?"
	output := run("exec-out", cmd)
	signal.Reset(syscall.SIGQUIT)
	close(quit)

	run("exec-out", "rm", "-rf", deviceGotmp) // Clean up.

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

// adbSyncGoroot ensures that files necessary for testing the Go standard
// packages are present on the attached device.
func adbSyncGoroot() {
	// Also known by cmd/dist. The bootstrap command deletes the file.
	statPath := filepath.Join(os.TempDir(), "go_android_exec-adb-sync-status")
	stat, err := os.OpenFile(statPath, os.O_CREATE|os.O_RDWR, 0666)
	if err != nil {
		log.Fatal(err)
	}
	defer stat.Close()
	// Serialize check and syncing.
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
	devRoot := "/data/local/tmp/goroot"
	run("exec-out", "rm", "-rf", devRoot)
	run("exec-out", "mkdir", "-p", devRoot+"/pkg")
	goroot := runtime.GOROOT()
	goCmd := filepath.Join(goroot, "bin", "go")
	runtimea, err := exec.Command(goCmd, "list", "-f", "{{.Target}}", "runtime").Output()
	if err != nil {
		log.Fatal(err)
	}
	pkgdir := filepath.Dir(string(runtimea))
	if pkgdir == "" {
		log.Fatal("could not find android pkg dir")
	}
	for _, dir := range []string{"src", "test", "lib"} {
		run("push", filepath.Join(goroot, dir), filepath.Join(devRoot))
	}
	run("push", filepath.Join(pkgdir), filepath.Join(devRoot, "pkg/"))
	if _, err := stat.Write([]byte("done")); err != nil {
		log.Fatal(err)
	}
}
