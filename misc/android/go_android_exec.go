// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program can be used as go_android_GOARCH_exec by the Go tool.
// It executes binaries on an android device using adb.
package main

import (
	"bytes"
	"fmt"
	"go/build"
	"io"
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
	// Directory structure on the target device androidtest.bash assumes.
	deviceGoroot = "/data/local/tmp/goroot"
	deviceGopath = "/data/local/tmp/gopath"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("go_android_exec: ")

	// Prepare a temporary directory that will be cleaned up at the end.
	deviceGotmp := fmt.Sprintf("/data/local/tmp/%s-%d",
		filepath.Base(os.Args[1]), os.Getpid())
	run("shell", "mkdir", "-p", deviceGotmp)

	// Determine the package by examining the current working
	// directory, which will look something like
	// "$GOROOT/src/mime/multipart" or "$GOPATH/src/golang.org/x/mobile".
	// We extract everything after the $GOROOT or $GOPATH to run on the
	// same relative directory on the target device.
	subdir, inGoRoot := subdir()
	deviceCwd := filepath.Join(deviceGoroot, subdir)
	if !inGoRoot {
		deviceCwd = filepath.Join(deviceGopath, subdir)
	}

	// Binary names can conflict.
	// E.g. template.test from the {html,text}/template packages.
	binName := fmt.Sprintf("%s-%d", filepath.Base(os.Args[1]), os.Getpid())
	deviceBin := fmt.Sprintf("%s/%s", deviceGotmp, binName)

	// The push of the binary happens in parallel with other tests.
	// Unfortunately, a simultaneous call to adb shell hold open
	// file descriptors, so it is necessary to push then move to
	// avoid a "text file busy" error on execution.
	// https://code.google.com/p/android/issues/detail?id=65857
	run("push", os.Args[1], deviceBin+"-tmp")
	run("shell", "cp '"+deviceBin+"-tmp' '"+deviceBin+"'")
	run("shell", "rm '"+deviceBin+"-tmp'")

	// Forward SIGQUIT from the go command to show backtraces from
	// the binary instead of from this wrapper.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGQUIT)
	go func() {
		for range quit {
			// We don't have the PID of the running process; use the
			// binary name instead.
			run("shell", "killall -QUIT "+binName)
		}
	}()
	// The adb shell command will return an exit code of 0 regardless
	// of the command run. E.g.
	//      $ adb shell false
	//      $ echo $?
	//      0
	// https://code.google.com/p/android/issues/detail?id=3254
	// So we append the exitcode to the output and parse it from there.
	const exitstr = "exitcode="
	cmd := `export TMPDIR="` + deviceGotmp + `"` +
		`; export GOROOT="` + deviceGoroot + `"` +
		`; export GOPATH="` + deviceGopath + `"` +
		`; cd "` + deviceCwd + `"` +
		"; '" + deviceBin + "' " + strings.Join(os.Args[2:], " ") +
		"; echo -n " + exitstr + "$?"
	output := run("shell", cmd)
	signal.Reset(syscall.SIGQUIT)
	close(quit)

	run("shell", "rm", "-rf", deviceGotmp) // Clean up.

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
	if root := runtime.GOROOT(); strings.HasPrefix(cwd, root) {
		subdir, err := filepath.Rel(root, cwd)
		if err != nil {
			log.Fatal(err)
		}
		return subdir, true
	}

	for _, p := range filepath.SplitList(build.Default.GOPATH) {
		if !strings.HasPrefix(cwd, p) {
			continue
		}
		subdir, err := filepath.Rel(p, cwd)
		if err == nil {
			return subdir, false
		}
	}
	log.Fatalf("the current path %q is not in either GOROOT(%q) or GOPATH(%q)",
		cwd, runtime.GOROOT(), build.Default.GOPATH)
	return "", false
}
