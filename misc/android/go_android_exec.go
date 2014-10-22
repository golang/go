// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program can be used as go_android_GOARCH_exec by the Go tool.
// It executes binaries on an android device using adb.
package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

func run(args ...string) string {
	buf := new(bytes.Buffer)
	cmd := exec.Command("adb", args...)
	cmd.Stdout = io.MultiWriter(os.Stdout, buf)
	cmd.Stderr = os.Stderr
	log.Printf("adb %s", strings.Join(args, " "))
	err := cmd.Run()
	if err != nil {
		log.Fatalf("adb %s: %v", strings.Join(args, " "), err)
	}
	return buf.String()
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("go_android_exec: ")

	// Determine thepackage by examining the current working
	// directory, which will look something like
	// "$GOROOT/src/mime/multipart". We extract everything
	// after the $GOROOT to run on the same relative directory
	// on the target device.
	//
	// TODO(crawshaw): Pick useful subdir when we are not
	// inside a GOROOT, e.g. we are in a GOPATH.
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	subdir, err := filepath.Rel(runtime.GOROOT(), cwd)
	if err != nil {
		log.Fatal(err)
	}
	subdir = filepath.ToSlash(subdir)

	// Binary names can conflict.
	// E.g. template.test from the {html,text}/template packages.
	binName := filepath.Base(os.Args[1])
	deviceGoroot := "/data/local/tmp/goroot"
	deviceBin := fmt.Sprintf("%s/%s-%d", deviceGoroot, binName, os.Getpid())

	// The push of the binary happens in parallel with other tests.
	// Unfortunately, a simultaneous call to adb shell hold open
	// file descriptors, so it is necessary to push then move to
	// avoid a "text file busy" error on execution.
	// https://code.google.com/p/android/issues/detail?id=65857
	run("push", os.Args[1], deviceBin+"-tmp")
	run("shell", "cp '"+deviceBin+"-tmp' '"+deviceBin+"'")
	run("shell", "rm '"+deviceBin+"-tmp'")

	// The adb shell command will return an exit code of 0 regardless
	// of the command run. E.g.
	//	$ adb shell false
	//	$ echo $?
	//	0
	// https://code.google.com/p/android/issues/detail?id=3254
	// So we append the exitcode to the output and parse it from there.
	const exitstr = "exitcode="
	cmd := `export TMPDIR="/data/local/tmp"` +
		`; export GOROOT="` + deviceGoroot + `"` +
		`; cd "$GOROOT/` + subdir + `"` +
		"; '" + deviceBin + "' " + strings.Join(os.Args[2:], " ") +
		"; echo -n " + exitstr + "$?"
	output := run("shell", cmd)
	run("shell", "rm '"+deviceBin+"'") // cleanup
	output = output[strings.LastIndex(output, "\n")+1:]
	if !strings.HasPrefix(output, exitstr) {
		log.Fatalf("no exit code: %q", output)
	}
	code, err := strconv.Atoi(output[len(exitstr):])
	if err != nil {
		log.Fatalf("bad exit code: %v", err)
	}
	os.Exit(code)
}
