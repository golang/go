// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js && !wasip1

package base

import (
	"internal/godebug"
	"log"
	"os"
	"os/exec"
	"runtime"
	"syscall"
)

// Exec replaces the current process with an invocation of exe with the current
// command-line arguments and environment, or runs it as a subprocess and exits
// on platforms without syscall.Exec. name is used in error messages.
func Exec(name, exe string) {
	// On Windows, there is no syscall.Exec, so the best we can do
	// is run a subprocess and exit with the same status.
	// Doing the same on Unix would be a problem because it wouldn't
	// propagate signals and such, but there are no signals on Windows.
	// We also use the exec case when GODEBUG=gotoolchainexec=0,
	// to allow testing this code even when not on Windows.
	if godebug.New("#gotoolchainexec").Value() == "0" || runtime.GOOS == "windows" {
		cmd := exec.Command(exe, os.Args[1:]...)
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err := cmd.Run()
		if err != nil {
			if e, ok := err.(*exec.ExitError); ok && e.ProcessState != nil {
				if e.ProcessState.Exited() {
					os.Exit(e.ProcessState.ExitCode())
				}
				log.SetPrefix("go: ")
				Fatalf("exec %s: %s", name, e.ProcessState)
			}
			log.SetPrefix("go: ")
			Fatalf("exec %s: %s", exe, err)
		}
		os.Exit(0)
	}
	argv := os.Args
	if runtime.GOOS == "aix" {
		// On AIX, os.Executable derives the executable path from Args[0],
		// so the child must see its own path as argv[0].
		argv = append([]string{exe}, os.Args[1:]...)
	}
	err := syscall.Exec(exe, argv, os.Environ())
	log.SetPrefix("go: ")
	Fatalf("exec %s: %v", name, err)
}
