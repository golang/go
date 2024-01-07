// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js && !wasip1

package toolchain

import (
	"cmd/go/internal/base"
	"internal/godebug"
	"os"
	"os/exec"
	"runtime"
	"syscall"
)

// execGoToolchain execs the Go toolchain with the given name (gotoolchain),
// GOROOT directory, and go command executable.
// The GOROOT directory is empty if we are invoking a command named
// gotoolchain found in $PATH.
func execGoToolchain(gotoolchain, dir, exe string) {
	os.Setenv(targetEnv, gotoolchain)
	if dir == "" {
		os.Unsetenv("GOROOT")
	} else {
		os.Setenv("GOROOT", dir)
	}

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
				base.Fatalf("exec %s: %s", gotoolchain, e.ProcessState)
			}
			base.Fatalf("exec %s: %s", exe, err)
		}
		os.Exit(0)
	}
	err := syscall.Exec(exe, os.Args, os.Environ())
	base.Fatalf("exec %s: %v", gotoolchain, err)
}
