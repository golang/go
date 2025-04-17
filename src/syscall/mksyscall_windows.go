// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

// mksyscall_windows wraps golang.org/x/sys/windows/mkwinsyscall.
package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

func main() {
	goTool := filepath.Join(runtime.GOROOT(), "bin", "go")

	listCmd := exec.Command(goTool, "list", "-m")
	listCmd.Env = append(os.Environ(), "GO111MODULE=on")

	var (
		cmdEnv  []string
		modArgs []string
	)
	if out, err := listCmd.Output(); err == nil && string(bytes.TrimSpace(out)) == "std" {
		// Force module mode to use mkwinsyscall at the same version as the x/sys
		// module vendored into the standard library.
		cmdEnv = append(os.Environ(), "GO111MODULE=on")

		// Force -mod=readonly instead of the default -mod=vendor.
		//
		// mkwinsyscall is not itself vendored into the standard library, and it is
		// not feasible to do so at the moment: std-vendored libraries are included
		// in the "std" meta-pattern (because in general they *are* linked into
		// users binaries separately from the original import paths), and we can't
		// allow a binary in the "std" meta-pattern.
		modArgs = []string{"-mod=readonly"}
	} else {
		// Nobody outside the standard library should be using this wrapper: other
		// modules can vendor in the mkwinsyscall tool directly (as described in
		// https://golang.org/issue/25922), so they don't need this wrapper to
		// set module mode and -mod=readonly explicitly.
		os.Stderr.WriteString("WARNING: Please switch from using:\n    go run $GOROOT/src/syscall/mksyscall_windows.go\nto using:\n    go run golang.org/x/sys/windows/mkwinsyscall\n")
	}

	args := append([]string{"run"}, modArgs...)
	args = append(args, "golang.org/x/sys/windows/mkwinsyscall")
	args = append(args, os.Args[1:]...)
	cmd := exec.Command(goTool, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = cmdEnv
	err := cmd.Run()
	if err != nil {
		os.Exit(1)
	}
}
