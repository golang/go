// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

func main() {
	os.Stderr.WriteString("WARNING: Please switch from using:\n    go run $GOROOT/src/syscall/mksyscall_windows.go\nto using:\n    go run golang.org/x/sys/windows/mkwinsyscall\n")
	args := append([]string{"run", "golang.org/x/sys/windows/mkwinsyscall"}, os.Args[1:]...)
	cmd := exec.Command(filepath.Join(runtime.GOROOT(), "bin", "go"), args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		os.Exit(1)
	}
}
