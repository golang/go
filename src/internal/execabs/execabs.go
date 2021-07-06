// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package execabs is a drop-in replacement for os/exec
// that requires PATH lookups to find absolute paths.
// That is, execabs.Command("cmd") runs the same PATH lookup
// as exec.Command("cmd"), but if the result is a path
// which is relative, the Run and Start methods will report
// an error instead of running the executable.
package execabs

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"reflect"
	"unsafe"
)

var ErrNotFound = exec.ErrNotFound

type (
	Cmd       = exec.Cmd
	Error     = exec.Error
	ExitError = exec.ExitError
)

func relError(file, path string) error {
	return fmt.Errorf("%s resolves to executable relative to current directory (.%c%s)", file, filepath.Separator, path)
}

func LookPath(file string) (string, error) {
	path, err := exec.LookPath(file)
	if err != nil {
		return "", err
	}
	if filepath.Base(file) == file && !filepath.IsAbs(path) {
		return "", relError(file, path)
	}
	return path, nil
}

func fixCmd(name string, cmd *exec.Cmd) {
	if filepath.Base(name) == name && !filepath.IsAbs(cmd.Path) {
		// exec.Command was called with a bare binary name and
		// exec.LookPath returned a path which is not absolute.
		// Set cmd.lookPathErr and clear cmd.Path so that it
		// cannot be run.
		lookPathErr := (*error)(unsafe.Pointer(reflect.ValueOf(cmd).Elem().FieldByName("lookPathErr").Addr().Pointer()))
		if *lookPathErr == nil {
			*lookPathErr = relError(name, cmd.Path)
		}
		cmd.Path = ""
	}
}

func CommandContext(ctx context.Context, name string, arg ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, name, arg...)
	fixCmd(name, cmd)
	return cmd

}

func Command(name string, arg ...string) *exec.Cmd {
	cmd := exec.Command(name, arg...)
	fixCmd(name, cmd)
	return cmd
}
