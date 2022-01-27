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
	"os/exec"
)

var ErrNotFound = exec.ErrNotFound

type (
	Cmd       = exec.Cmd
	Error     = exec.Error
	ExitError = exec.ExitError
)

func LookPath(file string) (string, error) {
	return exec.LookPath(file)
}

func CommandContext(ctx context.Context, name string, arg ...string) *exec.Cmd {
	return exec.CommandContext(ctx, name, arg...)
}

func Command(name string, arg ...string) *exec.Cmd {
	return exec.Command(name, arg...)
}
