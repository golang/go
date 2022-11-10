// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ((aix || dragonfly || freebsd || (linux && !android) || netbsd || openbsd) && cgo) || darwin

// Package pty is a simple pseudo-terminal package for Unix systems,
// implemented by calling C functions via cgo.
// This is only used for testing the os/signal package.
package pty

import (
	"fmt"
	"os"
	"syscall"
)

type PtyError struct {
	FuncName    string
	ErrorString string
	Errno       syscall.Errno
}

func ptyError(name string, err error) *PtyError {
	return &PtyError{name, err.Error(), err.(syscall.Errno)}
}

func (e *PtyError) Error() string {
	return fmt.Sprintf("%s: %s", e.FuncName, e.ErrorString)
}

func (e *PtyError) Unwrap() error { return e.Errno }

// Open returns a control pty and the name of the linked process tty.
func Open() (pty *os.File, processTTY string, err error) {
	return open()
}
