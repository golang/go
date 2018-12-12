// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd js,wasm linux nacl netbsd openbsd solaris windows

package net

import (
	"os"
	"syscall"
)

// wrapSyscallError takes an error and a syscall name. If the error is
// a syscall.Errno, it wraps it in a os.SyscallError using the syscall name.
func wrapSyscallError(name string, err error) error {
	if _, ok := err.(syscall.Errno); ok {
		err = os.NewSyscallError(name, err)
	}
	return err
}
