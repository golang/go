// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows
// +build !plan9,!windows

package net

import (
	"os"
	"syscall"
)

var (
	errTimedout       = syscall.ETIMEDOUT
	errOpNotSupported = syscall.EOPNOTSUPP

	abortedConnRequestErrors = []error{syscall.ECONNABORTED} // see accept in fd_unix.go
)

func isPlatformError(err error) bool {
	_, ok := err.(syscall.Errno)
	return ok
}

func samePlatformError(err, want error) bool {
	if op, ok := err.(*OpError); ok {
		err = op.Err
	}
	if sys, ok := err.(*os.SyscallError); ok {
		err = sys.Err
	}
	return err == want
}
