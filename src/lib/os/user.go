// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// User ids etc.

package os

import (
	"syscall";
	"os";
)

// Getuid returns the numeric user id of the caller.
func Getuid() (uid int, err Error) {
	u, _, e := syscall.Syscall(syscall.SYS_GETUID, 0, 0, 0);
	if e != 0 {
		return -1, ErrnoToError(e)
	}
	return int(u), nil
}

// Geteuid returns the numeric effective user id of the caller.
func Geteuid() (uid int, err Error) {
	u, _, e := syscall.Syscall(syscall.SYS_GETEUID, 0, 0, 0);
	if e != 0 {
		return -1, ErrnoToError(e)
	}
	return int(u), nil
}

// Getgid returns the numeric group id of the caller.
func Getgid() (uid int, err Error) {
	g, _, e := syscall.Syscall(syscall.SYS_GETGID, 0, 0, 0);
	if e != 0 {
		return -1, ErrnoToError(e)
	}
	return int(g), nil
}

// Getegid returns the numeric effective group id of the caller.
func Getegid() (uid int, err Error) {
	g, _, e := syscall.Syscall(syscall.SYS_GETEGID, 0, 0, 0);
	if e != 0 {
		return -1, ErrnoToError(e)
	}
	return int(g), nil
}

