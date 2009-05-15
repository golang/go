// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// User ids etc.

package os

import (
	"syscall";
	"os";
	"unsafe";
)

// Getuid returns the numeric user id of the caller.
func Getuid() int {
	u, r2, e := syscall.Syscall(syscall.SYS_GETUID, 0, 0, 0);
	return int(u);
}

// Geteuid returns the numeric effective user id of the caller.
func Geteuid() int {
	u, r2, e := syscall.Syscall(syscall.SYS_GETEUID, 0, 0, 0);
	return int(u);
}

// Getgid returns the numeric group id of the caller.
func Getgid() int {
	g, r2, e := syscall.Syscall(syscall.SYS_GETGID, 0, 0, 0);
	return int(g);
}

// Getegid returns the numeric effective group id of the caller.
func Getegid() int {
	g, r2, e := syscall.Syscall(syscall.SYS_GETEGID, 0, 0, 0);
	return int(g);
}

// Getgroups returns a list of the numeric ids of groups that the caller belongs to.
func Getgroups() ([]int, os.Error) {
	// first call asks how many there are.
	r1, r2, err := syscall.Syscall(syscall.SYS_GETGROUPS, 0, 0, 0);
	if err != 0 {
		return nil, ErrnoToError(err);
	}

	if r1 < 0 || r1 > 1024 {	// the current max is 16; 1024 is a future-proof sanity check
		return nil, EINVAL;
	}
	a := make([]int, r1);
	if r1 > 0 {
		tmp := make([]uint32, r1);
		r1, r2, err = syscall.Syscall(syscall.SYS_GETGROUPS, r1, int64(uintptr(unsafe.Pointer(&tmp[0]))), 0);
		if err != 0 {
			return nil, ErrnoToError(err);
		}
		for i := 0; i < len(a); i++ {
			a[i] = int(tmp[i]);
		}
	}
	return a[0:r1], nil;
}
