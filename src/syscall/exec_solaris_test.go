// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build solaris

package syscall

var (
	procGetpgid = modlibc.NewProc("getpgid")
	procGetpgrp = modlibc.NewProc("getpgrp")
)

func Getpgid(pid int) (pgid int, err error) {
	r0, _, e1 := sysvicall6(procGetpgid.Addr(), 1, uintptr(pid), 0, 0, 0, 0, 0)
	pgid = int(r0)
	if e1 != 0 {
		err = e1
	}
	return
}

func Getpgrp() (pgrp int) {
	r0, _, _ := sysvicall6(procGetpgrp.Addr(), 0, 0, 0, 0, 0, 0, 0)
	pgrp = int(r0)
	return
}

var Ioctl = ioctl
