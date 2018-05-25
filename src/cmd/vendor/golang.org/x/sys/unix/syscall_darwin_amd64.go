// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,darwin

package unix

import (
	"syscall"
	"unsafe"
)

func setTimespec(sec, nsec int64) Timespec {
	return Timespec{Sec: sec, Nsec: nsec}
}

func setTimeval(sec, usec int64) Timeval {
	return Timeval{Sec: sec, Usec: int32(usec)}
}

//sysnb	gettimeofday(tp *Timeval) (sec int64, usec int32, err error)
func Gettimeofday(tv *Timeval) (err error) {
	// The tv passed to gettimeofday must be non-nil
	// but is otherwise unused. The answers come back
	// in the two registers.
	sec, usec, err := gettimeofday(tv)
	tv.Sec = sec
	tv.Usec = usec
	return err
}

func SetKevent(k *Kevent_t, fd, mode, flags int) {
	k.Ident = uint64(fd)
	k.Filter = int16(mode)
	k.Flags = uint16(flags)
}

func (iov *Iovec) SetLen(length int) {
	iov.Len = uint64(length)
}

func (msghdr *Msghdr) SetControllen(length int) {
	msghdr.Controllen = uint32(length)
}

func (cmsg *Cmsghdr) SetLen(length int) {
	cmsg.Len = uint32(length)
}

func sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) {
	var length = uint64(count)

	_, _, e1 := Syscall6(SYS_SENDFILE, uintptr(infd), uintptr(outfd), uintptr(*offset), uintptr(unsafe.Pointer(&length)), 0, 0)

	written = int(length)

	if e1 != 0 {
		err = e1
	}
	return
}

func Syscall9(num, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err syscall.Errno)

// SYS___SYSCTL is used by syscall_bsd.go for all BSDs, but in modern versions
// of darwin/amd64 the syscall is called sysctl instead of __sysctl.
const SYS___SYSCTL = SYS_SYSCTL
