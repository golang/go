// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package syscall provides the syscall primitives required for the runtime.
package syscall

import (
	"internal/goarch"
	"unsafe"
)

// TODO(https://go.dev/issue/51087): This package is incomplete and currently
// only contains very minimal support for Linux.

// Syscall6 calls system call number 'num' with arguments a1-6.
func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)

func EpollCreate1(flags int32) (fd int32, errno uintptr) {
	r1, _, e := Syscall6(SYS_EPOLL_CREATE1, uintptr(flags), 0, 0, 0, 0, 0)
	return int32(r1), e
}

var _zero uintptr

func EpollWait(epfd int32, events []EpollEvent, maxev, waitms int32) (n int32, errno uintptr) {
	var ev unsafe.Pointer
	if len(events) > 0 {
		ev = unsafe.Pointer(&events[0])
	} else {
		ev = unsafe.Pointer(&_zero)
	}
	r1, _, e := Syscall6(SYS_EPOLL_PWAIT, uintptr(epfd), uintptr(ev), uintptr(maxev), uintptr(waitms), 0, 0)
	return int32(r1), e
}

func EpollCtl(epfd, op, fd int32, event *EpollEvent) (errno uintptr) {
	_, _, e := Syscall6(SYS_EPOLL_CTL, uintptr(epfd), uintptr(op), uintptr(fd), uintptr(unsafe.Pointer(event)), 0, 0)
	return e
}

func Eventfd(initval, flags int32) (fd int32, errno uintptr) {
	r1, _, e := Syscall6(SYS_EVENTFD2, uintptr(initval), uintptr(flags), 0, 0, 0, 0)
	return int32(r1), e
}

func Open(path *byte, mode int, perm uint32) (fd int, errno uintptr) {
	// Use SYS_OPENAT to match the syscall package.
	dfd := AT_FDCWD
	r1, _, e := Syscall6(SYS_OPENAT, uintptr(dfd), uintptr(unsafe.Pointer(path)), uintptr(mode|O_LARGEFILE), uintptr(perm), 0, 0)
	return int(r1), e
}

func Close(fd int) (errno uintptr) {
	_, _, e := Syscall6(SYS_CLOSE, uintptr(fd), 0, 0, 0, 0, 0)
	return e
}

func Read(fd int, p []byte) (n int, errno uintptr) {
	var p0 unsafe.Pointer
	if len(p) > 0 {
		p0 = unsafe.Pointer(&p[0])
	} else {
		p0 = unsafe.Pointer(&_zero)
	}
	r1, _, e := Syscall6(SYS_READ, uintptr(fd), uintptr(p0), uintptr(len(p)), 0, 0, 0)
	return int(r1), e
}

func Pread(fd int, p []byte, offset int64) (n int, errno uintptr) {
	var p0 unsafe.Pointer
	if len(p) > 0 {
		p0 = unsafe.Pointer(&p[0])
	} else {
		p0 = unsafe.Pointer(&_zero)
	}
	var r1, e uintptr
	switch goarch.GOARCH {
	case "386":
		r1, _, e = Syscall6(SYS_PREAD64, uintptr(fd), uintptr(p0), uintptr(len(p)), uintptr(offset), uintptr(offset>>32), 0)
	case "arm", "mipsle":
		r1, _, e = Syscall6(SYS_PREAD64, uintptr(fd), uintptr(p0), uintptr(len(p)), 0, uintptr(offset), uintptr(offset>>32))
	case "mips":
		r1, _, e = Syscall6(SYS_PREAD64, uintptr(fd), uintptr(p0), uintptr(len(p)), 0, uintptr(offset>>32), uintptr(offset))
	default:
		r1, _, e = Syscall6(SYS_PREAD64, uintptr(fd), uintptr(p0), uintptr(len(p)), uintptr(offset), 0, 0)
	}
	return int(r1), e
}
