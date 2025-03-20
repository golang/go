// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || solaris

package unix

import (
	"syscall"
	"unsafe"
)

//go:linkname procFstatat libc_fstatat
//go:linkname procOpenat libc_openat
//go:linkname procUnlinkat libc_unlinkat
//go:linkname procReadlinkat libc_readlinkat
//go:linkname procMkdirat libc_mkdirat
//go:linkname procFchmodat libc_fchmodat
//go:linkname procFchownat libc_fchownat
//go:linkname procRenameat libc_renameat
//go:linkname procLinkat libc_linkat

var (
	procFstatat,
	procOpenat,
	procUnlinkat,
	procReadlinkat,
	procMkdirat,
	procFchmodat,
	procFchownat,
	procRenameat,
	procLinkat uintptr
)

func Unlinkat(dirfd int, path string, flags int) error {
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}

	_, _, errno := syscall6(uintptr(unsafe.Pointer(&procUnlinkat)), 3,
		uintptr(dirfd),
		uintptr(unsafe.Pointer(p)),
		uintptr(flags),
		0, 0, 0)
	if errno != 0 {
		return errno
	}

	return nil
}

func Openat(dirfd int, path string, flags int, perm uint32) (int, error) {
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return 0, err
	}

	fd, _, errno := syscall6(uintptr(unsafe.Pointer(&procOpenat)), 4,
		uintptr(dirfd),
		uintptr(unsafe.Pointer(p)),
		uintptr(flags),
		uintptr(perm),
		0, 0)
	if errno != 0 {
		return 0, errno
	}

	return int(fd), nil
}

func Fstatat(dirfd int, path string, stat *syscall.Stat_t, flags int) error {
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}

	_, _, errno := syscall6(uintptr(unsafe.Pointer(&procFstatat)), 4,
		uintptr(dirfd),
		uintptr(unsafe.Pointer(p)),
		uintptr(unsafe.Pointer(stat)),
		uintptr(flags),
		0, 0)
	if errno != 0 {
		return errno
	}

	return nil
}

func Readlinkat(dirfd int, path string, buf []byte) (int, error) {
	p0, err := syscall.BytePtrFromString(path)
	if err != nil {
		return 0, err
	}
	var p1 unsafe.Pointer
	if len(buf) > 0 {
		p1 = unsafe.Pointer(&buf[0])
	} else {
		p1 = unsafe.Pointer(&_zero)
	}
	n, _, errno := syscall6(uintptr(unsafe.Pointer(&procReadlinkat)), 4,
		uintptr(dirfd),
		uintptr(unsafe.Pointer(p0)),
		uintptr(p1),
		uintptr(len(buf)),
		0, 0)
	if errno != 0 {
		return 0, errno
	}

	return int(n), nil
}

func Mkdirat(dirfd int, path string, mode uint32) error {
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}

	_, _, errno := syscall6(uintptr(unsafe.Pointer(&procMkdirat)), 3,
		uintptr(dirfd),
		uintptr(unsafe.Pointer(p)),
		uintptr(mode),
		0, 0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}

func Fchmodat(dirfd int, path string, mode uint32, flags int) error {
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}
	_, _, errno := syscall6(uintptr(unsafe.Pointer(&procFchmodat)), 4,
		uintptr(dirfd),
		uintptr(unsafe.Pointer(p)),
		uintptr(mode),
		uintptr(flags),
		0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}

func Fchownat(dirfd int, path string, uid, gid int, flags int) error {
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}
	_, _, errno := syscall6(uintptr(unsafe.Pointer(&procFchownat)), 5,
		uintptr(dirfd),
		uintptr(unsafe.Pointer(p)),
		uintptr(uid),
		uintptr(gid),
		uintptr(flags),
		0)
	if errno != 0 {
		return errno
	}
	return nil
}

func Renameat(olddirfd int, oldpath string, newdirfd int, newpath string) error {
	oldp, err := syscall.BytePtrFromString(oldpath)
	if err != nil {
		return err
	}
	newp, err := syscall.BytePtrFromString(newpath)
	if err != nil {
		return err
	}
	_, _, errno := syscall6(uintptr(unsafe.Pointer(&procRenameat)), 4,
		uintptr(olddirfd),
		uintptr(unsafe.Pointer(oldp)),
		uintptr(newdirfd),
		uintptr(unsafe.Pointer(newp)),
		0,
		0)
	if errno != 0 {
		return errno
	}
	return nil
}

func Linkat(olddirfd int, oldpath string, newdirfd int, newpath string, flag int) error {
	oldp, err := syscall.BytePtrFromString(oldpath)
	if err != nil {
		return err
	}
	newp, err := syscall.BytePtrFromString(newpath)
	if err != nil {
		return err
	}
	_, _, errno := syscall6(uintptr(unsafe.Pointer(&procLinkat)), 5,
		uintptr(olddirfd),
		uintptr(unsafe.Pointer(oldp)),
		uintptr(newdirfd),
		uintptr(unsafe.Pointer(newp)),
		uintptr(flag),
		0)
	if errno != 0 {
		return errno
	}
	return nil
}
