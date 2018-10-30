// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"syscall"
	"unsafe"
)

// Implemented in runtime/syscall_solaris.go.
func sysvicall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err syscall.Errno)

//go:cgo_import_dynamic libc_fstatat fstatat "libc.so"
//go:cgo_import_dynamic libc_openat openat "libc.so"
//go:cgo_import_dynamic libc_unlinkat unlinkat "libc.so"

//go:linkname procFstatat libc_fstatat
//go:linkname procOpenat libc_openat
//go:linkname procUnlinkat libc_unlinkat

var (
	procFstatat,
	procOpenat,
	procUnlinkat uintptr
)

const AT_REMOVEDIR = 0x1
const AT_SYMLINK_NOFOLLOW = 0x1000

func Unlinkat(dirfd int, path string, flags int) error {
	var p *byte
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}

	_, _, errno := sysvicall6(uintptr(unsafe.Pointer(&procUnlinkat)), 3, uintptr(dirfd), uintptr(unsafe.Pointer(p)), uintptr(flags), 0, 0, 0)
	if errno != 0 {
		return errno
	}

	return nil
}

func Openat(dirfd int, path string, flags int, perm uint32) (int, error) {
	var p *byte
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return 0, err
	}

	fd, _, errno := sysvicall6(uintptr(unsafe.Pointer(&procOpenat)), 4, uintptr(dirfd), uintptr(unsafe.Pointer(p)), uintptr(flags), uintptr(perm), 0, 0)
	if errno != 0 {
		return 0, errno
	}

	return int(fd), nil
}

func Fstatat(dirfd int, path string, stat *syscall.Stat_t, flags int) error {
	var p *byte
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}

	_, _, errno := sysvicall6(uintptr(unsafe.Pointer(&procFstatat)), 4, uintptr(dirfd), uintptr(unsafe.Pointer(p)), uintptr(unsafe.Pointer(stat)), uintptr(flags), 0, 0)
	if errno != 0 {
		return errno
	}

	return nil
}
