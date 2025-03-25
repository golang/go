// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd && !mips64

package unix

import (
	"internal/abi"
	"syscall"
	"unsafe"
)

//go:cgo_import_dynamic libc_readlinkat readlinkat "libc.so"

func libc_readlinkat_trampoline()

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
	n, _, errno := syscall_syscall6(abi.FuncPCABI0(libc_readlinkat_trampoline), uintptr(dirfd), uintptr(unsafe.Pointer(p0)), uintptr(p1), uintptr(len(buf)), 0, 0)
	if errno != 0 {
		return 0, errno
	}
	return int(n), nil
}

//go:cgo_import_dynamic libc_mkdirat mkdirat "libc.so"

func libc_mkdirat_trampoline()

func Mkdirat(dirfd int, path string, mode uint32) error {
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}
	_, _, errno := syscall_syscall6(abi.FuncPCABI0(libc_mkdirat_trampoline), uintptr(dirfd), uintptr(unsafe.Pointer(p)), 0, 0, 0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}

//go:cgo_import_dynamic libc_fchmodat fchmodat "libc.so"

func libc_fchmodat_trampoline()

func Fchmodat(dirfd int, path string, mode uint32, flags int) error {
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}
	_, _, errno := syscall_syscall6(abi.FuncPCABI0(libc_fchmodat_trampoline),
		uintptr(dirfd),
		uintptr(unsafe.Pointer(p)),
		uintptr(mode),
		uintptr(flags),
		0,
		0)
	if errno != 0 {
		return errno
	}
	return nil
}

//go:cgo_import_dynamic libc_fchownat fchownat "libc.so"

func libc_fchownat_trampoline()

func Fchownat(dirfd int, path string, uid, gid int, flags int) error {
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}
	_, _, errno := syscall_syscall6(abi.FuncPCABI0(libc_fchownat_trampoline),
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

//go:cgo_import_dynamic libc_renameat renameat "libc.so"

func libc_renameat_trampoline()

func Renameat(olddirfd int, oldpath string, newdirfd int, newpath string) error {
	oldp, err := syscall.BytePtrFromString(oldpath)
	if err != nil {
		return err
	}
	newp, err := syscall.BytePtrFromString(newpath)
	if err != nil {
		return err
	}
	_, _, errno := syscall_syscall6(abi.FuncPCABI0(libc_renameat_trampoline),
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

func libc_linkat_trampoline()

//go:cgo_import_dynamic libc_linkat linkat "libc.so"

func Linkat(olddirfd int, oldpath string, newdirfd int, newpath string, flag int) error {
	oldp, err := syscall.BytePtrFromString(oldpath)
	if err != nil {
		return err
	}
	newp, err := syscall.BytePtrFromString(newpath)
	if err != nil {
		return err
	}
	_, _, errno := syscall_syscall6(abi.FuncPCABI0(libc_linkat_trampoline),
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

func libc_symlinkat_trampoline()

//go:cgo_import_dynamic libc_symlinkat symlinkat "libc.so"

func Symlinkat(oldpath string, newdirfd int, newpath string) error {
	oldp, err := syscall.BytePtrFromString(oldpath)
	if err != nil {
		return err
	}
	newp, err := syscall.BytePtrFromString(newpath)
	if err != nil {
		return err
	}
	_, _, errno := syscall_syscall6(abi.FuncPCABI0(libc_symlinkat_trampoline),
		uintptr(unsafe.Pointer(oldp)),
		uintptr(newdirfd),
		uintptr(unsafe.Pointer(newp)),
		0,
		0,
		0)
	if errno != 0 {
		return errno
	}
	return nil
}
