// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || openbsd || netbsd || dragonfly

package unix

import (
	"syscall"
	"unsafe"
)

func Unlinkat(dirfd int, path string, flags int) error {
	var p *byte
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}

	_, _, errno := syscall.Syscall(unlinkatTrap, uintptr(dirfd), uintptr(unsafe.Pointer(p)), uintptr(flags))
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

	fd, _, errno := syscall.Syscall6(openatTrap, uintptr(dirfd), uintptr(unsafe.Pointer(p)), uintptr(flags), uintptr(perm), 0, 0)
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

	_, _, errno := syscall.Syscall6(fstatatTrap, uintptr(dirfd), uintptr(unsafe.Pointer(p)), uintptr(unsafe.Pointer(stat)), uintptr(flags), 0, 0)
	if errno != 0 {
		return errno
	}

	return nil

}
