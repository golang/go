// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"internal/abi"
	"unsafe"
)

//go:cgo_import_dynamic libc_grantpt grantpt "/usr/lib/libSystem.B.dylib"
func libc_grantpt_trampoline()

func Grantpt(fd int) error {
	_, _, errno := syscall_syscall6(abi.FuncPCABI0(libc_grantpt_trampoline), uintptr(fd), 0, 0, 0, 0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}

//go:cgo_import_dynamic libc_unlockpt unlockpt "/usr/lib/libSystem.B.dylib"
func libc_unlockpt_trampoline()

func Unlockpt(fd int) error {
	_, _, errno := syscall_syscall6(abi.FuncPCABI0(libc_unlockpt_trampoline), uintptr(fd), 0, 0, 0, 0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}

//go:cgo_import_dynamic libc_ptsname_r ptsname_r "/usr/lib/libSystem.B.dylib"
func libc_ptsname_r_trampoline()

func Ptsname(fd int) (string, error) {
	buf := make([]byte, 256)
	_, _, errno := syscall_syscall6(abi.FuncPCABI0(libc_ptsname_r_trampoline),
		uintptr(fd),
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(len(buf)-1),
		0, 0, 0)
	if errno != 0 {
		return "", errno
	}
	for i, c := range buf {
		if c == 0 {
			buf = buf[:i]
			break
		}
	}
	return string(buf), nil
}

//go:cgo_import_dynamic libc_posix_openpt posix_openpt "/usr/lib/libSystem.B.dylib"
func libc_posix_openpt_trampoline()

func PosixOpenpt(flag int) (fd int, err error) {
	ufd, _, errno := syscall_syscall6(abi.FuncPCABI0(libc_posix_openpt_trampoline), uintptr(flag), 0, 0, 0, 0, 0)
	if errno != 0 {
		return -1, errno
	}
	return int(ufd), nil
}
