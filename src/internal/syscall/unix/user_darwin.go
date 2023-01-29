// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"internal/abi"
	"syscall"
	"unsafe"
)

//go:cgo_import_dynamic libc_getgrouplist getgrouplist "/usr/lib/libSystem.B.dylib"
func libc_getgrouplist_trampoline()

func Getgrouplist(name *byte, gid uint32, gids *uint32, n *int32) error {
	_, _, errno := syscall_syscall6(abi.FuncPCABI0(libc_getgrouplist_trampoline),
		uintptr(unsafe.Pointer(name)), uintptr(gid), uintptr(unsafe.Pointer(gids)),
		uintptr(unsafe.Pointer(n)), 0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}

const (
	SC_GETGR_R_SIZE_MAX = 0x46
	SC_GETPW_R_SIZE_MAX = 0x47
)

type Passwd struct {
	Name   *byte
	Passwd *byte
	Uid    uint32 // uid_t
	Gid    uint32 // gid_t
	Change int64  // time_t
	Class  *byte
	Gecos  *byte
	Dir    *byte
	Shell  *byte
	Expire int64 // time_t
}

type Group struct {
	Name   *byte
	Passwd *byte
	Gid    uint32 // gid_t
	Mem    **byte
}

//go:cgo_import_dynamic libc_getpwnam_r getpwnam_r  "/usr/lib/libSystem.B.dylib"
func libc_getpwnam_r_trampoline()

func Getpwnam(name *byte, pwd *Passwd, buf *byte, size uintptr, result **Passwd) syscall.Errno {
	// Note: Returns an errno as its actual result, not in global errno.
	errno, _, _ := syscall_syscall6(abi.FuncPCABI0(libc_getpwnam_r_trampoline),
		uintptr(unsafe.Pointer(name)),
		uintptr(unsafe.Pointer(pwd)),
		uintptr(unsafe.Pointer(buf)),
		size,
		uintptr(unsafe.Pointer(result)),
		0)
	return syscall.Errno(errno)
}

//go:cgo_import_dynamic libc_getpwuid_r getpwuid_r  "/usr/lib/libSystem.B.dylib"
func libc_getpwuid_r_trampoline()

func Getpwuid(uid uint32, pwd *Passwd, buf *byte, size uintptr, result **Passwd) syscall.Errno {
	// Note: Returns an errno as its actual result, not in global errno.
	errno, _, _ := syscall_syscall6(abi.FuncPCABI0(libc_getpwuid_r_trampoline),
		uintptr(uid),
		uintptr(unsafe.Pointer(pwd)),
		uintptr(unsafe.Pointer(buf)),
		size,
		uintptr(unsafe.Pointer(result)),
		0)
	return syscall.Errno(errno)
}

//go:cgo_import_dynamic libc_getgrnam_r getgrnam_r  "/usr/lib/libSystem.B.dylib"
func libc_getgrnam_r_trampoline()

func Getgrnam(name *byte, grp *Group, buf *byte, size uintptr, result **Group) syscall.Errno {
	// Note: Returns an errno as its actual result, not in global errno.
	errno, _, _ := syscall_syscall6(abi.FuncPCABI0(libc_getgrnam_r_trampoline),
		uintptr(unsafe.Pointer(name)),
		uintptr(unsafe.Pointer(grp)),
		uintptr(unsafe.Pointer(buf)),
		size,
		uintptr(unsafe.Pointer(result)),
		0)
	return syscall.Errno(errno)
}

//go:cgo_import_dynamic libc_getgrgid_r getgrgid_r  "/usr/lib/libSystem.B.dylib"
func libc_getgrgid_r_trampoline()

func Getgrgid(gid uint32, grp *Group, buf *byte, size uintptr, result **Group) syscall.Errno {
	// Note: Returns an errno as its actual result, not in global errno.
	errno, _, _ := syscall_syscall6(abi.FuncPCABI0(libc_getgrgid_r_trampoline),
		uintptr(gid),
		uintptr(unsafe.Pointer(grp)),
		uintptr(unsafe.Pointer(buf)),
		size,
		uintptr(unsafe.Pointer(result)),
		0)
	return syscall.Errno(errno)
}

//go:cgo_import_dynamic libc_sysconf sysconf "/usr/lib/libSystem.B.dylib"
func libc_sysconf_trampoline()

func Sysconf(key int32) int64 {
	val, _, errno := syscall_syscall6X(abi.FuncPCABI0(libc_sysconf_trampoline),
		uintptr(key), 0, 0, 0, 0, 0)
	if errno != 0 {
		return -1
	}
	return int64(val)
}
