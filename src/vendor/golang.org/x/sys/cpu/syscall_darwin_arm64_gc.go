// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Minimal copy from internal/cpu and runtime to make sysctl calls.

//go:build darwin && arm64 && gc

package cpu

import (
	"syscall"
	"unsafe"
)

type Errno = syscall.Errno

// adapted from internal/cpu/cpu_arm64_darwin.go
func darwinSysctlEnabled(name []byte) bool {
	out := int32(0)
	nout := unsafe.Sizeof(out)
	if ret := sysctlbyname(&name[0], (*byte)(unsafe.Pointer(&out)), &nout, nil, 0); ret != nil {
		return false
	}
	return out > 0
}

//go:cgo_import_dynamic libc_sysctl sysctl "/usr/lib/libSystem.B.dylib"

var libc_sysctlbyname_trampoline_addr uintptr

// adapted from runtime/sys_darwin.go in the pattern of sysctl() above, as defined in x/sys/unix
func sysctlbyname(name *byte, old *byte, oldlen *uintptr, new *byte, newlen uintptr) error {
	if _, _, err := syscall_syscall6(
		libc_sysctlbyname_trampoline_addr,
		uintptr(unsafe.Pointer(name)),
		uintptr(unsafe.Pointer(old)),
		uintptr(unsafe.Pointer(oldlen)),
		uintptr(unsafe.Pointer(new)),
		uintptr(newlen),
		0,
	); err != 0 {
		return err
	}

	return nil
}

//go:cgo_import_dynamic libc_sysctlbyname sysctlbyname "/usr/lib/libSystem.B.dylib"

// Implemented in the runtime package (runtime/sys_darwin.go)
func syscall_syscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)

//go:linkname syscall_syscall6 syscall.syscall6
