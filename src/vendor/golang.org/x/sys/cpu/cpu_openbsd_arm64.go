// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

import (
	"syscall"
	"unsafe"
)

// Minimal copy of functionality from x/sys/unix so the cpu package can call
// sysctl without depending on x/sys/unix.

const (
	// From OpenBSD's sys/sysctl.h.
	_CTL_MACHDEP = 7

	// From OpenBSD's machine/cpu.h.
	_CPU_ID_AA64ISAR0 = 2
	_CPU_ID_AA64ISAR1 = 3
)

// Implemented in the runtime package (runtime/sys_openbsd3.go)
func syscall_syscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err syscall.Errno)

//go:linkname syscall_syscall6 syscall.syscall6

func sysctl(mib []uint32, old *byte, oldlen *uintptr, new *byte, newlen uintptr) (err error) {
	_, _, errno := syscall_syscall6(libc_sysctl_trampoline_addr, uintptr(unsafe.Pointer(&mib[0])), uintptr(len(mib)), uintptr(unsafe.Pointer(old)), uintptr(unsafe.Pointer(oldlen)), uintptr(unsafe.Pointer(new)), uintptr(newlen))
	if errno != 0 {
		return errno
	}
	return nil
}

var libc_sysctl_trampoline_addr uintptr

//go:cgo_import_dynamic libc_sysctl sysctl "libc.so"

func sysctlUint64(mib []uint32) (uint64, bool) {
	var out uint64
	nout := unsafe.Sizeof(out)
	if err := sysctl(mib, (*byte)(unsafe.Pointer(&out)), &nout, nil, 0); err != nil {
		return 0, false
	}
	return out, true
}

func doinit() {
	setMinimalFeatures()

	// Get ID_AA64ISAR0 and ID_AA64ISAR1 from sysctl.
	isar0, ok := sysctlUint64([]uint32{_CTL_MACHDEP, _CPU_ID_AA64ISAR0})
	if !ok {
		return
	}
	isar1, ok := sysctlUint64([]uint32{_CTL_MACHDEP, _CPU_ID_AA64ISAR1})
	if !ok {
		return
	}
	parseARM64SystemRegisters(isar0, isar1, 0, 0)

	Initialized = true
}
