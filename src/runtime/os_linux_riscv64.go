// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/syscall/linux"
	"unsafe"
)

type riscvHWProbePairs = struct {
	key   int64
	value uint64
}

//go:linkname internal_cpu_riscvHWProbe internal/cpu.riscvHWProbe
func internal_cpu_riscvHWProbe(pairs []riscvHWProbePairs, flags uint) bool {
	// sys_RISCV_HWPROBE is copied from golang.org/x/sys/unix/zsysnum_linux_riscv64.go.
	const sys_RISCV_HWPROBE uintptr = 258

	if len(pairs) == 0 {
		return false
	}
	// Passing a cpuCount of 0 and a cpu of nil queries the extensions supported
	// on all cores, which is what internal/cpu wants.
	//
	// Prefer the vDSO entry: it answers from its data page without a syscall,
	// so it works even when a seccomp filter blocks the riscv_hwprobe syscall.
	// Use the syscall only when the vDSO entry is absent (older kernels).
	if vdsoRiscvHWProbeSym != 0 {
		return vdsoRiscvHWProbe(&pairs[0], uintptr(len(pairs)), 0, nil, flags) == 0
	}
	_, _, e := linux.Syscall6(sys_RISCV_HWPROBE, uintptr(unsafe.Pointer(&pairs[0])), uintptr(len(pairs)), 0, 0, uintptr(flags), 0)
	return e == 0
}

// vdsoRiscvHWProbe calls the __vdso_riscv_hwprobe vDSO entry. It must only be
// called when vdsoRiscvHWProbeSym is non-zero. Implemented in sys_linux_riscv64.s.
//
//go:noescape
func vdsoRiscvHWProbe(pairs *riscvHWProbePairs, pairCount, cpusetsize uintptr, cpus *uint, flags uint) uintptr
