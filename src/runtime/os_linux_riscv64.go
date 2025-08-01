// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/syscall/linux"
	"unsafe"
)

func osArchInit() {}

type riscvHWProbePairs = struct {
	key   int64
	value uint64
}

// TODO: Consider whether to use the VDSO entry for riscv_hwprobe.
// There is a VDSO entry for riscv_hwprobe that should allow us to avoid the syscall
// entirely as it can handle the case where the caller only requests extensions that are
// supported on all cores, which is what we're doing here. However, as we're only calling
// this syscall once, it may not be worth the added effort to implement the VDSO call.

//go:linkname internal_cpu_riscvHWProbe internal/cpu.riscvHWProbe
func internal_cpu_riscvHWProbe(pairs []riscvHWProbePairs, flags uint) bool {
	// sys_RISCV_HWPROBE is copied from golang.org/x/sys/unix/zsysnum_linux_riscv64.go.
	const sys_RISCV_HWPROBE uintptr = 258

	if len(pairs) == 0 {
		return false
	}
	// Passing in a cpuCount of 0 and a cpu of nil ensures that only extensions supported by all the
	// cores are returned, which is the behaviour we want in internal/cpu.
	_, _, e1 := linux.Syscall6(sys_RISCV_HWPROBE, uintptr(unsafe.Pointer(&pairs[0])), uintptr(len(pairs)), uintptr(0), uintptr(unsafe.Pointer(nil)), uintptr(flags), 0)
	return e1 == 0
}
