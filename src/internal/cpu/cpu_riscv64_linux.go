// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64 && linux

package cpu

import _ "unsafe"

// RISC-V extension discovery for Linux uses the riscv_hwprobe interface, which
// reports explicitly versioned extensions
// See https://docs.kernel.org/arch/riscv/hwprobe.html.
//
// hwprobe is queried through the vDSO when available: that answers this query
// from the vDSO data page without a syscall, so it still works under a seccomp
// filter that blocks the syscall. Otherwise the riscv_hwprobe syscall is used.
// Both paths are in runtime.internal_cpu_riscvHWProbe.
//
// On kernels predating riscv_hwprobe, no extensions are detected and vector
// code degrades to scalar.

const (
	// Copied from golang.org/x/sys/unix/ztypes_linux_riscv64.go.
	riscv_HWPROBE_KEY_IMA_EXT_0   = 0x4
	riscv_HWPROBE_IMA_V           = 0x4
	riscv_HWPROBE_EXT_ZBB         = 0x10
	riscv_HWPROBE_EXT_ZBC         = 0x80
	riscv_HWPROBE_EXT_ZVBB        = 0x20000
	riscv_HWPROBE_EXT_ZVBC        = 0x40000
	riscv_HWPROBE_EXT_ZVKB        = 0x80000
	riscv_HWPROBE_EXT_ZVKG        = 0x100000
	riscv_HWPROBE_EXT_ZVKNED      = 0x200000
	riscv_HWPROBE_EXT_ZVKNHA      = 0x400000
	riscv_HWPROBE_EXT_ZVKNHB      = 0x800000
	riscv_HWPROBE_EXT_ZVKSED      = 0x1000000
	riscv_HWPROBE_EXT_ZVKSH       = 0x2000000
	riscv_HWPROBE_EXT_ZVKT        = 0x4000000
	riscv_HWPROBE_KEY_CPUPERF_0   = 0x5
	riscv_HWPROBE_MISALIGNED_FAST = 0x3
	riscv_HWPROBE_MISALIGNED_MASK = 0x7
)

// riscvHWProbePairs is copied from golang.org/x/sys/unix/ztypes_linux_riscv64.go.
type riscvHWProbePairs struct {
	key   int64
	value uint64
}

//go:linkname riscvHWProbe
func riscvHWProbe(pairs []riscvHWProbePairs, flags uint) bool

func osInit() {
	// A slice of key/value pair structures is passed to the RISCVHWProbe syscall. The key
	// field should be initialised with one of the key constants defined above, e.g.,
	// RISCV_HWPROBE_KEY_IMA_EXT_0. The syscall will set the value field to the appropriate value.
	// If the kernel does not recognise a key it will set the key field to -1 and the value field to 0.

	pairs := []riscvHWProbePairs{
		{riscv_HWPROBE_KEY_IMA_EXT_0, 0},
		{riscv_HWPROBE_KEY_CPUPERF_0, 0},
	}

	// This call only indicates that extensions are supported if they are implemented on all cores.
	if !riscvHWProbe(pairs, 0) {
		return
	}

	if pairs[0].key != -1 {
		v := uint(pairs[0].value)
		RISCV64.HasV = isSet(v, riscv_HWPROBE_IMA_V)
		RISCV64.HasZbb = isSet(v, riscv_HWPROBE_EXT_ZBB)
		RISCV64.HasZbc = isSet(v, riscv_HWPROBE_EXT_ZBC)
		RISCV64.HasZvbb = isSet(v, riscv_HWPROBE_EXT_ZVBB)
		RISCV64.HasZvbc = isSet(v, riscv_HWPROBE_EXT_ZVBC)
		RISCV64.HasZvkg = isSet(v, riscv_HWPROBE_EXT_ZVKG)
		RISCV64.HasZvkned = isSet(v, riscv_HWPROBE_EXT_ZVKNED)
		RISCV64.HasZvknha = isSet(v, riscv_HWPROBE_EXT_ZVKNHA)
		RISCV64.HasZvknhb = isSet(v, riscv_HWPROBE_EXT_ZVKNHB)
		RISCV64.HasZvksed = isSet(v, riscv_HWPROBE_EXT_ZVKSED)
		RISCV64.HasZvksh = isSet(v, riscv_HWPROBE_EXT_ZVKSH)
		RISCV64.HasZvkt = isSet(v, riscv_HWPROBE_EXT_ZVKT)
	}
	if pairs[1].key != -1 {
		v := pairs[1].value & riscv_HWPROBE_MISALIGNED_MASK
		RISCV64.HasFastMisaligned = v == riscv_HWPROBE_MISALIGNED_FAST
	}
}
