// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64 && linux

package cpu

import _ "unsafe"

// RISC-V extension discovery code for Linux.
//
// A note on detection of the Vector extension using HWCAP.
//
// Support for the Vector extension version 1.0 was added to the Linux kernel in release 6.5.
// Support for the riscv_hwprobe syscall was added in 6.4. It follows that if the riscv_hwprobe
// syscall is not available then neither is the Vector extension (which needs kernel support).
// The riscv_hwprobe syscall should then be all we need to detect the Vector extension.
// However, some RISC-V board manufacturers ship boards with an older kernel on top of which
// they have back-ported various versions of the Vector extension patches but not the riscv_hwprobe
// patches. These kernels advertise support for the Vector extension using HWCAP. Falling
// back to HWCAP to detect the Vector extension, if riscv_hwprobe is not available, or simply not
// bothering with riscv_hwprobe at all and just using HWCAP may then seem like an attractive option.
//
// Unfortunately, simply checking the 'V' bit in AT_HWCAP will not work as this bit is used by
// RISC-V board and cloud instance providers to mean different things. The Lichee Pi 4A board
// and the Scaleway RV1 cloud instances use the 'V' bit to advertise their support for the unratified
// 0.7.1 version of the Vector Specification. The Banana Pi BPI-F3 and the CanMV-K230 board use
// it to advertise support for 1.0 of the Vector extension. Versions 0.7.1 and 1.0 of the Vector
// extension are binary incompatible. HWCAP can then not be used in isolation to populate the
// HasV field as this field indicates that the underlying CPU is compatible with RVV 1.0.
// Go will only support the ratified versions >= 1.0 and so any vector code it might generate
// would crash on a Scaleway RV1 instance or a Lichee Pi 4a, if allowed to run.
//
// There is a way at runtime to distinguish between versions 0.7.1 and 1.0 of the Vector
// specification by issuing a RVV 1.0 vsetvli instruction and checking the vill bit of the vtype
// register. This check would allow us to safely detect version 1.0 of the Vector extension
// with HWCAP, if riscv_hwprobe were not available. However, the check cannot
// be added until the assembler supports the Vector instructions.
//
// Note the riscv_hwprobe syscall does not suffer from these ambiguities by design as all of the
// extensions it advertises support for are explicitly versioned. It's also worth noting that
// the riscv_hwprobe syscall is the only way to detect multi-letter RISC-V extensions, e.g., Zvbb.
// These cannot be detected using HWCAP and so riscv_hwprobe must be used to detect the majority
// of RISC-V extensions.
//
// Please see https://docs.kernel.org/arch/riscv/hwprobe.html for more information.

const (
	// Copied from golang.org/x/sys/unix/ztypes_linux_riscv64.go.
	riscv_HWPROBE_KEY_IMA_EXT_0   = 0x4
	riscv_HWPROBE_IMA_V           = 0x4
	riscv_HWPROBE_EXT_ZBB         = 0x10
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
	}
	if pairs[1].key != -1 {
		v := pairs[1].value & riscv_HWPROBE_MISALIGNED_MASK
		RISCV64.HasFastMisaligned = v == riscv_HWPROBE_MISALIGNED_FAST
	}
}
