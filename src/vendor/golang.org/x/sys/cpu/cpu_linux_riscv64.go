// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

import (
	"syscall"
	"unsafe"
)

// RISC-V extension discovery code for Linux. The approach here is to first try the riscv_hwprobe
// syscall falling back to HWCAP to check for the C extension if riscv_hwprobe is not available.
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
//
// There is a way at runtime to distinguish between versions 0.7.1 and 1.0 of the Vector
// specification by issuing a RVV 1.0 vsetvli instruction and checking the vill bit of the vtype
// register. This check would allow us to safely detect version 1.0 of the Vector extension
// with HWCAP, if riscv_hwprobe were not available. However, the check cannot
// be added until the assembler supports the Vector instructions.
//
// Note the riscv_hwprobe syscall does not suffer from these ambiguities by design as all of the
// extensions it advertises support for are explicitly versioned. It's also worth noting that
// the riscv_hwprobe syscall is the only way to detect multi-letter RISC-V extensions, e.g., Zba.
// These cannot be detected using HWCAP and so riscv_hwprobe must be used to detect the majority
// of RISC-V extensions.
//
// Please see https://docs.kernel.org/arch/riscv/hwprobe.html for more information.

// golang.org/x/sys/cpu is not allowed to depend on golang.org/x/sys/unix so we must
// reproduce the constants, types and functions needed to make the riscv_hwprobe syscall
// here.

const (
	// Copied from golang.org/x/sys/unix/ztypes_linux_riscv64.go.
	riscv_HWPROBE_KEY_IMA_EXT_0   = 0x4
	riscv_HWPROBE_IMA_C           = 0x2
	riscv_HWPROBE_IMA_V           = 0x4
	riscv_HWPROBE_EXT_ZBA         = 0x8
	riscv_HWPROBE_EXT_ZBB         = 0x10
	riscv_HWPROBE_EXT_ZBS         = 0x20
	riscv_HWPROBE_EXT_ZVBB        = 0x20000
	riscv_HWPROBE_EXT_ZVBC        = 0x40000
	riscv_HWPROBE_EXT_ZVKB        = 0x80000
	riscv_HWPROBE_EXT_ZVKG        = 0x100000
	riscv_HWPROBE_EXT_ZVKNED      = 0x200000
	riscv_HWPROBE_EXT_ZVKNHB      = 0x800000
	riscv_HWPROBE_EXT_ZVKSED      = 0x1000000
	riscv_HWPROBE_EXT_ZVKSH       = 0x2000000
	riscv_HWPROBE_EXT_ZVKT        = 0x4000000
	riscv_HWPROBE_KEY_CPUPERF_0   = 0x5
	riscv_HWPROBE_MISALIGNED_FAST = 0x3
	riscv_HWPROBE_MISALIGNED_MASK = 0x7
)

const (
	// sys_RISCV_HWPROBE is copied from golang.org/x/sys/unix/zsysnum_linux_riscv64.go.
	sys_RISCV_HWPROBE = 258
)

// riscvHWProbePairs is copied from golang.org/x/sys/unix/ztypes_linux_riscv64.go.
type riscvHWProbePairs struct {
	key   int64
	value uint64
}

const (
	// CPU features
	hwcap_RISCV_ISA_C = 1 << ('C' - 'A')
)

func doinit() {
	// A slice of key/value pair structures is passed to the RISCVHWProbe syscall. The key
	// field should be initialised with one of the key constants defined above, e.g.,
	// RISCV_HWPROBE_KEY_IMA_EXT_0. The syscall will set the value field to the appropriate value.
	// If the kernel does not recognise a key it will set the key field to -1 and the value field to 0.

	pairs := []riscvHWProbePairs{
		{riscv_HWPROBE_KEY_IMA_EXT_0, 0},
		{riscv_HWPROBE_KEY_CPUPERF_0, 0},
	}

	// This call only indicates that extensions are supported if they are implemented on all cores.
	if riscvHWProbe(pairs, 0) {
		if pairs[0].key != -1 {
			v := uint(pairs[0].value)
			RISCV64.HasC = isSet(v, riscv_HWPROBE_IMA_C)
			RISCV64.HasV = isSet(v, riscv_HWPROBE_IMA_V)
			RISCV64.HasZba = isSet(v, riscv_HWPROBE_EXT_ZBA)
			RISCV64.HasZbb = isSet(v, riscv_HWPROBE_EXT_ZBB)
			RISCV64.HasZbs = isSet(v, riscv_HWPROBE_EXT_ZBS)
			RISCV64.HasZvbb = isSet(v, riscv_HWPROBE_EXT_ZVBB)
			RISCV64.HasZvbc = isSet(v, riscv_HWPROBE_EXT_ZVBC)
			RISCV64.HasZvkb = isSet(v, riscv_HWPROBE_EXT_ZVKB)
			RISCV64.HasZvkg = isSet(v, riscv_HWPROBE_EXT_ZVKG)
			RISCV64.HasZvkt = isSet(v, riscv_HWPROBE_EXT_ZVKT)
			// Cryptography shorthand extensions
			RISCV64.HasZvkn = isSet(v, riscv_HWPROBE_EXT_ZVKNED) &&
				isSet(v, riscv_HWPROBE_EXT_ZVKNHB) && RISCV64.HasZvkb && RISCV64.HasZvkt
			RISCV64.HasZvknc = RISCV64.HasZvkn && RISCV64.HasZvbc
			RISCV64.HasZvkng = RISCV64.HasZvkn && RISCV64.HasZvkg
			RISCV64.HasZvks = isSet(v, riscv_HWPROBE_EXT_ZVKSED) &&
				isSet(v, riscv_HWPROBE_EXT_ZVKSH) && RISCV64.HasZvkb && RISCV64.HasZvkt
			RISCV64.HasZvksc = RISCV64.HasZvks && RISCV64.HasZvbc
			RISCV64.HasZvksg = RISCV64.HasZvks && RISCV64.HasZvkg
		}
		if pairs[1].key != -1 {
			v := pairs[1].value & riscv_HWPROBE_MISALIGNED_MASK
			RISCV64.HasFastMisaligned = v == riscv_HWPROBE_MISALIGNED_FAST
		}
	}

	// Let's double check with HWCAP if the C extension does not appear to be supported.
	// This may happen if we're running on a kernel older than 6.4.

	if !RISCV64.HasC {
		RISCV64.HasC = isSet(hwCap, hwcap_RISCV_ISA_C)
	}
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}

// riscvHWProbe is a simplified version of the generated wrapper function found in
// golang.org/x/sys/unix/zsyscall_linux_riscv64.go. We simplify it by removing the
// cpuCount and cpus parameters which we do not need. We always want to pass 0 for
// these parameters here so the kernel only reports the extensions that are present
// on all cores.
func riscvHWProbe(pairs []riscvHWProbePairs, flags uint) bool {
	var _zero uintptr
	var p0 unsafe.Pointer
	if len(pairs) > 0 {
		p0 = unsafe.Pointer(&pairs[0])
	} else {
		p0 = unsafe.Pointer(&_zero)
	}

	_, _, e1 := syscall.Syscall6(sys_RISCV_HWPROBE, uintptr(p0), uintptr(len(pairs)), uintptr(0), uintptr(0), uintptr(flags), 0)
	return e1 == 0
}
