// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
)

var memmoveBits uint8

const (
	// avxSupported indicates that the CPU supports AVX instructions.
	avxSupported = 1 << 0

	// repmovsPreferred indicates that REP MOVSx instruction is more
	// efficient on the CPU.
	repmovsPreferred = 1 << 1
)

// Bit masks for the XCR0 control register, which is accessed via XGETBV[CX=0]
// (or XGETBV[CX=1] to get the logical AND of XCR0 and XINUSE). The low order
// 32 bits are loaded in AX and the high order 32 bits in DX.
const (
	xcr0_AX_SSE       = 1 << 1 // XMM registers
	xcr0_AX_YMM       = 1 << 2 // Upper halves of YMM registers
	xcr0_AX_OpMask    = 1 << 5 // AVX-512 opmask registers K0-K7
	xcr0_AX_ZMM_Hi256 = 1 << 6 // Upper halves of AVX-512 registers Z0-Z15
	xcr0_AX_Hi16_ZMM  = 1 << 7 // AVX-512 registers Z16-Z31
)

func init() {
	// Here we assume that on modern CPUs with both FSRM and ERMS features,
	// copying data blocks of 2KB or larger using the REP MOVSB instruction
	// will be more efficient to avoid having to keep up with CPU generations.
	// Therefore, we may retain a BlockList mechanism to ensure that microarchitectures
	// that do not fit this case may appear in the future.
	// We enable it on Intel CPUs first, and we may support more platforms
	// in the future.
	isERMSNiceCPU := isIntel
	useREPMOV := isERMSNiceCPU && cpu.X86.HasERMS && cpu.X86.HasFSRM
	if cpu.X86.HasAVX {
		memmoveBits |= avxSupported
	}
	if useREPMOV {
		memmoveBits |= repmovsPreferred
	}
}
