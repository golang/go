// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 amd64 amd64p32

package cpu

const CacheLineSize = 64

// cpuid is implemented in cpu_x86.s.
func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)

// xgetbv with ecx = 0 is implemented in cpu_x86.s.
func xgetbv() (eax, edx uint32)

func init() {
	maxId, _, _, _ := cpuid(0, 0)

	if maxId < 1 {
		return
	}

	_, _, ecx1, edx1 := cpuid(1, 0)
	X86.HasSSE2 = isSet(26, edx1)

	X86.HasSSE3 = isSet(0, ecx1)
	X86.HasPCLMULQDQ = isSet(1, ecx1)
	X86.HasSSSE3 = isSet(9, ecx1)
	X86.HasSSE41 = isSet(19, ecx1)
	X86.HasSSE42 = isSet(20, ecx1)
	X86.HasPOPCNT = isSet(23, ecx1)
	X86.HasAES = isSet(25, ecx1)
	X86.HasOSXSAVE = isSet(27, ecx1)

	osSupportsAVX := false
	// For XGETBV, OSXSAVE bit is required and sufficient.
	if X86.HasOSXSAVE {
		eax, _ := xgetbv()
		// Check if XMM and YMM registers have OS support.
		osSupportsAVX = isSet(1, eax) && isSet(2, eax)
	}

	X86.HasAVX = isSet(28, ecx1) && osSupportsAVX

	if maxId < 7 {
		return
	}

	_, ebx7, _, _ := cpuid(7, 0)
	X86.HasBMI1 = isSet(3, ebx7)
	X86.HasAVX2 = isSet(5, ebx7) && osSupportsAVX
	X86.HasBMI2 = isSet(8, ebx7)
	X86.HasERMS = isSet(9, ebx7)
}

func isSet(bitpos uint, value uint32) bool {
	return value&(1<<bitpos) != 0
}
