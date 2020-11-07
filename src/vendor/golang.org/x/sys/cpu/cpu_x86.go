// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 amd64 amd64p32

package cpu

import "runtime"

const cacheLineSize = 64

func initOptions() {
	options = []option{
		{Name: "adx", Feature: &X86.HasADX},
		{Name: "aes", Feature: &X86.HasAES},
		{Name: "avx", Feature: &X86.HasAVX},
		{Name: "avx2", Feature: &X86.HasAVX2},
		{Name: "avx512", Feature: &X86.HasAVX512},
		{Name: "avx512f", Feature: &X86.HasAVX512F},
		{Name: "avx512cd", Feature: &X86.HasAVX512CD},
		{Name: "avx512er", Feature: &X86.HasAVX512ER},
		{Name: "avx512pf", Feature: &X86.HasAVX512PF},
		{Name: "avx512vl", Feature: &X86.HasAVX512VL},
		{Name: "avx512bw", Feature: &X86.HasAVX512BW},
		{Name: "avx512dq", Feature: &X86.HasAVX512DQ},
		{Name: "avx512ifma", Feature: &X86.HasAVX512IFMA},
		{Name: "avx512vbmi", Feature: &X86.HasAVX512VBMI},
		{Name: "avx512vnniw", Feature: &X86.HasAVX5124VNNIW},
		{Name: "avx5124fmaps", Feature: &X86.HasAVX5124FMAPS},
		{Name: "avx512vpopcntdq", Feature: &X86.HasAVX512VPOPCNTDQ},
		{Name: "avx512vpclmulqdq", Feature: &X86.HasAVX512VPCLMULQDQ},
		{Name: "avx512vnni", Feature: &X86.HasAVX512VNNI},
		{Name: "avx512gfni", Feature: &X86.HasAVX512GFNI},
		{Name: "avx512vaes", Feature: &X86.HasAVX512VAES},
		{Name: "avx512vbmi2", Feature: &X86.HasAVX512VBMI2},
		{Name: "avx512bitalg", Feature: &X86.HasAVX512BITALG},
		{Name: "avx512bf16", Feature: &X86.HasAVX512BF16},
		{Name: "bmi1", Feature: &X86.HasBMI1},
		{Name: "bmi2", Feature: &X86.HasBMI2},
		{Name: "erms", Feature: &X86.HasERMS},
		{Name: "fma", Feature: &X86.HasFMA},
		{Name: "osxsave", Feature: &X86.HasOSXSAVE},
		{Name: "pclmulqdq", Feature: &X86.HasPCLMULQDQ},
		{Name: "popcnt", Feature: &X86.HasPOPCNT},
		{Name: "rdrand", Feature: &X86.HasRDRAND},
		{Name: "rdseed", Feature: &X86.HasRDSEED},
		{Name: "sse3", Feature: &X86.HasSSE3},
		{Name: "sse41", Feature: &X86.HasSSE41},
		{Name: "sse42", Feature: &X86.HasSSE42},
		{Name: "ssse3", Feature: &X86.HasSSSE3},

		// These capabilities should always be enabled on amd64:
		{Name: "sse2", Feature: &X86.HasSSE2, Required: runtime.GOARCH == "amd64"},
	}
}

func archInit() {

	Initialized = true

	maxID, _, _, _ := cpuid(0, 0)

	if maxID < 1 {
		return
	}

	_, _, ecx1, edx1 := cpuid(1, 0)
	X86.HasSSE2 = isSet(26, edx1)

	X86.HasSSE3 = isSet(0, ecx1)
	X86.HasPCLMULQDQ = isSet(1, ecx1)
	X86.HasSSSE3 = isSet(9, ecx1)
	X86.HasFMA = isSet(12, ecx1)
	X86.HasSSE41 = isSet(19, ecx1)
	X86.HasSSE42 = isSet(20, ecx1)
	X86.HasPOPCNT = isSet(23, ecx1)
	X86.HasAES = isSet(25, ecx1)
	X86.HasOSXSAVE = isSet(27, ecx1)
	X86.HasRDRAND = isSet(30, ecx1)

	var osSupportsAVX, osSupportsAVX512 bool
	// For XGETBV, OSXSAVE bit is required and sufficient.
	if X86.HasOSXSAVE {
		eax, _ := xgetbv()
		// Check if XMM and YMM registers have OS support.
		osSupportsAVX = isSet(1, eax) && isSet(2, eax)

		// Check if OPMASK and ZMM registers have OS support.
		osSupportsAVX512 = osSupportsAVX && isSet(5, eax) && isSet(6, eax) && isSet(7, eax)
	}

	X86.HasAVX = isSet(28, ecx1) && osSupportsAVX

	if maxID < 7 {
		return
	}

	_, ebx7, ecx7, edx7 := cpuid(7, 0)
	X86.HasBMI1 = isSet(3, ebx7)
	X86.HasAVX2 = isSet(5, ebx7) && osSupportsAVX
	X86.HasBMI2 = isSet(8, ebx7)
	X86.HasERMS = isSet(9, ebx7)
	X86.HasRDSEED = isSet(18, ebx7)
	X86.HasADX = isSet(19, ebx7)

	X86.HasAVX512 = isSet(16, ebx7) && osSupportsAVX512 // Because avx-512 foundation is the core required extension
	if X86.HasAVX512 {
		X86.HasAVX512F = true
		X86.HasAVX512CD = isSet(28, ebx7)
		X86.HasAVX512ER = isSet(27, ebx7)
		X86.HasAVX512PF = isSet(26, ebx7)
		X86.HasAVX512VL = isSet(31, ebx7)
		X86.HasAVX512BW = isSet(30, ebx7)
		X86.HasAVX512DQ = isSet(17, ebx7)
		X86.HasAVX512IFMA = isSet(21, ebx7)
		X86.HasAVX512VBMI = isSet(1, ecx7)
		X86.HasAVX5124VNNIW = isSet(2, edx7)
		X86.HasAVX5124FMAPS = isSet(3, edx7)
		X86.HasAVX512VPOPCNTDQ = isSet(14, ecx7)
		X86.HasAVX512VPCLMULQDQ = isSet(10, ecx7)
		X86.HasAVX512VNNI = isSet(11, ecx7)
		X86.HasAVX512GFNI = isSet(8, ecx7)
		X86.HasAVX512VAES = isSet(9, ecx7)
		X86.HasAVX512VBMI2 = isSet(6, ecx7)
		X86.HasAVX512BITALG = isSet(12, ecx7)

		eax71, _, _, _ := cpuid(7, 1)
		X86.HasAVX512BF16 = isSet(5, eax71)
	}
}

func isSet(bitpos uint, value uint32) bool {
	return value&(1<<bitpos) != 0
}
