// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64 || amd64p32

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
		{Name: "amxtile", Feature: &X86.HasAMXTile},
		{Name: "amxint8", Feature: &X86.HasAMXInt8},
		{Name: "amxbf16", Feature: &X86.HasAMXBF16},
		{Name: "bmi1", Feature: &X86.HasBMI1},
		{Name: "bmi2", Feature: &X86.HasBMI2},
		{Name: "cx16", Feature: &X86.HasCX16},
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
		{Name: "avxifma", Feature: &X86.HasAVXIFMA},
		{Name: "avxvnni", Feature: &X86.HasAVXVNNI},
		{Name: "avxvnniint8", Feature: &X86.HasAVXVNNIInt8},

		// These capabilities should always be enabled on amd64:
		{Name: "sse2", Feature: &X86.HasSSE2, Required: runtime.GOARCH == "amd64"},
	}
}

func archInit() {

	// From internal/cpu
	const (
		// eax bits
		cpuid_AVXVNNI = 1 << 4

		// ecx bits
		cpuid_SSE3            = 1 << 0
		cpuid_PCLMULQDQ       = 1 << 1
		cpuid_AVX512VBMI      = 1 << 1
		cpuid_AVX512VBMI2     = 1 << 6
		cpuid_SSSE3           = 1 << 9
		cpuid_AVX512GFNI      = 1 << 8
		cpuid_AVX512VAES      = 1 << 9
		cpuid_AVX512VNNI      = 1 << 11
		cpuid_AVX512BITALG    = 1 << 12
		cpuid_FMA             = 1 << 12
		cpuid_AVX512VPOPCNTDQ = 1 << 14
		cpuid_SSE41           = 1 << 19
		cpuid_SSE42           = 1 << 20
		cpuid_POPCNT          = 1 << 23
		cpuid_AES             = 1 << 25
		cpuid_OSXSAVE         = 1 << 27
		cpuid_AVX             = 1 << 28

		// "Extended Feature Flag" bits returned in EBX for CPUID EAX=0x7 ECX=0x0
		cpuid_BMI1     = 1 << 3
		cpuid_AVX2     = 1 << 5
		cpuid_BMI2     = 1 << 8
		cpuid_ERMS     = 1 << 9
		cpuid_AVX512F  = 1 << 16
		cpuid_AVX512DQ = 1 << 17
		cpuid_ADX      = 1 << 19
		cpuid_AVX512CD = 1 << 28
		cpuid_SHA      = 1 << 29
		cpuid_AVX512BW = 1 << 30
		cpuid_AVX512VL = 1 << 31

		// "Extended Feature Flag" bits returned in ECX for CPUID EAX=0x7 ECX=0x0
		cpuid_AVX512_VBMI      = 1 << 1
		cpuid_AVX512_VBMI2     = 1 << 6
		cpuid_GFNI             = 1 << 8
		cpuid_AVX512VPCLMULQDQ = 1 << 10
		cpuid_AVX512_BITALG    = 1 << 12

		// edx bits
		cpuid_FSRM = 1 << 4
		// edx bits for CPUID 0x80000001
		cpuid_RDTSCP = 1 << 27
	)
	// Additional constants not in internal/cpu
	const (
		// eax=1: edx
		cpuid_SSE2 = 1 << 26
		// eax=1: ecx
		cpuid_CX16   = 1 << 13
		cpuid_RDRAND = 1 << 30
		// eax=7,ecx=0: ebx
		cpuid_RDSEED     = 1 << 18
		cpuid_AVX512IFMA = 1 << 21
		cpuid_AVX512PF   = 1 << 26
		cpuid_AVX512ER   = 1 << 27
		// eax=7,ecx=0: edx
		cpuid_AVX5124VNNIW = 1 << 2
		cpuid_AVX5124FMAPS = 1 << 3
		cpuid_AMXBF16      = 1 << 22
		cpuid_AMXTile      = 1 << 24
		cpuid_AMXInt8      = 1 << 25
		// eax=7,ecx=1: eax
		cpuid_AVX512BF16 = 1 << 5
		cpuid_AVXIFMA    = 1 << 23
		// eax=7,ecx=1: edx
		cpuid_AVXVNNIInt8 = 1 << 4
	)

	Initialized = true

	maxID, _, _, _ := cpuid(0, 0)

	if maxID < 1 {
		return
	}

	_, _, ecx1, edx1 := cpuid(1, 0)
	X86.HasSSE2 = isSet(edx1, cpuid_SSE2)

	X86.HasSSE3 = isSet(ecx1, cpuid_SSE3)
	X86.HasPCLMULQDQ = isSet(ecx1, cpuid_PCLMULQDQ)
	X86.HasSSSE3 = isSet(ecx1, cpuid_SSSE3)
	X86.HasFMA = isSet(ecx1, cpuid_FMA)
	X86.HasCX16 = isSet(ecx1, cpuid_CX16)
	X86.HasSSE41 = isSet(ecx1, cpuid_SSE41)
	X86.HasSSE42 = isSet(ecx1, cpuid_SSE42)
	X86.HasPOPCNT = isSet(ecx1, cpuid_POPCNT)
	X86.HasAES = isSet(ecx1, cpuid_AES)
	X86.HasOSXSAVE = isSet(ecx1, cpuid_OSXSAVE)
	X86.HasRDRAND = isSet(ecx1, cpuid_RDRAND)

	var osSupportsAVX, osSupportsAVX512 bool
	// For XGETBV, OSXSAVE bit is required and sufficient.
	if X86.HasOSXSAVE {
		eax, _ := xgetbv()
		// Check if XMM and YMM registers have OS support.
		osSupportsAVX = isSet(eax, 1<<1) && isSet(eax, 1<<2)

		if runtime.GOOS == "darwin" {
			// Darwin requires special AVX512 checks, see cpu_darwin_x86.go
			osSupportsAVX512 = osSupportsAVX && darwinSupportsAVX512()
		} else {
			// Check if OPMASK and ZMM registers have OS support.
			osSupportsAVX512 = osSupportsAVX && isSet(eax, 1<<5) && isSet(eax, 1<<6) && isSet(eax, 1<<7)
		}
	}

	X86.HasAVX = isSet(ecx1, cpuid_AVX) && osSupportsAVX

	if maxID < 7 {
		return
	}

	eax7, ebx7, ecx7, edx7 := cpuid(7, 0)
	X86.HasBMI1 = isSet(ebx7, cpuid_BMI1)
	X86.HasAVX2 = isSet(ebx7, cpuid_AVX2) && osSupportsAVX
	X86.HasBMI2 = isSet(ebx7, cpuid_BMI2)
	X86.HasERMS = isSet(ebx7, cpuid_ERMS)
	X86.HasRDSEED = isSet(ebx7, cpuid_RDSEED)
	X86.HasADX = isSet(ebx7, cpuid_ADX)

	X86.HasAVX512 = isSet(ebx7, cpuid_AVX512F) && osSupportsAVX512 // Because avx-512 foundation is the core required extension
	if X86.HasAVX512 {
		X86.HasAVX512F = true
		X86.HasAVX512CD = isSet(ebx7, cpuid_AVX512CD)
		X86.HasAVX512ER = isSet(ebx7, cpuid_AVX512ER)
		X86.HasAVX512PF = isSet(ebx7, cpuid_AVX512PF)
		X86.HasAVX512VL = isSet(ebx7, cpuid_AVX512VL)
		X86.HasAVX512BW = isSet(ebx7, cpuid_AVX512BW)
		X86.HasAVX512DQ = isSet(ebx7, cpuid_AVX512DQ)
		X86.HasAVX512IFMA = isSet(ebx7, cpuid_AVX512IFMA)
		X86.HasAVX512VBMI = isSet(ecx7, cpuid_AVX512_VBMI)
		X86.HasAVX5124VNNIW = isSet(edx7, cpuid_AVX5124VNNIW)
		X86.HasAVX5124FMAPS = isSet(edx7, cpuid_AVX5124FMAPS)
		X86.HasAVX512VPOPCNTDQ = isSet(ecx7, cpuid_AVX512VPOPCNTDQ)
		X86.HasAVX512VPCLMULQDQ = isSet(ecx7, cpuid_AVX512VPCLMULQDQ)
		X86.HasAVX512VNNI = isSet(ecx7, cpuid_AVX512VNNI)
		X86.HasAVX512GFNI = isSet(ecx7, cpuid_AVX512GFNI)
		X86.HasAVX512VAES = isSet(ecx7, cpuid_AVX512VAES)
		X86.HasAVX512VBMI2 = isSet(ecx7, cpuid_AVX512VBMI2)
		X86.HasAVX512BITALG = isSet(ecx7, cpuid_AVX512BITALG)
	}

	X86.HasAMXTile = isSet(edx7, cpuid_AMXTile)
	X86.HasAMXInt8 = isSet(edx7, cpuid_AMXInt8)
	X86.HasAMXBF16 = isSet(edx7, cpuid_AMXBF16)

	// These features depend on the second level of extended features.
	if eax7 >= 1 {
		eax71, _, _, edx71 := cpuid(7, 1)
		if X86.HasAVX512 {
			X86.HasAVX512BF16 = isSet(eax71, cpuid_AVX512BF16)
		}
		if X86.HasAVX {
			X86.HasAVXIFMA = isSet(eax71, cpuid_AVXIFMA)
			X86.HasAVXVNNI = isSet(eax71, cpuid_AVXVNNI)
			X86.HasAVXVNNIInt8 = isSet(edx71, cpuid_AVXVNNIInt8)
		}
	}
}

func isSet(hwc uint32, value uint32) bool {
	return hwc&value != 0
}
