// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64

package cpu

const CacheLinePadSize = 64

// cpuid is implemented in cpu_x86.s.
func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)

// xgetbv with ecx = 0 is implemented in cpu_x86.s.
func xgetbv() (eax, edx uint32)

// getGOAMD64level is implemented in cpu_x86.s. Returns number in [1,4].
func getGOAMD64level() int32

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

var maxExtendedFunctionInformation uint32

func doinit() {
	options = []option{
		{Name: "adx", Feature: &X86.HasADX},
		{Name: "aes", Feature: &X86.HasAES},
		{Name: "erms", Feature: &X86.HasERMS},
		{Name: "fsrm", Feature: &X86.HasFSRM},
		{Name: "pclmulqdq", Feature: &X86.HasPCLMULQDQ},
		{Name: "rdtscp", Feature: &X86.HasRDTSCP},
		{Name: "sha", Feature: &X86.HasSHA},
		{Name: "vpclmulqdq", Feature: &X86.HasAVX512VPCLMULQDQ},
	}
	level := getGOAMD64level()
	if level < 2 {
		// These options are required at level 2. At lower levels
		// they can be turned off.
		options = append(options,
			option{Name: "popcnt", Feature: &X86.HasPOPCNT},
			option{Name: "sse3", Feature: &X86.HasSSE3},
			option{Name: "sse41", Feature: &X86.HasSSE41},
			option{Name: "sse42", Feature: &X86.HasSSE42},
			option{Name: "ssse3", Feature: &X86.HasSSSE3})
	}
	if level < 3 {
		// These options are required at level 3. At lower levels
		// they can be turned off.
		options = append(options,
			option{Name: "avx", Feature: &X86.HasAVX},
			option{Name: "avx2", Feature: &X86.HasAVX2},
			option{Name: "bmi1", Feature: &X86.HasBMI1},
			option{Name: "bmi2", Feature: &X86.HasBMI2},
			option{Name: "fma", Feature: &X86.HasFMA})
	}
	if level < 4 {
		// These options are required at level 4. At lower levels
		// they can be turned off.
		options = append(options,
			option{Name: "avx512f", Feature: &X86.HasAVX512F},
			option{Name: "avx512cd", Feature: &X86.HasAVX512CD},
			option{Name: "avx512bw", Feature: &X86.HasAVX512BW},
			option{Name: "avx512dq", Feature: &X86.HasAVX512DQ},
			option{Name: "avx512vl", Feature: &X86.HasAVX512VL},
		)
	}

	maxID, _, _, _ := cpuid(0, 0)

	if maxID < 1 {
		return
	}

	maxExtendedFunctionInformation, _, _, _ = cpuid(0x80000000, 0)

	_, _, ecx1, _ := cpuid(1, 0)

	X86.HasSSE3 = isSet(ecx1, cpuid_SSE3)
	X86.HasPCLMULQDQ = isSet(ecx1, cpuid_PCLMULQDQ)
	X86.HasSSSE3 = isSet(ecx1, cpuid_SSSE3)
	X86.HasSSE41 = isSet(ecx1, cpuid_SSE41)
	X86.HasSSE42 = isSet(ecx1, cpuid_SSE42)
	X86.HasPOPCNT = isSet(ecx1, cpuid_POPCNT)
	X86.HasAES = isSet(ecx1, cpuid_AES)

	// OSXSAVE can be false when using older Operating Systems
	// or when explicitly disabled on newer Operating Systems by
	// e.g. setting the xsavedisable boot option on Windows 10.
	X86.HasOSXSAVE = isSet(ecx1, cpuid_OSXSAVE)

	// The FMA instruction set extension only has VEX prefixed instructions.
	// VEX prefixed instructions require OSXSAVE to be enabled.
	// See Intel 64 and IA-32 Architecture Software Developer’s Manual Volume 2
	// Section 2.4 "AVX and SSE Instruction Exception Specification"
	X86.HasFMA = isSet(ecx1, cpuid_FMA) && X86.HasOSXSAVE

	osSupportsAVX := false
	osSupportsAVX512 := false
	// For XGETBV, OSXSAVE bit is required and sufficient.
	if X86.HasOSXSAVE {
		eax, _ := xgetbv()
		// Check if XMM and YMM registers have OS support.
		osSupportsAVX = isSet(eax, 1<<1) && isSet(eax, 1<<2)

		// AVX512 detection does not work on Darwin,
		// see https://github.com/golang/go/issues/49233
		//
		// Check if opmask, ZMMhi256 and Hi16_ZMM have OS support.
		osSupportsAVX512 = osSupportsAVX && isSet(eax, 1<<5) && isSet(eax, 1<<6) && isSet(eax, 1<<7)
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
	X86.HasADX = isSet(ebx7, cpuid_ADX)
	X86.HasSHA = isSet(ebx7, cpuid_SHA)

	X86.HasAVX512F = isSet(ebx7, cpuid_AVX512F) && osSupportsAVX512
	if X86.HasAVX512F {
		X86.HasAVX512CD = isSet(ebx7, cpuid_AVX512CD)
		X86.HasAVX512BW = isSet(ebx7, cpuid_AVX512BW)
		X86.HasAVX512DQ = isSet(ebx7, cpuid_AVX512DQ)
		X86.HasAVX512VL = isSet(ebx7, cpuid_AVX512VL)
		X86.HasAVX512GFNI = isSet(ecx7, cpuid_AVX512GFNI)
		X86.HasAVX512BITALG = isSet(ecx7, cpuid_AVX512BITALG)
		X86.HasAVX512VPOPCNTDQ = isSet(ecx7, cpuid_AVX512VPOPCNTDQ)
		X86.HasAVX512VBMI = isSet(ecx7, cpuid_AVX512VBMI)
		X86.HasAVX512VBMI2 = isSet(ecx7, cpuid_AVX512VBMI2)
		X86.HasAVX512VNNI = isSet(ecx7, cpuid_AVX512VNNI)
		X86.HasAVX512VPCLMULQDQ = isSet(ecx7, cpuid_AVX512VPCLMULQDQ)
		X86.HasAVX512VBMI = isSet(ecx7, cpuid_AVX512_VBMI)
		X86.HasAVX512VBMI2 = isSet(ecx7, cpuid_AVX512_VBMI2)
		X86.HasGFNI = isSet(ecx7, cpuid_GFNI)
		X86.HasAVX512BITALG = isSet(ecx7, cpuid_AVX512_BITALG)
	}

	X86.HasFSRM = isSet(edx7, cpuid_FSRM)

	var maxExtendedInformation uint32
	maxExtendedInformation, _, _, _ = cpuid(0x80000000, 0)

	if maxExtendedInformation < 0x80000001 {
		return
	}

	_, _, _, edxExt1 := cpuid(0x80000001, 0)
	X86.HasRDTSCP = isSet(edxExt1, cpuid_RDTSCP)

	doDerived = func() {
		// Rather than carefully gating on fundamental AVX-512 features, we have
		// a virtual "AVX512" feature that captures F+CD+BW+DQ+VL. BW, DQ, and
		// VL have a huge effect on which AVX-512 instructions are available,
		// and these have all been supported on everything except the earliest
		// Phi chips with AVX-512. No CPU has had CD without F, so we include
		// it. GOAMD64=v4 also implies exactly this set, and these are all
		// included in AVX10.1.
		X86.HasAVX512 = X86.HasAVX512F && X86.HasAVX512CD && X86.HasAVX512BW && X86.HasAVX512DQ && X86.HasAVX512VL
	}

	if eax7 >= 1 {
		eax71, _, _, _ := cpuid(7, 1)
		if X86.HasAVX {
			X86.HasAVXVNNI = isSet(4, eax71)
		}
	}
}

func isSet(hwc uint32, value uint32) bool {
	return hwc&value != 0
}

// Name returns the CPU name given by the vendor.
// If the CPU name can not be determined an
// empty string is returned.
func Name() string {
	if maxExtendedFunctionInformation < 0x80000004 {
		return ""
	}

	data := make([]byte, 0, 3*4*4)

	var eax, ebx, ecx, edx uint32
	eax, ebx, ecx, edx = cpuid(0x80000002, 0)
	data = appendBytes(data, eax, ebx, ecx, edx)
	eax, ebx, ecx, edx = cpuid(0x80000003, 0)
	data = appendBytes(data, eax, ebx, ecx, edx)
	eax, ebx, ecx, edx = cpuid(0x80000004, 0)
	data = appendBytes(data, eax, ebx, ecx, edx)

	// Trim leading spaces.
	for len(data) > 0 && data[0] == ' ' {
		data = data[1:]
	}

	// Trim tail after and including the first null byte.
	for i, c := range data {
		if c == '\x00' {
			data = data[:i]
			break
		}
	}

	return string(data)
}

func appendBytes(b []byte, args ...uint32) []byte {
	for _, arg := range args {
		b = append(b,
			byte((arg >> 0)),
			byte((arg >> 8)),
			byte((arg >> 16)),
			byte((arg >> 24)))
	}
	return b
}
