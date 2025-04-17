// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cpu implements processor feature detection for
// various CPU architectures.
package cpu

import (
	"os"
	"strings"
)

// Initialized reports whether the CPU features were initialized.
//
// For some GOOS/GOARCH combinations initialization of the CPU features depends
// on reading an operating specific file, e.g. /proc/self/auxv on linux/arm
// Initialized will report false if reading the file fails.
var Initialized bool

// CacheLinePad is used to pad structs to avoid false sharing.
type CacheLinePad struct{ _ [cacheLineSize]byte }

// X86 contains the supported CPU features of the
// current X86/AMD64 platform. If the current platform
// is not X86/AMD64 then all feature flags are false.
//
// X86 is padded to avoid false sharing. Further the HasAVX
// and HasAVX2 are only set if the OS supports XMM and YMM
// registers in addition to the CPUID feature bit being set.
var X86 struct {
	_                   CacheLinePad
	HasAES              bool // AES hardware implementation (AES NI)
	HasADX              bool // Multi-precision add-carry instruction extensions
	HasAVX              bool // Advanced vector extension
	HasAVX2             bool // Advanced vector extension 2
	HasAVX512           bool // Advanced vector extension 512
	HasAVX512F          bool // Advanced vector extension 512 Foundation Instructions
	HasAVX512CD         bool // Advanced vector extension 512 Conflict Detection Instructions
	HasAVX512ER         bool // Advanced vector extension 512 Exponential and Reciprocal Instructions
	HasAVX512PF         bool // Advanced vector extension 512 Prefetch Instructions
	HasAVX512VL         bool // Advanced vector extension 512 Vector Length Extensions
	HasAVX512BW         bool // Advanced vector extension 512 Byte and Word Instructions
	HasAVX512DQ         bool // Advanced vector extension 512 Doubleword and Quadword Instructions
	HasAVX512IFMA       bool // Advanced vector extension 512 Integer Fused Multiply Add
	HasAVX512VBMI       bool // Advanced vector extension 512 Vector Byte Manipulation Instructions
	HasAVX5124VNNIW     bool // Advanced vector extension 512 Vector Neural Network Instructions Word variable precision
	HasAVX5124FMAPS     bool // Advanced vector extension 512 Fused Multiply Accumulation Packed Single precision
	HasAVX512VPOPCNTDQ  bool // Advanced vector extension 512 Double and quad word population count instructions
	HasAVX512VPCLMULQDQ bool // Advanced vector extension 512 Vector carry-less multiply operations
	HasAVX512VNNI       bool // Advanced vector extension 512 Vector Neural Network Instructions
	HasAVX512GFNI       bool // Advanced vector extension 512 Galois field New Instructions
	HasAVX512VAES       bool // Advanced vector extension 512 Vector AES instructions
	HasAVX512VBMI2      bool // Advanced vector extension 512 Vector Byte Manipulation Instructions 2
	HasAVX512BITALG     bool // Advanced vector extension 512 Bit Algorithms
	HasAVX512BF16       bool // Advanced vector extension 512 BFloat16 Instructions
	HasAMXTile          bool // Advanced Matrix Extension Tile instructions
	HasAMXInt8          bool // Advanced Matrix Extension Int8 instructions
	HasAMXBF16          bool // Advanced Matrix Extension BFloat16 instructions
	HasBMI1             bool // Bit manipulation instruction set 1
	HasBMI2             bool // Bit manipulation instruction set 2
	HasCX16             bool // Compare and exchange 16 Bytes
	HasERMS             bool // Enhanced REP for MOVSB and STOSB
	HasFMA              bool // Fused-multiply-add instructions
	HasOSXSAVE          bool // OS supports XSAVE/XRESTOR for saving/restoring XMM registers.
	HasPCLMULQDQ        bool // PCLMULQDQ instruction - most often used for AES-GCM
	HasPOPCNT           bool // Hamming weight instruction POPCNT.
	HasRDRAND           bool // RDRAND instruction (on-chip random number generator)
	HasRDSEED           bool // RDSEED instruction (on-chip random number generator)
	HasSSE2             bool // Streaming SIMD extension 2 (always available on amd64)
	HasSSE3             bool // Streaming SIMD extension 3
	HasSSSE3            bool // Supplemental streaming SIMD extension 3
	HasSSE41            bool // Streaming SIMD extension 4 and 4.1
	HasSSE42            bool // Streaming SIMD extension 4 and 4.2
	HasAVXIFMA          bool // Advanced vector extension Integer Fused Multiply Add
	HasAVXVNNI          bool // Advanced vector extension Vector Neural Network Instructions
	HasAVXVNNIInt8      bool // Advanced vector extension Vector Neural Network Int8 instructions
	_                   CacheLinePad
}

// ARM64 contains the supported CPU features of the
// current ARMv8(aarch64) platform. If the current platform
// is not arm64 then all feature flags are false.
var ARM64 struct {
	_           CacheLinePad
	HasFP       bool // Floating-point instruction set (always available)
	HasASIMD    bool // Advanced SIMD (always available)
	HasEVTSTRM  bool // Event stream support
	HasAES      bool // AES hardware implementation
	HasPMULL    bool // Polynomial multiplication instruction set
	HasSHA1     bool // SHA1 hardware implementation
	HasSHA2     bool // SHA2 hardware implementation
	HasCRC32    bool // CRC32 hardware implementation
	HasATOMICS  bool // Atomic memory operation instruction set
	HasFPHP     bool // Half precision floating-point instruction set
	HasASIMDHP  bool // Advanced SIMD half precision instruction set
	HasCPUID    bool // CPUID identification scheme registers
	HasASIMDRDM bool // Rounding double multiply add/subtract instruction set
	HasJSCVT    bool // Javascript conversion from floating-point to integer
	HasFCMA     bool // Floating-point multiplication and addition of complex numbers
	HasLRCPC    bool // Release Consistent processor consistent support
	HasDCPOP    bool // Persistent memory support
	HasSHA3     bool // SHA3 hardware implementation
	HasSM3      bool // SM3 hardware implementation
	HasSM4      bool // SM4 hardware implementation
	HasASIMDDP  bool // Advanced SIMD double precision instruction set
	HasSHA512   bool // SHA512 hardware implementation
	HasSVE      bool // Scalable Vector Extensions
	HasSVE2     bool // Scalable Vector Extensions 2
	HasASIMDFHM bool // Advanced SIMD multiplication FP16 to FP32
	HasDIT      bool // Data Independent Timing support
	HasI8MM     bool // Advanced SIMD Int8 matrix multiplication instructions
	_           CacheLinePad
}

// ARM contains the supported CPU features of the current ARM (32-bit) platform.
// All feature flags are false if:
//  1. the current platform is not arm, or
//  2. the current operating system is not Linux.
var ARM struct {
	_           CacheLinePad
	HasSWP      bool // SWP instruction support
	HasHALF     bool // Half-word load and store support
	HasTHUMB    bool // ARM Thumb instruction set
	Has26BIT    bool // Address space limited to 26-bits
	HasFASTMUL  bool // 32-bit operand, 64-bit result multiplication support
	HasFPA      bool // Floating point arithmetic support
	HasVFP      bool // Vector floating point support
	HasEDSP     bool // DSP Extensions support
	HasJAVA     bool // Java instruction set
	HasIWMMXT   bool // Intel Wireless MMX technology support
	HasCRUNCH   bool // MaverickCrunch context switching and handling
	HasTHUMBEE  bool // Thumb EE instruction set
	HasNEON     bool // NEON instruction set
	HasVFPv3    bool // Vector floating point version 3 support
	HasVFPv3D16 bool // Vector floating point version 3 D8-D15
	HasTLS      bool // Thread local storage support
	HasVFPv4    bool // Vector floating point version 4 support
	HasIDIVA    bool // Integer divide instruction support in ARM mode
	HasIDIVT    bool // Integer divide instruction support in Thumb mode
	HasVFPD32   bool // Vector floating point version 3 D15-D31
	HasLPAE     bool // Large Physical Address Extensions
	HasEVTSTRM  bool // Event stream support
	HasAES      bool // AES hardware implementation
	HasPMULL    bool // Polynomial multiplication instruction set
	HasSHA1     bool // SHA1 hardware implementation
	HasSHA2     bool // SHA2 hardware implementation
	HasCRC32    bool // CRC32 hardware implementation
	_           CacheLinePad
}

// MIPS64X contains the supported CPU features of the current mips64/mips64le
// platforms. If the current platform is not mips64/mips64le or the current
// operating system is not Linux then all feature flags are false.
var MIPS64X struct {
	_      CacheLinePad
	HasMSA bool // MIPS SIMD architecture
	_      CacheLinePad
}

// PPC64 contains the supported CPU features of the current ppc64/ppc64le platforms.
// If the current platform is not ppc64/ppc64le then all feature flags are false.
//
// For ppc64/ppc64le, it is safe to check only for ISA level starting on ISA v3.00,
// since there are no optional categories. There are some exceptions that also
// require kernel support to work (DARN, SCV), so there are feature bits for
// those as well. The struct is padded to avoid false sharing.
var PPC64 struct {
	_        CacheLinePad
	HasDARN  bool // Hardware random number generator (requires kernel enablement)
	HasSCV   bool // Syscall vectored (requires kernel enablement)
	IsPOWER8 bool // ISA v2.07 (POWER8)
	IsPOWER9 bool // ISA v3.00 (POWER9), implies IsPOWER8
	_        CacheLinePad
}

// S390X contains the supported CPU features of the current IBM Z
// (s390x) platform. If the current platform is not IBM Z then all
// feature flags are false.
//
// S390X is padded to avoid false sharing. Further HasVX is only set
// if the OS supports vector registers in addition to the STFLE
// feature bit being set.
var S390X struct {
	_         CacheLinePad
	HasZARCH  bool // z/Architecture mode is active [mandatory]
	HasSTFLE  bool // store facility list extended
	HasLDISP  bool // long (20-bit) displacements
	HasEIMM   bool // 32-bit immediates
	HasDFP    bool // decimal floating point
	HasETF3EH bool // ETF-3 enhanced
	HasMSA    bool // message security assist (CPACF)
	HasAES    bool // KM-AES{128,192,256} functions
	HasAESCBC bool // KMC-AES{128,192,256} functions
	HasAESCTR bool // KMCTR-AES{128,192,256} functions
	HasAESGCM bool // KMA-GCM-AES{128,192,256} functions
	HasGHASH  bool // KIMD-GHASH function
	HasSHA1   bool // K{I,L}MD-SHA-1 functions
	HasSHA256 bool // K{I,L}MD-SHA-256 functions
	HasSHA512 bool // K{I,L}MD-SHA-512 functions
	HasSHA3   bool // K{I,L}MD-SHA3-{224,256,384,512} and K{I,L}MD-SHAKE-{128,256} functions
	HasVX     bool // vector facility
	HasVXE    bool // vector-enhancements facility 1
	_         CacheLinePad
}

// RISCV64 contains the supported CPU features and performance characteristics for riscv64
// platforms. The booleans in RISCV64, with the exception of HasFastMisaligned, indicate
// the presence of RISC-V extensions.
//
// It is safe to assume that all the RV64G extensions are supported and so they are omitted from
// this structure. As riscv64 Go programs require at least RV64G, the code that populates
// this structure cannot run successfully if some of the RV64G extensions are missing.
// The struct is padded to avoid false sharing.
var RISCV64 struct {
	_                 CacheLinePad
	HasFastMisaligned bool // Fast misaligned accesses
	HasC              bool // Compressed instruction-set extension
	HasV              bool // Vector extension compatible with RVV 1.0
	HasZba            bool // Address generation instructions extension
	HasZbb            bool // Basic bit-manipulation extension
	HasZbs            bool // Single-bit instructions extension
	_                 CacheLinePad
}

func init() {
	archInit()
	initOptions()
	processOptions()
}

// options contains the cpu debug options that can be used in GODEBUG.
// Options are arch dependent and are added by the arch specific initOptions functions.
// Features that are mandatory for the specific GOARCH should have the Required field set
// (e.g. SSE2 on amd64).
var options []option

// Option names should be lower case. e.g. avx instead of AVX.
type option struct {
	Name      string
	Feature   *bool
	Specified bool // whether feature value was specified in GODEBUG
	Enable    bool // whether feature should be enabled
	Required  bool // whether feature is mandatory and can not be disabled
}

func processOptions() {
	env := os.Getenv("GODEBUG")
field:
	for env != "" {
		field := ""
		i := strings.IndexByte(env, ',')
		if i < 0 {
			field, env = env, ""
		} else {
			field, env = env[:i], env[i+1:]
		}
		if len(field) < 4 || field[:4] != "cpu." {
			continue
		}
		i = strings.IndexByte(field, '=')
		if i < 0 {
			print("GODEBUG sys/cpu: no value specified for \"", field, "\"\n")
			continue
		}
		key, value := field[4:i], field[i+1:] // e.g. "SSE2", "on"

		var enable bool
		switch value {
		case "on":
			enable = true
		case "off":
			enable = false
		default:
			print("GODEBUG sys/cpu: value \"", value, "\" not supported for cpu option \"", key, "\"\n")
			continue field
		}

		if key == "all" {
			for i := range options {
				options[i].Specified = true
				options[i].Enable = enable || options[i].Required
			}
			continue field
		}

		for i := range options {
			if options[i].Name == key {
				options[i].Specified = true
				options[i].Enable = enable
				continue field
			}
		}

		print("GODEBUG sys/cpu: unknown cpu feature \"", key, "\"\n")
	}

	for _, o := range options {
		if !o.Specified {
			continue
		}

		if o.Enable && !*o.Feature {
			print("GODEBUG sys/cpu: can not enable \"", o.Name, "\", missing CPU support\n")
			continue
		}

		if !o.Enable && o.Required {
			print("GODEBUG sys/cpu: can not disable \"", o.Name, "\", required CPU feature\n")
			continue
		}

		*o.Feature = o.Enable
	}
}
