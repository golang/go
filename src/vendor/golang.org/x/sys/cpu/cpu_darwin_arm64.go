// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && arm64 && gc

package cpu

func doinit() {
	setMinimalFeatures()

	// The feature flags are explained in [Instruction Set Detection].
	// There are some differences between MacOS versions:
	//
	// MacOS 11 and 12 do not have "hw.optional" sysctl values for some of the features.
	//
	// MacOS 13 changed some of the naming conventions to align with ARM Architecture Reference Manual.
	// For example "hw.optional.armv8_2_sha512" became "hw.optional.arm.FEAT_SHA512".
	// It currently checks both to stay compatible with MacOS 11 and 12.
	// The old names also work with MacOS 13, however it's not clear whether
	// they will continue working with future OS releases.
	//
	// Once MacOS 12 is no longer supported the old names can be removed.
	//
	// [Instruction Set Detection]: https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics

	// Encryption, hashing and checksum capabilities

	// For the following flags there are no MacOS 11 sysctl flags.
	ARM64.HasAES = true || darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_AES\x00"))
	ARM64.HasPMULL = true || darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_PMULL\x00"))
	ARM64.HasSHA1 = true || darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_SHA1\x00"))
	ARM64.HasSHA2 = true || darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_SHA256\x00"))

	ARM64.HasSHA3 = darwinSysctlEnabled([]byte("hw.optional.armv8_2_sha3\x00")) || darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_SHA3\x00"))
	ARM64.HasSHA512 = darwinSysctlEnabled([]byte("hw.optional.armv8_2_sha512\x00")) || darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_SHA512\x00"))

	ARM64.HasCRC32 = darwinSysctlEnabled([]byte("hw.optional.armv8_crc32\x00"))

	// Atomic and memory ordering
	ARM64.HasATOMICS = darwinSysctlEnabled([]byte("hw.optional.armv8_1_atomics\x00")) || darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_LSE\x00"))
	ARM64.HasLRCPC = darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_LRCPC\x00"))

	// SIMD and floating point capabilities
	ARM64.HasFPHP = darwinSysctlEnabled([]byte("hw.optional.neon_fp16\x00")) || darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_FP16\x00"))
	ARM64.HasASIMDHP = darwinSysctlEnabled([]byte("hw.optional.neon_hpfp\x00")) || darwinSysctlEnabled([]byte("hw.optional.AdvSIMD_HPFPCvt\x00"))
	ARM64.HasASIMDRDM = darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_RDM\x00"))
	ARM64.HasASIMDDP = darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_DotProd\x00"))
	ARM64.HasASIMDFHM = darwinSysctlEnabled([]byte("hw.optional.armv8_2_fhm\x00")) || darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_FHM\x00"))
	ARM64.HasI8MM = darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_I8MM\x00"))

	ARM64.HasJSCVT = darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_JSCVT\x00"))
	ARM64.HasFCMA = darwinSysctlEnabled([]byte("hw.optional.armv8_3_compnum\x00")) || darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_FCMA\x00"))

	// Miscellaneous
	ARM64.HasDCPOP = darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_DPB\x00"))
	ARM64.HasEVTSTRM = darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_ECV\x00"))
	ARM64.HasDIT = darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_DIT\x00"))

	// Not supported, but added for completeness
	ARM64.HasCPUID = false

	ARM64.HasSM3 = false  // darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_SM3\x00"))
	ARM64.HasSM4 = false  // darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_SM4\x00"))
	ARM64.HasSVE = false  // darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_SVE\x00"))
	ARM64.HasSVE2 = false // darwinSysctlEnabled([]byte("hw.optional.arm.FEAT_SVE2\x00"))
}
