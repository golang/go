// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

func doinit() {
	// set HasASIMD and HasFP to true as per
	// https://learn.microsoft.com/en-us/cpp/build/arm64-windows-abi-conventions?view=msvc-170#base-requirements
	//
	// The ARM64 version of Windows always presupposes that it's running on an ARMv8 or later architecture.
	// Both floating-point and NEON support are presumed to be present in hardware.
	//
	ARM64.HasASIMD = true
	ARM64.HasFP = true

	if isProcessorFeaturePresent(_PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE) {
		ARM64.HasAES = true
		ARM64.HasPMULL = true
		ARM64.HasSHA1 = true
		ARM64.HasSHA2 = true
	}
	ARM64.HasSHA3 = isProcessorFeaturePresent(_PF_ARM_SHA3_INSTRUCTIONS_AVAILABLE)
	ARM64.HasCRC32 = isProcessorFeaturePresent(_PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE)
	ARM64.HasSHA512 = isProcessorFeaturePresent(_PF_ARM_SHA512_INSTRUCTIONS_AVAILABLE)
	ARM64.HasATOMICS = isProcessorFeaturePresent(_PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE)
	if isProcessorFeaturePresent(_PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) {
		ARM64.HasASIMDDP = true
		ARM64.HasASIMDRDM = true
	}
	if isProcessorFeaturePresent(_PF_ARM_V83_LRCPC_INSTRUCTIONS_AVAILABLE) {
		ARM64.HasLRCPC = true
		ARM64.HasSM3 = true
	}
	ARM64.HasSVE = isProcessorFeaturePresent(_PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)
	ARM64.HasSVE2 = isProcessorFeaturePresent(_PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE)
	ARM64.HasJSCVT = isProcessorFeaturePresent(_PF_ARM_V83_JSCVT_INSTRUCTIONS_AVAILABLE)
}
