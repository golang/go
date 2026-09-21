// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm64

package cpu

func osInit() {
	// TODO(brainman): we don't know how to retrieve those bits form user space - leaving them as false for now
	// https://github.com/golang/go/issues/76791#issuecomment-3877873959
	ARM64.HasCPUID = false
	ARM64.HasDIT = false
	ARM64.IsNeoverse = false

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
}
