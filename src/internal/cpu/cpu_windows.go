// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

import _ "unsafe" // for linkname

const (
	_PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE  = 30
	_PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE   = 31
	_PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE = 34
	_PF_ARM_SHA3_INSTRUCTIONS_AVAILABLE       = 64
	_PF_ARM_SHA512_INSTRUCTIONS_AVAILABLE     = 65
)

// isProcessorFeaturePresent calls windows IsProcessorFeaturePresent API.
//
//go:linkname isProcessorFeaturePresent
func isProcessorFeaturePresent(processorFeature uint32) bool // Implemented in runtime package.
