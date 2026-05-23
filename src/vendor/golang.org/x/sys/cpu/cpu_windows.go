// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

//go:generate go run golang.org/x/sys/windows/mkwinsyscall -systemdll=false -output zcpu_windows.go cpu_windows.go

//sys	isProcessorFeaturePresent(ProcessorFeature uint32) (ret bool) = kernel32.IsProcessorFeaturePresent

// The processor features to be tested for IsProcessorFeaturePresent, see
// https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent
const (
	_PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE  = 30
	_PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE   = 31
	_PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE = 34
	_PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE     = 43

	_PF_ARM_V83_JSCVT_INSTRUCTIONS_AVAILABLE = 44
	_PF_ARM_V83_LRCPC_INSTRUCTIONS_AVAILABLE = 45
	_PF_ARM_SVE_INSTRUCTIONS_AVAILABLE       = 46
	_PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE      = 47

	_PF_ARM_SHA3_INSTRUCTIONS_AVAILABLE   = 64
	_PF_ARM_SHA512_INSTRUCTIONS_AVAILABLE = 65
)
