// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

import (
	"internal/cpu"
	"internal/goarch"
)

const (
	BigEndian = goarch.BigEndian
	AMD64     = goarch.IsAmd64 == 1
	ARM64     = goarch.IsArm64 == 1
	PPC64     = goarch.IsPpc64 == 1
	PPC64le   = goarch.IsPpc64le == 1
)

var (
	ARM64HasAES    = cpu.ARM64.HasAES
	ARM64HasPMULL  = cpu.ARM64.HasPMULL
	ARM64HasSHA2   = cpu.ARM64.HasSHA2
	ARM64HasSHA512 = cpu.ARM64.HasSHA512

	S390XHasAES    = cpu.S390X.HasAES
	S390XHasAESCBC = cpu.S390X.HasAESCBC
	S390XHasAESCTR = cpu.S390X.HasAESCTR
	S390XHasAESGCM = cpu.S390X.HasAESGCM
	S390XHasECDSA  = cpu.S390X.HasECDSA
	S390XHasGHASH  = cpu.S390X.HasGHASH
	S390XHasSHA256 = cpu.S390X.HasSHA256
	S390XHasSHA3   = cpu.S390X.HasSHA3
	S390XHasSHA512 = cpu.S390X.HasSHA512

	X86HasAES       = cpu.X86.HasAES
	X86HasADX       = cpu.X86.HasADX
	X86HasAVX       = cpu.X86.HasAVX
	X86HasAVX2      = cpu.X86.HasAVX2
	X86HasBMI2      = cpu.X86.HasBMI2
	X86HasPCLMULQDQ = cpu.X86.HasPCLMULQDQ
	X86HasSHA       = cpu.X86.HasSHA
	X86HasSSE41     = cpu.X86.HasSSE41
	X86HasSSSE3     = cpu.X86.HasSSSE3
)
