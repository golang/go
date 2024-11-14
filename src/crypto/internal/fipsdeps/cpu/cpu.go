// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

import (
	"internal/cpu"
	"internal/goarch"
)

const BigEndian = goarch.BigEndian
const AMD64 = goarch.IsAmd64 == 1
const ARM64 = goarch.IsArm64 == 1
const PPC64 = goarch.IsPpc64 == 1
const PPC64le = goarch.IsPpc64le == 1

var ARM64HasAES = cpu.ARM64.HasAES
var ARM64HasPMULL = cpu.ARM64.HasPMULL
var ARM64HasSHA2 = cpu.ARM64.HasSHA2
var ARM64HasSHA512 = cpu.ARM64.HasSHA512
var S390XHasAES = cpu.S390X.HasAES
var S390XHasAESCBC = cpu.S390X.HasAESCBC
var S390XHasAESCTR = cpu.S390X.HasAESCTR
var S390XHasAESGCM = cpu.S390X.HasAESGCM
var S390XHasGHASH = cpu.S390X.HasGHASH
var S390XHasSHA256 = cpu.S390X.HasSHA256
var S390XHasSHA3 = cpu.S390X.HasSHA3
var S390XHasSHA512 = cpu.S390X.HasSHA512
var X86HasAES = cpu.X86.HasAES
var X86HasAVX = cpu.X86.HasAVX
var X86HasAVX2 = cpu.X86.HasAVX2
var X86HasBMI2 = cpu.X86.HasBMI2
var X86HasPCLMULQDQ = cpu.X86.HasPCLMULQDQ
var X86HasSHA = cpu.X86.HasSHA
var X86HasSSE41 = cpu.X86.HasSSE41
var X86HasSSSE3 = cpu.X86.HasSSSE3
