// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (386 || amd64 || amd64p32) && gc
// +build 386 amd64 amd64p32
// +build gc

#include "textflag.h"

// func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL eaxArg+0(FP), AX
	MOVL ecxArg+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() (eax, edx uint32)
TEXT ·xgetbv(SB),NOSPLIT,$0-8
	MOVL $0, CX
	XGETBV
	MOVL AX, eax+0(FP)
	MOVL DX, edx+4(FP)
	RET

// func darwinSupportsAVX512() bool
TEXT ·darwinSupportsAVX512(SB), NOSPLIT, $0-1
    MOVB    $0, ret+0(FP) // default to false
#ifdef GOOS_darwin   // return if not darwin
#ifdef GOARCH_amd64  // return if not amd64
// These values from:
// https://github.com/apple/darwin-xnu/blob/xnu-4570.1.46/osfmk/i386/cpu_capabilities.h
#define commpage64_base_address         0x00007fffffe00000
#define commpage64_cpu_capabilities64   (commpage64_base_address+0x010)
#define commpage64_version              (commpage64_base_address+0x01E)
#define hasAVX512F                      0x0000004000000000
    MOVQ    $commpage64_version, BX
    CMPW    (BX), $13  // cpu_capabilities64 undefined in versions < 13
    JL      no_avx512
    MOVQ    $commpage64_cpu_capabilities64, BX
    MOVQ    $hasAVX512F, CX
    TESTQ   (BX), CX
    JZ      no_avx512
    MOVB    $1, ret+0(FP)
no_avx512:
#endif
#endif
    RET
