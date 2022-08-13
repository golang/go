// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc
// +build gc

#include "textflag.h"

// try to run "vmov.f64 d0, d0" instruction
TEXT ·useVFPv1(SB),NOSPLIT,$0
	WORD $0xeeb00b40	// vmov.f64 d0, d0
	RET

// try to run VFPv3-only "vmov.f64 d0, #112" instruction
TEXT ·useVFPv3(SB),NOSPLIT,$0
	WORD $0xeeb70b00	// vmov.f64 d0, #112
	RET

// try to run ARMv6K (or above) "ldrexd" instruction
TEXT ·useARMv6K(SB),NOSPLIT,$32
	MOVW R13, R2
	BIC  $15, R13
	WORD $0xe1bd0f9f	// ldrexd r0, r1, [sp]
	WORD $0xf57ff01f	// clrex
	MOVW R2, R13
	RET
