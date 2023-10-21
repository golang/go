// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc

#include "textflag.h"

// func getisar0() uint64
TEXT ·getisar0(SB),NOSPLIT,$0-8
	// get Instruction Set Attributes 0 into x0
	// mrs x0, ID_AA64ISAR0_EL1 = d5380600
	WORD	$0xd5380600
	MOVD	R0, ret+0(FP)
	RET

// func getisar1() uint64
TEXT ·getisar1(SB),NOSPLIT,$0-8
	// get Instruction Set Attributes 1 into x0
	// mrs x0, ID_AA64ISAR1_EL1 = d5380620
	WORD	$0xd5380620
	MOVD	R0, ret+0(FP)
	RET

// func getpfr0() uint64
TEXT ·getpfr0(SB),NOSPLIT,$0-8
	// get Processor Feature Register 0 into x0
	// mrs x0, ID_AA64PFR0_EL1 = d5380400
	WORD	$0xd5380400
	MOVD	R0, ret+0(FP)
	RET
