// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.simd

#include "textflag.h"

// func memHash32AES(k uint32, h uintptr) uintptr
// ABIInternal for performance.
TEXT ·memHash32AES<ABIInternal>(SB),NOSPLIT,$0-24
	// AX = ptr to data
	// BX = seed
	MOVQ	BX, X0	    // X0 = seed
	PINSRD	$2, AX, X0	// data
	AESENC	·aeskeysched+0(SB), X0
	AESENC	·aeskeysched+16(SB), X0
	AESENC	·aeskeysched+32(SB), X0
	MOVQ	X0, AX	// return X0
	RET

// func memHash64AES(k uint64, h uintptr) uintptr
// ABIInternal for performance.
TEXT ·memHash64AES<ABIInternal>(SB),NOSPLIT,$0-24
	// AX = ptr to data
	// BX = seed
	MOVQ	BX, X0   	// X0 = seed
	PINSRQ	$1, AX, X0	// data
	AESENC	·aeskeysched+0(SB), X0
	AESENC	·aeskeysched+16(SB), X0
	AESENC	·aeskeysched+32(SB), X0
	MOVQ	X0, AX	// return X0
	RET
