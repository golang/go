// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "asm_riscv64.h"
#include "textflag.h"

// Packs the non-nil values of bufp[0:n] at the front of bufp, returning the
// packed length.
//
// func filterNilRVV(bufp *uintptr, n int32) int32
TEXT ·filterNilRVV(SB), NOSPLIT, $0-20
	MOV	bufp+0(FP), X10			// X10 = read ptr
	MOVW	n+8(FP), X11			// X11 = remaining
	MOV	$0, X13				// X13 = counter (init before n==0 skip)
	BLEZ	X11, done			// in case n < 0
	MOV	X10, X12			// X12 = write ptr
loop:
	VSETVLI	X11, E64, M2, TA, MA, X14	// X14 = vl = min(remaining, VLMAX)
	VLE64V	(X10), V16
	VMSNEVX	X0, V16, V0			// mask = words != 0
	VCPOPM	V0, X15				// X15 = count of non-nil (mask view, over vl)
	BEQZ	X15, skip			// all-nil chunk: no write, no out advance
	VCOMPRESSVM V0, V16, V24
	VSE64V	V24, (X12)			// write at out ptr
	ADD	X15, X13			// counter += popcount
#ifdef hasZba
	SH3ADD	X12, X15, X12			// write ptr += 8*popcount
#else
	SLLI	$3, X15				// 8*popcount
	ADD	X15, X12, X12			// write ptr += 8*popcount
#endif
skip:
#ifdef hasZba
	SH3ADD	X10, X14, X10			// read ptr += 8*vl
#else
	SLLI	$3, X14, X15			// X15 = 8*vl
	ADD	X15, X10, X10			// read ptr += 8*vl
#endif
	SUB	X14, X11, X11			// remaining -= vl
	BNEZ	X11, loop
done:
	MOVW	X13, ret+16(FP)
	RET
