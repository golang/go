// Inferno's libkern/vlop-386.s
// https://bitbucket.org/inferno-os/inferno-os/src/default/libkern/vlop-386.s
//
//         Copyright © 1994-1999 Lucent Technologies Inc. All rights reserved.
//         Revisions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).  All rights reserved.
//         Portions Copyright 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "textflag.h"

/*
 * C runtime for 64-bit divide.
 */

// runtime·_mul64x32(lo64 *uint64, a uint64, b uint32) (hi32 uint32)
// sets *lo64 = low 64 bits of 96-bit product a*b; returns high 32 bits.
TEXT runtime·_mul64by32(SB), NOSPLIT, $0
	MOVL	lo64+0(FP), CX
	MOVL	a_lo+4(FP), AX
	MULL	b+12(FP)
	MOVL	AX, 0(CX)
	MOVL	DX, BX
	MOVL	a_hi+8(FP), AX
	MULL	b+12(FP)
	ADDL	AX, BX
	ADCL	$0, DX
	MOVL	BX, 4(CX)
	MOVL	DX, AX
	MOVL	AX, hi32+16(FP)
	RET

TEXT runtime·_div64by32(SB), NOSPLIT, $0
	MOVL	r+12(FP), CX
	MOVL	a_lo+0(FP), AX
	MOVL	a_hi+4(FP), DX
	DIVL	b+8(FP)
	MOVL	DX, 0(CX)
	MOVL	AX, q+16(FP)
	RET
