// Inferno's libkern/vlop-386.s
// http://code.google.com/p/inferno-os/source/browse/libkern/vlop-386.s
//
//         Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
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

/*
 * C runtime for 64-bit divide.
 */

TEXT _mul64by32(SB), 7, $0
	MOVL	r+0(FP), CX
	MOVL	a+4(FP), AX
	MULL	b+12(FP)
	MOVL	AX, 0(CX)
	MOVL	DX, BX
	MOVL	a+8(FP), AX
	MULL	b+12(FP)
	ADDL	AX, BX
	MOVL	BX, 4(CX)
	RET

TEXT _div64by32(SB), 7, $0
	MOVL	r+12(FP), CX
	MOVL	a+0(FP), AX
	MOVL	a+4(FP), DX
	DIVL	b+8(FP)
	MOVL	DX, 0(CX)
	RET
