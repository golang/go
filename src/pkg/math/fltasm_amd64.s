// Derived from Inferno's libkern/getfcr-amd64.s
// http://code.google.com/p/inferno-os/source/browse/libkern/getfcr-amd64.s
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

TEXT	·SetFPControl(SB), 7, $8
	// Set new
	MOVL	p+0(FP), DI
	XORL	$(0x3F<<7), DI
	ANDL	$0xFFC0, DI
	WAIT
	STMXCSR	0(SP)
	MOVL	0(SP), AX
	ANDL	$~0x3F, AX
	ORL	DI, AX
	MOVL	AX, 0(SP)
	LDMXCSR	0(SP)
	RET

TEXT	·GetFPControl(SB), 7, $0
	WAIT
	STMXCSR	0(SP)
	MOVWLZX	0(SP), AX
	ANDL	$0xFFC0, AX
	XORL	$(0x3F<<7), AX
	MOVL	AX, ret+0(FP)
	RET

TEXT	·SetFPStatus(SB), $0
	MOVL	p+0(FP), DI
	ANDL	$0x3F, DI
	WAIT
	STMXCSR	0(SP)
	MOVL	0(SP), AX
	ANDL	$~0x3F, AX
	ORL	DI, AX
	MOVL	AX, 0(SP)
	LDMXCSR	0(SP)
	RET

TEXT	·GetFPStatus(SB), $0
	WAIT
	STMXCSR	0(SP)
	MOVL	0(SP), AX
	ANDL	$0x3F, AX
	MOVL	AX, ret+0(FP)
	RET
