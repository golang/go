// Inferno's libkern/vlop-arm.s
// http://code.google.com/p/inferno-os/source/browse/libkern/vlop-arm.s
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

arg=0

/* replaced use of R10 by R11 because the former can be the data segment base register */

TEXT _mulv(SB), $0
	MOVW	0(FP), R0
	MOVW	4(FP), R2	/* l0 */
	MOVW	8(FP), R11	/* h0 */
	MOVW	12(FP), R4	/* l1 */
	MOVW	16(FP), R5	/* h1 */
	MULLU	R4, R2, (R7,R6)
	MUL	R11, R4, R8
	ADD	R8, R7
	MUL	R2, R5, R8
	ADD	R8, R7
	MOVW	R6, 0(R(arg))
	MOVW	R7, 4(R(arg))
	RET

// trampoline for _sfloat2. passes LR as arg0 and
// saves registers R0-R13 and CPSR on the stack. R0-R12 and CPSR flags can
// be changed by _sfloat2.
TEXT _sfloat(SB), 7, $64 // 4 arg + 14*4 saved regs + cpsr
	MOVW	R14, 4(R13)
	MOVW	R0, 8(R13)
	MOVW	$12(R13), R0
	MOVM.IA.W	[R1-R12], (R0)
	MOVW	$68(R13), R1 // correct for frame size
	MOVW	R1, 60(R13)
	WORD	$0xe10f1000 // mrs r1, cpsr
	MOVW	R1, 64(R13)
	BL	runtime·_sfloat2(SB)
	MOVW	R0, 0(R13)
	MOVW	64(R13), R1
	WORD	$0xe128f001	// msr cpsr_f, r1
	MOVW	$12(R13), R0
	MOVM.IA.W	(R0), [R1-R12]
	MOVW	8(R13), R0
	RET

// func udiv(n, d uint32) (q, r uint32)
// Reference: 
// Sloss, Andrew et. al; ARM System Developer's Guide: Designing and Optimizing System Software
// Morgan Kaufmann; 1 edition (April 8, 2004), ISBN 978-1558608740
q = 0 // input d, output q
r = 1 // input n, output r
s = 2 // three temporary variables
m = 3
a = 11
// Please be careful when changing this, it is pretty fragile:
// 1, don't use unconditional branch as the linker is free to reorder the blocks;
// 2. if a == 11, beware that the linker will use R11 if you use certain instructions.
TEXT udiv<>(SB),7,$-4
	CLZ 	R(q), R(s) // find normalizing shift
	MOVW.S	R(q)<<R(s), R(a)
	ADD 	R(a)>>25, PC, R(a) // most significant 7 bits of divisor
	MOVBU.NE	(4*36-64)(R(a)), R(a) // 36 == number of inst. between fast_udiv_tab and begin

begin:
	SUB.S	$7, R(s)
	RSB 	$0, R(q), R(m) // m = -q
	MOVW.PL	R(a)<<R(s), R(q)

	// 1st Newton iteration
	MUL.PL	R(m), R(q), R(a) // a = -q*d
	BMI 	udiv_by_large_d
	MULAWT	R(a), R(q), R(q), R(q) // q approx q-(q*q*d>>32)
	TEQ 	R(m)->1, R(m) // check for d=0 or d=1

	// 2nd Newton iteration
	MUL.NE	R(m), R(q), R(a)
	MOVW.NE	$0, R(s)
	MULAL.NE R(q), R(a), (R(q),R(s))
	BEQ 	udiv_by_0_or_1

	// q now accurate enough for a remainder r, 0<=r<3*d
	MULLU	R(q), R(r), (R(q),R(s)) // q = (r * q) >> 32	
	ADD 	R(m), R(r), R(r) // r = n - d
	MULA	R(m), R(q), R(r), R(r) // r = n - (q+1)*d

	// since 0 <= n-q*d < 3*d; thus -d <= r < 2*d
	CMN 	R(m), R(r) // t = r-d
	SUB.CS	R(m), R(r), R(r) // if (t<-d || t>=0) r=r+d
	ADD.CC	$1, R(q)
	ADD.PL	R(m)<<1, R(r)
	ADD.PL	$2, R(q)

	// return, can't use RET here or fast_udiv_tab will be dropped during linking
	MOVW	R14, R15

udiv_by_large_d:
	// at this point we know d>=2^(31-6)=2^25
	SUB 	$4, R(a), R(a)
	RSB 	$0, R(s), R(s)
	MOVW	R(a)>>R(s), R(q)
	MULLU	R(q), R(r), (R(q),R(s))
	MULA	R(m), R(q), R(r), R(r)

	// q now accurate enough for a remainder r, 0<=r<4*d
	CMN 	R(r)>>1, R(m) // if(r/2 >= d)
	ADD.CS	R(m)<<1, R(r)
	ADD.CS	$2, R(q)
	CMN 	R(r), R(m)
	ADD.CS	R(m), R(r)
	ADD.CS	$1, R(q)

	// return, can't use RET here or fast_udiv_tab will be dropped during linking
	MOVW	R14, R15

udiv_by_0_or_1:
	// carry set if d==1, carry clear if d==0
	MOVW.CS	R(r), R(q)
	MOVW.CS	$0, R(r)
	BL.CC 	runtime·panicdivide(SB) // no way back

	// return, can't use RET here or fast_udiv_tab will be dropped during linking
	MOVW	R14, R15

fast_udiv_tab:
	// var tab [64]byte
	// tab[0] = 255; for i := 1; i <= 63; i++ { tab[i] = (1<<14)/(64+i) }
	// laid out here as little-endian uint32s
	WORD $0xf4f8fcff
	WORD $0xe6eaedf0
	WORD $0xdadde0e3
	WORD $0xcfd2d4d7
	WORD $0xc5c7cacc
	WORD $0xbcbec0c3
	WORD $0xb4b6b8ba
	WORD $0xacaeb0b2
	WORD $0xa5a7a8aa
	WORD $0x9fa0a2a3
	WORD $0x999a9c9d
	WORD $0x93949697
	WORD $0x8e8f9092
	WORD $0x898a8c8d
	WORD $0x85868788
	WORD $0x81828384

// The linker will pass numerator in R(TMP), and it also
// expects the result in R(TMP)
TMP = 11

TEXT _divu(SB), 7, $16
	MOVW	R(q), 4(R13)
	MOVW	R(r), 8(R13)
	MOVW	R(s), 12(R13)
	MOVW	R(m), 16(R13)

	MOVW	R(TMP), R(r)		/* numerator */
	MOVW	0(FP), R(q) 		/* denominator */
	BL  	udiv<>(SB)
	MOVW	R(q), R(TMP)
	MOVW	4(R13), R(q)
	MOVW	8(R13), R(r)
	MOVW	12(R13), R(s)
	MOVW	16(R13), R(m)
	RET

TEXT _modu(SB), 7, $16
	MOVW	R(q), 4(R13)
	MOVW	R(r), 8(R13)
	MOVW	R(s), 12(R13)
	MOVW	R(m), 16(R13)

	MOVW	R(TMP), R(r)		/* numerator */
	MOVW	0(FP), R(q) 		/* denominator */
	BL  	udiv<>(SB)
	MOVW	R(r), R(TMP)
	MOVW	4(R13), R(q)
	MOVW	8(R13), R(r)
	MOVW	12(R13), R(s)
	MOVW	16(R13), R(m)
	RET

TEXT _div(SB),7,$16
	MOVW	R(q), 4(R13)
	MOVW	R(r), 8(R13)
	MOVW	R(s), 12(R13)
	MOVW	R(m), 16(R13)
	MOVW	R(TMP), R(r)		/* numerator */
	MOVW	0(FP), R(q) 		/* denominator */
	CMP 	$0, R(r)
	BGE 	d1
	RSB 	$0, R(r), R(r)
	CMP 	$0, R(q)
	BGE 	d2
	RSB 	$0, R(q), R(q)
d0:
	BL  	udiv<>(SB)  		/* none/both neg */
	MOVW	R(q), R(TMP)
	B		out
d1:
	CMP 	$0, R(q)
	BGE 	d0
	RSB 	$0, R(q), R(q)
d2:
	BL  	udiv<>(SB)  		/* one neg */
	RSB		$0, R(q), R(TMP)
	B   	out

TEXT _mod(SB),7,$16
	MOVW	R(q), 4(R13)
	MOVW	R(r), 8(R13)
	MOVW	R(s), 12(R13)
	MOVW	R(m), 16(R13)
	MOVW	R(TMP), R(r)		/* numerator */
	MOVW	0(FP), R(q) 		/* denominator */
	CMP 	$0, R(q)
	RSB.LT	$0, R(q), R(q)
	CMP 	$0, R(r)
	BGE 	m1
	RSB 	$0, R(r), R(r)
	BL  	udiv<>(SB)  		/* neg numerator */
	RSB 	$0, R(r), R(TMP)
	B   	out
m1:
	BL  	udiv<>(SB)  		/* pos numerator */
	MOVW	R(r), R(TMP)
out:
	MOVW	4(R13), R(q)
	MOVW	8(R13), R(r)
	MOVW	12(R13), R(s)
	MOVW	16(R13), R(m)
	RET
