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

#include "zasm_GOOS_GOARCH.h"
#include "textflag.h"

arg=0

/* replaced use of R10 by R11 because the former can be the data segment base register */

TEXT _mulv(SB), NOSPLIT, $0
	MOVW	l0+0(FP), R2	/* l0 */
	MOVW	h0+4(FP), R11	/* h0 */
	MOVW	l1+8(FP), R4	/* l1 */
	MOVW	h1+12(FP), R5	/* h1 */
	MULLU	R4, R2, (R7,R6)
	MUL	R11, R4, R8
	ADD	R8, R7
	MUL	R2, R5, R8
	ADD	R8, R7
	MOVW	R6, ret_lo+16(FP)
	MOVW	R7, ret_hi+20(FP)
	RET

// trampoline for _sfloat2. passes LR as arg0 and
// saves registers R0-R13 and CPSR on the stack. R0-R12 and CPSR flags can
// be changed by _sfloat2.
TEXT _sfloat(SB), NOSPLIT, $68-0 // 4 arg + 14*4 saved regs + cpsr + return value
	MOVW	R14, 4(R13)
	MOVW	R0, 8(R13)
	MOVW	$12(R13), R0
	MOVM.IA.W	[R1-R12], (R0)
	MOVW	$72(R13), R1 // correct for frame size
	MOVW	R1, 60(R13)
	WORD	$0xe10f1000 // mrs r1, cpsr
	MOVW	R1, 64(R13)
	// Disable preemption of this goroutine during _sfloat2 by
	// m->locks++ and m->locks-- around the call.
	// Rescheduling this goroutine may cause the loss of the
	// contents of the software floating point registers in 
	// m->freghi, m->freglo, m->fflag, if the goroutine is moved
	// to a different m or another goroutine runs on this m.
	// Rescheduling at ordinary function calls is okay because
	// all registers are caller save, but _sfloat2 and the things
	// that it runs are simulating the execution of individual
	// program instructions, and those instructions do not expect
	// the floating point registers to be lost.
	// An alternative would be to move the software floating point
	// registers into G, but they do not need to be kept at the 
	// usual places a goroutine reschedules (at function calls),
	// so it would be a waste of 132 bytes per G.
	MOVW	g_m(g), R8
	MOVW	m_locks(R8), R1
	ADD	$1, R1
	MOVW	R1, m_locks(R8)
	MOVW	$1, R1
	MOVW	R1, m_softfloat(R8)
	BL	runtime·_sfloat2(SB)
	MOVW	68(R13), R0
	MOVW	g_m(g), R8
	MOVW	m_locks(R8), R1
	SUB	$1, R1
	MOVW	R1, m_locks(R8)
	MOVW	$0, R1
	MOVW	R1, m_softfloat(R8)
	MOVW	R0, 0(R13)
	MOVW	64(R13), R1
	WORD	$0xe128f001	// msr cpsr_f, r1
	MOVW	$12(R13), R0
	// Restore R1-R12, R0.
	MOVM.IA.W	(R0), [R1-R12]
	MOVW	8(R13), R0
	RET

// trampoline for _sfloat2 panic.
// _sfloat2 instructs _sfloat to return here.
// We need to push a fake saved LR onto the stack,
// load the signal fault address into LR, and jump
// to the real sigpanic.
// This simulates what sighandler does for a memory fault.
TEXT _sfloatpanic(SB),NOSPLIT,$-4
	MOVW	$0, R0
	MOVW.W	R0, -4(R13)
	MOVW	g_sigpc(g), LR
	B	runtime·sigpanic(SB)

// func udiv(n, d uint32) (q, r uint32)
// Reference: 
// Sloss, Andrew et. al; ARM System Developer's Guide: Designing and Optimizing System Software
// Morgan Kaufmann; 1 edition (April 8, 2004), ISBN 978-1558608740
q = 0 // input d, output q
r = 1 // input n, output r
s = 2 // three temporary variables
M = 3
a = 11
// Be careful: R(a) == R11 will be used by the linker for synthesized instructions.
TEXT udiv<>(SB),NOSPLIT,$-4
	CLZ 	R(q), R(s) // find normalizing shift
	MOVW.S	R(q)<<R(s), R(a)
	MOVW	$fast_udiv_tab<>-64(SB), R(M)
	ADD.NE	R(a)>>25, R(M), R(a) // index by most significant 7 bits of divisor
	MOVBU.NE	(R(a)), R(a)

	SUB.S	$7, R(s)
	RSB 	$0, R(q), R(M) // M = -q
	MOVW.PL	R(a)<<R(s), R(q)

	// 1st Newton iteration
	MUL.PL	R(M), R(q), R(a) // a = -q*d
	BMI 	udiv_by_large_d
	MULAWT	R(a), R(q), R(q), R(q) // q approx q-(q*q*d>>32)
	TEQ 	R(M)->1, R(M) // check for d=0 or d=1

	// 2nd Newton iteration
	MUL.NE	R(M), R(q), R(a)
	MOVW.NE	$0, R(s)
	MULAL.NE R(q), R(a), (R(q),R(s))
	BEQ 	udiv_by_0_or_1

	// q now accurate enough for a remainder r, 0<=r<3*d
	MULLU	R(q), R(r), (R(q),R(s)) // q = (r * q) >> 32	
	ADD 	R(M), R(r), R(r) // r = n - d
	MULA	R(M), R(q), R(r), R(r) // r = n - (q+1)*d

	// since 0 <= n-q*d < 3*d; thus -d <= r < 2*d
	CMN 	R(M), R(r) // t = r-d
	SUB.CS	R(M), R(r), R(r) // if (t<-d || t>=0) r=r+d
	ADD.CC	$1, R(q)
	ADD.PL	R(M)<<1, R(r)
	ADD.PL	$2, R(q)
	RET

udiv_by_large_d:
	// at this point we know d>=2^(31-6)=2^25
	SUB 	$4, R(a), R(a)
	RSB 	$0, R(s), R(s)
	MOVW	R(a)>>R(s), R(q)
	MULLU	R(q), R(r), (R(q),R(s))
	MULA	R(M), R(q), R(r), R(r)

	// q now accurate enough for a remainder r, 0<=r<4*d
	CMN 	R(r)>>1, R(M) // if(r/2 >= d)
	ADD.CS	R(M)<<1, R(r)
	ADD.CS	$2, R(q)
	CMN 	R(r), R(M)
	ADD.CS	R(M), R(r)
	ADD.CS	$1, R(q)
	RET

udiv_by_0_or_1:
	// carry set if d==1, carry clear if d==0
	BCC udiv_by_0
	MOVW	R(r), R(q)
	MOVW	$0, R(r)
	RET

udiv_by_0:
	// The ARM toolchain expects it can emit references to DIV and MOD
	// instructions. The linker rewrites each pseudo-instruction into
	// a sequence that pushes two values onto the stack and then calls
	// _divu, _modu, _div, or _mod (below), all of which have a 16-byte
	// frame plus the saved LR. The traceback routine knows the expanded
	// stack frame size at the pseudo-instruction call site, but it
	// doesn't know that the frame has a non-standard layout. In particular,
	// it expects to find a saved LR in the bottom word of the frame.
	// Unwind the stack back to the pseudo-instruction call site, copy the
	// saved LR where the traceback routine will look for it, and make it
	// appear that panicdivide was called from that PC.
	MOVW	0(R13), LR
	ADD	$20, R13
	MOVW	8(R13), R1 // actual saved LR
	MOVW	R1, 0(R13) // expected here for traceback
	B 	runtime·panicdivide(SB)

// var tab [64]byte
// tab[0] = 255; for i := 1; i <= 63; i++ { tab[i] = (1<<14)/(64+i) }
// laid out here as little-endian uint32s
DATA fast_udiv_tab<>+0x00(SB)/4, $0xf4f8fcff
DATA fast_udiv_tab<>+0x04(SB)/4, $0xe6eaedf0
DATA fast_udiv_tab<>+0x08(SB)/4, $0xdadde0e3
DATA fast_udiv_tab<>+0x0c(SB)/4, $0xcfd2d4d7
DATA fast_udiv_tab<>+0x10(SB)/4, $0xc5c7cacc
DATA fast_udiv_tab<>+0x14(SB)/4, $0xbcbec0c3
DATA fast_udiv_tab<>+0x18(SB)/4, $0xb4b6b8ba
DATA fast_udiv_tab<>+0x1c(SB)/4, $0xacaeb0b2
DATA fast_udiv_tab<>+0x20(SB)/4, $0xa5a7a8aa
DATA fast_udiv_tab<>+0x24(SB)/4, $0x9fa0a2a3
DATA fast_udiv_tab<>+0x28(SB)/4, $0x999a9c9d
DATA fast_udiv_tab<>+0x2c(SB)/4, $0x93949697
DATA fast_udiv_tab<>+0x30(SB)/4, $0x8e8f9092
DATA fast_udiv_tab<>+0x34(SB)/4, $0x898a8c8d
DATA fast_udiv_tab<>+0x38(SB)/4, $0x85868788
DATA fast_udiv_tab<>+0x3c(SB)/4, $0x81828384
GLOBL fast_udiv_tab<>(SB), RODATA, $64

// The linker will pass numerator in R(TMP), and it also
// expects the result in R(TMP)
TMP = 11

TEXT _divu(SB), NOSPLIT, $16
	MOVW	R(q), 4(R13)
	MOVW	R(r), 8(R13)
	MOVW	R(s), 12(R13)
	MOVW	R(M), 16(R13)

	MOVW	R(TMP), R(r)		/* numerator */
	MOVW	0(FP), R(q) 		/* denominator */
	BL  	udiv<>(SB)
	MOVW	R(q), R(TMP)
	MOVW	4(R13), R(q)
	MOVW	8(R13), R(r)
	MOVW	12(R13), R(s)
	MOVW	16(R13), R(M)
	RET

TEXT _modu(SB), NOSPLIT, $16
	MOVW	R(q), 4(R13)
	MOVW	R(r), 8(R13)
	MOVW	R(s), 12(R13)
	MOVW	R(M), 16(R13)

	MOVW	R(TMP), R(r)		/* numerator */
	MOVW	0(FP), R(q) 		/* denominator */
	BL  	udiv<>(SB)
	MOVW	R(r), R(TMP)
	MOVW	4(R13), R(q)
	MOVW	8(R13), R(r)
	MOVW	12(R13), R(s)
	MOVW	16(R13), R(M)
	RET

TEXT _div(SB),NOSPLIT,$16
	MOVW	R(q), 4(R13)
	MOVW	R(r), 8(R13)
	MOVW	R(s), 12(R13)
	MOVW	R(M), 16(R13)
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
	B		out1
d1:
	CMP 	$0, R(q)
	BGE 	d0
	RSB 	$0, R(q), R(q)
d2:
	BL  	udiv<>(SB)  		/* one neg */
	RSB		$0, R(q), R(TMP)
out1:
	MOVW	4(R13), R(q)
	MOVW	8(R13), R(r)
	MOVW	12(R13), R(s)
	MOVW	16(R13), R(M)
	RET

TEXT _mod(SB),NOSPLIT,$16
	MOVW	R(q), 4(R13)
	MOVW	R(r), 8(R13)
	MOVW	R(s), 12(R13)
	MOVW	R(M), 16(R13)
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
	MOVW	16(R13), R(M)
	RET

// _mul64by32 and _div64by32 not implemented on arm
TEXT runtime·_mul64by32(SB), NOSPLIT, $0
	MOVW	$0, R0
	MOVW	(R0), R1 // crash

TEXT runtime·_div64by32(SB), NOSPLIT, $0
	MOVW	$0, R0
	MOVW	(R0), R1 // crash
