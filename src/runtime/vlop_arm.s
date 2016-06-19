// Inferno's libkern/vlop-arm.s
// http://code.google.com/p/inferno-os/source/browse/libkern/vlop-arm.s
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

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

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
TEXT runtime·_sfloatpanic(SB),NOSPLIT,$-4
	MOVW	$0, R0
	MOVW.W	R0, -4(R13)
	MOVW	g_sigpc(g), LR
	B	runtime·sigpanic(SB)

// func udiv(n, d uint32) (q, r uint32)
// Reference: 
// Sloss, Andrew et. al; ARM System Developer's Guide: Designing and Optimizing System Software
// Morgan Kaufmann; 1 edition (April 8, 2004), ISBN 978-1558608740
#define Rq	R0 // input d, output q
#define Rr	R1 // input n, output r
#define Rs	R2 // three temporary variables
#define RM	R3
#define Ra	R11

// Be careful: Ra == R11 will be used by the linker for synthesized instructions.
TEXT udiv<>(SB),NOSPLIT,$-4
	CLZ 	Rq, Rs // find normalizing shift
	MOVW.S	Rq<<Rs, Ra
	MOVW	$fast_udiv_tab<>-64(SB), RM
	ADD.NE	Ra>>25, RM, Ra // index by most significant 7 bits of divisor
	MOVBU.NE	(Ra), Ra

	SUB.S	$7, Rs
	RSB 	$0, Rq, RM // M = -q
	MOVW.PL	Ra<<Rs, Rq

	// 1st Newton iteration
	MUL.PL	RM, Rq, Ra // a = -q*d
	BMI 	udiv_by_large_d
	MULAWT	Ra, Rq, Rq, Rq // q approx q-(q*q*d>>32)
	TEQ 	RM->1, RM // check for d=0 or d=1

	// 2nd Newton iteration
	MUL.NE	RM, Rq, Ra
	MOVW.NE	$0, Rs
	MULAL.NE Rq, Ra, (Rq,Rs)
	BEQ 	udiv_by_0_or_1

	// q now accurate enough for a remainder r, 0<=r<3*d
	MULLU	Rq, Rr, (Rq,Rs) // q = (r * q) >> 32
	ADD 	RM, Rr, Rr // r = n - d
	MULA	RM, Rq, Rr, Rr // r = n - (q+1)*d

	// since 0 <= n-q*d < 3*d; thus -d <= r < 2*d
	CMN 	RM, Rr // t = r-d
	SUB.CS	RM, Rr, Rr // if (t<-d || t>=0) r=r+d
	ADD.CC	$1, Rq
	ADD.PL	RM<<1, Rr
	ADD.PL	$2, Rq
	RET

udiv_by_large_d:
	// at this point we know d>=2^(31-6)=2^25
	SUB 	$4, Ra, Ra
	RSB 	$0, Rs, Rs
	MOVW	Ra>>Rs, Rq
	MULLU	Rq, Rr, (Rq,Rs)
	MULA	RM, Rq, Rr, Rr

	// q now accurate enough for a remainder r, 0<=r<4*d
	CMN 	Rr>>1, RM // if(r/2 >= d)
	ADD.CS	RM<<1, Rr
	ADD.CS	$2, Rq
	CMN 	Rr, RM
	ADD.CS	RM, Rr
	ADD.CS	$1, Rq
	RET

udiv_by_0_or_1:
	// carry set if d==1, carry clear if d==0
	BCC udiv_by_0
	MOVW	Rr, Rq
	MOVW	$0, Rr
	RET

udiv_by_0:
	MOVW	$runtime·panicdivide(SB), R11
	B	(R11)

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

// The linker will pass numerator in RTMP, and it also
// expects the result in RTMP
#define RTMP R11

TEXT _divu(SB), NOSPLIT, $16-0
	// It's not strictly true that there are no local pointers.
	// It could be that the saved registers Rq, Rr, Rs, and Rm
	// contain pointers. However, the only way this can matter
	// is if the stack grows (which it can't, udiv is nosplit)
	// or if a fault happens and more frames are added to
	// the stack due to deferred functions.
	// In the latter case, the stack can grow arbitrarily,
	// and garbage collection can happen, and those
	// operations care about pointers, but in that case
	// the calling frame is dead, and so are the saved
	// registers. So we can claim there are no pointers here.
	NO_LOCAL_POINTERS
	MOVW	Rq, 4(R13)
	MOVW	Rr, 8(R13)
	MOVW	Rs, 12(R13)
	MOVW	RM, 16(R13)

	MOVW	RTMP, Rr		/* numerator */
	MOVW	g_m(g), Rq
	MOVW	m_divmod(Rq), Rq	/* denominator */
	BL  	udiv<>(SB)
	MOVW	Rq, RTMP
	MOVW	4(R13), Rq
	MOVW	8(R13), Rr
	MOVW	12(R13), Rs
	MOVW	16(R13), RM
	RET

TEXT _modu(SB), NOSPLIT, $16-0
	NO_LOCAL_POINTERS
	MOVW	Rq, 4(R13)
	MOVW	Rr, 8(R13)
	MOVW	Rs, 12(R13)
	MOVW	RM, 16(R13)

	MOVW	RTMP, Rr		/* numerator */
	MOVW	g_m(g), Rq
	MOVW	m_divmod(Rq), Rq	/* denominator */
	BL  	udiv<>(SB)
	MOVW	Rr, RTMP
	MOVW	4(R13), Rq
	MOVW	8(R13), Rr
	MOVW	12(R13), Rs
	MOVW	16(R13), RM
	RET

TEXT _div(SB),NOSPLIT,$16-0
	NO_LOCAL_POINTERS
	MOVW	Rq, 4(R13)
	MOVW	Rr, 8(R13)
	MOVW	Rs, 12(R13)
	MOVW	RM, 16(R13)
	MOVW	RTMP, Rr		/* numerator */
	MOVW	g_m(g), Rq
	MOVW	m_divmod(Rq), Rq	/* denominator */
	CMP 	$0, Rr
	BGE 	d1
	RSB 	$0, Rr, Rr
	CMP 	$0, Rq
	BGE 	d2
	RSB 	$0, Rq, Rq
d0:
	BL  	udiv<>(SB)  		/* none/both neg */
	MOVW	Rq, RTMP
	B		out1
d1:
	CMP 	$0, Rq
	BGE 	d0
	RSB 	$0, Rq, Rq
d2:
	BL  	udiv<>(SB)  		/* one neg */
	RSB		$0, Rq, RTMP
out1:
	MOVW	4(R13), Rq
	MOVW	8(R13), Rr
	MOVW	12(R13), Rs
	MOVW	16(R13), RM
	RET

TEXT _mod(SB),NOSPLIT,$16-0
	NO_LOCAL_POINTERS
	MOVW	Rq, 4(R13)
	MOVW	Rr, 8(R13)
	MOVW	Rs, 12(R13)
	MOVW	RM, 16(R13)
	MOVW	RTMP, Rr		/* numerator */
	MOVW	g_m(g), Rq
	MOVW	m_divmod(Rq), Rq	/* denominator */
	CMP 	$0, Rq
	RSB.LT	$0, Rq, Rq
	CMP 	$0, Rr
	BGE 	m1
	RSB 	$0, Rr, Rr
	BL  	udiv<>(SB)  		/* neg numerator */
	RSB 	$0, Rr, RTMP
	B   	out
m1:
	BL  	udiv<>(SB)  		/* pos numerator */
	MOVW	Rr, RTMP
out:
	MOVW	4(R13), Rq
	MOVW	8(R13), Rr
	MOVW	12(R13), Rs
	MOVW	16(R13), RM
	RET

// _mul64by32 and _div64by32 not implemented on arm
TEXT runtime·_mul64by32(SB), NOSPLIT, $0
	MOVW	$0, R0
	MOVW	(R0), R1 // crash

TEXT runtime·_div64by32(SB), NOSPLIT, $0
	MOVW	$0, R0
	MOVW	(R0), R1 // crash
