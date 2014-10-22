// Original source:
//	http://www.zorinaq.com/papers/rc4-amd64.html
//	http://www.zorinaq.com/papers/rc4-amd64.tar.bz2

#include "textflag.h"

// Local modifications:
//
// Transliterated from GNU to 6a assembly syntax by the Go authors.
// The comments and spacing are from the original.
//
// The new EXTEND macros avoid a bad stall on some systems after 8-bit math.
//
// The original code accumulated 64 bits of key stream in an integer
// register and then XOR'ed the key stream into the data 8 bytes at a time.
// Modified to accumulate 128 bits of key stream into an XMM register
// and then XOR the key stream into the data 16 bytes at a time.
// Approximately doubles throughput.
//
// Converted to amd64p32.
//
// To make safe for Native Client, avoid use of BP, R15,
// and two-register addressing modes.

// NOTE: Changing EXTEND to a no-op makes the code run 1.2x faster on Core i5
// but makes the code run 2.0x slower on Xeon.
#define EXTEND(r) MOVBLZX r, r

/*
** RC4 implementation optimized for AMD64.
**
** Author: Marc Bevand <bevand_m (at) epita.fr>
** Licence: I hereby disclaim the copyright on this code and place it
** in the public domain.
**
** The code has been designed to be easily integrated into openssl:
** the exported RC4() function can replace the actual implementations
** openssl already contains. Please note that when linking with openssl,
** it requires that sizeof(RC4_INT) == 8. So openssl must be compiled
** with -DRC4_INT='unsigned long'.
**
** The throughput achieved by this code is about 320 MBytes/sec, on
** a 1.8 GHz AMD Opteron (rev C0) processor.
*/

TEXT ·xorKeyStream(SB),NOSPLIT,$0
	MOVL	n+8(FP),	BX		// rbx = ARG(len)
	MOVL	src+4(FP),	SI		// in = ARG(in)
	MOVL	dst+0(FP),	DI		// out = ARG(out)
	MOVL	state+12(FP),	R10		// d = ARG(data)
	MOVL	i+16(FP),	AX
	MOVBQZX	0(AX),		CX		// x = *xp
	MOVL	j+20(FP),	AX
	MOVBQZX	0(AX),		DX		// y = *yp

	LEAQ	(SI)(BX*1),	R9		// limit = in+len

l1:	CMPQ	SI,		R9		// cmp in with in+len
	JGE	finished			// jump if (in >= in+len)

	INCB	CX
	EXTEND(CX)
	TESTL	$15,		CX
	JZ	wordloop
	LEAL	(R10)(CX*4), R12

	MOVBLZX	(R12),	AX

	ADDB	AX,		DX		// y += tx
	EXTEND(DX)
	LEAL (R10)(DX*4), R11
	MOVBLZX	(R11),	BX		// ty = d[y]
	MOVB	BX,		(R12)	// d[x] = ty
	ADDB	AX,		BX		// val = ty+tx
	EXTEND(BX)
	LEAL (R10)(BX*4), R13
	MOVB	AX,		(R11)	// d[y] = tx
	MOVBLZX	(R13),	R8		// val = d[val]
	XORB	(SI),		R8		// xor 1 byte
	MOVB	R8,		(DI)
	INCQ	SI				// in++
	INCQ	DI				// out++
	JMP l1

wordloop:
	SUBQ	$16,		R9
	CMPQ	SI,		R9
	JGT	end

start:
	ADDQ	$16,		SI		// increment in
	ADDQ	$16,		DI		// increment out

	// Each KEYROUND generates one byte of key and
	// inserts it into an XMM register at the given 16-bit index.
	// The key state array is uint32 words only using the bottom
	// byte of each word, so the 16-bit OR only copies 8 useful bits.
	// We accumulate alternating bytes into X0 and X1, and then at
	// the end we OR X1<<8 into X0 to produce the actual key.
	//
	// At the beginning of the loop, CX%16 == 0, so the 16 loads
	// at state[CX], state[CX+1], ..., state[CX+15] can precompute
	// (state+CX) as R12 and then become R12[0], R12[1], ... R12[15],
	// without fear of the byte computation CX+15 wrapping around.
	//
	// The first round needs R12[0], the second needs R12[1], and so on.
	// We can avoid memory stalls by starting the load for round n+1
	// before the end of round n, using the LOAD macro.
	LEAQ	(R10)(CX*4),	R12

#define KEYROUND(xmm, load, off, r1, r2, index) \
	LEAL (R10)(DX*4), R11; \
	MOVBLZX	(R11),	R8; \
	MOVB	r1,		(R11); \
	load((off+1), r2); \
	MOVB	R8,		(off*4)(R12); \
	ADDB	r1,		R8; \
	EXTEND(R8); \
	LEAL (R10)(R8*4), R14; \
	PINSRW	$index, (R14), xmm

#define LOAD(off, reg) \
	MOVBLZX	(off*4)(R12),	reg; \
	ADDB	reg,		DX; \
	EXTEND(DX)

#define SKIP(off, reg)

	LOAD(0, AX)
	KEYROUND(X0, LOAD, 0, AX, BX, 0)
	KEYROUND(X1, LOAD, 1, BX, AX, 0)
	KEYROUND(X0, LOAD, 2, AX, BX, 1)
	KEYROUND(X1, LOAD, 3, BX, AX, 1)
	KEYROUND(X0, LOAD, 4, AX, BX, 2)
	KEYROUND(X1, LOAD, 5, BX, AX, 2)
	KEYROUND(X0, LOAD, 6, AX, BX, 3)
	KEYROUND(X1, LOAD, 7, BX, AX, 3)
	KEYROUND(X0, LOAD, 8, AX, BX, 4)
	KEYROUND(X1, LOAD, 9, BX, AX, 4)
	KEYROUND(X0, LOAD, 10, AX, BX, 5)
	KEYROUND(X1, LOAD, 11, BX, AX, 5)
	KEYROUND(X0, LOAD, 12, AX, BX, 6)
	KEYROUND(X1, LOAD, 13, BX, AX, 6)
	KEYROUND(X0, LOAD, 14, AX, BX, 7)
	KEYROUND(X1, SKIP, 15, BX, AX, 7)
	
	ADDB	$16,		CX

	PSLLQ	$8,		X1
	PXOR	X1,		X0
	MOVOU	-16(SI),	X2
	PXOR	X0,		X2
	MOVOU	X2,		-16(DI)

	CMPQ	SI,		R9		// cmp in with in+len-16
	JLE	start				// jump if (in <= in+len-16)

end:
	DECB	CX
	ADDQ	$16,		R9		// tmp = in+len

	// handle the last bytes, one by one
l2:	CMPQ	SI,		R9		// cmp in with in+len
	JGE	finished			// jump if (in >= in+len)

	INCB	CX
	EXTEND(CX)
	LEAL (R10)(CX*4), R12
	MOVBLZX	(R12),	AX

	ADDB	AX,		DX		// y += tx
	EXTEND(DX)
	LEAL (R10)(DX*4), R11
	MOVBLZX	(R11),	BX		// ty = d[y]
	MOVB	BX,		(R12)	// d[x] = ty
	ADDB	AX,		BX		// val = ty+tx
	EXTEND(BX)
	LEAL (R10)(BX*4), R13
	MOVB	AX,		(R11)	// d[y] = tx
	MOVBLZX	(R13),	R8		// val = d[val]
	XORB	(SI),		R8		// xor 1 byte
	MOVB	R8,		(DI)
	INCQ	SI				// in++
	INCQ	DI				// out++
	JMP l2

finished:
	MOVL	j+20(FP),	BX
	MOVB	DX, 0(BX)
	MOVL	i+16(FP),	AX
	MOVB	CX, 0(AX)
	RET
