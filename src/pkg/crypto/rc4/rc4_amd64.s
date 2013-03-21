// Original source:
//	http://www.zorinaq.com/papers/rc4-amd64.html
//	http://www.zorinaq.com/papers/rc4-amd64.tar.bz2
//
// Transliterated from GNU to 6a assembly syntax by the Go authors.
// The comments and spacing are from the original.

// The new EXTEND macros avoid a bad stall on some systems after 8-bit math.

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

TEXT ·xorKeyStream(SB),7,$0
	MOVQ	len+16(FP),	BX		// rbx = ARG(len)
	MOVQ	in+8(FP),	SI		// in = ARG(in)
	MOVQ	out+0(FP),	DI		// out = ARG(out)
	MOVQ	d+24(FP),	BP		// d = ARG(data)
	MOVQ	xp+32(FP),	AX
	MOVBQZX	0(AX),		CX		// x = *xp
	MOVQ	yp+40(FP),	AX
	MOVBQZX	0(AX),		DX		// y = *yp

	INCQ	CX				// x++
	ANDQ	$255,		CX		// x &= 0xff
	LEAQ	-8(BX)(SI*1),	BX		// rbx = in+len-8
	MOVQ	BX,		R9		// tmp = in+len-8
	MOVBLZX	(BP)(CX*1),	AX		// tx = d[x]
	CMPQ	BX,		SI		// cmp in with in+len-8
	JLT	end				// jump if (in+len-8 < in)

start:
	ADDQ	$8,		SI		// increment in
	ADDQ	$8,		DI		// increment out
	
	// generate the next 8 bytes of the rc4 stream into R8
	MOVQ	$8,		R11		// byte counter
l1:	ADDB	AX,		DX
	EXTEND(DX)
	MOVBLZX	(BP)(DX*1),	BX		// ty = d[y]
	MOVB	BX,		(BP)(CX*1)	// d[x] = ty
	ADDB	AX,		BX		// val = ty + tx
	EXTEND(BX)
	MOVB	AX,		(BP)(DX*1)	// d[y] = tx
	INCB	CX				// x++ (NEXT ROUND)
	EXTEND(CX)
	MOVBLZX	(BP)(CX*1),	AX		// tx = d[x] (NEXT ROUND)
	SHLQ	$8,		R8
	MOVB	(BP)(BX*1),	R8		// val = d[val]
	DECQ	R11
	JNZ	l1

	// xor 8 bytes
	BSWAPQ	R8
	XORQ	-8(SI),		R8
	CMPQ	SI,		R9		// cmp in+len-8 with in XXX
	MOVQ	R8,		-8(DI)
	JLE	start				// jump if (in <= in+len-8)

end:
	ADDQ	$8,		R9		// tmp = in+len

	// handle the last bytes, one by one
l2:	CMPQ	R9,		SI		// cmp in with in+len
	JLE	finished			// jump if (in+len <= in)
	ADDB	AX,		DX		// y += tx
	EXTEND(DX)
	MOVBLZX	(BP)(DX*1),	BX		// ty = d[y]
	MOVB	BX,		(BP)(CX*1)	// d[x] = ty
	ADDB	AX,		BX		// val = ty+tx
	EXTEND(BX)
	MOVB	AX,		(BP)(DX*1)	// d[y] = tx
	INCB	CX				// x++ (NEXT ROUND)
	EXTEND(CX)
	MOVBLZX	(BP)(CX*1),	AX		// tx = d[x] (NEXT ROUND)
	MOVBLZX	(BP)(BX*1),	R8		// val = d[val]
	XORB	(SI),		R8		// xor 1 byte
	MOVB	R8,		(DI)
	INCQ	SI				// in++
	INCQ	DI				// out++
	JMP l2

finished:
	DECQ	CX				// x--
	MOVQ	yp+40(FP),	BX
	MOVB	DX, 0(BX)
	MOVQ	xp+32(FP),	AX
	MOVB	CX, 0(AX)
	RET
