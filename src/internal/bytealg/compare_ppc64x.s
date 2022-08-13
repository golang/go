// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-56
	// incoming:
	// R3 a addr -> R5
	// R4 a len  -> R3
	// R5 a cap unused
	// R6 b addr -> R6
	// R7 b len  -> R4
	// R8 b cap unused
	MOVD	R3, R5
	MOVD	R4, R3
	MOVD	R7, R4
	CMP     R5,R6,CR7
	CMP	R3,R4,CR6
	BEQ	CR7,equal
	MOVBZ	internal∕cpu·PPC64+const_offsetPPC64HasPOWER9(SB), R16
	CMP	R16,$1
	BNE	power8
	BR	cmpbodyp9<>(SB)
power8:
	BR	cmpbody<>(SB)
equal:
	BEQ	CR6,done
	MOVD	$1, R8
	BGT	CR6,greater
	NEG	R8
greater:
	MOVD	R8, R3
	RET
done:
	MOVD	$0, R3
	RET

TEXT runtime·cmpstring<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-40
	// incoming:
	// R3 a addr -> R5
	// R4 a len  -> R3
	// R5 b addr -> R6
	// R6 b len  -> R4
	MOVD	R6, R7
	MOVD	R5, R6
	MOVD	R3, R5
	MOVD	R4, R3
	MOVD	R7, R4
	CMP     R5,R6,CR7
	CMP	R3,R4,CR6
	BEQ	CR7,equal
	MOVBZ	internal∕cpu·PPC64+const_offsetPPC64HasPOWER9(SB), R16
	CMP	R16,$1
	BNE	power8
	BR	cmpbodyp9<>(SB)
power8:
	BR	cmpbody<>(SB)
equal:
	BEQ	CR6,done
	MOVD	$1, R8
	BGT	CR6,greater
	NEG	R8
greater:
	MOVD	R8, R3
	RET

done:
	MOVD	$0, R3
	RET

#ifdef GOARCH_ppc64le
DATA byteswap<>+0(SB)/8, $0x0706050403020100
DATA byteswap<>+8(SB)/8, $0x0f0e0d0c0b0a0908
GLOBL byteswap<>+0(SB), RODATA, $16
#define SWAP V21
#endif

// Do an efficient memcmp for ppc64le/ppc64/POWER8
// R3 = a len
// R4 = b len
// R5 = a addr
// R6 = b addr
// On exit:
// R3 = return value
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	R3,R8		// set up length
	CMP	R3,R4,CR2	// unequal?
	BLT	CR2,setuplen	// BLT CR2
	MOVD	R4,R8		// use R4 for comparison len
setuplen:
	CMP	R8,$32		// optimize >= 32
	MOVD	R8,R9
	BLT	setup8a		// optimize < 32
	MOVD	$16,R10		// set offsets to load into vectors
	CMP	R8,$64
	BLT	cmp32		// process size 32-63

	DCBT	(R5)		// optimize >= 64
	DCBT	(R6)		// cache hint
	MOVD	$32,R11		// set offsets to load into vector
	MOVD	$48,R12		// set offsets to load into vector

loop64a:// process size 64 and greater
	LXVD2X	(R5)(R0),V3	// load bytes of A at offset 0 into vector
	LXVD2X	(R6)(R0),V4	// load bytes of B at offset 0 into vector
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different	// jump out if its different

	LXVD2X	(R5)(R10),V3	// load bytes of A at offset 16 into vector
	LXVD2X	(R6)(R10),V4	// load bytes of B at offset 16 into vector

	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	LXVD2X	(R5)(R11),V3	// load bytes of A at offset 32 into vector
	LXVD2X	(R6)(R11),V4	// load bytes of B at offset 32 into vector

	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	LXVD2X	(R5)(R12),V3	// load bytes of A at offset 64 into vector
	LXVD2X	(R6)(R12),V4	// load bytes of B at offset 64 into vector

	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	ADD	$-64,R9,R9	// reduce remaining size by 64
	ADD	$64,R5,R5	// increment to next 64 bytes of A
	ADD	$64,R6,R6	// increment to next 64 bytes of B
	CMPU	R9,$64
	BGE	loop64a		// loop back to loop64a only if there are >= 64 bytes remaining
	
	CMPU	R9,$32
	BGE	cmp32		// loop to cmp32 if there are 32-64 bytes remaining
	CMPU	R9,$0
	BNE	rem		// loop to rem if the remainder is not 0

	BEQ	CR2,equal	// remainder is zero, jump to equal if len(A)==len(B)
	BLT	CR2,less	// jump to less if len(A)<len(B)
	BR	greater		// jump to greater otherwise
cmp32:
	LXVD2X	(R5)(R0),V3	// load bytes of A at offset 0 into vector
	LXVD2X	(R6)(R0),V4	// load bytes of B at offset 0 into vector

	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	LXVD2X	(R5)(R10),V3	// load bytes of A at offset 16 into vector
	LXVD2X	(R6)(R10),V4	// load bytes of B at offset 16 into vector

	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	ADD	$-32,R9,R9	// reduce remaining size by 32
	ADD	$32,R5,R5	// increment to next 32 bytes of A
	ADD	$32,R6,R6	// increment to next 32 bytes of B
	CMPU	R9,$0
	BNE	rem		// loop to rem if the remainder is not 0
	BEQ	CR2,equal	// remainder is zero, jump to equal if len(A)==len(B)
	BLT	CR2,less	// jump to less if len(A)<len(B)
	BR	greater		// jump to greater otherwise
rem:
	MOVD	R9,R8
	ANDCC	$24,R8,R9	// Any 8 byte chunks?
	BEQ	leftover	// and result is 0
	BR	setup8a

different:
#ifdef	GOARCH_ppc64le
	MOVD	$byteswap<>+00(SB), R16
	LXVD2X	(R16)(R0),SWAP	// Set up swap string

	VPERM	V3,V3,SWAP,V3
	VPERM	V4,V4,SWAP,V4
#endif
	MFVSRD	VS35,R16	// move upper doublwords of A and B into GPR for comparison
	MFVSRD	VS36,R10

	CMPU	R16,R10
	BEQ	lower
	BGT	greater
	MOVD	$-1,R3		// return value if A < B
	RET
lower:
	VSLDOI	$8,V3,V3,V3	// move lower doublwords of A and B into GPR for comparison
	MFVSRD	VS35,R16
	VSLDOI	$8,V4,V4,V4
	MFVSRD	VS36,R10

	CMPU	R16,R10
	BGT	greater
	MOVD	$-1,R3		// return value if A < B
	RET
setup8a:
	SRADCC	$3,R8,R9	// get the 8 byte count
	BEQ	leftover	// shifted value is 0
	CMPU	R8,$8		// optimize 8byte move
	BEQ	size8
	CMPU	R8,$16
	BEQ	size16
	MOVD	R9,CTR		// loop count for doublewords
loop8:
#ifdef  GOARCH_ppc64le
	MOVDBR	(R5+R0),R16	// doublewords to compare
	MOVDBR	(R6+R0),R10	// LE compare order
#else
	MOVD	(R5+R0),R16	// doublewords to compare
	MOVD	(R6+R0),R10	// BE compare order
#endif
	ADD	$8,R5
	ADD	$8,R6
	CMPU	R16,R10		// match?
	BC	8,2,loop8	// bt ctr <> 0 && cr
	BGT	greater
	BLT	less
leftover:
	ANDCC	$7,R8,R9	// check for leftover bytes
	BEQ	zeroremainder
simplecheck:
	MOVD	R0,R14
	CMP	R9,$4		// process 4 bytes
	BLT	halfword
#ifdef  GOARCH_ppc64le
	MOVWBR	(R5)(R14),R10
	MOVWBR	(R6)(R14),R11
#else
	MOVWZ	(R5)(R14),R10
	MOVWZ	(R6)(R14),R11
#endif
	CMPU	R10,R11
	BGT	greater
	BLT	less
	ADD	$-4,R9
	ADD	$4,R14
	PCALIGN	$16

halfword:
	CMP	R9,$2		// process 2 bytes
	BLT	byte
#ifdef  GOARCH_ppc64le
	MOVHBR	(R5)(R14),R10
	MOVHBR	(R6)(R14),R11
#else
	MOVHZ	(R5)(R14),R10
	MOVHZ	(R6)(R14),R11
#endif
	CMPU	R10,R11
	BGT	greater
	BLT	less
	ADD	$-2,R9
	ADD	$2,R14
	PCALIGN	$16
byte:
	CMP	R9,$0		// process 1 byte
	BEQ	skip
	MOVBZ	(R5)(R14),R10
	MOVBZ	(R6)(R14),R11
	CMPU	R10,R11
	BGT	greater
	BLT	less
	PCALIGN	$16
skip:
	BEQ	CR2,equal
	BGT	CR2,greater

less:	MOVD	$-1,R3		// return value if A < B
	RET
size16:
	LXVD2X	(R5)(R0),V3	// load bytes of A at offset 0 into vector
	LXVD2X	(R6)(R0),V4	// load bytes of B at offset 0 into vector
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different
zeroremainder:
	BEQ	CR2,equal	// remainder is zero, jump to equal if len(A)==len(B)
	BLT	CR2,less	// jump to less if len(A)<len(B)
	BR	greater		// jump to greater otherwise
size8:
#ifdef  GOARCH_ppc64le
	MOVDBR	(R5+R0),R16	// doublewords to compare
	MOVDBR	(R6+R0),R10	// LE compare order
#else
	MOVD	(R5+R0),R16	// doublewords to compare
	MOVD	(R6+R0),R10	// BE compare order
#endif
	CMPU	R16,R10		// match?
	BGT	greater
	BLT	less
	BGT	CR2,greater	// 2nd len > 1st len
	BLT	CR2,less	// 2nd len < 1st len
equal:
	MOVD	$0, R3		// return value if A == B
	RET
greater:
	MOVD	$1,R3		// return value if A > B
	RET

// Do an efficient memcmp for ppc64le/ppc64/POWER9
// R3 = a len
// R4 = b len
// R5 = a addr
// R6 = b addr
// On exit:
// R3 = return value
TEXT cmpbodyp9<>(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	R3,R8		// set up length
	CMP	R3,R4,CR2	// unequal?
	BLT	CR2,setuplen	// BLT CR2
	MOVD	R4,R8		// use R4 for comparison len
setuplen:
	CMP	R8,$16		// optimize for size<16
	MOVD	R8,R9
	BLT	simplecheck
	MOVD	$16,R10		// set offsets to load into vectors
	CMP	R8,$32		// optimize for size 16-31
	BLT	cmp16
	CMP	R8,$64
	BLT	cmp32		// optimize for size 32-63
	DCBT	(R5)		// optimize for size>=64
	DCBT	(R6)		// cache hint

	MOVD	$32,R11		// set offsets to load into vector
	MOVD	$48,R12		// set offsets to load into vector

loop64a:// process size 64 and greater
	LXVB16X	(R0)(R5),V3	// load bytes of A at offset 0 into vector
	LXVB16X	(R0)(R6),V4	// load bytes of B at offset 0 into vector
	VCMPNEBCC	V3,V4,V1	// record comparison into V1
	BNE	CR6,different	// jump out if its different

	LXVB16X	(R10)(R5),V3	// load bytes of A at offset 16 into vector
	LXVB16X	(R10)(R6),V4	// load bytes of B at offset 16 into vector
	VCMPNEBCC	V3,V4,V1
	BNE	CR6,different

	LXVB16X	(R11)(R5),V3	// load bytes of A at offset 32 into vector
	LXVB16X	(R11)(R6),V4	// load bytes of B at offset 32 into vector
	VCMPNEBCC	V3,V4,V1
	BNE	CR6,different

	LXVB16X	(R12)(R5),V3	// load bytes of A at offset 48 into vector
	LXVB16X	(R12)(R6),V4	// load bytes of B at offset 48 into vector
	VCMPNEBCC	V3,V4,V1
	BNE	CR6,different

	ADD	$-64,R9,R9	// reduce remaining size by 64
	ADD	$64,R5,R5	// increment to next 64 bytes of A
	ADD	$64,R6,R6	// increment to next 64 bytes of B
	CMPU	R9,$64
	BGE	loop64a		// loop back to loop64a only if there are >= 64 bytes remaining

	CMPU	R9,$32
	BGE	cmp32		// loop to cmp32 if there are 32-64 bytes remaining
	CMPU	R9,$16
	BGE	cmp16		// loop to cmp16 if there are 16-31 bytes left
	CMPU	R9,$0
	BNE	simplecheck	// loop to simplecheck for remaining bytes

	BEQ	CR2,equal	// remainder is zero, jump to equal if len(A)==len(B)
	BLT	CR2,less	// jump to less if len(A)<len(B)
	BR	greater		// jump to greater otherwise
cmp32:
	LXVB16X	(R0)(R5),V3	// load bytes of A at offset 0 into vector
	LXVB16X	(R0)(R6),V4	// load bytes of B at offset 0 into vector

	VCMPNEBCC	V3,V4,V1	// record comparison into V1
	BNE	CR6,different	// jump out if its different

	LXVB16X	(R10)(R5),V3	// load bytes of A at offset 16 into vector
	LXVB16X	(R10)(R6),V4	// load bytes of B at offset 16 into vector
	VCMPNEBCC	V3,V4,V1
	BNE	CR6,different

	ADD	$-32,R9,R9	// reduce remaining size by 32
	ADD	$32,R5,R5	// increment to next 32 bytes of A
	ADD	$32,R6,R6	// increment to next 32 bytes of B
	CMPU	R9,$16		// loop to cmp16 if there are 16-31 bytes left
	BGE	cmp16
	CMPU	R9,$0
	BNE	simplecheck	// loop to simplecheck for remainder bytes
	BEQ	CR2,equal	// remainder is zero, jump to equal if len(A)==len(B)
	BLT	CR2,less	// jump to less if len(A)<len(B)
	BR	greater		// jump to greater otherwise
different:

	MFVSRD	VS35,R16	// move upper doublwords of A and B into GPR for comparison
	MFVSRD	VS36,R10

	CMPU	R16,R10
	BEQ	lower
	BGT	greater
	MOVD	$-1,R3		// return value if A < B
	RET
lower:
	MFVSRLD	VS35,R16	// next move lower doublewords of A and B into GPR for comparison
	MFVSRLD	VS36,R10

	CMPU	R16,R10
	BGT	greater
	MOVD	$-1,R3		// return value if A < B
	RET

greater:
	MOVD	$1,R3		// return value if A > B
	RET
cmp16:
	ANDCC	$16,R9,R31
	BEQ	tail

	LXVB16X	(R0)(R5),V3	// load bytes of A at offset 16 into vector
	LXVB16X	(R0)(R6),V4	// load bytes of B at offset 16 into vector
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	ADD	$16,R5
	ADD	$16,R6
tail:
	ANDCC	$15,R9		// Load the last 16 bytes (we know there are at least 32b)
	BEQ	end

	ADD	R9,R5
	ADD	R9,R6
	MOVD	$-16,R10

	LXVB16X	(R10)(R5),V3	// load bytes of A at offset 16 into vector
	LXVB16X	(R10)(R6),V4	// load bytes of B at offset 16 into vector
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different
end:
	BEQ	CR2,equal	// remainder is zero, jump to equal if len(A)==len(B)
	BLT	CR2,less	// jump to less if BLT CR2 that is, len(A)<len(B)
	BR	greater		// jump to greater otherwise
simplecheck:
	MOVD	$0,R14		// process 8 bytes
	CMP	R9,$8
	BLT	word
#ifdef  GOARCH_ppc64le
	MOVDBR	(R5+R14),R10
	MOVDBR	(R6+R14),R11
#else
	MOVD	(R5+R14),R10
	MOVD	(R6+R14),R11
#endif
	CMPU	R10,R11
	BGT	greater
	BLT	less
	ADD	$8,R14
	ADD	$-8,R9
	PCALIGN	$16
word:
	CMP	R9,$4		// process 4 bytes
	BLT	halfword
#ifdef  GOARCH_ppc64le
	MOVWBR	(R5+R14),R10
	MOVWBR	(R6+R14),R11
#else
	MOVWZ	(R5+R14),R10
	MOVWZ	(R6+R14),R11
#endif
	CMPU	R10,R11
	BGT	greater
	BLT	less
	ADD	$4,R14
	ADD	$-4,R9
	PCALIGN	$16
halfword:
	CMP	R9,$2		// process 2 bytes
	BLT	byte
#ifdef  GOARCH_ppc64le
	MOVHBR	(R5+R14),R10
	MOVHBR	(R6+R14),R11
#else
	MOVHZ	(R5+R14),R10
	MOVHZ	(R6+R14),R11
#endif
	CMPU	R10,R11
	BGT	greater
	BLT	less
	ADD	$2,R14
	ADD	$-2,R9
	PCALIGN	$16
byte:
	CMP	R9,$0		// process 1 byte
	BEQ	skip
	MOVBZ	(R5+R14),R10
	MOVBZ	(R6+R14),R11
	CMPU	R10,R11
	BGT	greater
	BLT	less
	PCALIGN	$16
skip:
	BEQ	CR2,equal
	BGT	CR2,greater
less:
	MOVD	$-1,R3		// return value if A < B
	RET
equal:
	MOVD	$0, R3		// return value if A == B
	RET
