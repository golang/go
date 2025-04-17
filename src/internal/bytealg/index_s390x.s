// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// Caller must confirm availability of vx facility before calling.
TEXT ·Index(SB),NOSPLIT|NOFRAME,$0-56
	LMG	a_base+0(FP), R1, R2  // R1=&s[0],   R2=len(s)
	LMG	b_base+24(FP), R3, R4 // R3=&sep[0], R4=len(sep)
	MOVD	$ret+48(FP), R5
	BR	indexbody<>(SB)

// Caller must confirm availability of vx facility before calling.
TEXT ·IndexString(SB),NOSPLIT|NOFRAME,$0-40
	LMG	a_base+0(FP), R1, R2  // R1=&s[0],   R2=len(s)
	LMG	b_base+16(FP), R3, R4 // R3=&sep[0], R4=len(sep)
	MOVD	$ret+32(FP), R5
	BR	indexbody<>(SB)

// s: string we are searching
// sep: string to search for
// R1=&s[0], R2=len(s)
// R3=&sep[0], R4=len(sep)
// R5=&ret (int)
// Caller must confirm availability of vx facility before calling.
TEXT indexbody<>(SB),NOSPLIT|NOFRAME,$0
	CMPBGT	R4, R2, notfound
	ADD	R1, R2
	SUB	R4, R2 // R2=&s[len(s)-len(sep)] (last valid index)
	CMPBEQ	R4, $0, notfound
	SUB	$1, R4 // R4=len(sep)-1 for use as VLL index
	VLL	R4, (R3), V0 // contains first 16 bytes of sep
	MOVD	R1, R7
index2plus:
	CMPBNE	R4, $1, index3plus
	MOVD	$15(R7), R9
	CMPBGE	R9, R2, index2to16
	VGBM	$0xaaaa, V31       // 0xff00ff00ff00ff00...
	VONE	V16
	VREPH	$0, V0, V1
	CMPBGE	R9, R2, index2to16
index2loop:
	VL	0(R7), V2          // 16 bytes, even indices
	VL	1(R7), V4          // 16 bytes, odd indices
	VCEQH	V1, V2, V5         // compare even indices
	VCEQH	V1, V4, V6         // compare odd indices
	VSEL	V5, V6, V31, V7    // merge even and odd indices
	VFEEBS	V16, V7, V17       // find leftmost index, set condition to 1 if found
	BLT	foundV17
	MOVD	$16(R7), R7        // R7+=16
	ADD	$15, R7, R9
	CMPBLE	R9, R2, index2loop // continue if (R7+15) <= R2 (last index to search)
	CMPBLE	R7, R2, index2to16
	BR	notfound

index3plus:
	CMPBNE	R4, $2, index4plus
	ADD	$15, R7, R9
	CMPBGE	R9, R2, index2to16
	MOVD	$1, R0
	VGBM	$0xaaaa, V31       // 0xff00ff00ff00ff00...
	VONE	V16
	VREPH	$0, V0, V1
	VREPB	$2, V0, V8
index3loop:
	VL	(R7), V2           // load 16-bytes into V2
	VLL	R0, 16(R7), V3     // load 2-bytes into V3
	VSLDB	$1, V2, V3, V4     // V4=(V2:V3)<<1
	VSLDB	$2, V2, V3, V9     // V9=(V2:V3)<<2
	VCEQH	V1, V2, V5         // compare 2-byte even indices
	VCEQH	V1, V4, V6         // compare 2-byte odd indices
	VCEQB	V8, V9, V10        // compare last bytes
	VSEL	V5, V6, V31, V7    // merge even and odd indices
	VN	V7, V10, V7        // AND indices with last byte
	VFEEBS	V16, V7, V17       // find leftmost index, set condition to 1 if found
	BLT	foundV17
	MOVD	$16(R7), R7        // R7+=16
	ADD	$15, R7, R9
	CMPBLE	R9, R2, index3loop // continue if (R7+15) <= R2 (last index to search)
	CMPBLE	R7, R2, index2to16
	BR	notfound

index4plus:
	CMPBNE	R4, $3, index5plus
	ADD	$15, R7, R9
	CMPBGE	R9, R2, index2to16
	MOVD	$2, R0
	VGBM	$0x8888, V29       // 0xff000000ff000000...
	VGBM	$0x2222, V30       // 0x0000ff000000ff00...
	VGBM	$0xcccc, V31       // 0xffff0000ffff0000...
	VONE	V16
	VREPF	$0, V0, V1
index4loop:
	VL	(R7), V2           // load 16-bytes into V2
	VLL	R0, 16(R7), V3     // load 3-bytes into V3
	VSLDB	$1, V2, V3, V4     // V4=(V2:V3)<<1
	VSLDB	$2, V2, V3, V9     // V9=(V2:V3)<<1
	VSLDB	$3, V2, V3, V10    // V10=(V2:V3)<<1
	VCEQF	V1, V2, V5         // compare index 0, 4, ...
	VCEQF	V1, V4, V6         // compare index 1, 5, ...
	VCEQF	V1, V9, V11        // compare index 2, 6, ...
	VCEQF	V1, V10, V12       // compare index 3, 7, ...
	VSEL	V5, V6, V29, V13   // merge index 0, 1, 4, 5, ...
	VSEL	V11, V12, V30, V14 // merge index 2, 3, 6, 7, ...
	VSEL	V13, V14, V31, V7  // final merge
	VFEEBS	V16, V7, V17       // find leftmost index, set condition to 1 if found
	BLT	foundV17
	MOVD	$16(R7), R7        // R7+=16
	ADD	$15, R7, R9
	CMPBLE	R9, R2, index4loop // continue if (R7+15) <= R2 (last index to search)
	CMPBLE	R7, R2, index2to16
	BR	notfound

index5plus:
	CMPBGT	R4, $15, index17plus
index2to16:
	CMPBGT	R7, R2, notfound
	MOVD	$1(R7), R8
	CMPBGT	R8, R2, index2to16tail
index2to16loop:
	// unrolled 2x
	VLL	R4, (R7), V1
	VLL	R4, 1(R7), V2
	VCEQGS	V0, V1, V3
	BEQ	found
	MOVD	$1(R7), R7
	VCEQGS	V0, V2, V4
	BEQ	found
	MOVD	$1(R7), R7
	CMPBLT	R7, R2, index2to16loop
	CMPBGT	R7, R2, notfound
index2to16tail:
	VLL	R4, (R7), V1
	VCEQGS	V0, V1, V2
	BEQ	found
	BR	notfound

index17plus:
	CMPBGT	R4, $31, index33plus
	SUB	$16, R4, R0
	VLL	R0, 16(R3), V1
	VONE	V7
index17to32loop:
	VL	(R7), V2
	VLL	R0, 16(R7), V3
	VCEQG	V0, V2, V4
	VCEQG	V1, V3, V5
	VN	V4, V5, V6
	VCEQGS	V6, V7, V8
	BEQ	found
	MOVD	$1(R7), R7
	CMPBLE  R7, R2, index17to32loop
	BR	notfound

index33plus:
	CMPBGT	R4, $47, index49plus
	SUB	$32, R4, R0
	VL	16(R3), V1
	VLL	R0, 32(R3), V2
	VONE	V11
index33to48loop:
	VL	(R7), V3
	VL	16(R7), V4
	VLL	R0, 32(R7), V5
	VCEQG	V0, V3, V6
	VCEQG	V1, V4, V7
	VCEQG	V2, V5, V8
	VN	V6, V7, V9
	VN	V8, V9, V10
	VCEQGS	V10, V11, V12
	BEQ	found
	MOVD	$1(R7), R7
	CMPBLE  R7, R2, index33to48loop
	BR	notfound

index49plus:
	CMPBGT	R4, $63, index65plus
	SUB	$48, R4, R0
	VL	16(R3), V1
	VL	32(R3), V2
	VLL	R0, 48(R3), V3
	VONE	V15
index49to64loop:
	VL	(R7), V4
	VL	16(R7), V5
	VL	32(R7), V6
	VLL	R0, 48(R7), V7
	VCEQG	V0, V4, V8
	VCEQG	V1, V5, V9
	VCEQG	V2, V6, V10
	VCEQG	V3, V7, V11
	VN	V8, V9, V12
	VN	V10, V11, V13
	VN	V12, V13, V14
	VCEQGS	V14, V15, V16
	BEQ	found
	MOVD	$1(R7), R7
	CMPBLE  R7, R2, index49to64loop
notfound:
	MOVD	$-1, (R5)
	RET

index65plus:
	// not implemented
	MOVD	$0, (R0)
	RET

foundV17: // index is in doubleword V17[0]
	VLGVG	$0, V17, R8
	ADD	R8, R7
found:
	SUB	R1, R7
	MOVD	R7, (R5)
	RET
