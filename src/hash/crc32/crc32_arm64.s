// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Fold constants (K(8k+63), K(8k-1)) for k = 128, 112, ..., 16 bytes,
// where K(e) = bitreverse32(x^e mod 0x11EDC6F41) << 32.
DATA	crc32cFoldConsts<>+0(SB)/8, $0x6577b24500000000
DATA	crc32cFoldConsts<>+8(SB)/8, $0x7417153f00000000
DATA	crc32cFoldConsts<>+16(SB)/8, $0x169472b600000000
DATA	crc32cFoldConsts<>+24(SB)/8, $0x963e61cd00000000
DATA	crc32cFoldConsts<>+32(SB)/8, $0xc92f998d00000000
DATA	crc32cFoldConsts<>+40(SB)/8, $0x3365346a00000000
DATA	crc32cFoldConsts<>+48(SB)/8, $0x1c42da4300000000
DATA	crc32cFoldConsts<>+56(SB)/8, $0x6d883e3800000000
DATA	crc32cFoldConsts<>+64(SB)/8, $0x1c19243b00000000
DATA	crc32cFoldConsts<>+72(SB)/8, $0x75bba45b00000000
DATA	crc32cFoldConsts<>+80(SB)/8, $0xa46ef4aa00000000
DATA	crc32cFoldConsts<>+88(SB)/8, $0x6051243f00000000
DATA	crc32cFoldConsts<>+96(SB)/8, $0x33ccbbbc00000000
DATA	crc32cFoldConsts<>+104(SB)/8, $0xa2158b3400000000
DATA	crc32cFoldConsts<>+112(SB)/8, $0x3743f7bd00000000
DATA	crc32cFoldConsts<>+120(SB)/8, $0x3171d43000000000
GLOBL	crc32cFoldConsts<>(SB), RODATA, $128

// castagnoliUpdate updates the non-inverted crc with the given data.

// func castagnoliUpdate(crc uint32, p []byte) uint32
TEXT ·castagnoliUpdate(SB),NOSPLIT,$0-36
	MOVWU	crc+0(FP), R9  // CRC value
	MOVD	p+8(FP), R13  // data pointer
	MOVD	p_len+16(FP), R11  // len(p)
	MOVD	$ret+32(FP), R14  // result address
	B	castagnoliTail<>(SB)

// castagnoliTail takes R9 crc, R13 data, R11 length, R14 result address.
TEXT castagnoliTail<>(SB),NOSPLIT|NOFRAME,$0-0
update:
	CMP	$16, R11
	BLT	less_than_16
	LDP.P	16(R13), (R8, R10)
	CRC32CX	R8, R9
	CRC32CX	R10, R9
	SUB	$16, R11

	JMP	update

less_than_16:
	TBZ	$3, R11, less_than_8

	MOVD.P	8(R13), R10
	CRC32CX	R10, R9

less_than_8:
	TBZ	$2, R11, less_than_4

	MOVWU.P	4(R13), R10
	CRC32CW	R10, R9

less_than_4:
	TBZ	$1, R11, less_than_2

	MOVHU.P	2(R13), R10
	CRC32CH	R10, R9

less_than_2:
	TBZ	$0, R11, done

	MOVBU	(R13), R10
	CRC32CB	R10, R9

done:
	MOVW	R9, (R14)
	RET

// castagnoliUpdatePMULL updates the non-inverted crc with the given data.

// func castagnoliUpdatePMULL(crc uint32, p []byte) uint32
TEXT ·castagnoliUpdatePMULL(SB),NOSPLIT,$0-36
	MOVWU	crc+0(FP), R9  // CRC value
	MOVD	p+8(FP), R13  // data pointer
	MOVD	p_len+16(FP), R11  // len(p)
	MOVD	$ret+32(FP), R14  // result address

	CMP	$256, R11
	BGE	fold
	B	castagnoliTail<>(SB)

fold:
	MOVD	$crc32cFoldConsts<>(SB), R5
	VLD1.P	64(R5), [V8.B16, V9.B16, V10.B16, V11.B16]
	VLD1	(R5), [V12.B16, V13.B16, V14.B16, V15.B16]

	VLD1.P	64(R13), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD1.P	64(R13), [V4.B16, V5.B16, V6.B16, V7.B16]
	FMOVS	R9, F24
	VEOR	V24.B16, V0.B16, V0.B16
	SUB	$128, R11

	// Data is xored into the low product to keep it off the loop-carried path.
loop128:
	VLD1.P	64(R13), [V16.B16, V17.B16, V18.B16, V19.B16]
	VLD1.P	64(R13), [V20.B16, V21.B16, V22.B16, V23.B16]
	VPMULL	V0.D1, V8.D1, V24.Q1
	VPMULL2	V0.D2, V8.D2, V0.Q1
	VPMULL	V1.D1, V8.D1, V25.Q1
	VPMULL2	V1.D2, V8.D2, V1.Q1
	VPMULL	V2.D1, V8.D1, V26.Q1
	VPMULL2	V2.D2, V8.D2, V2.Q1
	VPMULL	V3.D1, V8.D1, V27.Q1
	VPMULL2	V3.D2, V8.D2, V3.Q1
	VPMULL	V4.D1, V8.D1, V28.Q1
	VPMULL2	V4.D2, V8.D2, V4.Q1
	VPMULL	V5.D1, V8.D1, V29.Q1
	VPMULL2	V5.D2, V8.D2, V5.Q1
	VPMULL	V6.D1, V8.D1, V30.Q1
	VPMULL2	V6.D2, V8.D2, V6.Q1
	VPMULL	V7.D1, V8.D1, V31.Q1
	VPMULL2	V7.D2, V8.D2, V7.Q1
	VEOR	V16.B16, V24.B16, V24.B16
	VEOR	V17.B16, V25.B16, V25.B16
	VEOR	V18.B16, V26.B16, V26.B16
	VEOR	V19.B16, V27.B16, V27.B16
	VEOR	V20.B16, V28.B16, V28.B16
	VEOR	V21.B16, V29.B16, V29.B16
	VEOR	V22.B16, V30.B16, V30.B16
	VEOR	V23.B16, V31.B16, V31.B16
	VEOR	V24.B16, V0.B16, V0.B16
	VEOR	V25.B16, V1.B16, V1.B16
	VEOR	V26.B16, V2.B16, V2.B16
	VEOR	V27.B16, V3.B16, V3.B16
	VEOR	V28.B16, V4.B16, V4.B16
	VEOR	V29.B16, V5.B16, V5.B16
	VEOR	V30.B16, V6.B16, V6.B16
	VEOR	V31.B16, V7.B16, V7.B16
	SUB	$128, R11
	CMP	$128, R11
	BGE	loop128

	// Fold V0-V6 into V7, then reduce to 32 bits.
	VPMULL	V0.D1, V9.D1, V24.Q1
	VPMULL2	V0.D2, V9.D2, V0.Q1
	VPMULL	V1.D1, V10.D1, V25.Q1
	VPMULL2	V1.D2, V10.D2, V1.Q1
	VPMULL	V2.D1, V11.D1, V26.Q1
	VPMULL2	V2.D2, V11.D2, V2.Q1
	VPMULL	V3.D1, V12.D1, V27.Q1
	VPMULL2	V3.D2, V12.D2, V3.Q1
	VPMULL	V4.D1, V13.D1, V28.Q1
	VPMULL2	V4.D2, V13.D2, V4.Q1
	VPMULL	V5.D1, V14.D1, V29.Q1
	VPMULL2	V5.D2, V14.D2, V5.Q1
	VPMULL	V6.D1, V15.D1, V30.Q1
	VPMULL2	V6.D2, V15.D2, V6.Q1
	VEOR	V24.B16, V0.B16, V0.B16
	VEOR	V25.B16, V1.B16, V1.B16
	VEOR	V26.B16, V2.B16, V2.B16
	VEOR	V27.B16, V3.B16, V3.B16
	VEOR	V28.B16, V4.B16, V4.B16
	VEOR	V29.B16, V5.B16, V5.B16
	VEOR	V30.B16, V6.B16, V6.B16
	VEOR	V0.B16, V1.B16, V1.B16
	VEOR	V2.B16, V3.B16, V3.B16
	VEOR	V4.B16, V5.B16, V5.B16
	VEOR	V6.B16, V7.B16, V7.B16
	VEOR	V1.B16, V3.B16, V3.B16
	VEOR	V5.B16, V7.B16, V7.B16
	VEOR	V3.B16, V7.B16, V7.B16
	VMOV	V7.D[0], R8
	VMOV	V7.D[1], R10
	MOVD	$0, R9
	CRC32CX	R8, R9
	CRC32CX	R10, R9
	B	castagnoliTail<>(SB)

// ieeeUpdate updates the non-inverted crc with the given data.

// func ieeeUpdate(crc uint32, p []byte) uint32
TEXT ·ieeeUpdate(SB),NOSPLIT,$0-36
	MOVWU	crc+0(FP), R9  // CRC value
	MOVD	p+8(FP), R13  // data pointer
	MOVD	p_len+16(FP), R11  // len(p)

update:
	CMP	$16, R11
	BLT	less_than_16
	LDP.P	16(R13), (R8, R10)
	CRC32X	R8, R9
	CRC32X	R10, R9
	SUB	$16, R11

	JMP	update

less_than_16:
	TBZ $3, R11, less_than_8

	MOVD.P	8(R13), R10
	CRC32X	R10, R9

less_than_8:
	TBZ	$2, R11, less_than_4

	MOVWU.P	4(R13), R10
	CRC32W	R10, R9

less_than_4:
	TBZ	$1, R11, less_than_2

	MOVHU.P	2(R13), R10
	CRC32H	R10, R9

less_than_2:
	TBZ	$0, R11, done

	MOVBU	(R13), R10
	CRC32B	R10, R9

done:
	MOVWU	R9, ret+32(FP)
	RET
