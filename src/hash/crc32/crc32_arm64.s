// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// castagnoliUpdate updates the non-inverted crc with the given data.

// func castagnoliUpdate(crc uint32, p []byte) uint32
TEXT ·castagnoliUpdate(SB),NOSPLIT,$0-36
	MOVWU	crc+0(FP), R9  // CRC value
	MOVD	p+8(FP), R13  // data pointer
	MOVD	p_len+16(FP), R11  // len(p)

	CMP	$8, R11
	BLT	less_than_8

update:
	MOVD.P	8(R13), R10
	CRC32CX	R10, R9
	SUB	$8, R11

	CMP	$8, R11
	BLT	less_than_8

	JMP	update

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
	MOVWU	R9, ret+32(FP)
	RET

// ieeeUpdate updates the non-inverted crc with the given data.

// func ieeeUpdate(crc uint32, p []byte) uint32
TEXT ·ieeeUpdate(SB),NOSPLIT,$0-36
	MOVWU	crc+0(FP), R9  // CRC value
	MOVD	p+8(FP), R13  // data pointer
	MOVD	p_len+16(FP), R11  // len(p)

	CMP	$8, R11
	BLT	less_than_8

update:
	MOVD.P	8(R13), R10
	CRC32X	R10, R9
	SUB	$8, R11

	CMP	$8, R11
	BLT	less_than_8

	JMP	update

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

// func supportsCRC32() bool
TEXT ·supportsCRC32(SB),NOSPLIT,$0-1
	MOVB	runtime·supportCRC32(SB), R0
	MOVB	R0, ret+0(FP)
	RET
