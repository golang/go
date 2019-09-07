// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../../../../../runtime/textflag.h"

TEXT asmtest(SB),DUPOK|NOSPLIT,$0

	// Arbitrary bytes (entered in little-endian mode)
	WORD	$0x12345678	// WORD $305419896	// 78563412
	WORD	$0x9abcdef0	// WORD $2596069104	// f0debc9a

	// Unprivileged ISA

	// 2.4: Integer Computational Instructions

	ADDI	$2047, X5, X6				// 1383f27f
	ADDI	$-2048, X5, X6				// 13830280
	ADDI	$2047, X5				// 9382f27f
	ADDI	$-2048, X5				// 93820280

	SLTI	$55, X5, X7				// 93a37203
	SLTIU	$55, X5, X7				// 93b37203

	ANDI	$1, X5, X6				// 13f31200
	ANDI	$1, X5					// 93f21200
	ORI	$1, X5, X6				// 13e31200
	ORI	$1, X5					// 93e21200
	XORI	$1, X5, X6				// 13c31200
	XORI	$1, X5					// 93c21200

	SLLI	$1, X5, X6				// 13931200
	SLLI	$1, X5					// 93921200
	SRLI	$1, X5, X6				// 13d31200
	SRLI	$1, X5					// 93d21200
	SRAI	$1, X5, X6				// 13d31240
	SRAI	$1, X5					// 93d21240

	ADD	X6, X5, X7				// b3836200
	ADD	X5, X6					// 33035300
	ADD	$2047, X5, X6				// 1383f27f
	ADD	$-2048, X5, X6				// 13830280
	ADD	$2047, X5				// 9382f27f
	ADD	$-2048, X5				// 93820280

	SLT	X6, X5, X7				// b3a36200
	SLT	$55, X5, X7				// 93a37203
	SLTU	X6, X5, X7				// b3b36200
	SLTU	$55, X5, X7				// 93b37203

	AND	X6, X5, X7				// b3f36200
	AND	X5, X6					// 33735300
	AND	$1, X5, X6				// 13f31200
	AND	$1, X5					// 93f21200
	OR	X6, X5, X7				// b3e36200
	OR	X5, X6					// 33635300
	OR	$1, X5, X6				// 13e31200
	OR	$1, X5					// 93e21200
	XOR	X6, X5, X7				// b3c36200
	XOR	X5, X6					// 33435300
	XOR	$1, X5, X6				// 13c31200
	XOR	$1, X5					// 93c21200

	SLL	X6, X5, X7				// b3936200
	SLL	X5, X6					// 33135300
	SLL	$1, X5, X6				// 13931200
	SLL	$1, X5					// 93921200
	SRL	X6, X5, X7				// b3d36200
	SRL	X5, X6					// 33535300
	SRL	$1, X5, X6				// 13d31200
	SRL	$1, X5					// 93d21200

	SUB	X6, X5, X7				// b3836240
	SUB	X5, X6					// 33035340

	SRA	X6, X5, X7				// b3d36240
	SRA	X5, X6					// 33535340
	SRA	$1, X5, X6				// 13d31240
	SRA	$1, X5					// 93d21240
