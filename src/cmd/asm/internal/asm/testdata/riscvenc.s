// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../../../../../runtime/textflag.h"

TEXT asmtest(SB),DUPOK|NOSPLIT,$0

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

	// 2.6: Load and Store Instructions
	LW	$0, X5, X6				// 03a30200
	LW	$4, X5, X6				// 03a34200
	LWU	$0, X5, X6				// 03e30200
	LWU	$4, X5, X6				// 03e34200
	LH	$0, X5, X6				// 03930200
	LH	$4, X5, X6				// 03934200
	LHU	$0, X5, X6				// 03d30200
	LHU	$4, X5, X6				// 03d34200
	LB	$0, X5, X6				// 03830200
	LB	$4, X5, X6				// 03834200
	LBU	$0, X5, X6				// 03c30200
	LBU	$4, X5, X6				// 03c34200

	SW	$0, X5, X6				// 23205300
	SW	$4, X5, X6				// 23225300
	SH	$0, X5, X6				// 23105300
	SH	$4, X5, X6				// 23125300
	SB	$0, X5, X6				// 23005300
	SB	$4, X5, X6				// 23025300

        // 5.2: Integer Computational Instructions (RV64I)
	ADDIW	$1, X5, X6				// 1b831200
	SLLIW	$1, X5, X6				// 1b931200
	SRLIW	$1, X5, X6				// 1bd31200
	SRAIW	$1, X5, X6				// 1bd31240
	ADDW	X5, X6, X7				// bb035300
	SLLW	X5, X6, X7				// bb135300
	SRLW	X5, X6, X7				// bb535300
	SUBW	X5, X6, X7				// bb035340
	SRAW	X5, X6, X7				// bb535340

	// 5.3: Load and Store Instructions (RV64I)
	LD	$0, X5, X6				// 03b30200
	LD	$4, X5, X6				// 03b34200
	SD	$0, X5, X6				// 23305300
	SD	$4, X5, X6				// 23325300

	// 7.1: Multiplication Operations
	MUL	X5, X6, X7				// b3035302
	MULH	X5, X6, X7				// b3135302
	MULHU	X5, X6, X7				// b3335302
	MULHSU	X5, X6, X7				// b3235302
	MULW	X5, X6, X7				// bb035302
	DIV	X5, X6, X7				// b3435302
	DIVU	X5, X6, X7				// b3535302
	REM	X5, X6, X7				// b3635302
	REMU	X5, X6, X7				// b3735302
	DIVW	X5, X6, X7				// bb435302
	DIVUW	X5, X6, X7				// bb535302
	REMW	X5, X6, X7				// bb635302
	REMUW	X5, X6, X7				// bb735302

	// 10.1: Base Counters and Timers
	RDCYCLE		X5				// f32200c0
	RDTIME		X5				// f32210c0
	RDINSTRET	X5				// f32220c0

	// Privileged ISA

	// 3.2.1: Environment Call and Breakpoint
	ECALL						// 73000000
	SCALL						// 73000000
	EBREAK						// 73001000
	SBREAK						// 73001000

	// Arbitrary bytes (entered in little-endian mode)
	WORD	$0x12345678	// WORD $305419896	// 78563412
	WORD	$0x9abcdef0	// WORD $2596069104	// f0debc9a
