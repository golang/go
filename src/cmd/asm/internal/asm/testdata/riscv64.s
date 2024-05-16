// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../../../../../runtime/textflag.h"

TEXT asmtest(SB),DUPOK|NOSPLIT,$0
start:
	// Unprivileged ISA

	// 2.4: Integer Computational Instructions

	ADDI	$2047, X5				// 9382f27f
	ADDI	$-2048, X5				// 93820280
	ADDI	$2048, X5				// 9382024093820240
	ADDI	$-2049, X5				// 938202c09382f2bf
	ADDI	$4094, X5				// 9382f27f9382f27f
	ADDI	$-4096, X5				// 9382028093820280
	ADDI	$4095, X5				// b71f00009b8fffffb382f201
	ADDI	$-4097, X5				// b7ffffff9b8fffffb382f201
	ADDI	$2047, X5, X6				// 1383f27f
	ADDI	$-2048, X5, X6				// 13830280
	ADDI	$2048, X5, X6				// 1383024013030340
	ADDI	$-2049, X5, X6				// 138302c01303f3bf
	ADDI	$4094, X5, X6				// 1383f27f1303f37f
	ADDI	$-4096, X5, X6				// 1383028013030380
	ADDI	$4095, X5, X6				// b71f00009b8fffff3383f201
	ADDI	$-4097, X5, X6				// b7ffffff9b8fffff3383f201

	SLTI	$55, X5, X7				// 93a37203
	SLTIU	$55, X5, X7				// 93b37203

	ANDI	$1, X5, X6				// 13f31200
	ANDI	$1, X5					// 93f21200
	ANDI	$2048, X5				// b71f00009b8f0f80b3f2f201
	ORI	$1, X5, X6				// 13e31200
	ORI	$1, X5					// 93e21200
	ORI	$2048, X5				// b71f00009b8f0f80b3e2f201
	XORI	$1, X5, X6				// 13c31200
	XORI	$1, X5					// 93c21200
	XORI	$2048, X5				// b71f00009b8f0f80b3c2f201

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

	AUIPC	$0, X10					// 17050000
	AUIPC	$0, X11					// 97050000
	AUIPC	$1, X10					// 17150000
	AUIPC	$-524288, X15				// 97070080
	AUIPC	$524287, X10				// 17f5ff7f

	LUI	$0, X15					// b7070000
	LUI	$167, X15				// b7770a00
	LUI	$-524288, X15				// b7070080
	LUI	$524287, X15				// b7f7ff7f

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
	SUB	$-2047, X5, X6				// 1383f27f
	SUB	$2048, X5, X6				// 13830280
	SUB	$-2047, X5				// 9382f27f
	SUB	$2048, X5				// 93820280

	SRA	X6, X5, X7				// b3d36240
	SRA	X5, X6					// 33535340
	SRA	$1, X5, X6				// 13d31240
	SRA	$1, X5					// 93d21240

	// 2.5: Control Transfer Instructions
	JAL	X5, 2(PC)				// ef028000
	JALR	X6, (X5)				// 67830200
	JALR	X6, 4(X5)				// 67834200
	BEQ	X5, X6, 2(PC)				// 63846200
	BNE	X5, X6, 2(PC)				// 63946200
	BLT	X5, X6, 2(PC)				// 63c46200
	BLTU	X5, X6, 2(PC)				// 63e46200
	BGE	X5, X6, 2(PC)				// 63d46200
	BGEU	X5, X6, 2(PC)				// 63f46200

	// 2.6: Load and Store Instructions
	LW	(X5), X6				// 03a30200
	LW	4(X5), X6				// 03a34200
	LWU	(X5), X6				// 03e30200
	LWU	4(X5), X6				// 03e34200
	LH	(X5), X6				// 03930200
	LH	4(X5), X6				// 03934200
	LHU	(X5), X6				// 03d30200
	LHU	4(X5), X6				// 03d34200
	LB	(X5), X6				// 03830200
	LB	4(X5), X6				// 03834200
	LBU	(X5), X6				// 03c30200
	LBU	4(X5), X6				// 03c34200

	SW	X5, (X6)				// 23205300
	SW	X5, 4(X6)				// 23225300
	SH	X5, (X6)				// 23105300
	SH	X5, 4(X6)				// 23125300
	SB	X5, (X6)				// 23005300
	SB	X5, 4(X6)				// 23025300

	// 2.7: Memory Ordering Instructions
	FENCE						// 0f00f00f

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
	ADDIW	$1, X6					// 1b031300
	SLLIW	$1, X6					// 1b131300
	SRLIW	$1, X6					// 1b531300
	SRAIW	$1, X6					// 1b531340
	ADDW	X5, X7					// bb835300
	SLLW	X5, X7					// bb935300
	SRLW	X5, X7					// bbd35300
	SUBW	X5, X7					// bb835340
	SRAW	X5, X7					// bbd35340
	ADDW	$1, X6					// 1b031300
	SLLW	$1, X6					// 1b131300
	SRLW	$1, X6					// 1b531300
	SUBW	$1, X6					// 1b03f3ff
	SRAW	$1, X6					// 1b531340

	// 5.3: Load and Store Instructions (RV64I)
	LD	(X5), X6				// 03b30200
	LD	4(X5), X6				// 03b34200
	SD	X5, (X6)				// 23305300
	SD	X5, 4(X6)				// 23325300

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

	// 8.2: Load-Reserved/Store-Conditional
	LRW	(X5), X6				// 2fa30214
	LRD	(X5), X6				// 2fb30214
	SCW	X5, (X6), X7				// af23531a
	SCD	X5, (X6), X7				// af33531a

	// 8.3: Atomic Memory Operations
	AMOSWAPW	X5, (X6), X7			// af23530e
	AMOSWAPD	X5, (X6), X7			// af33530e
	AMOADDW		X5, (X6), X7			// af235306
	AMOADDD		X5, (X6), X7			// af335306
	AMOANDW		X5, (X6), X7			// af235366
	AMOANDD		X5, (X6), X7			// af335366
	AMOORW		X5, (X6), X7			// af235346
	AMOORD		X5, (X6), X7			// af335346
	AMOXORW		X5, (X6), X7			// af235326
	AMOXORD		X5, (X6), X7			// af335326
	AMOMAXW		X5, (X6), X7			// af2353a6
	AMOMAXD		X5, (X6), X7			// af3353a6
	AMOMAXUW	X5, (X6), X7			// af2353e6
	AMOMAXUD	X5, (X6), X7			// af3353e6
	AMOMINW		X5, (X6), X7			// af235386
	AMOMIND		X5, (X6), X7			// af335386
	AMOMINUW	X5, (X6), X7			// af2353c6
	AMOMINUD	X5, (X6), X7			// af3353c6

	// 10.1: Base Counters and Timers
	RDCYCLE		X5				// f32200c0
	RDTIME		X5				// f32210c0
	RDINSTRET	X5				// f32220c0

	// 11.5: Single-Precision Load and Store Instructions
	FLW	(X5), F0				// 07a00200
	FLW	4(X5), F0				// 07a04200
	FSW	F0, (X5)				// 27a00200
	FSW	F0, 4(X5)				// 27a20200

	// 11.6: Single-Precision Floating-Point Computational Instructions
	FADDS	F1, F0, F2				// 53011000
	FSUBS	F1, F0, F2				// 53011008
	FMULS	F1, F0, F2				// 53011010
	FDIVS	F1, F0, F2				// 53011018
	FMINS	F1, F0, F2				// 53011028
	FMAXS	F1, F0, F2				// 53111028
	FSQRTS	F0, F1					// d3000058

	// 11.7: Single-Precision Floating-Point Conversion and Move Instructions
	FCVTWS	F0, X5					// d31200c0
	FCVTWS.RNE	F0, X5				// d30200c0
	FCVTWS.RTZ	F0, X5				// d31200c0
	FCVTWS.RDN	F0, X5				// d32200c0
	FCVTWS.RUP	F0, X5				// d33200c0
	FCVTWS.RMM	F0, X5				// d34200c0
	FCVTLS	F0, X5					// d31220c0
	FCVTLS.RNE	F0, X5				// d30220c0
	FCVTLS.RTZ	F0, X5				// d31220c0
	FCVTLS.RDN	F0, X5				// d32220c0
	FCVTLS.RUP	F0, X5				// d33220c0
	FCVTLS.RMM	F0, X5				// d34220c0
	FCVTSW	X5, F0					// 538002d0
	FCVTSL	X5, F0					// 538022d0
	FCVTWUS	F0, X5					// d31210c0
	FCVTWUS.RNE	F0, X5				// d30210c0
	FCVTWUS.RTZ	F0, X5				// d31210c0
	FCVTWUS.RDN	F0, X5				// d32210c0
	FCVTWUS.RUP	F0, X5				// d33210c0
	FCVTWUS.RMM	F0, X5				// d34210c0
	FCVTLUS	F0, X5					// d31230c0
	FCVTLUS.RNE	F0, X5				// d30230c0
	FCVTLUS.RTZ	F0, X5				// d31230c0
	FCVTLUS.RDN	F0, X5				// d32230c0
	FCVTLUS.RUP	F0, X5				// d33230c0
	FCVTLUS.RMM	F0, X5				// d34230c0
	FCVTSWU	X5, F0					// 538012d0
	FCVTSLU	X5, F0					// 538032d0
	FSGNJS	F1, F0, F2				// 53011020
	FSGNJNS	F1, F0, F2				// 53111020
	FSGNJXS	F1, F0, F2				// 53211020
	FMVXS	F0, X5					// d30200e0
	FMVSX	X5, F0					// 538002f0
	FMVXW	F0, X5					// d30200e0
	FMVWX	X5, F0					// 538002f0
	FMADDS	F1, F2, F3, F4				// 43822018
	FMSUBS	F1, F2, F3, F4				// 47822018
	FNMSUBS	F1, F2, F3, F4				// 4b822018
	FNMADDS	F1, F2, F3, F4				// 4f822018

	// 11.8: Single-Precision Floating-Point Compare Instructions
	FEQS	F0, F1, X7				// d3a300a0
	FLTS	F0, F1, X7				// d39300a0
	FLES	F0, F1, X7				// d38300a0

	// 11.9: Single-Precision Floating-Point Classify Instruction
	FCLASSS	F0, X5					// d31200e0

	// 12.3: Double-Precision Load and Store Instructions
	FLD	(X5), F0				// 07b00200
	FLD	4(X5), F0				// 07b04200
	FSD	F0, (X5)				// 27b00200
	FSD	F0, 4(X5)				// 27b20200

	// 12.4: Double-Precision Floating-Point Computational Instructions
	FADDD	F1, F0, F2				// 53011002
	FSUBD	F1, F0, F2				// 5301100a
	FMULD	F1, F0, F2				// 53011012
	FDIVD	F1, F0, F2				// 5301101a
	FMIND	F1, F0, F2				// 5301102a
	FMAXD	F1, F0, F2				// 5311102a
	FSQRTD	F0, F1					// d300005a

	// 12.5: Double-Precision Floating-Point Conversion and Move Instructions
	FCVTWD	F0, X5					// d31200c2
	FCVTWD.RNE	F0, X5				// d30200c2
	FCVTWD.RTZ	F0, X5				// d31200c2
	FCVTWD.RDN	F0, X5				// d32200c2
	FCVTWD.RUP	F0, X5				// d33200c2
	FCVTWD.RMM	F0, X5				// d34200c2
	FCVTLD	F0, X5					// d31220c2
	FCVTLD.RNE	F0, X5				// d30220c2
	FCVTLD.RTZ	F0, X5				// d31220c2
	FCVTLD.RDN	F0, X5				// d32220c2
	FCVTLD.RUP	F0, X5				// d33220c2
	FCVTLD.RMM	F0, X5				// d34220c2
	FCVTDW	X5, F0					// 538002d2
	FCVTDL	X5, F0					// 538022d2
	FCVTWUD F0, X5					// d31210c2
	FCVTWUD.RNE F0, X5				// d30210c2
	FCVTWUD.RTZ F0, X5				// d31210c2
	FCVTWUD.RDN F0, X5				// d32210c2
	FCVTWUD.RUP F0, X5				// d33210c2
	FCVTWUD.RMM F0, X5				// d34210c2
	FCVTLUD F0, X5					// d31230c2
	FCVTLUD.RNE F0, X5				// d30230c2
	FCVTLUD.RTZ F0, X5				// d31230c2
	FCVTLUD.RDN F0, X5				// d32230c2
	FCVTLUD.RUP F0, X5				// d33230c2
	FCVTLUD.RMM F0, X5				// d34230c2
	FCVTDWU X5, F0					// 538012d2
	FCVTDLU X5, F0					// 538032d2
	FCVTSD	F0, F1					// d3001040
	FCVTDS	F0, F1					// d3000042
	FSGNJD	F1, F0, F2				// 53011022
	FSGNJND	F1, F0, F2				// 53111022
	FSGNJXD	F1, F0, F2				// 53211022
	FMVXD	F0, X5					// d30200e2
	FMVDX	X5, F0					// 538002f2
	FMADDD	F1, F2, F3, F4				// 4382201a
	FMSUBD	F1, F2, F3, F4				// 4782201a
	FNMSUBD	F1, F2, F3, F4				// 4b82201a
	FNMADDD	F1, F2, F3, F4				// 4f82201a

	// 12.6: Double-Precision Floating-Point Classify Instruction
	FCLASSD	F0, X5					// d31200e2

	// RISC-V Bit-Manipulation ISA-extensions (1.0)
	// 1.1: Address Generation Instructions (Zba)
	ADDUW		X10, X11, X12			// 3b86a508
	ADDUW		X10, X11			// bb85a508
	SH1ADD		X11, X12, X13			// b326b620
	SH1ADD		X11, X12			// 3326b620
	SH1ADDUW	X12, X13, X14			// 3ba7c620
	SH1ADDUW	X12, X13			// bba6c620
	SH2ADD		X13, X14, X15			// b347d720
	SH2ADD		X13, X14			// 3347d720
	SH2ADDUW	X14, X15, X16			// 3bc8e720
	SH2ADDUW	X14, X15			// bbc7e720
	SH3ADD		X15, X16, X17			// b368f820
	SH3ADD		X15, X16			// 3368f820
	SH3ADDUW	X16, X17, X18			// 3be90821
	SH3ADDUW	X16, X17			// bbe80821
	SLLIUW		$31, X17, X18			// 1b99f809
	SLLIUW		$63, X17			// 9b98f80b
	SLLIUW		$63, X17, X18			// 1b99f80b
	SLLIUW		$1, X18, X19			// 9b191908

	// 1.2: Basic Bit Manipulation (Zbb)
	ANDN	X19, X20, X21				// b37a3a41
	ANDN	X19, X20				// 337a3a41
	CLZ	X20, X21				// 931a0a60
	CLZW	X21, X22				// 1b9b0a60
	CPOP	X22, X23				// 931b2b60
	CPOPW	X23, X24				// 1b9c2b60
	CTZ	X24, X25				// 931c1c60
	CTZW	X25, X26				// 1b9d1c60
	MAX	X26, X28, X29				// b36eae0b
	MAX	X26, X28				// 336eae0b
	MAXU	X28, X29, X30				// 33ffce0b
	MAXU	X28, X29				// b3fece0b
	MIN	X29, X30, X5				// b342df0b
	MIN	X29, X30				// 334fdf0b
	MINU	X30, X5, X6				// 33d3e20b
	MINU	X30, X5					// b3d2e20b
	ORN	X6, X7, X8				// 33e46340
	ORN	X6, X7					// b3e36340
	SEXTB	X16, X17				// 93184860
	SEXTH	X17, X18				// 13995860
	XNOR	X18, X19, X20				// 33ca2941
	XNOR	X18, X19				// b3c92941
	ZEXTH	X19, X20				// 3bca0908

	// 1.3: Bitwise Rotation (Zbb)
	ROL	X8, X9, X10				// 33958460 or b30f8040b3dff4013395840033e5af00
	ROL	X8, X9					// b3948460 or b30f8040b3dff401b3948400b3e49f00
	ROLW	X9, X10, X11				// bb159560 or b30f9040bb5ff501bb159500b3e5bf00
	ROLW	X9, X10					// 3b159560 or b30f9040bb5ff5013b15950033e5af00
	ROR	X10, X11, X12				// 33d6a560 or b30fa040b39ff50133d6a50033e6cf00
	ROR	X10, X11				// b3d5a560 or b30fa040b39ff501b3d5a500b3e5bf00
	ROR	$63, X11				// 93d5f563 or 93dff50393951500b3e5bf00
	RORI	$63, X11, X12				// 13d6f563 or 93dff5031396150033e6cf00
	RORI	$1, X12, X13				// 93561660 or 935f16009316f603b3e6df00
	RORIW	$31, X13, X14				// 1bd7f661 or 9bdff6011b97160033e7ef00
	RORIW	$1, X14, X15				// 9b571760 or 9b5f17009b17f701b3e7ff00
	RORW	X15, X16, X17				// bb58f860 or b30ff040bb1ff801bb58f800b3e81f01
	RORW	X15, X16				// 3b58f860 or b30ff040bb1ff8013b58f80033e80f01
	RORW	$31, X13				// 9bd6f661 or 9bdff6019b961600b3e6df00
	ORCB	X5, X6					// 13d37228
	REV8	X7, X8					// 13d4836b

	// 1.5: Single-bit Instructions (Zbs)
	BCLR	X23, X24, X25				// b31c7c49
	BCLR	$63, X24				// 131cfc4b
	BCLRI	$1, X25, X26				// 139d1c48
	BEXT	X26, X28, X29				// b35eae49
	BEXT	$63, X28				// 135efe4b
	BEXTI	$1, X29, X30				// 13df1e48
	BINV	X30, X5, X6				// 3393e269
	BINV	$63, X6					// 1313f36b
	BINVI	$1, X7, X8				// 13941368
	BSET	X8, X9, X10				// 33958428
	BSET	$63, X9					// 9394f42b
	BSETI	$1, X10, X11				// 93151528

	// Privileged ISA

	// 3.2.1: Environment Call and Breakpoint
	ECALL						// 73000000
	SCALL						// 73000000
	EBREAK						// 73001000
	SBREAK						// 73001000

	// Arbitrary bytes (entered in little-endian mode)
	WORD	$0x12345678	// WORD $305419896	// 78563412
	WORD	$0x9abcdef0	// WORD $2596069104	// f0debc9a

	// MOV pseudo-instructions
	MOV	X5, X6					// 13830200
	MOV	$2047, X5				// 9302f07f
	MOV	$-2048, X5				// 93020080
	MOV	$2048, X5				// b71200009b820280
	MOV	$-2049, X5				// b7f2ffff9b82f27f
	MOV	$4096, X5				// b7120000
	MOV	$2147479552, X5				// b7f2ff7f
	MOV	$2147483647, X5				// b70200809b82f2ff
	MOV	$-2147483647, X5			// b70200809b821200

	// Converted to load of symbol (AUIPC + LD)
	MOV	$4294967295, X5				// 9702000083b20200
	// Converted to MOV $1, X5 + SLLI $32, X5
	MOV	$4294967296, X5				// 9302100093920202

	MOV	(X5), X6				// 03b30200
	MOV	4(X5), X6				// 03b34200
	MOVB	(X5), X6				// 03830200
	MOVB	4(X5), X6				// 03834200
	MOVH	(X5), X6				// 03930200
	MOVH	4(X5), X6				// 03934200
	MOVW	(X5), X6				// 03a30200
	MOVW	4(X5), X6				// 03a34200
	MOV	X5, (X6)				// 23305300
	MOV	X5, 4(X6)				// 23325300
	MOVB	X5, (X6)				// 23005300
	MOVB	X5, 4(X6)				// 23025300
	MOVH	X5, (X6)				// 23105300
	MOVH	X5, 4(X6)				// 23125300
	MOVW	X5, (X6)				// 23205300
	MOVW	X5, 4(X6)				// 23225300

	MOVB	X5, X6					// 1393820313538343 or 13934260
	MOVH	X5, X6					// 1393020313530343 or 13935260
	MOVW	X5, X6					// 1b830200
	MOVBU	X5, X6					// 13f3f20f
	MOVHU	X5, X6					// 1393020313530303 or 3bc30208
	MOVWU	X5, X6					// 1393020213530302 or 3b830208

	MOVF	4(X5), F0				// 07a04200
	MOVF	F0, 4(X5)				// 27a20200
	MOVF	F0, F1					// d3000020

	MOVD	4(X5), F0				// 07b04200
	MOVD	F0, 4(X5)				// 27b20200
	MOVD	F0, F1					// d3000022

	// TLS load with local-exec (LUI + ADDIW + ADD of TP + load)
	MOV	tls(SB), X5				// b70f00009b8f0f00b38f4f0083b20f00
	MOVB	tls(SB), X5				// b70f00009b8f0f00b38f4f0083820f00

	// TLS store with local-exec (LUI + ADDIW + ADD of TP + store)
	MOV	X5, tls(SB)				// b70f00009b8f0f00b38f4f0023b05f00
	MOVB	X5, tls(SB)				// b70f00009b8f0f00b38f4f0023805f00

	// NOT pseudo-instruction
	NOT	X5					// 93c2f2ff
	NOT	X5, X6					// 13c3f2ff

	// NEG/NEGW pseudo-instructions
	NEG	X5					// b3025040
	NEG	X5, X6					// 33035040
	NEGW	X5					// bb025040
	NEGW	X5, X6					// 3b035040

	// This jumps to the second instruction in the function (the
	// first instruction is an invisible stack pointer adjustment).
	JMP	start					// JMP	2

	JMP	2(PC)					// 6f008000
	JMP	(X5)					// 67800200
	JMP	4(X5)					// 67804200

	// CALL and JMP to symbol are encoded as JAL (using LR or ZERO
	// respectively), with a R_RISCV_JAL relocation. The linker resolves
	// the real address and updates the immediate, using a trampoline in
	// the case where the address is not directly reachable.
	CALL	asmtest(SB)				// ef000000
	JMP	asmtest(SB)				// 6f000000

	// Branch pseudo-instructions
	BEQZ	X5, 2(PC)				// 63840200
	BGEZ	X5, 2(PC)				// 63d40200
	BGT	X5, X6, 2(PC)				// 63445300
	BGTU	X5, X6, 2(PC)				// 63645300
	BGTZ	X5, 2(PC)				// 63445000
	BLE	X5, X6, 2(PC)				// 63545300
	BLEU	X5, X6, 2(PC)				// 63745300
	BLEZ	X5, 2(PC)				// 63545000
	BLTZ	X5, 2(PC)				// 63c40200
	BNEZ	X5, 2(PC)				// 63940200

	// Set pseudo-instructions
	SEQZ	X15, X15				// 93b71700
	SNEZ	X15, X15				// b337f000

	// F extension
	FABSS	F0, F1					// d3200020
	FNEGS	F0, F1					// d3100020
	FNES	F0, F1, X7				// d3a300a093c31300

	// D extension
	FABSD	F0, F1					// d3200022
	FNEGD	F0, F1					// d3100022
	FNED	F0, F1, X5				// d3a200a293c21200
	FLTD	F0, F1, X5				// d39200a2
	FLED	F0, F1, X5				// d38200a2
	FEQD	F0, F1, X5				// d3a200a2

GLOBL tls(SB), TLSBSS, $8
