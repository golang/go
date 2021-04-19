// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This input was created by taking the instruction productions in
// the old assembler's (9a's) grammar and hand-writing complete
// instructions for each rule, to guarantee we cover the same space.

TEXT foo(SB),7,$0

//inst:
//
// load ints and bytes
//
//	LMOVW rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, R2

//	LMOVW addr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	foo<>+3(SB), R2
	MOVW	16(R1), R2

//	LMOVW regaddr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	(R1), R2
	MOVW	(R1+R2), R3 // MOVW (R1)(R2*1), R3

//	LMOVB rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, R2

//	LMOVB addr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVB	foo<>+3(SB), R2
	MOVB	16(R1), R2

//	LMOVB regaddr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVB	(R1), R2
	MOVB	(R1+R2), R3 // MOVB (R1)(R2*1), R3

//
// load floats
//
//	LFMOV addr ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	FMOVD	foo<>+3(SB), F2
	FMOVD	16(R1), F2

//	LFMOV regaddr ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	FMOVD	(R1), F2

//	LFMOV fimm ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	FMOVD	$0.1, F2 // FMOVD $(0.10000000000000001), F2

//	LFMOV freg ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	FMOVD	F1, F2

//	LFMOV freg ',' addr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	FMOVD	F2, foo<>+3(SB)
	FMOVD	F2, 16(R1)

//	LFMOV freg ',' regaddr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	FMOVD	F2, (R1)

//
// store ints and bytes
//
//	LMOVW rreg ',' addr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, foo<>+3(SB)
	MOVW	R1, 16(R2)

//	LMOVW rreg ',' regaddr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, (R1)
	MOVW	R1, (R2+R3) // MOVW R1, (R2)(R3*1)

//	LMOVB rreg ',' addr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVB	R1, foo<>+3(SB)
	MOVB	R1, 16(R2)

//	LMOVB rreg ',' regaddr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVB	R1, (R1)
	MOVB	R1, (R2+R3) // MOVB R1, (R2)(R3*1)
//
// store floats
//
//	LMOVW freg ',' addr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	FMOVD	F1, foo<>+3(SB)
	FMOVD	F1, 16(R2)

//	LMOVW freg ',' regaddr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	FMOVD	F1, (R1)

//
// floating point status
//
//	LMOVW fpscr ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVFL	FPSCR, F1

//	LMOVW freg ','  fpscr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVFL	F1, FPSCR

//	LMOVW freg ',' imm ',' fpscr
//	{
//		outgcode(int($1), &$2, 0, &$4, &$6);
//	}
	MOVFL	F1, $4, FPSCR

//	LMOVW fpscr ',' creg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVFL	FPSCR, CR0

//	LMTFSB imm ',' con
//	{
//		outcode(int($1), &$2, int($4), &nullgen);
//	}
//TODO	9a doesn't work MTFSB0	$4, 4

//
// field moves (mtcrf)
//
//	LMOVW rreg ',' imm ',' lcr
//	{
//		outgcode(int($1), &$2, 0, &$4, &$6);
//	}
// TODO 9a doesn't work	MOVFL	R1,$4,CR

//	LMOVW rreg ',' creg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
		MOVW	R1, CR1

//	LMOVW rreg ',' lcr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, CR

//
// integer operations
// logical instructions
// shift instructions
// unary instructions
//
//	LADDW rreg ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	ADD	R1, R2, R3

//	LADDW imm ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	ADD	$1, R2, R3

//	LADDW rreg ',' imm ',' rreg
//	{
//		outgcode(int($1), &$2, 0, &$4, &$6);
//	}
//TODO 9a trouble	ADD	R1, $2, R3 maybe swap rreg and imm

//	LADDW rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	ADD	R1, R2

//	LADDW imm ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	ADD	$4, R1

//	LLOGW rreg ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	ADDE	R1, R2, R3

//	LLOGW rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	ADDE	R1, R2

//	LSHW rreg ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	SLW	R1, R2, R3

//	LSHW rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	SLW	R1, R2

//	LSHW imm ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	SLW	$4, R1, R2

//	LSHW imm ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	SLW	$4, R1

//	LABS rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	SLW	$4, R1

//	LABS rreg
//	{
//		outcode(int($1), &$2, 0, &$2);
//	}
	SUBME	R1 // SUBME R1, R1

//
// multiply-accumulate
//
//	LMA rreg ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
//TODO this instruction is undefined in lex.go	LMA R1, R2, R3 NOT SUPPORTED (called MAC)

//
// move immediate: macro for cau+or, addi, addis, and other combinations
//
//	LMOVW imm ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	$1, R1

//	LMOVW ximm ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	$1, R1
	MOVW	$foo(SB), R1

// condition register operations
//
//	LCROP cbit ',' cbit
//	{
//		outcode(int($1), &$2, int($4.Reg), &$4);
//	}
//TODO 9a trouble	CREQV	1, 2 delete? liblink encodes like a divide (maybe wrong too)

//	LCROP cbit ',' con ',' cbit
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
//TODO 9a trouble	CREQV	1, 2, 3

//
// condition register moves
// move from machine state register
//
//	LMOVW creg ',' creg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVFL	CR0, CR1

//	LMOVW psr ',' creg // TODO: should psr should be fpscr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
//TODO 9a trouble	MOVW	FPSCR, CR1

//	LMOVW lcr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	CR, R1

//	LMOVW psr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	SPR(0), R1
	MOVW	SPR(7), R1

//	LMOVW xlreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	LR, R1
	MOVW	CTR, R1

//	LMOVW rreg ',' xlreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, LR
	MOVW	R1, CTR

//	LMOVW creg ',' psr // TODO doesn't exist
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
//TODO 9a trouble	MOVW	CR1, SPR(7)

//	LMOVW rreg ',' psr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, SPR(7)

//
// branch, branch conditional
// branch conditional register
// branch conditional to count register
//
//	LBRA rel
//	{
//		outcode(int($1), &nullgen, 0, &$2);
//	}
	BEQ	CR1, 2(PC)
label0:
	BR	1(PC) // JMP 1(PC)
	BEQ	CR1, 2(PC)
	BR	label0+0 // JMP 62

//	LBRA addr
//	{
//		outcode(int($1), &nullgen, 0, &$2);
//	}
	BEQ	CR1, 2(PC)
	BR	LR // JMP LR
	BEQ	CR1, 2(PC)
//	BR	0(R1)	// TODO should work
	BEQ	CR1, 2(PC)
	BR	foo+0(SB) // JMP foo(SB)

//	LBRA '(' xlreg ')'
//	{
//		outcode(int($1), &nullgen, 0, &$3);
//	}
	BEQ	CR1, 2(PC)
	BR	(CTR) // JMP CTR

//	LBRA ',' rel  // asm doesn't support the leading comma
//	{
//		outcode(int($1), &nullgen, 0, &$3);
//	}
//	LBRA ',' addr  // asm doesn't support the leading comma
//	{
//		outcode(int($1), &nullgen, 0, &$3);
//	}
//	LBRA ',' '(' xlreg ')'  // asm doesn't support the leading comma
//	{
//		outcode(int($1), &nullgen, 0, &$4);
//	}
//	LBRA creg ',' rel
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
label1:
	BEQ	CR1, 1(PC)
	BEQ	CR1, label1 // BEQ CR1, 72

//	LBRA creg ',' addr // TODO DOES NOT WORK in 9a
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}

//	LBRA creg ',' '(' xlreg ')' // TODO DOES NOT WORK in 9a
//	{
//		outcode(int($1), &$2, 0, &$5);
//	}

//	LBRA con ',' rel // TODO DOES NOT WORK in 9a
//	{
//		outcode(int($1), &nullgen, int($2), &$4);
//	}

//	LBRA con ',' addr // TODO DOES NOT WORK in 9a
//	{
//		outcode(int($1), &nullgen, int($2), &$4);
//	}

//	LBRA con ',' '(' xlreg ')'
//	{
//		outcode(int($1), &nullgen, int($2), &$5);
//	}
//	BC	4, (CTR)	// TODO - should work

//	LBRA con ',' con ',' rel
//	{
//		var g obj.Addr
//		g = nullgen;
//		g.Type = obj.TYPE_CONST;
//		g.Offset = $2;
//		outcode(int($1), &g, int(REG_R0+$4), &$6);
//	}
//	BC	3, 4, label1 // TODO - should work

//	LBRA con ',' con ',' addr // TODO mystery
//	{
//		var g obj.Addr
//		g = nullgen;
//		g.Type = obj.TYPE_CONST;
//		g.Offset = $2;
//		outcode(int($1), &g, int(REG_R0+$4), &$6);
//	}
//TODO 9a trouble	BC	3, 3, 4(R1)

//	LBRA con ',' con ',' '(' xlreg ')'
//	{
//		var g obj.Addr
//		g = nullgen;
//		g.Type = obj.TYPE_CONST;
//		g.Offset = $2;
//		outcode(int($1), &g, int(REG_R0+$4), &$7);
//	}
	BC	3, 3, (LR) // BC $3, R3, LR

//
// conditional trap // TODO NOT DEFINED
// TODO these instructions are not in lex.go
//
//	LTRAP rreg ',' sreg
//	{
//		outcode(int($1), &$2, int($4), &nullgen);
//	}
//	LTRAP imm ',' sreg
//	{
//		outcode(int($1), &$2, int($4), &nullgen);
//	}
//	LTRAP rreg comma
//	{
//		outcode(int($1), &$2, 0, &nullgen);
//	}
//	LTRAP comma
//	{
//		outcode(int($1), &nullgen, 0, &nullgen);
//	}

//
// floating point operate
//
//	LFCONV freg ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	FABS	F1, F2

//	LFADD freg ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	FADD	F1, F2

//	LFADD freg ',' freg ',' freg
//	{
//		outcode(int($1), &$2, int($4.Reg), &$6);
//	}
	FADD	F1, F2, F3

//	LFMA freg ',' freg ',' freg ',' freg
//	{
//		outgcode(int($1), &$2, int($4.Reg), &$6, &$8);
//	}
	FMADD	F1, F2, F3, F4

//	LFCMP freg ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	FCMPU	F1, F2

//	LFCMP freg ',' freg ',' creg
//	{
//		outcode(int($1), &$2, int($6.Reg), &$4);
//	}
//	FCMPU	F1, F2, CR0

//
// CMP
//
//	LCMP rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	CMP	R1, R2

//	LCMP rreg ',' imm
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	CMP	R1, $4

//	LCMP rreg ',' rreg ',' creg
//	{
//		outcode(int($1), &$2, int($6.Reg), &$4);
//	}
	CMP	R1, R2, CR0 // CMP R1, CR0, R2

//	LCMP rreg ',' imm ',' creg
//	{
//		outcode(int($1), &$2, int($6.Reg), &$4);
//	}
	CMP	R1, $4, CR0 // CMP R1, CR0, $4

//
// rotate and mask
//
//	LRLWM  imm ',' rreg ',' imm ',' rreg
//	{
//		outgcode(int($1), &$2, int($4.Reg), &$6, &$8);
//	}
	RLDC $4, R1, $16, R2

//	LRLWM  imm ',' rreg ',' mask ',' rreg
//	{
//		outgcode(int($1), &$2, int($4.Reg), &$6, &$8);
//	}
	RLDC $26, R1, 4, 5, R2 // RLDC $26, R1, $201326592, R2

//	LRLWM  rreg ',' rreg ',' imm ',' rreg
//	{
//		outgcode(int($1), &$2, int($4.Reg), &$6, &$8);
//	}
	RLDCL	R1, R2, $7, R3

//	LRLWM  rreg ',' rreg ',' mask ',' rreg
//	{
//		outgcode(int($1), &$2, int($4.Reg), &$6, &$8);
//	}
	RLWMI	R1, R2, 4, 5, R3 // RLWMI	R1, R2, $201326592, R3


// opcodes added with constant shift counts, not masks

	RLDICR	$3, R2, $24, R4

	RLDICL	$1, R2, $61, R6

	RLDIMI  $7, R2, $52, R7

//
// load/store multiple
//
//	LMOVMW addr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
//	MOVMW	foo+0(SB), R2 // TODO TLS broke this!
	MOVMW	4(R1), R2

//	LMOVMW rreg ',' addr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
//	MOVMW	R1, foo+0(SB) // TODO TLS broke this!
	MOVMW	R1, 4(R2)

//
// various indexed load/store
// indexed unary (eg, cache clear)
//
//	LXLD regaddr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	LSW	(R1), R2
	LSW	(R1+R2), R3 // LSW	(R1)(R2*1), R3

//	LXLD regaddr ',' imm ',' rreg
//	{
//		outgcode(int($1), &$2, 0, &$4, &$6);
//	}
	LSW	(R1), $1, R2
	LSW	(R1+R2), $1, R3 // LSW	(R1)(R2*1), $1, R3

//	LXST rreg ',' regaddr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	STSW	R1, (R2)
	STSW	R1, (R2+R3) // STSW	R1, (R2)(R3*1)

//	LXST rreg ',' imm ',' regaddr
//	{
//		outgcode(int($1), &$2, 0, &$4, &$6);
//	}
	STSW	R1, $1, (R2)
	STSW	R1, $1, (R2+R3) // STSW	R1, $1, (R2)(R3*1)

//	LXMV regaddr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVHBR	(R1), R2
	MOVHBR	(R1+R2), R3 // MOVHBR	(R1)(R2*1), R3

//	LXMV rreg ',' regaddr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVHBR	R1, (R2)
	MOVHBR	R1, (R2+R3) // MOVHBR	R1, (R2)(R3*1)

//	LXOP regaddr
//	{
//		outcode(int($1), &$2, 0, &nullgen);
//	}
	DCBF	(R1)
	DCBF	(R1+R2) // DCBF	(R1)(R2*1)

//	VMX instructions

//	Described as:
//	<instruction type>, <instruction format>
//	<go asm operand order> produces
//	<Power ISA operand order>

//	Vector load, VX-form
//	<MNEMONIC> (RB)(RA*1),VRT produces
//	<mnemonic> VRT,RA,RB
	LVEBX	(R1)(R2*1), V0
	LVEHX	(R3)(R4*1), V1
	LVEWX	(R5)(R6*1), V2
	LVX	(R7)(R8*1), V3
	LVXL	(R9)(R10*1), V4
	LVSL	(R11)(R12*1), V5
	LVSR	(R14)(R15*1), V6

//	Vector store, VX-form
//	<MNEMONIC> VRT,(RB)(RA*1) produces
//	<mnemonic> VRT,RA,RB
	STVEBX	V31, (R1)(R2*1)
	STVEHX	V30, (R2)(R3*1)
	STVEWX	V29, (R4)(R5*1)
	STVX	V28, (R6)(R7*1)
	STVXL	V27, (R9)(R9*1)

//	Vector AND, VX-form
//	<MNEMONIC> VRA,VRB,VRT produces
//	<mnemonic> VRT,VRA,VRB
	VANDL	V10, V9, V8
	VANDC	V15, V14, V13
	VNAND	V19, V18, V17

//	Vector OR, VX-form
//	<MNEMONIC> VRA,VRB,VRT produces
//	<mnemonic> VRT,VRA,VRB
	VORL	V26, V25, V24
	VORC	V23, V22, V21
	VNOR	V20, V19, V18
	VXOR	V17, V16, V15
	VEQV	V14, V13, V12

//	Vector ADD, VX-form
//	<MNEMONIC> VRA,VRB,VRT produces
//	<mnemonic> VRT,VRA,VRB
	VADDUBM	V3, V2, V1
	VADDUHM	V3, V2, V1
	VADDUWM	V3, V2, V1
	VADDUDM	V3, V2, V1
	VADDUQM	V3, V2, V1
	VADDCUQ	V3, V2, V1
	VADDCUW	V3, V2, V1
	VADDUBS	V3, V2, V1
	VADDUHS	V3, V2, V1
	VADDUWS	V3, V2, V1
	VADDSBS	V3, V2, V1
	VADDSHS	V3, V2, V1
	VADDSWS	V3, V2, V1

//	Vector ADD extended, VA-form
//	<MNEMONIC> VRA,VRB,VRC,VRT produces
//	<mnemonic> VRT,VRA,VRB,VRC
	VADDEUQM V4, V3, V2, V1
	VADDECUQ V4, V3, V2, V1

//	Vector SUB, VX-form
//	<MNEMONIC> VRA,VRB,VRT produces
//	<mnemonic> VRT,VRA,VRB
	VSUBUBM	V3, V2, V1
	VSUBUHM	V3, V2, V1
	VSUBUWM	V3, V2, V1
	VSUBUDM	V3, V2, V1
	VSUBUQM	V3, V2, V1
	VSUBCUQ	V3, V2, V1
	VSUBCUW	V3, V2, V1
	VSUBUBS	V3, V2, V1
	VSUBUHS	V3, V2, V1
	VSUBUWS	V3, V2, V1
	VSUBSBS	V3, V2, V1
	VSUBSHS	V3, V2, V1
	VSUBSWS	V3, V2, V1

//	Vector SUB extended, VA-form
//	<MNEMONIC> VRA,VRB,VRC,VRT produces
//	<mnemonic> VRT,VRA,VRB,VRC
	VSUBEUQM V4, V3, V2, V1
	VSUBECUQ V4, V3, V2, V1

//	Vector rotate, VX-form
//	<MNEMONIC> VRA,VRB,VRT produces
//	<mnemonic> VRT,VRA,VRB
	VRLB	V2, V1, V0
	VRLH	V2, V1, V0
	VRLW	V2, V1, V0
	VRLD	V2, V1, V0

//	Vector shift, VX-form
//	<MNEMONIC> VRA,VRB,VRT
//	<mnemonic> VRT,VRA,VRB
	VSLB	V2, V1, V0
	VSLH	V2, V1, V0
	VSLW	V2, V1, V0
	VSL	V2, V1, V0
	VSLO	V2, V1, V0
	VSRB	V2, V1, V0
	VSRH	V2, V1, V0
	VSRW	V2, V1, V0
	VSR	V2, V1, V0
	VSRO	V2, V1, V0
	VSLD	V2, V1, V0
	VSRD	V2, V1, V0
	VSRAB	V2, V1, V0
	VSRAH	V2, V1, V0
	VSRAW	V2, V1, V0
	VSRAD	V2, V1, V0

//	Vector shift by octect immediate, VA-form with SHB 4-bit field
//	<MNEMONIC> SHB,VRA,VRB,VRT produces
//	<mnemonic> VRT,VRA,VRB,SHB
	VSLDOI	$4, V2, V1, V0

//	Vector count, VX-form
//	<MNEMONIC> VRB,VRT produces
//	<mnemonic> VRT,VRB
	VCLZB	V4, V5
	VCLZH	V4, V5
	VCLZW	V4, V5
	VCLZD	V4, V5
	VPOPCNTB V4, V5
	VPOPCNTH V4, V5
	VPOPCNTW V4, V5
	VPOPCNTD V4, V5

//	Vector compare, VC-form
//	<MNEMONIC> VRA,VRB,VRT produces
//	<mnemonic> VRT,VRA,VRB
//	* Note: 'CC' suffix denotes Rc=1
//	  i.e. vcmpequb. v3,v1,v2 equals VCMPEQUBCC V1,V2,V3
	VCMPEQUB    V3, V2, V1
	VCMPEQUBCC  V3, V2, V1
	VCMPEQUH    V3, V2, V1
	VCMPEQUHCC  V3, V2, V1
	VCMPEQUW    V3, V2, V1
	VCMPEQUWCC  V3, V2, V1
	VCMPEQUD    V3, V2, V1
	VCMPEQUDCC  V3, V2, V1
	VCMPGTUB    V3, V2, V1
	VCMPGTUBCC  V3, V2, V1
	VCMPGTUH    V3, V2, V1
	VCMPGTUHCC  V3, V2, V1
	VCMPGTUW    V3, V2, V1
	VCMPGTUWCC  V3, V2, V1
	VCMPGTUD    V3, V2, V1
	VCMPGTUDCC  V3, V2, V1
	VCMPGTSB    V3, V2, V1
	VCMPGTSBCC  V3, V2, V1
	VCMPGTSH    V3, V2, V1
	VCMPGTSHCC  V3, V2, V1
	VCMPGTSW    V3, V2, V1
	VCMPGTSWCC  V3, V2, V1
	VCMPGTSD    V3, V2, V1
	VCMPGTSDCC  V3, V2, V1

//	Vector permute, VA-form
//	<MNEMONIC> VRA,VRB,VRC,VRT produces
//	<mnemonic> VRT,VRA,VRB,VRC
	VPERM V3, V2, V1, V0

//	Vector select, VA-form
//	<MNEMONIC> VRA,VRB,VRC,VRT produces
//	<mnemonic> VRT,VRA,VRB,VRC
	VSEL  V3, V2, V1, V0

//	Vector splat, VX-form with 4-bit UIM field
//	<MNEMONIC> UIM,VRB,VRT produces
//	<mnemonic> VRT,VRB,UIM
	VSPLTB	  $15, V1, V0
	VSPLTH	  $7, V1, V0
	VSPLTW	  $3, V1, V0

//	Vector splat immediate signed, VX-form with 5-bit SIM field
//	<MNEMONIC> SIM,VRT produces
//	<mnemonic> VRT,SIM
	VSPLTISB  $31, V4
	VSPLTISH  $31, V4
	VSPLTISW  $31, V4

//	Vector AES cipher, VX-form
//	<MNEMONIC> VRA,VRB,VRT produces
//	<mnemonic> VRT,VRA,VRB
	VCIPHER	      V3, V2, V1
	VCIPHERLAST   V3, V2, V1
	VNCIPHER      V3, V2, V1
	VNCIPHERLAST  V3, V2, V1

//	Vector AES subbytes, VX-form
//	<MNEMONIC> VRA,VRT produces
//	<mnemonic> VRT,VRA
	VSBOX	      V2, V1

//	Vector SHA, VX-form with ST bit field and 4-bit SIX field
//	<MNEMONIC> SIX,VRA,ST,VRT produces
//	<mnemonic> VRT,VRA,ST,SIX
	VSHASIGMAW    $15, V1, $1, V0
	VSHASIGMAD    $15, V1, $1, V0

//	VSX instructions
//	Described as:
//	<instruction type>, <instruction format>
//	<go asm operand order> produces
//	<Power ISA operand order>

//	VSX load, XX1-form
//	<MNEMONIC> (RB)(RA*1),XT produces
//	<mnemonic> XT,RA,RB
	LXVD2X	    (R1)(R2*1), VS0
	LXVDSX	    (R1)(R2*1), VS0
	LXVW4X	    (R1)(R2*1), VS0
	LXSDX	    (R1)(R2*1), VS0
	LXSIWAX	    (R1)(R2*1), VS0
	LXSIWZX	    (R1)(R2*1), VS0

//	VSX store, XX1-form
//	<MNEMONIC> XS,(RB)(RA*1) produces
//	<mnemonic> XS,RA,RB
	STXVD2X	    VS63, (R1)(R2*1)
	STXVW4X	    VS63, (R1)(R2*1)
	STXSDX	    VS63, (R1)(R2*1)
	STXSIWX	    VS63, (R1)(R2*1)

//	VSX move from VSR, XX1-form
//	<MNEMONIC> XS,RA produces
//	<mnemonic> RA,XS
	MFVSRD	    VS0, R1
	MFVSRWZ	    VS33, R1

//	VSX move to VSR, XX1-form
//	<MNEMONIC> RA,XT produces
//	<mnemonic> XT,RA
	MTVSRD	    R1, VS0
	MTVSRWA	    R1, VS31
	MTVSRWZ	    R1, VS63

//	VSX AND, XX3-form
//	<MNEMONIC> XA,XB,XT produces
//	<mnemonic> XT,XA,XB
	XXLANDQ	    VS0,VS1,VS32
	XXLANDC	    VS0,VS1,VS32
	XXLEQV	    VS0,VS1,VS32
	XXLNAND	    VS0,VS1,VS32

//	VSX OR, XX3-form
//	<MNEMONIC> XA,XB,XT produces
//	<mnemonic> XT,XA,XB
	XXLORC	    VS0,VS1,VS32
	XXLNOR	    VS0,VS1,VS32
	XXLORQ	    VS0,VS1,VS32
	XXLXOR	    VS0,VS1,VS32

//	VSX select, XX4-form
//	<MNEMONIC> XA,XB,XC,XT produces
//	<mnemonic> XT,XA,XB,XC
	XXSEL	    VS0,VS1,VS3,VS32

//	VSX merge, XX3-form
//	<MNEMONIC> XA,XB,XT produces
//	<mnemonic> XT,XA,XB
	XXMRGHW	    VS0,VS1,VS32
	XXMRGLW	    VS0,VS1,VS32

//	VSX splat, XX2-form
//	<MNEMONIC> XB,UIM,XT produces
//	<mnemonic> XT,XB,UIM
	XXSPLTW	    VS0,$3,VS32

//	VSX permute, XX3-form
//	<MNEMONIC> XA,XB,DM,XT produces
//	<mnemonic> XT,XA,XB,DM
	XXPERMDI    VS0,VS1,$3,VS32

//	VSX shift, XX3-form
//	<MNEMONIC> XA,XB,SHW,XT produces
//	<mnemonic> XT,XA,XB,SHW
	XXSLDWI	    VS0,VS1,$3,VS32

//	VSX scalar FP-FP conversion, XX2-form
//	<MNEMONIC> XB,XT produces
//	<mnemonic> XT,XB
	XSCVDPSP    VS0,VS32
	XSCVSPDP    VS0,VS32
	XSCVDPSPN   VS0,VS32
	XSCVSPDPN   VS0,VS32

//	VSX vector FP-FP conversion, XX2-form
//	<MNEMONIC> XB,XT produces
//	<mnemonic> XT,XB
	XVCVDPSP    VS0,VS32
	XVCVSPDP    VS0,VS32

//	VSX scalar FP-integer conversion, XX2-form
//	<MNEMONIC> XB,XT produces
//	<mnemonic> XT,XB
	XSCVDPSXDS  VS0,VS32
	XSCVDPSXWS  VS0,VS32
	XSCVDPUXDS  VS0,VS32
	XSCVDPUXWS  VS0,VS32

//	VSX scalar integer-FP conversion, XX2-form
//	<MNEMONIC> XB,XT produces
//	<mnemonic> XT,XB
	XSCVSXDDP   VS0,VS32
	XSCVUXDDP   VS0,VS32
	XSCVSXDSP   VS0,VS32
	XSCVUXDSP   VS0,VS32

//	VSX vector FP-integer conversion, XX2-form
//	<MNEMONIC> XB,XT produces
//	<mnemonic> XT,XB
	XVCVDPSXDS  VS0,VS32
	XVCVDPSXWS  VS0,VS32
	XVCVDPUXDS  VS0,VS32
	XVCVDPUXWS  VS0,VS32
	XVCVSPSXDS  VS0,VS32
	XVCVSPSXWS  VS0,VS32
	XVCVSPUXDS  VS0,VS32
	XVCVSPUXWS  VS0,VS32

//	VSX scalar integer-FP conversion, XX2-form
//	<MNEMONIC> XB,XT produces
//	<mnemonic> XT,XB
	XVCVSXDDP   VS0,VS32
	XVCVSXWDP   VS0,VS32
	XVCVUXDDP   VS0,VS32
	XVCVUXWDP   VS0,VS32
	XVCVSXDSP   VS0,VS32
	XVCVSXWSP   VS0,VS32
	XVCVUXDSP   VS0,VS32
	XVCVUXWSP   VS0,VS32

//
// NOP
//
//	LNOP comma // asm doesn't support the trailing comma.
//	{
//		outcode(int($1), &nullgen, 0, &nullgen);
//	}
	NOP

//	LNOP rreg comma // asm doesn't support the trailing comma.
//	{
//		outcode(int($1), &$2, 0, &nullgen);
//	}
	NOP R2

//	LNOP freg comma // asm doesn't support the trailing comma.
//	{
//		outcode(int($1), &$2, 0, &nullgen);
//	}
	NOP	F2

//	LNOP ',' rreg // asm doesn't support the leading comma.
//	{
//		outcode(int($1), &nullgen, 0, &$3);
//	}
	NOP	R2

//	LNOP ',' freg // asm doesn't support the leading comma.
//	{
//		outcode(int($1), &nullgen, 0, &$3);
//	}
	NOP	F2

//	LNOP imm // SYSCALL $num: load $num to R0 before syscall and restore R0 to 0 afterwards.
//	{
//		outcode(int($1), &$2, 0, &nullgen);
//	}
	NOP	$4

// RET
//
//	LRETRN	comma // asm doesn't support the trailing comma.
//	{
//		outcode(int($1), &nullgen, 0, &nullgen);
//	}
	BEQ	2(PC)
	RET

// More BR/BL cases, and canonical names JMP, CALL.

	BEQ	2(PC)
	BR	foo(SB) // JMP foo(SB)
	BL	foo(SB) //  CALL foo(SB)
	BEQ	2(PC)
	JMP	foo(SB)
	CALL	foo(SB)

// END
//
//	LEND	comma // asm doesn't support the trailing comma.
//	{
//		outcode(int($1), &nullgen, 0, &nullgen);
//	}
	END
