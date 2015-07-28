// Inferno utils/5a/lex.c
// http://code.google.com/p/inferno-os/source/browse/utils/5a/lex.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.	All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//go:generate go tool yacc a.y

package main

import (
	"cmd/internal/asm"
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
)

var (
	yyerror  = asm.Yyerror
	nullgen  obj.Addr
	stmtline int32
)

const Always = arm.C_SCOND_NONE

func main() {
	cinit()

	asm.LSCONST = LSCONST
	asm.LCONST = LCONST
	asm.LFCONST = LFCONST
	asm.LNAME = LNAME
	asm.LVAR = LVAR
	asm.LLAB = LLAB

	asm.Lexinit = lexinit
	asm.Cclean = cclean
	asm.Yyparse = yyparse

	asm.Thechar = '5'
	asm.Thestring = "arm"
	asm.Thelinkarch = &arm.Linkarm

	asm.Main()
}

type yy struct{}

func (yy) Lex(v *yySymType) int {
	var av asm.Yylval
	tok := asm.Yylex(&av)
	v.sym = av.Sym
	v.lval = int32(av.Lval)
	v.sval = av.Sval
	v.dval = av.Dval
	return tok
}

func (yy) Error(msg string) {
	asm.Yyerror("%s", msg)
}

func yyparse() {
	yyParse(yy{})
}

var lexinit = []asm.Lextab{
	{"SP", LSP, obj.NAME_AUTO},
	{"SB", LSB, obj.NAME_EXTERN},
	{"FP", LFP, obj.NAME_PARAM},
	{"PC", LPC, obj.TYPE_BRANCH},

	{"R", LR, 0},

	{"R0", LREG, arm.REG_R0},
	{"R1", LREG, arm.REG_R1},
	{"R2", LREG, arm.REG_R2},
	{"R3", LREG, arm.REG_R3},
	{"R4", LREG, arm.REG_R4},
	{"R5", LREG, arm.REG_R5},
	{"R6", LREG, arm.REG_R6},
	{"R7", LREG, arm.REG_R7},
	{"R8", LREG, arm.REG_R8},
	{"R9", LREG, arm.REG_R9},
	{"g", LREG, arm.REG_R10}, // avoid unintentionally clobber g using R10
	{"R11", LREG, arm.REG_R11},
	{"R12", LREG, arm.REG_R12},
	{"R13", LREG, arm.REG_R13},
	{"R14", LREG, arm.REG_R14},
	{"R15", LREG, arm.REG_R15},
	{"F", LF, 0},
	{"F0", LFREG, arm.REG_F0},
	{"F1", LFREG, arm.REG_F1},
	{"F2", LFREG, arm.REG_F2},
	{"F3", LFREG, arm.REG_F3},
	{"F4", LFREG, arm.REG_F4},
	{"F5", LFREG, arm.REG_F5},
	{"F6", LFREG, arm.REG_F6},
	{"F7", LFREG, arm.REG_F7},
	{"F8", LFREG, arm.REG_F8},
	{"F9", LFREG, arm.REG_F9},
	{"F10", LFREG, arm.REG_F10},
	{"F11", LFREG, arm.REG_F11},
	{"F12", LFREG, arm.REG_F12},
	{"F13", LFREG, arm.REG_F13},
	{"F14", LFREG, arm.REG_F14},
	{"F15", LFREG, arm.REG_F15},
	{"C", LC, 0},
	{"C0", LCREG, 0},
	{"C1", LCREG, 1},
	{"C2", LCREG, 2},
	{"C3", LCREG, 3},
	{"C4", LCREG, 4},
	{"C5", LCREG, 5},
	{"C6", LCREG, 6},
	{"C7", LCREG, 7},
	{"C8", LCREG, 8},
	{"C9", LCREG, 9},
	{"C10", LCREG, 10},
	{"C11", LCREG, 11},
	{"C12", LCREG, 12},
	{"C13", LCREG, 13},
	{"C14", LCREG, 14},
	{"C15", LCREG, 15},
	{"CPSR", LPSR, arm.REG_CPSR},
	{"SPSR", LPSR, arm.REG_SPSR},
	{"FPSR", LFCR, arm.REG_FPSR},
	{"FPCR", LFCR, arm.REG_FPCR},
	{".EQ", LCOND, arm.C_SCOND_EQ},
	{".NE", LCOND, arm.C_SCOND_NE},
	{".CS", LCOND, arm.C_SCOND_HS},
	{".HS", LCOND, arm.C_SCOND_HS},
	{".CC", LCOND, arm.C_SCOND_LO},
	{".LO", LCOND, arm.C_SCOND_LO},
	{".MI", LCOND, arm.C_SCOND_MI},
	{".PL", LCOND, arm.C_SCOND_PL},
	{".VS", LCOND, arm.C_SCOND_VS},
	{".VC", LCOND, arm.C_SCOND_VC},
	{".HI", LCOND, arm.C_SCOND_HI},
	{".LS", LCOND, arm.C_SCOND_LS},
	{".GE", LCOND, arm.C_SCOND_GE},
	{".LT", LCOND, arm.C_SCOND_LT},
	{".GT", LCOND, arm.C_SCOND_GT},
	{".LE", LCOND, arm.C_SCOND_LE},
	{".AL", LCOND, arm.C_SCOND_NONE},
	{".U", LS, arm.C_UBIT},
	{".S", LS, arm.C_SBIT},
	{".W", LS, arm.C_WBIT},
	{".P", LS, arm.C_PBIT},
	{".PW", LS, arm.C_WBIT | arm.C_PBIT},
	{".WP", LS, arm.C_WBIT | arm.C_PBIT},
	{".F", LS, arm.C_FBIT},
	{".IBW", LS, arm.C_WBIT | arm.C_PBIT | arm.C_UBIT},
	{".IAW", LS, arm.C_WBIT | arm.C_UBIT},
	{".DBW", LS, arm.C_WBIT | arm.C_PBIT},
	{".DAW", LS, arm.C_WBIT},
	{".IB", LS, arm.C_PBIT | arm.C_UBIT},
	{".IA", LS, arm.C_UBIT},
	{".DB", LS, arm.C_PBIT},
	{".DA", LS, 0},
	{"@", LAT, 0},
	{"AND", LTYPE1, arm.AAND},
	{"EOR", LTYPE1, arm.AEOR},
	{"SUB", LTYPE1, arm.ASUB},
	{"RSB", LTYPE1, arm.ARSB},
	{"ADD", LTYPE1, arm.AADD},
	{"ADC", LTYPE1, arm.AADC},
	{"SBC", LTYPE1, arm.ASBC},
	{"RSC", LTYPE1, arm.ARSC},
	{"ORR", LTYPE1, arm.AORR},
	{"BIC", LTYPE1, arm.ABIC},
	{"SLL", LTYPE1, arm.ASLL},
	{"SRL", LTYPE1, arm.ASRL},
	{"SRA", LTYPE1, arm.ASRA},
	{"MUL", LTYPE1, arm.AMUL},
	{"MULA", LTYPEN, arm.AMULA},
	{"DIV", LTYPE1, arm.ADIV},
	{"MOD", LTYPE1, arm.AMOD},
	{"MULL", LTYPEM, arm.AMULL},
	{"MULAL", LTYPEM, arm.AMULAL},
	{"MULLU", LTYPEM, arm.AMULLU},
	{"MULALU", LTYPEM, arm.AMULALU},
	{"MVN", LTYPE2, arm.AMVN}, /* op2 ignored */
	{"MOVB", LTYPE3, arm.AMOVB},
	{"MOVBU", LTYPE3, arm.AMOVBU},
	{"MOVH", LTYPE3, arm.AMOVH},
	{"MOVHU", LTYPE3, arm.AMOVHU},
	{"MOVW", LTYPE3, arm.AMOVW},
	{"MOVD", LTYPE3, arm.AMOVD},
	{"MOVDF", LTYPE3, arm.AMOVDF},
	{"MOVDW", LTYPE3, arm.AMOVDW},
	{"MOVF", LTYPE3, arm.AMOVF},
	{"MOVFD", LTYPE3, arm.AMOVFD},
	{"MOVFW", LTYPE3, arm.AMOVFW},
	{"MOVWD", LTYPE3, arm.AMOVWD},
	{"MOVWF", LTYPE3, arm.AMOVWF},
	{"LDREX", LTYPE3, arm.ALDREX},
	{"LDREXD", LTYPE3, arm.ALDREXD},
	{"STREX", LTYPE9, arm.ASTREX},
	{"STREXD", LTYPE9, arm.ASTREXD},

	/*
		{"NEGF",		LTYPEI, ANEGF},
		{"NEGD",		LTYPEI, ANEGD},
		{"SQTF",		LTYPEI,	ASQTF},
		{"SQTD",		LTYPEI,	ASQTD},
		{"RNDF",		LTYPEI,	ARNDF},
		{"RNDD",		LTYPEI,	ARNDD},
		{"URDF",		LTYPEI,	AURDF},
		{"URDD",		LTYPEI,	AURDD},
		{"NRMF",		LTYPEI,	ANRMF},
		{"NRMD",		LTYPEI,	ANRMD},
	*/
	{"ABSF", LTYPEI, arm.AABSF},
	{"ABSD", LTYPEI, arm.AABSD},
	{"SQRTF", LTYPEI, arm.ASQRTF},
	{"SQRTD", LTYPEI, arm.ASQRTD},
	{"CMPF", LTYPEL, arm.ACMPF},
	{"CMPD", LTYPEL, arm.ACMPD},
	{"ADDF", LTYPEK, arm.AADDF},
	{"ADDD", LTYPEK, arm.AADDD},
	{"SUBF", LTYPEK, arm.ASUBF},
	{"SUBD", LTYPEK, arm.ASUBD},
	{"MULF", LTYPEK, arm.AMULF},
	{"MULD", LTYPEK, arm.AMULD},
	{"DIVF", LTYPEK, arm.ADIVF},
	{"DIVD", LTYPEK, arm.ADIVD},
	{"B", LTYPE4, arm.AB},
	{"JMP", LTYPE4, arm.AB},
	{"BL", LTYPE4, arm.ABL},
	{"CALL", LTYPE4, arm.ABL},
	{"BX", LTYPEBX, arm.ABX},
	{"BEQ", LTYPE5, arm.ABEQ},
	{"BNE", LTYPE5, arm.ABNE},
	{"BCS", LTYPE5, arm.ABCS},
	{"BHS", LTYPE5, arm.ABHS},
	{"BCC", LTYPE5, arm.ABCC},
	{"BLO", LTYPE5, arm.ABLO},
	{"BMI", LTYPE5, arm.ABMI},
	{"BPL", LTYPE5, arm.ABPL},
	{"BVS", LTYPE5, arm.ABVS},
	{"BVC", LTYPE5, arm.ABVC},
	{"BHI", LTYPE5, arm.ABHI},
	{"BLS", LTYPE5, arm.ABLS},
	{"BGE", LTYPE5, arm.ABGE},
	{"BLT", LTYPE5, arm.ABLT},
	{"BGT", LTYPE5, arm.ABGT},
	{"BLE", LTYPE5, arm.ABLE},
	{"BCASE", LTYPE5, arm.ABCASE},
	{"SWI", LTYPE6, arm.ASWI},
	{"CMP", LTYPE7, arm.ACMP},
	{"TST", LTYPE7, arm.ATST},
	{"TEQ", LTYPE7, arm.ATEQ},
	{"CMN", LTYPE7, arm.ACMN},
	{"MOVM", LTYPE8, arm.AMOVM},
	{"SWPBU", LTYPE9, arm.ASWPBU},
	{"SWPW", LTYPE9, arm.ASWPW},
	{"RET", LTYPEA, obj.ARET},
	{"RFE", LTYPEA, arm.ARFE},
	{"TEXT", LTYPEB, obj.ATEXT},
	{"GLOBL", LGLOBL, obj.AGLOBL},
	{"DATA", LTYPEC, obj.ADATA},
	{"CASE", LTYPED, arm.ACASE},
	{"END", LTYPEE, obj.AEND},
	{"WORD", LTYPEH, arm.AWORD},
	{"NOP", LTYPEI, obj.ANOP},
	{"MCR", LTYPEJ, 0},
	{"MRC", LTYPEJ, 1},
	{"PLD", LTYPEPLD, arm.APLD},
	{"UNDEF", LTYPEE, obj.AUNDEF},
	{"CLZ", LTYPE2, arm.ACLZ},
	{"MULWT", LTYPE1, arm.AMULWT},
	{"MULWB", LTYPE1, arm.AMULWB},
	{"MULAWT", LTYPEN, arm.AMULAWT},
	{"MULAWB", LTYPEN, arm.AMULAWB},
	{"USEFIELD", LTYPEN, obj.AUSEFIELD},
	{"PCDATA", LTYPEPC, obj.APCDATA},
	{"FUNCDATA", LTYPEF, obj.AFUNCDATA},
}

func cinit() {
}

func isreg(g *obj.Addr) bool {
	return true
}

func cclean() {
	outcode(obj.AEND, Always, &nullgen, 0, &nullgen)
}

var bcode = []int{
	arm.ABEQ,
	arm.ABNE,
	arm.ABCS,
	arm.ABCC,
	arm.ABMI,
	arm.ABPL,
	arm.ABVS,
	arm.ABVC,
	arm.ABHI,
	arm.ABLS,
	arm.ABGE,
	arm.ABLT,
	arm.ABGT,
	arm.ABLE,
	arm.AB,
	obj.ANOP,
}

var lastpc *obj.Prog

func outcode(a, scond int32, g1 *obj.Addr, reg int32, g2 *obj.Addr) {
	var p *obj.Prog
	var pl *obj.Plist

	/* hack to make B.NE etc. work: turn it into the corresponding conditional */
	if a == arm.AB {
		a = int32(bcode[(scond^arm.C_SCOND_XOR)&0xf])
		scond = (scond &^ 0xf) | Always
	}

	if asm.Pass == 1 {
		goto out
	}

	p = new(obj.Prog)
	*p = obj.Prog{}
	p.Ctxt = asm.Ctxt
	p.As = int16(a)
	p.Lineno = stmtline
	p.Scond = uint8(scond)
	p.From = *g1
	p.Reg = int16(reg)
	p.To = *g2
	p.Pc = int64(asm.PC)

	if lastpc == nil {
		pl = obj.Linknewplist(asm.Ctxt)
		pl.Firstpc = p
	} else {
		lastpc.Link = p
	}
	lastpc = p

out:
	if a != obj.AGLOBL && a != obj.ADATA {
		asm.PC++
	}
}
