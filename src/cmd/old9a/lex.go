// cmd/9a/lex.c from Vita Nuova.
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
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
	"cmd/internal/obj/ppc64"
)

var (
	yyerror  = asm.Yyerror
	nullgen  obj.Addr
	stmtline int32
)

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

	asm.Thechar = '9'
	asm.Thestring = "ppc64"
	asm.Thelinkarch = &ppc64.Linkppc64
	asm.Arches = map[string]*obj.LinkArch{
		"ppc64le": &ppc64.Linkppc64le,
	}

	asm.Main()
}

type yy struct{}

func (yy) Lex(v *yySymType) int {
	var av asm.Yylval
	tok := asm.Yylex(&av)
	v.sym = av.Sym
	v.lval = av.Lval
	v.sval = av.Sval
	v.dval = av.Dval
	return tok
}

func (yy) Error(msg string) {
	asm.Yyerror("%s", msg)
}

func yyparse() {
	nosched = 0
	yyParse(yy{})
}

var lexinit = []asm.Lextab{
	{"SP", LSP, obj.NAME_AUTO},
	{"SB", LSB, obj.NAME_EXTERN},
	{"FP", LFP, obj.NAME_PARAM},
	{"PC", LPC, obj.TYPE_BRANCH},
	{"LR", LLR, ppc64.REG_LR},
	{"CTR", LCTR, ppc64.REG_CTR},
	{"XER", LSPREG, ppc64.REG_XER},
	{"MSR", LMSR, ppc64.REG_MSR},
	{"FPSCR", LFPSCR, ppc64.REG_FPSCR},
	{"SPR", LSPR, ppc64.REG_SPR0},
	{"DCR", LSPR, ppc64.REG_DCR0},
	{"CR", LCR, ppc64.REG_CR},
	{"CR0", LCREG, ppc64.REG_CR0},
	{"CR1", LCREG, ppc64.REG_CR1},
	{"CR2", LCREG, ppc64.REG_CR2},
	{"CR3", LCREG, ppc64.REG_CR3},
	{"CR4", LCREG, ppc64.REG_CR4},
	{"CR5", LCREG, ppc64.REG_CR5},
	{"CR6", LCREG, ppc64.REG_CR6},
	{"CR7", LCREG, ppc64.REG_CR7},
	{"R", LR, 0},
	{"R0", LREG, ppc64.REG_R0},
	{"R1", LREG, ppc64.REG_R1},
	{"R2", LREG, ppc64.REG_R2},
	{"R3", LREG, ppc64.REG_R3},
	{"R4", LREG, ppc64.REG_R4},
	{"R5", LREG, ppc64.REG_R5},
	{"R6", LREG, ppc64.REG_R6},
	{"R7", LREG, ppc64.REG_R7},
	{"R8", LREG, ppc64.REG_R8},
	{"R9", LREG, ppc64.REG_R9},
	{"R10", LREG, ppc64.REG_R10},
	{"R11", LREG, ppc64.REG_R11},
	{"R12", LREG, ppc64.REG_R12},
	{"R13", LREG, ppc64.REG_R13},
	{"R14", LREG, ppc64.REG_R14},
	{"R15", LREG, ppc64.REG_R15},
	{"R16", LREG, ppc64.REG_R16},
	{"R17", LREG, ppc64.REG_R17},
	{"R18", LREG, ppc64.REG_R18},
	{"R19", LREG, ppc64.REG_R19},
	{"R20", LREG, ppc64.REG_R20},
	{"R21", LREG, ppc64.REG_R21},
	{"R22", LREG, ppc64.REG_R22},
	{"R23", LREG, ppc64.REG_R23},
	{"R24", LREG, ppc64.REG_R24},
	{"R25", LREG, ppc64.REG_R25},
	{"R26", LREG, ppc64.REG_R26},
	{"R27", LREG, ppc64.REG_R27},
	{"R28", LREG, ppc64.REG_R28},
	{"R29", LREG, ppc64.REG_R29},
	{"g", LREG, ppc64.REG_R30}, // avoid unintentionally clobbering g using R30
	{"R31", LREG, ppc64.REG_R31},
	{"F", LF, 0},
	{"F0", LFREG, ppc64.REG_F0},
	{"F1", LFREG, ppc64.REG_F1},
	{"F2", LFREG, ppc64.REG_F2},
	{"F3", LFREG, ppc64.REG_F3},
	{"F4", LFREG, ppc64.REG_F4},
	{"F5", LFREG, ppc64.REG_F5},
	{"F6", LFREG, ppc64.REG_F6},
	{"F7", LFREG, ppc64.REG_F7},
	{"F8", LFREG, ppc64.REG_F8},
	{"F9", LFREG, ppc64.REG_F9},
	{"F10", LFREG, ppc64.REG_F10},
	{"F11", LFREG, ppc64.REG_F11},
	{"F12", LFREG, ppc64.REG_F12},
	{"F13", LFREG, ppc64.REG_F13},
	{"F14", LFREG, ppc64.REG_F14},
	{"F15", LFREG, ppc64.REG_F15},
	{"F16", LFREG, ppc64.REG_F16},
	{"F17", LFREG, ppc64.REG_F17},
	{"F18", LFREG, ppc64.REG_F18},
	{"F19", LFREG, ppc64.REG_F19},
	{"F20", LFREG, ppc64.REG_F20},
	{"F21", LFREG, ppc64.REG_F21},
	{"F22", LFREG, ppc64.REG_F22},
	{"F23", LFREG, ppc64.REG_F23},
	{"F24", LFREG, ppc64.REG_F24},
	{"F25", LFREG, ppc64.REG_F25},
	{"F26", LFREG, ppc64.REG_F26},
	{"F27", LFREG, ppc64.REG_F27},
	{"F28", LFREG, ppc64.REG_F28},
	{"F29", LFREG, ppc64.REG_F29},
	{"F30", LFREG, ppc64.REG_F30},
	{"F31", LFREG, ppc64.REG_F31},
	{"CREQV", LCROP, ppc64.ACREQV},
	{"CRXOR", LCROP, ppc64.ACRXOR},
	{"CRAND", LCROP, ppc64.ACRAND},
	{"CROR", LCROP, ppc64.ACROR},
	{"CRANDN", LCROP, ppc64.ACRANDN},
	{"CRORN", LCROP, ppc64.ACRORN},
	{"CRNAND", LCROP, ppc64.ACRNAND},
	{"CRNOR", LCROP, ppc64.ACRNOR},
	{"ADD", LADDW, ppc64.AADD},
	{"ADDV", LADDW, ppc64.AADDV},
	{"ADDCC", LADDW, ppc64.AADDCC},
	{"ADDVCC", LADDW, ppc64.AADDVCC},
	{"ADDC", LADDW, ppc64.AADDC},
	{"ADDCV", LADDW, ppc64.AADDCV},
	{"ADDCCC", LADDW, ppc64.AADDCCC},
	{"ADDCVCC", LADDW, ppc64.AADDCVCC},
	{"ADDE", LLOGW, ppc64.AADDE},
	{"ADDEV", LLOGW, ppc64.AADDEV},
	{"ADDECC", LLOGW, ppc64.AADDECC},
	{"ADDEVCC", LLOGW, ppc64.AADDEVCC},
	{"ADDME", LABS, ppc64.AADDME},
	{"ADDMEV", LABS, ppc64.AADDMEV},
	{"ADDMECC", LABS, ppc64.AADDMECC},
	{"ADDMEVCC", LABS, ppc64.AADDMEVCC},
	{"ADDZE", LABS, ppc64.AADDZE},
	{"ADDZEV", LABS, ppc64.AADDZEV},
	{"ADDZECC", LABS, ppc64.AADDZECC},
	{"ADDZEVCC", LABS, ppc64.AADDZEVCC},
	{"SUB", LADDW, ppc64.ASUB},
	{"SUBV", LADDW, ppc64.ASUBV},
	{"SUBCC", LADDW, ppc64.ASUBCC},
	{"SUBVCC", LADDW, ppc64.ASUBVCC},
	{"SUBE", LLOGW, ppc64.ASUBE},
	{"SUBECC", LLOGW, ppc64.ASUBECC},
	{"SUBEV", LLOGW, ppc64.ASUBEV},
	{"SUBEVCC", LLOGW, ppc64.ASUBEVCC},
	{"SUBC", LADDW, ppc64.ASUBC},
	{"SUBCCC", LADDW, ppc64.ASUBCCC},
	{"SUBCV", LADDW, ppc64.ASUBCV},
	{"SUBCVCC", LADDW, ppc64.ASUBCVCC},
	{"SUBME", LABS, ppc64.ASUBME},
	{"SUBMEV", LABS, ppc64.ASUBMEV},
	{"SUBMECC", LABS, ppc64.ASUBMECC},
	{"SUBMEVCC", LABS, ppc64.ASUBMEVCC},
	{"SUBZE", LABS, ppc64.ASUBZE},
	{"SUBZEV", LABS, ppc64.ASUBZEV},
	{"SUBZECC", LABS, ppc64.ASUBZECC},
	{"SUBZEVCC", LABS, ppc64.ASUBZEVCC},
	{"AND", LADDW, ppc64.AAND},
	{"ANDCC", LADDW, ppc64.AANDCC}, /* includes andil & andiu */
	{"ANDN", LLOGW, ppc64.AANDN},
	{"ANDNCC", LLOGW, ppc64.AANDNCC},
	{"EQV", LLOGW, ppc64.AEQV},
	{"EQVCC", LLOGW, ppc64.AEQVCC},
	{"NAND", LLOGW, ppc64.ANAND},
	{"NANDCC", LLOGW, ppc64.ANANDCC},
	{"NOR", LLOGW, ppc64.ANOR},
	{"NORCC", LLOGW, ppc64.ANORCC},
	{"OR", LADDW, ppc64.AOR}, /* includes oril & oriu */
	{"ORCC", LADDW, ppc64.AORCC},
	{"ORN", LLOGW, ppc64.AORN},
	{"ORNCC", LLOGW, ppc64.AORNCC},
	{"XOR", LADDW, ppc64.AXOR}, /* includes xoril & xoriu */
	{"XORCC", LLOGW, ppc64.AXORCC},
	{"EXTSB", LABS, ppc64.AEXTSB},
	{"EXTSBCC", LABS, ppc64.AEXTSBCC},
	{"EXTSH", LABS, ppc64.AEXTSH},
	{"EXTSHCC", LABS, ppc64.AEXTSHCC},
	{"CNTLZW", LABS, ppc64.ACNTLZW},
	{"CNTLZWCC", LABS, ppc64.ACNTLZWCC},
	{"RLWMI", LRLWM, ppc64.ARLWMI},
	{"RLWMICC", LRLWM, ppc64.ARLWMICC},
	{"RLWNM", LRLWM, ppc64.ARLWNM},
	{"RLWNMCC", LRLWM, ppc64.ARLWNMCC},
	{"SLW", LSHW, ppc64.ASLW},
	{"SLWCC", LSHW, ppc64.ASLWCC},
	{"SRW", LSHW, ppc64.ASRW},
	{"SRWCC", LSHW, ppc64.ASRWCC},
	{"SRAW", LSHW, ppc64.ASRAW},
	{"SRAWCC", LSHW, ppc64.ASRAWCC},
	{"BR", LBRA, ppc64.ABR},
	{"JMP", LBRA, ppc64.ABR},
	{"BC", LBRA, ppc64.ABC},
	{"BCL", LBRA, ppc64.ABC},
	{"BL", LBRA, ppc64.ABL},
	{"CALL", LBRA, ppc64.ABL},
	{"BEQ", LBRA, ppc64.ABEQ},
	{"BNE", LBRA, ppc64.ABNE},
	{"BGT", LBRA, ppc64.ABGT},
	{"BGE", LBRA, ppc64.ABGE},
	{"BLT", LBRA, ppc64.ABLT},
	{"BLE", LBRA, ppc64.ABLE},
	{"BVC", LBRA, ppc64.ABVC},
	{"BVS", LBRA, ppc64.ABVS},
	{"CMP", LCMP, ppc64.ACMP},
	{"CMPU", LCMP, ppc64.ACMPU},
	{"CMPW", LCMP, ppc64.ACMPW},
	{"CMPWU", LCMP, ppc64.ACMPWU},
	{"DIVW", LLOGW, ppc64.ADIVW},
	{"DIVWV", LLOGW, ppc64.ADIVWV},
	{"DIVWCC", LLOGW, ppc64.ADIVWCC},
	{"DIVWVCC", LLOGW, ppc64.ADIVWVCC},
	{"DIVWU", LLOGW, ppc64.ADIVWU},
	{"DIVWUV", LLOGW, ppc64.ADIVWUV},
	{"DIVWUCC", LLOGW, ppc64.ADIVWUCC},
	{"DIVWUVCC", LLOGW, ppc64.ADIVWUVCC},
	{"FABS", LFCONV, ppc64.AFABS},
	{"FABSCC", LFCONV, ppc64.AFABSCC},
	{"FNEG", LFCONV, ppc64.AFNEG},
	{"FNEGCC", LFCONV, ppc64.AFNEGCC},
	{"FNABS", LFCONV, ppc64.AFNABS},
	{"FNABSCC", LFCONV, ppc64.AFNABSCC},
	{"FADD", LFADD, ppc64.AFADD},
	{"FADDCC", LFADD, ppc64.AFADDCC},
	{"FSUB", LFADD, ppc64.AFSUB},
	{"FSUBCC", LFADD, ppc64.AFSUBCC},
	{"FMUL", LFADD, ppc64.AFMUL},
	{"FMULCC", LFADD, ppc64.AFMULCC},
	{"FDIV", LFADD, ppc64.AFDIV},
	{"FDIVCC", LFADD, ppc64.AFDIVCC},
	{"FRSP", LFCONV, ppc64.AFRSP},
	{"FRSPCC", LFCONV, ppc64.AFRSPCC},
	{"FCTIW", LFCONV, ppc64.AFCTIW},
	{"FCTIWCC", LFCONV, ppc64.AFCTIWCC},
	{"FCTIWZ", LFCONV, ppc64.AFCTIWZ},
	{"FCTIWZCC", LFCONV, ppc64.AFCTIWZCC},
	{"FMADD", LFMA, ppc64.AFMADD},
	{"FMADDCC", LFMA, ppc64.AFMADDCC},
	{"FMSUB", LFMA, ppc64.AFMSUB},
	{"FMSUBCC", LFMA, ppc64.AFMSUBCC},
	{"FNMADD", LFMA, ppc64.AFNMADD},
	{"FNMADDCC", LFMA, ppc64.AFNMADDCC},
	{"FNMSUB", LFMA, ppc64.AFNMSUB},
	{"FNMSUBCC", LFMA, ppc64.AFNMSUBCC},
	{"FMADDS", LFMA, ppc64.AFMADDS},
	{"FMADDSCC", LFMA, ppc64.AFMADDSCC},
	{"FMSUBS", LFMA, ppc64.AFMSUBS},
	{"FMSUBSCC", LFMA, ppc64.AFMSUBSCC},
	{"FNMADDS", LFMA, ppc64.AFNMADDS},
	{"FNMADDSCC", LFMA, ppc64.AFNMADDSCC},
	{"FNMSUBS", LFMA, ppc64.AFNMSUBS},
	{"FNMSUBSCC", LFMA, ppc64.AFNMSUBSCC},
	{"FCMPU", LFCMP, ppc64.AFCMPU},
	{"FCMPO", LFCMP, ppc64.AFCMPO},
	{"MTFSB0", LMTFSB, ppc64.AMTFSB0},
	{"MTFSB1", LMTFSB, ppc64.AMTFSB1},
	{"FMOVD", LFMOV, ppc64.AFMOVD},
	{"FMOVS", LFMOV, ppc64.AFMOVS},
	{"FMOVDCC", LFCONV, ppc64.AFMOVDCC}, /* fmr. */
	{"GLOBL", LGLOBL, obj.AGLOBL},
	{"MOVB", LMOVB, ppc64.AMOVB},
	{"MOVBZ", LMOVB, ppc64.AMOVBZ},
	{"MOVBU", LMOVB, ppc64.AMOVBU},
	{"MOVBZU", LMOVB, ppc64.AMOVBZU},
	{"MOVH", LMOVB, ppc64.AMOVH},
	{"MOVHZ", LMOVB, ppc64.AMOVHZ},
	{"MOVHU", LMOVB, ppc64.AMOVHU},
	{"MOVHZU", LMOVB, ppc64.AMOVHZU},
	{"MOVHBR", LXMV, ppc64.AMOVHBR},
	{"MOVWBR", LXMV, ppc64.AMOVWBR},
	{"MOVW", LMOVW, ppc64.AMOVW},
	{"MOVWU", LMOVW, ppc64.AMOVWU},
	{"MOVMW", LMOVMW, ppc64.AMOVMW},
	{"MOVFL", LMOVW, ppc64.AMOVFL},
	{"MULLW", LADDW, ppc64.AMULLW}, /* includes multiply immediate 10-139 */
	{"MULLWV", LLOGW, ppc64.AMULLWV},
	{"MULLWCC", LLOGW, ppc64.AMULLWCC},
	{"MULLWVCC", LLOGW, ppc64.AMULLWVCC},
	{"MULHW", LLOGW, ppc64.AMULHW},
	{"MULHWCC", LLOGW, ppc64.AMULHWCC},
	{"MULHWU", LLOGW, ppc64.AMULHWU},
	{"MULHWUCC", LLOGW, ppc64.AMULHWUCC},
	{"NEG", LABS, ppc64.ANEG},
	{"NEGV", LABS, ppc64.ANEGV},
	{"NEGCC", LABS, ppc64.ANEGCC},
	{"NEGVCC", LABS, ppc64.ANEGVCC},
	{"NOP", LNOP, obj.ANOP}, /* ori 0,0,0 */
	{"SYSCALL", LNOP, ppc64.ASYSCALL},
	{"UNDEF", LNOP, obj.AUNDEF},
	{"RET", LRETRN, obj.ARET},
	{"RETURN", LRETRN, obj.ARET},
	{"RFI", LRETRN, ppc64.ARFI},
	{"RFCI", LRETRN, ppc64.ARFCI},
	{"DATA", LDATA, obj.ADATA},
	{"END", LEND, obj.AEND},
	{"TEXT", LTEXT, obj.ATEXT},

	/* 64-bit instructions */
	{"CNTLZD", LABS, ppc64.ACNTLZD},
	{"CNTLZDCC", LABS, ppc64.ACNTLZDCC},
	{"DIVD", LLOGW, ppc64.ADIVD},
	{"DIVDCC", LLOGW, ppc64.ADIVDCC},
	{"DIVDVCC", LLOGW, ppc64.ADIVDVCC},
	{"DIVDV", LLOGW, ppc64.ADIVDV},
	{"DIVDU", LLOGW, ppc64.ADIVDU},
	{"DIVDUCC", LLOGW, ppc64.ADIVDUCC},
	{"DIVDUVCC", LLOGW, ppc64.ADIVDUVCC},
	{"DIVDUV", LLOGW, ppc64.ADIVDUV},
	{"EXTSW", LABS, ppc64.AEXTSW},
	{"EXTSWCC", LABS, ppc64.AEXTSWCC},
	{"FCTID", LFCONV, ppc64.AFCTID},
	{"FCTIDCC", LFCONV, ppc64.AFCTIDCC},
	{"FCTIDZ", LFCONV, ppc64.AFCTIDZ},
	{"FCTIDZCC", LFCONV, ppc64.AFCTIDZCC},
	{"FCFID", LFCONV, ppc64.AFCFID},
	{"FCFIDCC", LFCONV, ppc64.AFCFIDCC},
	{"LDAR", LXLD, ppc64.ALDAR},
	{"MOVD", LMOVW, ppc64.AMOVD},
	{"MOVDU", LMOVW, ppc64.AMOVDU},
	{"MOVWZ", LMOVW, ppc64.AMOVWZ},
	{"MOVWZU", LMOVW, ppc64.AMOVWZU},
	{"MULHD", LLOGW, ppc64.AMULHD},
	{"MULHDCC", LLOGW, ppc64.AMULHDCC},
	{"MULHDU", LLOGW, ppc64.AMULHDU},
	{"MULHDUCC", LLOGW, ppc64.AMULHDUCC},
	{"MULLD", LADDW, ppc64.AMULLD}, /* includes multiply immediate? */
	{"MULLDCC", LLOGW, ppc64.AMULLDCC},
	{"MULLDVCC", LLOGW, ppc64.AMULLDVCC},
	{"MULLDV", LLOGW, ppc64.AMULLDV},
	{"RFID", LRETRN, ppc64.ARFID},
	{"HRFID", LRETRN, ppc64.AHRFID},
	{"RLDMI", LRLWM, ppc64.ARLDMI},
	{"RLDMICC", LRLWM, ppc64.ARLDMICC},
	{"RLDC", LRLWM, ppc64.ARLDC},
	{"RLDCCC", LRLWM, ppc64.ARLDCCC},
	{"RLDCR", LRLWM, ppc64.ARLDCR},
	{"RLDCRCC", LRLWM, ppc64.ARLDCRCC},
	{"RLDCL", LRLWM, ppc64.ARLDCL},
	{"RLDCLCC", LRLWM, ppc64.ARLDCLCC},
	{"SLBIA", LNOP, ppc64.ASLBIA},
	{"SLBIE", LNOP, ppc64.ASLBIE},
	{"SLBMFEE", LABS, ppc64.ASLBMFEE},
	{"SLBMFEV", LABS, ppc64.ASLBMFEV},
	{"SLBMTE", LABS, ppc64.ASLBMTE},
	{"SLD", LSHW, ppc64.ASLD},
	{"SLDCC", LSHW, ppc64.ASLDCC},
	{"SRD", LSHW, ppc64.ASRD},
	{"SRAD", LSHW, ppc64.ASRAD},
	{"SRADCC", LSHW, ppc64.ASRADCC},
	{"SRDCC", LSHW, ppc64.ASRDCC},
	{"STDCCC", LXST, ppc64.ASTDCCC},
	{"TD", LADDW, ppc64.ATD},

	/* pseudo instructions */
	{"REM", LLOGW, ppc64.AREM},
	{"REMCC", LLOGW, ppc64.AREMCC},
	{"REMV", LLOGW, ppc64.AREMV},
	{"REMVCC", LLOGW, ppc64.AREMVCC},
	{"REMU", LLOGW, ppc64.AREMU},
	{"REMUCC", LLOGW, ppc64.AREMUCC},
	{"REMUV", LLOGW, ppc64.AREMUV},
	{"REMUVCC", LLOGW, ppc64.AREMUVCC},
	{"REMD", LLOGW, ppc64.AREMD},
	{"REMDCC", LLOGW, ppc64.AREMDCC},
	{"REMDV", LLOGW, ppc64.AREMDV},
	{"REMDVCC", LLOGW, ppc64.AREMDVCC},
	{"REMDU", LLOGW, ppc64.AREMDU},
	{"REMDUCC", LLOGW, ppc64.AREMDUCC},
	{"REMDUV", LLOGW, ppc64.AREMDUV},
	{"REMDUVCC", LLOGW, ppc64.AREMDUVCC},

	/* special instructions */
	{"DCBF", LXOP, ppc64.ADCBF},
	{"DCBI", LXOP, ppc64.ADCBI},
	{"DCBST", LXOP, ppc64.ADCBST},
	{"DCBT", LXOP, ppc64.ADCBT},
	{"DCBTST", LXOP, ppc64.ADCBTST},
	{"DCBZ", LXOP, ppc64.ADCBZ},
	{"ICBI", LXOP, ppc64.AICBI},
	{"ECIWX", LXLD, ppc64.AECIWX},
	{"ECOWX", LXST, ppc64.AECOWX},
	{"LWAR", LXLD, ppc64.ALWAR},
	{"STWCCC", LXST, ppc64.ASTWCCC},
	{"EIEIO", LRETRN, ppc64.AEIEIO},
	{"TLBIE", LNOP, ppc64.ATLBIE},
	{"TLBIEL", LNOP, ppc64.ATLBIEL},
	{"LSW", LXLD, ppc64.ALSW},
	{"STSW", LXST, ppc64.ASTSW},
	{"ISYNC", LRETRN, ppc64.AISYNC},
	{"SYNC", LRETRN, ppc64.ASYNC},
	{"TLBSYNC", LRETRN, ppc64.ATLBSYNC},
	{"PTESYNC", LRETRN, ppc64.APTESYNC},

	/*	"TW",		LADDW,	ATW,*/
	{"WORD", LWORD, ppc64.AWORD},
	{"DWORD", LWORD, ppc64.ADWORD},
	{"SCHED", LSCHED, 0},
	{"NOSCHED", LSCHED, 0x80},
	{"PCDATA", LPCDAT, obj.APCDATA},
	{"FUNCDATA", LFUNCDAT, obj.AFUNCDATA},
}

func cinit() {
}

func cclean() {
	outcode(obj.AEND, &nullgen, 0, &nullgen)
}

var lastpc *obj.Prog
var nosched int

func outcode(a int, g1 *obj.Addr, reg int, g2 *obj.Addr) {
	var p *obj.Prog
	var pl *obj.Plist

	if asm.Pass == 1 {
		goto out
	}

	if g1.Scale != 0 {
		if reg != 0 || g2.Scale != 0 {
			yyerror("bad addressing modes")
		}
		reg = int(g1.Scale)
	} else if g2.Scale != 0 {
		if reg != 0 {
			yyerror("bad addressing modes")
		}
		reg = int(g2.Scale)
	}

	p = asm.Ctxt.NewProg()
	p.As = int16(a)
	p.Lineno = stmtline
	if nosched != 0 {
		p.Mark |= ppc64.NOSCHED
	}
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

func outgcode(a int, g1 *obj.Addr, reg int, g2, g3 *obj.Addr) {
	var p *obj.Prog
	var pl *obj.Plist

	if asm.Pass == 1 {
		goto out
	}

	p = asm.Ctxt.NewProg()
	p.As = int16(a)
	p.Lineno = stmtline
	if nosched != 0 {
		p.Mark |= ppc64.NOSCHED
	}
	p.From = *g1
	p.Reg = int16(reg)
	if g2.Type != 0 {
		p.From3 = new(obj.Addr)
		*p.From3 = *g2
	}
	p.To = *g3
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
