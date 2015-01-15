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

#define	EXTERN
#include <u.h>
#include <libc.h>
#include "a.h"
#include "y.tab.h"

enum
{
	Plan9	= 1<<0,
	Unix	= 1<<1,
	Windows	= 1<<2,
};

int
systemtype(int sys)
{
#ifdef _WIN32
	return sys&Windows;
#else
	return sys&Plan9;
#endif
}

int
pathchar(void)
{
	return '/';
}

int
Lconv(Fmt *fp)
{
	return linklinefmt(ctxt, fp);
}

void
dodef(char *p)
{
	if(nDlist%8 == 0)
		Dlist = allocn(Dlist, nDlist*sizeof(char *),
			8*sizeof(char *));
	Dlist[nDlist++] = p;
}

LinkArch*       thelinkarch = &linkppc64;

void
usage(void)
{
	print("usage: %ca [options] file.c...\n", thechar);
	flagprint(1);
	errorexit();
}

void
main(int argc, char *argv[])
{
	char *p;

	thechar = '9';
	thestring = "ppc64";

	// Allow GOARCH=thestring or GOARCH=thestringsuffix,
	// but not other values.	
	p = getgoarch();
	if(strncmp(p, thestring, strlen(thestring)) != 0)
		sysfatal("cannot use %cc with GOARCH=%s", thechar, p);
	if(strcmp(p, "ppc64le") == 0)
		thelinkarch = &linkppc64le;

	ctxt = linknew(thelinkarch);
	ctxt->diag = yyerror;
	ctxt->bso = &bstdout;
	ctxt->enforce_data_order = 1;
	Binit(&bstdout, 1, OWRITE);
	listinit9();
	fmtinstall('L', Lconv);

	ensuresymb(NSYMB);
	memset(debug, 0, sizeof(debug));
	cinit();
	outfile = 0;
	setinclude(".");

	flagfn1("D", "name[=value]: add #define", dodef);
	flagfn1("I", "dir: add dir to include path", setinclude);
	flagcount("S", "print assembly and machine code", &debug['S']);
	flagcount("m", "debug preprocessor macros", &debug['m']);
	flagstr("o", "file: set output file", &outfile);
	flagstr("trimpath", "prefix: remove prefix from recorded source file paths", &ctxt->trimpath);

	flagparse(&argc, &argv, usage);
	ctxt->debugasm = debug['S'];

	if(argc < 1)
		usage();
	if(argc > 1){
		print("can't assemble multiple files\n");
		errorexit();
	}

	if(assemble(argv[0]))
		errorexit();
	Bflush(&bstdout);
	if(nerrors > 0)
		errorexit();
	exits(0);
}

int
assemble(char *file)
{
	char *ofile, *p;
	int i, of;

	ofile = alloc(strlen(file)+3); // +3 for .x\0 (x=thechar)
	strcpy(ofile, file);
	p = utfrrune(ofile, pathchar());
	if(p) {
		include[0] = ofile;
		*p++ = 0;
	} else
		p = ofile;
	if(outfile == 0) {
		outfile = p;
		if(outfile){
			p = utfrrune(outfile, '.');
			if(p)
				if(p[1] == 's' && p[2] == 0)
					p[0] = 0;
			p = utfrune(outfile, 0);
			p[0] = '.';
			p[1] = thechar;
			p[2] = 0;
		} else
			outfile = "/dev/null";
	}

	of = create(outfile, OWRITE, 0664);
	if(of < 0) {
		yyerror("%ca: cannot create %s", thechar, outfile);
		errorexit();
	}
	Binit(&obuf, of, OWRITE);
	Bprint(&obuf, "go object %s %s %s\n", getgoos(), getgoarch(), getgoversion());
	Bprint(&obuf, "!\n");

	for(pass = 1; pass <= 2; pass++) {
		nosched = 0;
		pinit(file);
		for(i=0; i<nDlist; i++)
			dodefine(Dlist[i]);
		yyparse();
		cclean();
		if(nerrors)
			return nerrors;
	}

	writeobj(ctxt, &obuf);
	Bflush(&obuf);
	return 0;
}

struct
{
	char	*name;
	ushort	type;
	ushort	value;
} itab[] =
{
	"SP",		LSP,	D_AUTO,
	"SB",		LSB,	D_EXTERN,
	"FP",		LFP,	D_PARAM,
	"PC",		LPC,	D_BRANCH,

	"LR",		LLR,	D_LR,
	"CTR",		LCTR,	D_CTR,

	"XER",		LSPREG,	D_XER,
	"MSR",		LMSR,	D_MSR,
	"FPSCR",	LFPSCR,	D_FPSCR,
	"SPR",		LSPR,	D_SPR,
	"DCR",		LSPR,	D_DCR,

	"CR",		LCR,	0,
	"CR0",		LCREG,	0,
	"CR1",		LCREG,	1,
	"CR2",		LCREG,	2,
	"CR3",		LCREG,	3,
	"CR4",		LCREG,	4,
	"CR5",		LCREG,	5,
	"CR6",		LCREG,	6,
	"CR7",		LCREG,	7,

	"R",		LR,	0,
	"R0",		LREG,	0,
	"R1",		LREG,	1,
	"R2",		LREG,	2,
	"R3",		LREG,	3,
	"R4",		LREG,	4,
	"R5",		LREG,	5,
	"R6",		LREG,	6,
	"R7",		LREG,	7,
	"R8",		LREG,	8,
	"R9",		LREG,	9,
	"R10",		LREG,	10,
	"R11",		LREG,	11,
	"R12",		LREG,	12,
	"R13",		LREG,	13,
	"R14",		LREG,	14,
	"R15",		LREG,	15,
	"R16",		LREG,	16,
	"R17",		LREG,	17,
	"R18",		LREG,	18,
	"R19",		LREG,	19,
	"R20",		LREG,	20,
	"R21",		LREG,	21,
	"R22",		LREG,	22,
	"R23",		LREG,	23,
	"R24",		LREG,	24,
	"R25",		LREG,	25,
	"R26",		LREG,	26,
	"R27",		LREG,	27,
	"R28",		LREG,	28,
	"R29",		LREG,	29,
	"g",		LREG,	30, // avoid unintentionally clobbering g using R30
	"R31",		LREG,	31,

	"F",		LF,	0,
	"F0",		LFREG,	0,
	"F1",		LFREG,	1,
	"F2",		LFREG,	2,
	"F3",		LFREG,	3,
	"F4",		LFREG,	4,
	"F5",		LFREG,	5,
	"F6",		LFREG,	6,
	"F7",		LFREG,	7,
	"F8",		LFREG,	8,
	"F9",		LFREG,	9,
	"F10",		LFREG,	10,
	"F11",		LFREG,	11,
	"F12",		LFREG,	12,
	"F13",		LFREG,	13,
	"F14",		LFREG,	14,
	"F15",		LFREG,	15,
	"F16",		LFREG,	16,
	"F17",		LFREG,	17,
	"F18",		LFREG,	18,
	"F19",		LFREG,	19,
	"F20",		LFREG,	20,
	"F21",		LFREG,	21,
	"F22",		LFREG,	22,
	"F23",		LFREG,	23,
	"F24",		LFREG,	24,
	"F25",		LFREG,	25,
	"F26",		LFREG,	26,
	"F27",		LFREG,	27,
	"F28",		LFREG,	28,
	"F29",		LFREG,	29,
	"F30",		LFREG,	30,
	"F31",		LFREG,	31,

	"CREQV",	LCROP, ACREQV,
	"CRXOR",	LCROP, ACRXOR,
	"CRAND",	LCROP, ACRAND,
	"CROR",		LCROP, ACROR,
	"CRANDN",	LCROP, ACRANDN,
	"CRORN",	LCROP, ACRORN,
	"CRNAND",	LCROP, ACRNAND,
	"CRNOR",	LCROP, ACRNOR,

	"ADD",		LADDW, AADD,
	"ADDV",		LADDW, AADDV,
	"ADDCC",	LADDW, AADDCC,
	"ADDVCC",	LADDW, AADDVCC,
	"ADDC",		LADDW, AADDC,
	"ADDCV",	LADDW, AADDCV,
	"ADDCCC",	LADDW, AADDCCC,
	"ADDCVCC",	LADDW, AADDCVCC,
	"ADDE",		LLOGW, AADDE,
	"ADDEV",	LLOGW, AADDEV,
	"ADDECC",	LLOGW, AADDECC,
	"ADDEVCC",	LLOGW, AADDEVCC,

	"ADDME",	LABS, AADDME,
	"ADDMEV",	LABS, AADDMEV,
	"ADDMECC",	LABS, AADDMECC,
	"ADDMEVCC",	LABS, AADDMEVCC,
	"ADDZE",	LABS, AADDZE,
	"ADDZEV",	LABS, AADDZEV,
	"ADDZECC",	LABS, AADDZECC,
	"ADDZEVCC",	LABS, AADDZEVCC,

	"SUB",		LADDW, ASUB,
	"SUBV",		LADDW, ASUBV,
	"SUBCC",	LADDW, ASUBCC,
	"SUBVCC",	LADDW, ASUBVCC,
	"SUBE",		LLOGW, ASUBE,
	"SUBECC",	LLOGW, ASUBECC,
	"SUBEV",	LLOGW, ASUBEV,
	"SUBEVCC",	LLOGW, ASUBEVCC,
	"SUBC",		LADDW, ASUBC,
	"SUBCCC",	LADDW, ASUBCCC,
	"SUBCV",	LADDW, ASUBCV,
	"SUBCVCC",	LADDW, ASUBCVCC,

	"SUBME",	LABS, ASUBME,
	"SUBMEV",	LABS, ASUBMEV,
	"SUBMECC",	LABS, ASUBMECC,
	"SUBMEVCC",	LABS, ASUBMEVCC,
	"SUBZE",	LABS, ASUBZE,
	"SUBZEV",	LABS, ASUBZEV,
	"SUBZECC",	LABS, ASUBZECC,
	"SUBZEVCC",	LABS, ASUBZEVCC,

	"AND",		LADDW, AAND,
	"ANDCC",	LADDW, AANDCC,	/* includes andil & andiu */
	"ANDN",		LLOGW, AANDN,
	"ANDNCC",	LLOGW, AANDNCC,
	"EQV",		LLOGW, AEQV,
	"EQVCC",	LLOGW, AEQVCC,
	"NAND",		LLOGW, ANAND,
	"NANDCC",	LLOGW, ANANDCC,
	"NOR",		LLOGW, ANOR,
	"NORCC",	LLOGW, ANORCC,
	"OR",		LADDW, AOR,	/* includes oril & oriu */
	"ORCC",		LADDW, AORCC,
	"ORN",		LLOGW, AORN,
	"ORNCC",	LLOGW, AORNCC,
	"XOR",		LADDW, AXOR,	/* includes xoril & xoriu */
	"XORCC",	LLOGW, AXORCC,

	"EXTSB",	LABS,	AEXTSB,
	"EXTSBCC",	LABS,	AEXTSBCC,
	"EXTSH",	LABS, AEXTSH,
	"EXTSHCC",	LABS, AEXTSHCC,

	"CNTLZW",	LABS, ACNTLZW,
	"CNTLZWCC",	LABS, ACNTLZWCC,

	"RLWMI",	LRLWM, ARLWMI,
	"RLWMICC",	LRLWM, ARLWMICC,
	"RLWNM",	LRLWM, ARLWNM,
	"RLWNMCC", LRLWM, ARLWNMCC,

	"SLW",		LSHW, ASLW,
	"SLWCC",	LSHW, ASLWCC,
	"SRW",		LSHW, ASRW,
	"SRWCC",	LSHW, ASRWCC,
	"SRAW",		LSHW, ASRAW,
	"SRAWCC",	LSHW, ASRAWCC,

	"BR",		LBRA, ABR,
	"BC",		LBRA, ABC,
	"BCL",		LBRA, ABC,
	"BL",		LBRA, ABL,
	"BEQ",		LBRA, ABEQ,
	"BNE",		LBRA, ABNE,
	"BGT",		LBRA, ABGT,
	"BGE",		LBRA, ABGE,
	"BLT",		LBRA, ABLT,
	"BLE",		LBRA, ABLE,
	"BVC",		LBRA, ABVC,
	"BVS",		LBRA, ABVS,

	"CMP",		LCMP, ACMP,
	"CMPU",		LCMP, ACMPU,
	"CMPW",		LCMP, ACMPW,
	"CMPWU",	LCMP, ACMPWU,

	"DIVW",		LLOGW, ADIVW,
	"DIVWV",	LLOGW, ADIVWV,
	"DIVWCC",	LLOGW, ADIVWCC,
	"DIVWVCC",	LLOGW, ADIVWVCC,
	"DIVWU",	LLOGW, ADIVWU,
	"DIVWUV",	LLOGW, ADIVWUV,
	"DIVWUCC",	LLOGW, ADIVWUCC,
	"DIVWUVCC",	LLOGW, ADIVWUVCC,

	"FABS",		LFCONV,	AFABS,
	"FABSCC",	LFCONV,	AFABSCC,
	"FNEG",		LFCONV,	AFNEG,
	"FNEGCC",	LFCONV,	AFNEGCC,
	"FNABS",	LFCONV,	AFNABS,
	"FNABSCC",	LFCONV,	AFNABSCC,

	"FADD",		LFADD,	AFADD,
	"FADDCC",	LFADD,	AFADDCC,
	"FSUB",		LFADD,  AFSUB,
	"FSUBCC",	LFADD,	AFSUBCC,
	"FMUL",		LFADD,	AFMUL,
	"FMULCC",	LFADD,	AFMULCC,
	"FDIV",		LFADD,	AFDIV,
	"FDIVCC",	LFADD,	AFDIVCC,
	"FRSP",		LFCONV,	AFRSP,
	"FRSPCC",	LFCONV,	AFRSPCC,
	"FCTIW",	LFCONV,	AFCTIW,
	"FCTIWCC",	LFCONV,	AFCTIWCC,
	"FCTIWZ",	LFCONV,	AFCTIWZ,
	"FCTIWZCC",	LFCONV,	AFCTIWZCC,

	"FMADD",	LFMA, AFMADD,
	"FMADDCC",	LFMA, AFMADDCC,
	"FMSUB",	LFMA, AFMSUB,
	"FMSUBCC",	LFMA, AFMSUBCC,
	"FNMADD",	LFMA, AFNMADD,
	"FNMADDCC",	LFMA, AFNMADDCC,
	"FNMSUB",	LFMA, AFNMSUB,
	"FNMSUBCC",	LFMA, AFNMSUBCC,
	"FMADDS",	LFMA, AFMADDS,
	"FMADDSCC",	LFMA, AFMADDSCC,
	"FMSUBS",	LFMA, AFMSUBS,
	"FMSUBSCC",	LFMA, AFMSUBSCC,
	"FNMADDS",	LFMA, AFNMADDS,
	"FNMADDSCC",	LFMA, AFNMADDSCC,
	"FNMSUBS",	LFMA, AFNMSUBS,
	"FNMSUBSCC",	LFMA, AFNMSUBSCC,

	"FCMPU",	LFCMP, AFCMPU,
	"FCMPO",	LFCMP, AFCMPO,
	"MTFSB0",	LMTFSB, AMTFSB0,
	"MTFSB1",	LMTFSB,	AMTFSB1,

	"FMOVD",	LFMOV, AFMOVD,
	"FMOVS",	LFMOV, AFMOVS,
	"FMOVDCC",	LFCONV,	AFMOVDCC,	/* fmr. */

	"GLOBL",	LTEXT, AGLOBL,

	"MOVB",		LMOVB, AMOVB,
	"MOVBZ",	LMOVB, AMOVBZ,
	"MOVBU",	LMOVB, AMOVBU,
	"MOVBZU", LMOVB, AMOVBZU,
	"MOVH",		LMOVB, AMOVH,
	"MOVHZ",	LMOVB, AMOVHZ,
	"MOVHU",	LMOVB, AMOVHU,
	"MOVHZU", LMOVB, AMOVHZU,
	"MOVHBR", 	LXMV, AMOVHBR,
	"MOVWBR",	LXMV, AMOVWBR,
	"MOVW",		LMOVW, AMOVW,
	"MOVWU",	LMOVW, AMOVWU,
	"MOVMW",	LMOVMW, AMOVMW,
	"MOVFL",	LMOVW,	AMOVFL,

	"MULLW",	LADDW, AMULLW,		/* includes multiply immediate 10-139 */
	"MULLWV",	LLOGW, AMULLWV,
	"MULLWCC",	LLOGW, AMULLWCC,
	"MULLWVCC",	LLOGW, AMULLWVCC,

	"MULHW",	LLOGW, AMULHW,
	"MULHWCC",	LLOGW, AMULHWCC,
	"MULHWU",	LLOGW, AMULHWU,
	"MULHWUCC",	LLOGW, AMULHWUCC,

	"NEG",		LABS, ANEG,
	"NEGV",		LABS, ANEGV,
	"NEGCC",	LABS, ANEGCC,
	"NEGVCC",	LABS, ANEGVCC,

	"NOP",		LNOP, ANOP,	/* ori 0,0,0 */
	"SYSCALL",	LNOP, ASYSCALL,
	"UNDEF",	LNOP, AUNDEF,

	"RET",		LRETRN, ARETURN,
	"RETURN",	LRETRN, ARETURN,
	"RFI",		LRETRN,	ARFI,
	"RFCI",		LRETRN,	ARFCI,

	"DATA",		LDATA, ADATA,
	"END",		LEND, AEND,
	"TEXT",		LTEXT, ATEXT,

	/* 64-bit instructions */
	"CNTLZD",	LABS,	ACNTLZD,
	"CNTLZDCC",	LABS,	ACNTLZDCC,
	"DIVD",	LLOGW,	ADIVD,
	"DIVDCC",	LLOGW,	ADIVDCC,
	"DIVDVCC",	LLOGW,	ADIVDVCC,
	"DIVDV",	LLOGW,	ADIVDV,
	"DIVDU",	LLOGW,	ADIVDU,
	"DIVDUCC",	LLOGW,	ADIVDUCC,
	"DIVDUVCC",	LLOGW,	ADIVDUVCC,
	"DIVDUV",	LLOGW,	ADIVDUV,
	"EXTSW",	LABS, AEXTSW,
	"EXTSWCC",	LABS, AEXTSWCC,
	"FCTID",	LFCONV,	AFCTID,
	"FCTIDCC",	LFCONV,	AFCTIDCC,
	"FCTIDZ",	LFCONV,	AFCTIDZ,
	"FCTIDZCC",	LFCONV,	AFCTIDZCC,
	"FCFID",	LFCONV,	AFCFID,
	"FCFIDCC",	LFCONV,	AFCFIDCC,
	"LDAR", LXLD, ALDAR,
	"MOVD",	LMOVW,	AMOVD,
	"MOVDU",	LMOVW,	AMOVDU,
	"MOVWZ",	LMOVW,	AMOVWZ,
	"MOVWZU",	LMOVW,	AMOVWZU,
	"MULHD",	LLOGW,	AMULHD,
	"MULHDCC",	LLOGW,	AMULHDCC,
	"MULHDU",	LLOGW,	AMULHDU,
	"MULHDUCC",	LLOGW,	AMULHDUCC,
	"MULLD",	LADDW,	AMULLD,	/* includes multiply immediate? */
	"MULLDCC",	LLOGW,	AMULLDCC,
	"MULLDVCC",	LLOGW,	AMULLDVCC,
	"MULLDV",	LLOGW,	AMULLDV,
	"RFID",	LRETRN,	ARFID,
	"HRFID", LRETRN, AHRFID,
	"RLDMI",	LRLWM,	ARLDMI,
	"RLDMICC",	LRLWM,	ARLDMICC,
	"RLDC",	LRLWM,	ARLDC,
	"RLDCCC",	LRLWM,	ARLDCCC,
	"RLDCR",	LRLWM,	ARLDCR,
	"RLDCRCC",	LRLWM,	ARLDCRCC,
	"RLDCL",	LRLWM,	ARLDCL,
	"RLDCLCC",	LRLWM,	ARLDCLCC,
	"SLBIA",	LNOP,	ASLBIA,
	"SLBIE",	LNOP,	ASLBIE,
	"SLBMFEE",	LABS,	ASLBMFEE,
	"SLBMFEV",	LABS,	ASLBMFEV,
	"SLBMTE",	LABS,	ASLBMTE,
	"SLD",	LSHW,	ASLD,
	"SLDCC",	LSHW,	ASLDCC,
	"SRD",	LSHW,	ASRD,
	"SRAD",	LSHW,	ASRAD,
	"SRADCC",	LSHW,	ASRADCC,
	"SRDCC",	LSHW,	ASRDCC,
	"STDCCC",	LXST,	ASTDCCC,
	"TD",	LADDW,	ATD,

	/* pseudo instructions */
	"REM",	LLOGW,	AREM,
	"REMCC",	LLOGW,	AREMCC,
	"REMV",	LLOGW,	AREMV,
	"REMVCC",	LLOGW,	AREMVCC,
	"REMU",	LLOGW,	AREMU,
	"REMUCC",	LLOGW,	AREMUCC,
	"REMUV",	LLOGW,	AREMUV,
	"REMUVCC",	LLOGW,	AREMUVCC,
	"REMD",	LLOGW,	AREMD,
	"REMDCC",	LLOGW,	AREMDCC,
	"REMDV",	LLOGW,	AREMDV,
	"REMDVCC",	LLOGW,	AREMDVCC,
	"REMDU",	LLOGW,	AREMDU,
	"REMDUCC",	LLOGW,	AREMDUCC,
	"REMDUV",	LLOGW,	AREMDUV,
	"REMDUVCC",	LLOGW,	AREMDUVCC,

/* special instructions */
	"DCBF",		LXOP,	ADCBF,
	"DCBI",		LXOP,	ADCBI,
	"DCBST",	LXOP,	ADCBST,
	"DCBT",		LXOP,	ADCBT,
	"DCBTST",	LXOP,	ADCBTST,
	"DCBZ",		LXOP,	ADCBZ,
	"ICBI",		LXOP,	AICBI,

	"ECIWX",	LXLD,	AECIWX,
	"ECOWX",	LXST,	AECOWX,
	"LWAR", LXLD, ALWAR,
	"LWAR", LXLD, ALWAR,
	"STWCCC", LXST, ASTWCCC,
	"EIEIO",	LRETRN,	AEIEIO,
	"TLBIE",	LNOP,	ATLBIE,
	"TLBIEL",	LNOP,	ATLBIEL,
	"LSW",	LXLD, ALSW,
	"STSW",	LXST, ASTSW,
	
	"ISYNC",	LRETRN, AISYNC,
	"SYNC",		LRETRN, ASYNC,
	"TLBSYNC",	LRETRN,	ATLBSYNC,
	"PTESYNC",	LRETRN,	APTESYNC,
/*	"TW",		LADDW,	ATW,*/

	"WORD",		LWORD, AWORD,
	"DWORD",	LWORD, ADWORD,
	"SCHED",	LSCHED, 0,
	"NOSCHED",	LSCHED,	0x80,

	"PCDATA",	LPCDAT,	APCDATA,
	"FUNCDATA",	LFUNCDAT,	AFUNCDATA,

	0
};

void
cinit(void)
{
	Sym *s;
	int i;

	nullgen.type = D_NONE;
	nullgen.name = D_NONE;
	nullgen.reg = NREG;
	nullgen.scale = NREG; // replaced Gen.xreg with Prog.scale

	nerrors = 0;
	iostack = I;
	iofree = I;
	peekc = IGN;
	nhunk = 0;
	for(i=0; i<NHASH; i++)
		hash[i] = S;
	for(i=0; itab[i].name; i++) {
		s = slookup(itab[i].name);
		s->type = itab[i].type;
		s->value = itab[i].value;
	}
}

void
syminit(Sym *s)
{

	s->type = LNAME;
	s->value = 0;
}

void
cclean(void)
{

	outcode(AEND, &nullgen, NREG, &nullgen);
}

static Prog *lastpc;

void
outcode(int a, Addr *g1, int reg, Addr *g2)
{
	Prog *p;
	Plist *pl;

	if(pass == 1)
		goto out;

	if(g1->scale != NREG) {
		if(reg != NREG || g2->scale != NREG)
			yyerror("bad addressing modes");
		reg = g1->scale;
	} else
	if(g2->scale != NREG) {
		if(reg != NREG)
			yyerror("bad addressing modes");
		reg = g2->scale;
	}

	p = ctxt->arch->prg();
	p->as = a;
	p->lineno = lineno;
	if(nosched)
		p->mark |= NOSCHED;
	p->from = *g1;
	p->reg = reg;
	p->to = *g2;
	p->pc = pc;

	if(lastpc == nil) {
		pl = linknewplist(ctxt);
		pl->firstpc = p;
	} else
		lastpc->link = p;
	lastpc = p;
out:
	if(a != AGLOBL && a != ADATA)
		pc++;
}

void
outgcode(int a, Addr *g1, int reg, Addr *g2, Addr *g3)
{
	Prog *p;
	Plist *pl;

	if(pass == 1)
		goto out;

	p = ctxt->arch->prg();
	p->as = a;
	p->lineno = lineno;
	if(nosched)
		p->mark |= NOSCHED;
	p->from = *g1;
	p->reg = reg;
	p->from3 = *g2;
	p->to = *g3;
	p->pc = pc;

	if(lastpc == nil) {
		pl = linknewplist(ctxt);
		pl->firstpc = p;
	} else
		lastpc->link = p;
	lastpc = p;
out:
	if(a != AGLOBL && a != ADATA)
		pc++;
}

#include "../cc/lexbody"
#include "../cc/macbody"
