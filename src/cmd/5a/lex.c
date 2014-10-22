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

	thechar = '5';
	thestring = "arm";

	ctxt = linknew(&linkarm);
	ctxt->diag = yyerror;
	ctxt->bso = &bstdout;
	ctxt->enforce_data_order = 1;
	Binit(&bstdout, 1, OWRITE);
	listinit5();
	fmtinstall('L', Lconv);

	// Allow GOARCH=thestring or GOARCH=thestringsuffix,
	// but not other values.	
	p = getgoarch();
	if(strncmp(p, thestring, strlen(thestring)) != 0)
		sysfatal("cannot use %cc with GOARCH=%s", thechar, p);

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
	exits(0);
}

int
assemble(char *file)
{
	char *ofile, *p;
	int i, of;

	ofile = alloc(strlen(file)+3); // +3 for .x\0 (x=thechar)
	strcpy(ofile, file);
	p = utfrrune(ofile, '/');
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
	"g",		LREG,	10, // avoid unintentionally clobber g using R10
	"R11",		LREG,	11,
	"R12",		LREG,	12,
	"R13",		LREG,	13,
	"R14",		LREG,	14,
	"R15",		LREG,	15,

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

	"C",		LC,	0,

	"C0",		LCREG,	0,
	"C1",		LCREG,	1,
	"C2",		LCREG,	2,
	"C3",		LCREG,	3,
	"C4",		LCREG,	4,
	"C5",		LCREG,	5,
	"C6",		LCREG,	6,
	"C7",		LCREG,	7,
	"C8",		LCREG,	8,
	"C9",		LCREG,	9,
	"C10",		LCREG,	10,
	"C11",		LCREG,	11,
	"C12",		LCREG,	12,
	"C13",		LCREG,	13,
	"C14",		LCREG,	14,
	"C15",		LCREG,	15,

	"CPSR",		LPSR,	0,
	"SPSR",		LPSR,	1,

	"FPSR",		LFCR,	0,
	"FPCR",		LFCR,	1,

	".EQ",		LCOND,	0,
	".NE",		LCOND,	1,
	".CS",		LCOND,	2,
	".HS",		LCOND,	2,
	".CC",		LCOND,	3,
	".LO",		LCOND,	3,
	".MI",		LCOND,	4,
	".PL",		LCOND,	5,
	".VS",		LCOND,	6,
	".VC",		LCOND,	7,
	".HI",		LCOND,	8,
	".LS",		LCOND,	9,
	".GE",		LCOND,	10,
	".LT",		LCOND,	11,
	".GT",		LCOND,	12,
	".LE",		LCOND,	13,
	".AL",		LCOND,	Always,

	".U",		LS,	C_UBIT,
	".S",		LS,	C_SBIT,
	".W",		LS,	C_WBIT,
	".P",		LS,	C_PBIT,
	".PW",		LS,	C_WBIT|C_PBIT,
	".WP",		LS,	C_WBIT|C_PBIT,

	".F",		LS,	C_FBIT,

	".IBW",		LS,	C_WBIT|C_PBIT|C_UBIT,
	".IAW",		LS,	C_WBIT|C_UBIT,
	".DBW",		LS,	C_WBIT|C_PBIT,
	".DAW",		LS,	C_WBIT,
	".IB",		LS,	C_PBIT|C_UBIT,
	".IA",		LS,	C_UBIT,
	".DB",		LS,	C_PBIT,
	".DA",		LS,	0,

	"@",		LAT,	0,

	"AND",		LTYPE1,	AAND,
	"EOR",		LTYPE1,	AEOR,
	"SUB",		LTYPE1,	ASUB,
	"RSB",		LTYPE1,	ARSB,
	"ADD",		LTYPE1,	AADD,
	"ADC",		LTYPE1,	AADC,
	"SBC",		LTYPE1,	ASBC,
	"RSC",		LTYPE1,	ARSC,
	"ORR",		LTYPE1,	AORR,
	"BIC",		LTYPE1,	ABIC,

	"SLL",		LTYPE1,	ASLL,
	"SRL",		LTYPE1,	ASRL,
	"SRA",		LTYPE1,	ASRA,

	"MUL",		LTYPE1, AMUL,
	"MULA",		LTYPEN, AMULA,
	"DIV",		LTYPE1,	ADIV,
	"MOD",		LTYPE1,	AMOD,

	"MULL",		LTYPEM, AMULL,
	"MULAL",	LTYPEM, AMULAL,
	"MULLU",	LTYPEM, AMULLU,
	"MULALU",	LTYPEM, AMULALU,

	"MVN",		LTYPE2, AMVN,	/* op2 ignored */

	"MOVB",		LTYPE3, AMOVB,
	"MOVBU",	LTYPE3, AMOVBU,
	"MOVH",		LTYPE3, AMOVH,
	"MOVHU",	LTYPE3, AMOVHU,
	"MOVW",		LTYPE3, AMOVW,

	"MOVD",		LTYPE3, AMOVD,
	"MOVDF",		LTYPE3, AMOVDF,
	"MOVDW",	LTYPE3, AMOVDW,
	"MOVF",		LTYPE3, AMOVF,
	"MOVFD",		LTYPE3, AMOVFD,
	"MOVFW",		LTYPE3, AMOVFW,
	"MOVWD",	LTYPE3, AMOVWD,
	"MOVWF",		LTYPE3, AMOVWF,

	"LDREX",		LTYPE3, ALDREX,
	"LDREXD",		LTYPE3, ALDREXD,
	"STREX",		LTYPE9, ASTREX,
	"STREXD",		LTYPE9, ASTREXD,

/*
	"NEGF",		LTYPEI, ANEGF,
	"NEGD",		LTYPEI, ANEGD,
	"SQTF",		LTYPEI,	ASQTF,
	"SQTD",		LTYPEI,	ASQTD,
	"RNDF",		LTYPEI,	ARNDF,
	"RNDD",		LTYPEI,	ARNDD,
	"URDF",		LTYPEI,	AURDF,
	"URDD",		LTYPEI,	AURDD,
	"NRMF",		LTYPEI,	ANRMF,
	"NRMD",		LTYPEI,	ANRMD,
*/

	"ABSF",		LTYPEI, AABSF,
	"ABSD",		LTYPEI, AABSD,
	"SQRTF",	LTYPEI, ASQRTF,
	"SQRTD",	LTYPEI, ASQRTD,
	"CMPF",		LTYPEL, ACMPF,
	"CMPD",		LTYPEL, ACMPD,
	"ADDF",		LTYPEK,	AADDF,
	"ADDD",		LTYPEK,	AADDD,
	"SUBF",		LTYPEK,	ASUBF,
	"SUBD",		LTYPEK,	ASUBD,
	"MULF",		LTYPEK,	AMULF,
	"MULD",		LTYPEK,	AMULD,
	"DIVF",		LTYPEK,	ADIVF,
	"DIVD",		LTYPEK,	ADIVD,

	"B",		LTYPE4, AB,
	"BL",		LTYPE4, ABL,
	"BX",		LTYPEBX,	ABX,

	"BEQ",		LTYPE5,	ABEQ,
	"BNE",		LTYPE5,	ABNE,
	"BCS",		LTYPE5,	ABCS,
	"BHS",		LTYPE5,	ABHS,
	"BCC",		LTYPE5,	ABCC,
	"BLO",		LTYPE5,	ABLO,
	"BMI",		LTYPE5,	ABMI,
	"BPL",		LTYPE5,	ABPL,
	"BVS",		LTYPE5,	ABVS,
	"BVC",		LTYPE5,	ABVC,
	"BHI",		LTYPE5,	ABHI,
	"BLS",		LTYPE5,	ABLS,
	"BGE",		LTYPE5,	ABGE,
	"BLT",		LTYPE5,	ABLT,
	"BGT",		LTYPE5,	ABGT,
	"BLE",		LTYPE5,	ABLE,
	"BCASE",	LTYPE5,	ABCASE,

	"SWI",		LTYPE6, ASWI,

	"CMP",		LTYPE7,	ACMP,
	"TST",		LTYPE7,	ATST,
	"TEQ",		LTYPE7,	ATEQ,
	"CMN",		LTYPE7,	ACMN,

	"MOVM",		LTYPE8, AMOVM,

	"SWPBU",	LTYPE9, ASWPBU,
	"SWPW",		LTYPE9, ASWPW,

	"RET",		LTYPEA, ARET,
	"RFE",		LTYPEA, ARFE,

	"TEXT",		LTYPEB, ATEXT,
	"GLOBL",	LTYPEB, AGLOBL,
	"DATA",		LTYPEC, ADATA,
	"CASE",		LTYPED, ACASE,
	"END",		LTYPEE, AEND,
	"WORD",		LTYPEH, AWORD,
	"NOP",		LTYPEI, ANOP,

	"MCR",		LTYPEJ, 0,
	"MRC",		LTYPEJ, 1,

	"PLD",		LTYPEPLD, APLD,
	"UNDEF",	LTYPEE,	AUNDEF,
	"CLZ",		LTYPE2, ACLZ,

	"MULWT",	LTYPE1, AMULWT,
	"MULWB",	LTYPE1, AMULWB,
	"MULAWT",	LTYPEN, AMULAWT,
	"MULAWB",	LTYPEN, AMULAWB,

	"USEFIELD",	LTYPEN, AUSEFIELD,
	"PCDATA",	LTYPEPC,	APCDATA,
	"FUNCDATA",	LTYPEF,	AFUNCDATA,

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

int
isreg(Addr *g)
{

	USED(g);
	return 1;
}

void
cclean(void)
{
	outcode(AEND, Always, &nullgen, NREG, &nullgen);
}

static int bcode[] =
{
	ABEQ,
	ABNE,
	ABCS,
	ABCC,
	ABMI,
	ABPL,
	ABVS,
	ABVC,
	ABHI,
	ABLS,
	ABGE,
	ABLT,
	ABGT,
	ABLE,
	AB,
	ANOP,
};

static Prog *lastpc;

void
outcode(int a, int scond, Addr *g1, int reg, Addr *g2)
{
	Prog *p;
	Plist *pl;

	/* hack to make B.NE etc. work: turn it into the corresponding conditional */
	if(a == AB){
		a = bcode[scond&0xf];
		scond = (scond & ~0xf) | Always;
	}

	if(pass == 1)
		goto out;
	
	p = malloc(sizeof *p);
	memset(p, 0, sizeof *p);
	p->as = a;
	p->lineno = stmtline;
	p->scond = scond;
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

#include "../cc/lexbody"
#include "../cc/macbody"
