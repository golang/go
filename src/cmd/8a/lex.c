// Inferno utils/8a/lex.c
// http://code.google.com/p/inferno-os/source/browse/utils/8a/lex.c
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

	thechar = '8';
	thestring = "386";

	ctxt = linknew(&link386);
	ctxt->diag = yyerror;
	ctxt->bso = &bstdout;
	ctxt->enforce_data_order = 1;
	Binit(&bstdout, 1, OWRITE);
	listinit8();
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

	"AL",		LBREG,	D_AL,
	"CL",		LBREG,	D_CL,
	"DL",		LBREG,	D_DL,
	"BL",		LBREG,	D_BL,
	"AH",		LBREG,	D_AH,
	"CH",		LBREG,	D_CH,
	"DH",		LBREG,	D_DH,
	"BH",		LBREG,	D_BH,

	"AX",		LLREG,	D_AX,
	"CX",		LLREG,	D_CX,
	"DX",		LLREG,	D_DX,
	"BX",		LLREG,	D_BX,
/*	"SP",		LLREG,	D_SP,	*/
	"BP",		LLREG,	D_BP,
	"SI",		LLREG,	D_SI,
	"DI",		LLREG,	D_DI,

	"F0",		LFREG,	D_F0+0,
	"F1",		LFREG,	D_F0+1,
	"F2",		LFREG,	D_F0+2,
	"F3",		LFREG,	D_F0+3,
	"F4",		LFREG,	D_F0+4,
	"F5",		LFREG,	D_F0+5,
	"F6",		LFREG,	D_F0+6,
	"F7",		LFREG,	D_F0+7,

	"X0",		LXREG,	D_X0+0,
	"X1",		LXREG,	D_X0+1,
	"X2",		LXREG,	D_X0+2,
	"X3",		LXREG,	D_X0+3,
	"X4",		LXREG,	D_X0+4,
	"X5",		LXREG,	D_X0+5,
	"X6",		LXREG,	D_X0+6,
	"X7",		LXREG,	D_X0+7,

	"CS",		LSREG,	D_CS,
	"SS",		LSREG,	D_SS,
	"DS",		LSREG,	D_DS,
	"ES",		LSREG,	D_ES,
	"FS",		LSREG,	D_FS,
	"GS",		LSREG,	D_GS,
	"TLS",		LSREG,	D_TLS,

	"GDTR",		LBREG,	D_GDTR,
	"IDTR",		LBREG,	D_IDTR,
	"LDTR",		LBREG,	D_LDTR,
	"MSW",		LBREG,	D_MSW,
	"TASK",		LBREG,	D_TASK,

	"CR0",		LBREG,	D_CR+0,
	"CR1",		LBREG,	D_CR+1,
	"CR2",		LBREG,	D_CR+2,
	"CR3",		LBREG,	D_CR+3,
	"CR4",		LBREG,	D_CR+4,
	"CR5",		LBREG,	D_CR+5,
	"CR6",		LBREG,	D_CR+6,
	"CR7",		LBREG,	D_CR+7,

	"DR0",		LBREG,	D_DR+0,
	"DR1",		LBREG,	D_DR+1,
	"DR2",		LBREG,	D_DR+2,
	"DR3",		LBREG,	D_DR+3,
	"DR4",		LBREG,	D_DR+4,
	"DR5",		LBREG,	D_DR+5,
	"DR6",		LBREG,	D_DR+6,
	"DR7",		LBREG,	D_DR+7,

	"TR0",		LBREG,	D_TR+0,
	"TR1",		LBREG,	D_TR+1,
	"TR2",		LBREG,	D_TR+2,
	"TR3",		LBREG,	D_TR+3,
	"TR4",		LBREG,	D_TR+4,
	"TR5",		LBREG,	D_TR+5,
	"TR6",		LBREG,	D_TR+6,
	"TR7",		LBREG,	D_TR+7,

	"AAA",		LTYPE0,	AAAA,
	"AAD",		LTYPE0,	AAAD,
	"AAM",		LTYPE0,	AAAM,
	"AAS",		LTYPE0,	AAAS,
	"ADCB",		LTYPE3,	AADCB,
	"ADCL",		LTYPE3,	AADCL,
	"ADCW",		LTYPE3,	AADCW,
	"ADDB",		LTYPE3,	AADDB,
	"ADDL",		LTYPE3,	AADDL,
	"ADDW",		LTYPE3,	AADDW,
	"ADJSP",	LTYPE2,	AADJSP,
	"ANDB",		LTYPE3,	AANDB,
	"ANDL",		LTYPE3,	AANDL,
	"ANDW",		LTYPE3,	AANDW,
	"ARPL",		LTYPE3,	AARPL,
	"BOUNDL",	LTYPE3,	ABOUNDL,
	"BOUNDW",	LTYPE3,	ABOUNDW,
	"BSFL",		LTYPE3,	ABSFL,
	"BSFW",		LTYPE3,	ABSFW,
	"BSRL",		LTYPE3,	ABSRL,
	"BSRW",		LTYPE3,	ABSRW,
	"BSWAPL",	LTYPE1,	ABSWAPL,
	"BTCL",		LTYPE3,	ABTCL,
	"BTCW",		LTYPE3,	ABTCW,
	"BTL",		LTYPE3,	ABTL,
	"BTRL",		LTYPE3,	ABTRL,
	"BTRW",		LTYPE3,	ABTRW,
	"BTSL",		LTYPE3,	ABTSL,
	"BTSW",		LTYPE3,	ABTSW,
	"BTW",		LTYPE3,	ABTW,
	"BYTE",		LTYPE2,	ABYTE,
	"CALL",		LTYPEC,	ACALL,
	"CLC",		LTYPE0,	ACLC,
	"CLD",		LTYPE0,	ACLD,
	"CLI",		LTYPE0,	ACLI,
	"CLTS",		LTYPE0,	ACLTS,
	"CMC",		LTYPE0,	ACMC,
	"CMPB",		LTYPE4,	ACMPB,
	"CMPL",		LTYPE4,	ACMPL,
	"CMPW",		LTYPE4,	ACMPW,
	"CMPSB",	LTYPE0,	ACMPSB,
	"CMPSL",	LTYPE0,	ACMPSL,
	"CMPSW",	LTYPE0,	ACMPSW,
	"CMPXCHG8B",	LTYPE1,	ACMPXCHG8B,
	"CMPXCHGB",	LTYPE3,	ACMPXCHGB,
	"CMPXCHGL",	LTYPE3,	ACMPXCHGL,
	"CMPXCHGW",	LTYPE3,	ACMPXCHGW,
	"CPUID",	LTYPE0,	ACPUID,
	"DAA",		LTYPE0,	ADAA,
	"DAS",		LTYPE0,	ADAS,
	"DATA",		LTYPED,	ADATA,
	"DECB",		LTYPE1,	ADECB,
	"DECL",		LTYPE1,	ADECL,
	"DECW",		LTYPE1,	ADECW,
	"DIVB",		LTYPE2,	ADIVB,
	"DIVL",		LTYPE2,	ADIVL,
	"DIVW",		LTYPE2,	ADIVW,
	"END",		LTYPE0,	AEND,
	"ENTER",	LTYPE2,	AENTER,
	"GLOBL",	LTYPEG,	AGLOBL,
	"HLT",		LTYPE0,	AHLT,
	"IDIVB",	LTYPE2,	AIDIVB,
	"IDIVL",	LTYPE2,	AIDIVL,
	"IDIVW",	LTYPE2,	AIDIVW,
	"IMULB",	LTYPE2,	AIMULB,
	"IMULL",	LTYPEI,	AIMULL,
	"IMULW",	LTYPEI,	AIMULW,
	"INB",		LTYPE0,	AINB,
	"INL",		LTYPE0,	AINL,
	"INW",		LTYPE0,	AINW,
	"INCB",		LTYPE1,	AINCB,
	"INCL",		LTYPE1,	AINCL,
	"INCW",		LTYPE1,	AINCW,
	"INSB",		LTYPE0,	AINSB,
	"INSL",		LTYPE0,	AINSL,
	"INSW",		LTYPE0,	AINSW,
	"INT",		LTYPE2,	AINT,
	"INTO",		LTYPE0,	AINTO,
	"IRETL",	LTYPE0,	AIRETL,
	"IRETW",	LTYPE0,	AIRETW,

	"JOS",		LTYPER,	AJOS,	/* overflow set (OF = 1) */
	"JO",		LTYPER,	AJOS,	/* alternate */
	"JOC",		LTYPER,	AJOC,	/* overflow clear (OF = 0) */
	"JNO",		LTYPER,	AJOC,	/* alternate */
	"JCS",		LTYPER,	AJCS,	/* carry set (CF = 1) */
	"JB",		LTYPER,	AJCS,	/* alternate */
	"JC",		LTYPER,	AJCS,	/* alternate */
	"JNAE",		LTYPER,	AJCS,	/* alternate */
	"JLO",		LTYPER,	AJCS,	/* alternate */
	"JCC",		LTYPER,	AJCC,	/* carry clear (CF = 0) */
	"JAE",		LTYPER,	AJCC,	/* alternate */
	"JNB",		LTYPER,	AJCC,	/* alternate */
	"JNC",		LTYPER,	AJCC,	/* alternate */
	"JHS",		LTYPER,	AJCC,	/* alternate */
	"JEQ",		LTYPER,	AJEQ,	/* equal (ZF = 1) */
	"JE",		LTYPER,	AJEQ,	/* alternate */
	"JZ",		LTYPER,	AJEQ,	/* alternate */
	"JNE",		LTYPER,	AJNE,	/* not equal (ZF = 0) */
	"JNZ",		LTYPER,	AJNE,	/* alternate */
	"JLS",		LTYPER,	AJLS,	/* lower or same (unsigned) (CF = 1 || ZF = 1) */
	"JBE",		LTYPER,	AJLS,	/* alternate */
	"JNA",		LTYPER,	AJLS,	/* alternate */
	"JHI",		LTYPER,	AJHI,	/* higher (unsigned) (CF = 0 && ZF = 0) */
	"JA",		LTYPER,	AJHI,	/* alternate */
	"JNBE",		LTYPER,	AJHI,	/* alternate */
	"JMI",		LTYPER,	AJMI,	/* negative (minus) (SF = 1) */
	"JS",		LTYPER,	AJMI,	/* alternate */
	"JPL",		LTYPER,	AJPL,	/* non-negative (plus) (SF = 0) */
	"JNS",		LTYPER,	AJPL,	/* alternate */
	"JPS",		LTYPER,	AJPS,	/* parity set (PF = 1) */
	"JP",		LTYPER,	AJPS,	/* alternate */
	"JPE",		LTYPER,	AJPS,	/* alternate */
	"JPC",		LTYPER,	AJPC,	/* parity clear (PF = 0) */
	"JNP",		LTYPER,	AJPC,	/* alternate */
	"JPO",		LTYPER,	AJPC,	/* alternate */
	"JLT",		LTYPER,	AJLT,	/* less than (signed) (SF != OF) */
	"JL",		LTYPER,	AJLT,	/* alternate */
	"JNGE",		LTYPER,	AJLT,	/* alternate */
	"JGE",		LTYPER,	AJGE,	/* greater than or equal (signed) (SF = OF) */
	"JNL",		LTYPER,	AJGE,	/* alternate */
	"JLE",		LTYPER,	AJLE,	/* less than or equal (signed) (ZF = 1 || SF != OF) */
	"JNG",		LTYPER,	AJLE,	/* alternate */
	"JGT",		LTYPER,	AJGT,	/* greater than (signed) (ZF = 0 && SF = OF) */
	"JG",		LTYPER,	AJGT,	/* alternate */
	"JNLE",		LTYPER,	AJGT,	/* alternate */

	"JCXZL",	LTYPER,	AJCXZL,
	"JCXZW",	LTYPER,	AJCXZW,
	"JMP",		LTYPEC,	AJMP,
	"LAHF",		LTYPE0,	ALAHF,
	"LARL",		LTYPE3,	ALARL,
	"LARW",		LTYPE3,	ALARW,
	"LEAL",		LTYPE3,	ALEAL,
	"LEAW",		LTYPE3,	ALEAW,
	"LEAVEL",	LTYPE0,	ALEAVEL,
	"LEAVEW",	LTYPE0,	ALEAVEW,
	"LOCK",		LTYPE0,	ALOCK,
	"LODSB",	LTYPE0,	ALODSB,
	"LODSL",	LTYPE0,	ALODSL,
	"LODSW",	LTYPE0,	ALODSW,
	"LONG",		LTYPE2,	ALONG,
	"LOOP",		LTYPER,	ALOOP,
	"LOOPEQ",	LTYPER,	ALOOPEQ,
	"LOOPNE",	LTYPER,	ALOOPNE,
	"LSLL",		LTYPE3,	ALSLL,
	"LSLW",		LTYPE3,	ALSLW,
	"MOVB",		LTYPE3,	AMOVB,
	"MOVL",		LTYPEM,	AMOVL,
	"MOVW",		LTYPEM,	AMOVW,
	"MOVQ",		LTYPEM, AMOVQ,
	"MOVBLSX",	LTYPE3, AMOVBLSX,
	"MOVBLZX",	LTYPE3, AMOVBLZX,
	"MOVBWSX",	LTYPE3, AMOVBWSX,
	"MOVBWZX",	LTYPE3, AMOVBWZX,
	"MOVWLSX",	LTYPE3, AMOVWLSX,
	"MOVWLZX",	LTYPE3, AMOVWLZX,
	"MOVSB",	LTYPE0,	AMOVSB,
	"MOVSL",	LTYPE0,	AMOVSL,
	"MOVSW",	LTYPE0,	AMOVSW,
	"MULB",		LTYPE2,	AMULB,
	"MULL",		LTYPE2,	AMULL,
	"MULW",		LTYPE2,	AMULW,
	"NEGB",		LTYPE1,	ANEGB,
	"NEGL",		LTYPE1,	ANEGL,
	"NEGW",		LTYPE1,	ANEGW,
	"NOP",		LTYPEN,	ANOP,
	"NOTB",		LTYPE1,	ANOTB,
	"NOTL",		LTYPE1,	ANOTL,
	"NOTW",		LTYPE1,	ANOTW,
	"ORB",		LTYPE3,	AORB,
	"ORL",		LTYPE3,	AORL,
	"ORW",		LTYPE3,	AORW,
	"OUTB",		LTYPE0,	AOUTB,
	"OUTL",		LTYPE0,	AOUTL,
	"OUTW",		LTYPE0,	AOUTW,
	"OUTSB",	LTYPE0,	AOUTSB,
	"OUTSL",	LTYPE0,	AOUTSL,
	"OUTSW",	LTYPE0,	AOUTSW,
	"PAUSE",	LTYPEN,	APAUSE,
	"PINSRD",	LTYPEX,	APINSRD,
	"POPAL",	LTYPE0,	APOPAL,
	"POPAW",	LTYPE0,	APOPAW,
	"POPFL",	LTYPE0,	APOPFL,
	"POPFW",	LTYPE0,	APOPFW,
	"POPL",		LTYPE1,	APOPL,
	"POPW",		LTYPE1,	APOPW,
	"PUSHAL",	LTYPE0,	APUSHAL,
	"PUSHAW",	LTYPE0,	APUSHAW,
	"PUSHFL",	LTYPE0,	APUSHFL,
	"PUSHFW",	LTYPE0,	APUSHFW,
	"PUSHL",	LTYPE2,	APUSHL,
	"PUSHW",	LTYPE2,	APUSHW,
	"RCLB",		LTYPE3,	ARCLB,
	"RCLL",		LTYPE3,	ARCLL,
	"RCLW",		LTYPE3,	ARCLW,
	"RCRB",		LTYPE3,	ARCRB,
	"RCRL",		LTYPE3,	ARCRL,
	"RCRW",		LTYPE3,	ARCRW,
	"RDTSC",	LTYPE0,	ARDTSC,
	"REP",		LTYPE0,	AREP,
	"REPN",		LTYPE0,	AREPN,
	"RET",		LTYPE0,	ARET,
	"ROLB",		LTYPE3,	AROLB,
	"ROLL",		LTYPE3,	AROLL,
	"ROLW",		LTYPE3,	AROLW,
	"RORB",		LTYPE3,	ARORB,
	"RORL",		LTYPE3,	ARORL,
	"RORW",		LTYPE3,	ARORW,
	"SAHF",		LTYPE0,	ASAHF,
	"SALB",		LTYPE3,	ASALB,
	"SALL",		LTYPE3,	ASALL,
	"SALW",		LTYPE3,	ASALW,
	"SARB",		LTYPE3,	ASARB,
	"SARL",		LTYPE3,	ASARL,
	"SARW",		LTYPE3,	ASARW,
	"SBBB",		LTYPE3,	ASBBB,
	"SBBL",		LTYPE3,	ASBBL,
	"SBBW",		LTYPE3,	ASBBW,
	"SCASB",	LTYPE0,	ASCASB,
	"SCASL",	LTYPE0,	ASCASL,
	"SCASW",	LTYPE0,	ASCASW,
	"SETCC",	LTYPE1,	ASETCC,	/* see JCC etc above for condition codes */
	"SETCS",	LTYPE1,	ASETCS,
	"SETEQ",	LTYPE1,	ASETEQ,
	"SETGE",	LTYPE1,	ASETGE,
	"SETGT",	LTYPE1,	ASETGT,
	"SETHI",	LTYPE1,	ASETHI,
	"SETLE",	LTYPE1,	ASETLE,
	"SETLS",	LTYPE1,	ASETLS,
	"SETLT",	LTYPE1,	ASETLT,
	"SETMI",	LTYPE1,	ASETMI,
	"SETNE",	LTYPE1,	ASETNE,
	"SETOC",	LTYPE1,	ASETOC,
	"SETOS",	LTYPE1,	ASETOS,
	"SETPC",	LTYPE1,	ASETPC,
	"SETPL",	LTYPE1,	ASETPL,
	"SETPS",	LTYPE1,	ASETPS,
	"CDQ",		LTYPE0,	ACDQ,
	"CWD",		LTYPE0,	ACWD,
	"SHLB",		LTYPE3,	ASHLB,
	"SHLL",		LTYPES,	ASHLL,
	"SHLW",		LTYPES,	ASHLW,
	"SHRB",		LTYPE3,	ASHRB,
	"SHRL",		LTYPES,	ASHRL,
	"SHRW",		LTYPES,	ASHRW,
	"STC",		LTYPE0,	ASTC,
	"STD",		LTYPE0,	ASTD,
	"STI",		LTYPE0,	ASTI,
	"STOSB",	LTYPE0,	ASTOSB,
	"STOSL",	LTYPE0,	ASTOSL,
	"STOSW",	LTYPE0,	ASTOSW,
	"SUBB",		LTYPE3,	ASUBB,
	"SUBL",		LTYPE3,	ASUBL,
	"SUBW",		LTYPE3,	ASUBW,
	"SYSCALL",	LTYPE0,	ASYSCALL,
	"TESTB",	LTYPE3,	ATESTB,
	"TESTL",	LTYPE3,	ATESTL,
	"TESTW",	LTYPE3,	ATESTW,
	"TEXT",		LTYPET,	ATEXT,
	"VERR",		LTYPE2,	AVERR,
	"VERW",		LTYPE2,	AVERW,
	"WAIT",		LTYPE0,	AWAIT,
	"WORD",		LTYPE2,	AWORD,
	"XADDB",	LTYPE3,	AXADDB,
	"XADDL",	LTYPE3,	AXADDL,
	"XADDW",	LTYPE3,	AXADDW,
	"XCHGB",	LTYPE3,	AXCHGB,
	"XCHGL",	LTYPE3,	AXCHGL,
	"XCHGW",	LTYPE3,	AXCHGW,
	"XLAT",		LTYPE2,	AXLAT,
	"XORB",		LTYPE3,	AXORB,
	"XORL",		LTYPE3,	AXORL,
	"XORW",		LTYPE3,	AXORW,

	"CMOVLCC",	LTYPE3,	ACMOVLCC,
	"CMOVLCS",	LTYPE3,	ACMOVLCS,
	"CMOVLEQ",	LTYPE3,	ACMOVLEQ,
	"CMOVLGE",	LTYPE3,	ACMOVLGE,
	"CMOVLGT",	LTYPE3,	ACMOVLGT,
	"CMOVLHI",	LTYPE3,	ACMOVLHI,
	"CMOVLLE",	LTYPE3,	ACMOVLLE,
	"CMOVLLS",	LTYPE3,	ACMOVLLS,
	"CMOVLLT",	LTYPE3,	ACMOVLLT,
	"CMOVLMI",	LTYPE3,	ACMOVLMI,
	"CMOVLNE",	LTYPE3,	ACMOVLNE,
	"CMOVLOC",	LTYPE3,	ACMOVLOC,
	"CMOVLOS",	LTYPE3,	ACMOVLOS,
	"CMOVLPC",	LTYPE3,	ACMOVLPC,
	"CMOVLPL",	LTYPE3,	ACMOVLPL,
	"CMOVLPS",	LTYPE3,	ACMOVLPS,
	"CMOVWCC",	LTYPE3,	ACMOVWCC,
	"CMOVWCS",	LTYPE3,	ACMOVWCS,
	"CMOVWEQ",	LTYPE3,	ACMOVWEQ,
	"CMOVWGE",	LTYPE3,	ACMOVWGE,
	"CMOVWGT",	LTYPE3,	ACMOVWGT,
	"CMOVWHI",	LTYPE3,	ACMOVWHI,
	"CMOVWLE",	LTYPE3,	ACMOVWLE,
	"CMOVWLS",	LTYPE3,	ACMOVWLS,
	"CMOVWLT",	LTYPE3,	ACMOVWLT,
	"CMOVWMI",	LTYPE3,	ACMOVWMI,
	"CMOVWNE",	LTYPE3,	ACMOVWNE,
	"CMOVWOC",	LTYPE3,	ACMOVWOC,
	"CMOVWOS",	LTYPE3,	ACMOVWOS,
	"CMOVWPC",	LTYPE3,	ACMOVWPC,
	"CMOVWPL",	LTYPE3,	ACMOVWPL,
	"CMOVWPS",	LTYPE3,	ACMOVWPS,

	"FMOVB",	LTYPE3, AFMOVB,
	"FMOVBP",	LTYPE3, AFMOVBP,
	"FMOVD",	LTYPE3, AFMOVD,
	"FMOVDP",	LTYPE3, AFMOVDP,
	"FMOVF",	LTYPE3, AFMOVF,
	"FMOVFP",	LTYPE3, AFMOVFP,
	"FMOVL",	LTYPE3, AFMOVL,
	"FMOVLP",	LTYPE3, AFMOVLP,
	"FMOVV",	LTYPE3, AFMOVV,
	"FMOVVP",	LTYPE3, AFMOVVP,
	"FMOVW",	LTYPE3, AFMOVW,
	"FMOVWP",	LTYPE3, AFMOVWP,
	"FMOVX",	LTYPE3, AFMOVX,
	"FMOVXP",	LTYPE3, AFMOVXP,
	"FCMOVCC",	LTYPE3, AFCMOVCC,
	"FCMOVCS",	LTYPE3, AFCMOVCS,
	"FCMOVEQ",	LTYPE3, AFCMOVEQ,
	"FCMOVHI",	LTYPE3, AFCMOVHI,
	"FCMOVLS",	LTYPE3, AFCMOVLS,
	"FCMOVNE",	LTYPE3, AFCMOVNE,
	"FCMOVNU",	LTYPE3, AFCMOVNU,
	"FCMOVUN",	LTYPE3, AFCMOVUN,
	"FCOMB",	LTYPE3, AFCOMB,
	"FCOMBP",	LTYPE3, AFCOMBP,
	"FCOMD",	LTYPE3, AFCOMD,
	"FCOMDP",	LTYPE3, AFCOMDP,
	"FCOMDPP",	LTYPE3, AFCOMDPP,
	"FCOMF",	LTYPE3, AFCOMF,
	"FCOMFP",	LTYPE3, AFCOMFP,
	"FCOMI",	LTYPE3, AFCOMI,
	"FCOMIP",	LTYPE3, AFCOMIP,
	"FCOML",	LTYPE3, AFCOML,
	"FCOMLP",	LTYPE3, AFCOMLP,
	"FCOMW",	LTYPE3, AFCOMW,
	"FCOMWP",	LTYPE3, AFCOMWP,
	"FUCOM",	LTYPE3, AFUCOM,
	"FUCOMI",	LTYPE3, AFUCOMI,
	"FUCOMIP",	LTYPE3, AFUCOMIP,
	"FUCOMP",	LTYPE3, AFUCOMP,
	"FUCOMPP",	LTYPE3, AFUCOMPP,
	"FADDW",	LTYPE3, AFADDW,
	"FADDL",	LTYPE3, AFADDL,
	"FADDF",	LTYPE3, AFADDF,
	"FADDD",	LTYPE3, AFADDD,
	"FADDDP",	LTYPE3, AFADDDP,
	"FSUBDP",	LTYPE3, AFSUBDP,
	"FSUBW",	LTYPE3, AFSUBW,
	"FSUBL",	LTYPE3, AFSUBL,
	"FSUBF",	LTYPE3, AFSUBF,
	"FSUBD",	LTYPE3, AFSUBD,
	"FSUBRDP",	LTYPE3, AFSUBRDP,
	"FSUBRW",	LTYPE3, AFSUBRW,
	"FSUBRL",	LTYPE3, AFSUBRL,
	"FSUBRF",	LTYPE3, AFSUBRF,
	"FSUBRD",	LTYPE3, AFSUBRD,
	"FMULDP",	LTYPE3, AFMULDP,
	"FMULW",	LTYPE3, AFMULW,
	"FMULL",	LTYPE3, AFMULL,
	"FMULF",	LTYPE3, AFMULF,
	"FMULD",	LTYPE3, AFMULD,
	"FDIVDP",	LTYPE3, AFDIVDP,
	"FDIVW",	LTYPE3, AFDIVW,
	"FDIVL",	LTYPE3, AFDIVL,
	"FDIVF",	LTYPE3, AFDIVF,
	"FDIVD",	LTYPE3, AFDIVD,
	"FDIVRDP",	LTYPE3, AFDIVRDP,
	"FDIVRW",	LTYPE3, AFDIVRW,
	"FDIVRL",	LTYPE3, AFDIVRL,
	"FDIVRF",	LTYPE3, AFDIVRF,
	"FDIVRD",	LTYPE3, AFDIVRD,
	"FXCHD",	LTYPE3, AFXCHD,
	"FFREE",	LTYPE1, AFFREE,
	"FLDCW",	LTYPE2, AFLDCW,
	"FLDENV",	LTYPE1, AFLDENV,
	"FRSTOR",	LTYPE2, AFRSTOR,
	"FSAVE",	LTYPE1, AFSAVE,
	"FSTCW",	LTYPE1, AFSTCW,
	"FSTENV",	LTYPE1, AFSTENV,
	"FSTSW",	LTYPE1, AFSTSW,
	"F2XM1",	LTYPE0, AF2XM1,
	"FABS",		LTYPE0, AFABS,
	"FCHS",		LTYPE0, AFCHS,
	"FCLEX",	LTYPE0, AFCLEX,
	"FCOS",		LTYPE0, AFCOS,
	"FDECSTP",	LTYPE0, AFDECSTP,
	"FINCSTP",	LTYPE0, AFINCSTP,
	"FINIT",	LTYPE0, AFINIT,
	"FLD1",		LTYPE0, AFLD1,
	"FLDL2E",	LTYPE0, AFLDL2E,
	"FLDL2T",	LTYPE0, AFLDL2T,
	"FLDLG2",	LTYPE0, AFLDLG2,
	"FLDLN2",	LTYPE0, AFLDLN2,
	"FLDPI",	LTYPE0, AFLDPI,
	"FLDZ",		LTYPE0, AFLDZ,
	"FNOP",		LTYPE0, AFNOP,
	"FPATAN",	LTYPE0, AFPATAN,
	"FPREM",	LTYPE0, AFPREM,
	"FPREM1",	LTYPE0, AFPREM1,
	"FPTAN",	LTYPE0, AFPTAN,
	"FRNDINT",	LTYPE0, AFRNDINT,
	"FSCALE",	LTYPE0, AFSCALE,
	"FSIN",		LTYPE0, AFSIN,
	"FSINCOS",	LTYPE0, AFSINCOS,
	"FSQRT",	LTYPE0, AFSQRT,
	"FTST",		LTYPE0, AFTST,
	"FXAM",		LTYPE0, AFXAM,
	"FXTRACT",	LTYPE0, AFXTRACT,
	"FYL2X",	LTYPE0, AFYL2X,
	"FYL2XP1",	LTYPE0, AFYL2XP1,
	"LFENCE",	LTYPE0, ALFENCE,
	"MFENCE",	LTYPE0, AMFENCE,
	"SFENCE",	LTYPE0, ASFENCE,
	"EMMS",		LTYPE0, AEMMS,
	"PREFETCHT0",		LTYPE2,	APREFETCHT0,
	"PREFETCHT1",		LTYPE2,	APREFETCHT1,
	"PREFETCHT2",		LTYPE2,	APREFETCHT2,
	"PREFETCHNTA",		LTYPE2,	APREFETCHNTA,
	"UNDEF",	LTYPE0,	AUNDEF,

	"ADDPD",	LTYPE3,	AADDPD,
	"ADDPS",	LTYPE3,	AADDPS,
	"ADDSD",	LTYPE3,	AADDSD,
	"ADDSS",	LTYPE3,	AADDSS,
	"AESENC",	LTYPE3,	AAESENC,
	"ANDNPD",	LTYPE3,	AANDNPD,
	"ANDNPS",	LTYPE3,	AANDNPS,
	"ANDPD",	LTYPE3,	AANDPD,
	"ANDPS",	LTYPE3,	AANDPS,
	"CMPPD",	LTYPEXC,ACMPPD,
	"CMPPS",	LTYPEXC,ACMPPS,
	"CMPSD",	LTYPEXC,ACMPSD,
	"CMPSS",	LTYPEXC,ACMPSS,
	"COMISD",	LTYPE3,	ACOMISD,
	"COMISS",	LTYPE3,	ACOMISS,
	"CVTPL2PD",	LTYPE3,	ACVTPL2PD,
	"CVTPL2PS",	LTYPE3,	ACVTPL2PS,
	"CVTPD2PL",	LTYPE3,	ACVTPD2PL,
	"CVTPD2PS",	LTYPE3,	ACVTPD2PS,
	"CVTPS2PL",	LTYPE3,	ACVTPS2PL,
	"CVTPS2PD",	LTYPE3,	ACVTPS2PD,
	"CVTSD2SL",	LTYPE3,	ACVTSD2SL,
	"CVTSD2SS",	LTYPE3,	ACVTSD2SS,
	"CVTSL2SD",	LTYPE3,	ACVTSL2SD,
	"CVTSL2SS",	LTYPE3,	ACVTSL2SS,
	"CVTSS2SD",	LTYPE3,	ACVTSS2SD,
	"CVTSS2SL",	LTYPE3,	ACVTSS2SL,
	"CVTTPD2PL",	LTYPE3,	ACVTTPD2PL,
	"CVTTPS2PL",	LTYPE3,	ACVTTPS2PL,
	"CVTTSD2SL",	LTYPE3,	ACVTTSD2SL,
	"CVTTSS2SL",	LTYPE3,	ACVTTSS2SL,
	"DIVPD",	LTYPE3,	ADIVPD,
	"DIVPS",	LTYPE3,	ADIVPS,
	"DIVSD",	LTYPE3,	ADIVSD,
	"DIVSS",	LTYPE3,	ADIVSS,
	"MASKMOVOU",	LTYPE3,	AMASKMOVOU,
	"MASKMOVDQU",	LTYPE3,	AMASKMOVOU,	/* syn */
	"MAXPD",	LTYPE3,	AMAXPD,
	"MAXPS",	LTYPE3,	AMAXPS,
	"MAXSD",	LTYPE3,	AMAXSD,
	"MAXSS",	LTYPE3,	AMAXSS,
	"MINPD",	LTYPE3,	AMINPD,
	"MINPS",	LTYPE3,	AMINPS,
	"MINSD",	LTYPE3,	AMINSD,
	"MINSS",	LTYPE3,	AMINSS,
	"MOVAPD",	LTYPE3,	AMOVAPD,
	"MOVAPS",	LTYPE3,	AMOVAPS,
	"MOVO",		LTYPE3,	AMOVO,
	"MOVOA",	LTYPE3,	AMOVO,	/* syn */
	"MOVOU",	LTYPE3,	AMOVOU,
	"MOVHLPS",	LTYPE3,	AMOVHLPS,
	"MOVHPD",	LTYPE3,	AMOVHPD,
	"MOVHPS",	LTYPE3,	AMOVHPS,
	"MOVLHPS",	LTYPE3,	AMOVLHPS,
	"MOVLPD",	LTYPE3,	AMOVLPD,
	"MOVLPS",	LTYPE3,	AMOVLPS,
	"MOVMSKPD",	LTYPE3,	AMOVMSKPD,
	"MOVMSKPS",	LTYPE3,	AMOVMSKPS,
	"MOVNTO",	LTYPE3,	AMOVNTO,
	"MOVNTDQ",	LTYPE3,	AMOVNTO,	/* syn */
	"MOVNTPD",	LTYPE3,	AMOVNTPD,
	"MOVNTPS",	LTYPE3,	AMOVNTPS,
	"MOVSD",	LTYPE3,	AMOVSD,
	"MOVSS",	LTYPE3,	AMOVSS,
	"MOVUPD",	LTYPE3,	AMOVUPD,
	"MOVUPS",	LTYPE3,	AMOVUPS,
	"MULPD",	LTYPE3,	AMULPD,
	"MULPS",	LTYPE3,	AMULPS,
	"MULSD",	LTYPE3,	AMULSD,
	"MULSS",	LTYPE3,	AMULSS,
	"ORPD",		LTYPE3,	AORPD,
	"ORPS",		LTYPE3,	AORPS,
	"PADDQ",	LTYPE3,	APADDQ,
	"PAND",		LTYPE3,	APAND,
	"PCMPEQB",	LTYPE3,	APCMPEQB,
	"PMAXSW",	LTYPE3,	APMAXSW,
	"PMAXUB",	LTYPE3,	APMAXUB,
	"PMINSW",	LTYPE3,	APMINSW,
	"PMINUB",	LTYPE3,	APMINUB,
	"PMOVMSKB",	LTYPE3,	APMOVMSKB,
	"PSADBW",	LTYPE3,	APSADBW,
	"PSHUFB",	LTYPE3, APSHUFB,
	"PSHUFHW",      LTYPEX, APSHUFHW,
	"PSHUFL",       LTYPEX, APSHUFL,
	"PSHUFLW",      LTYPEX, APSHUFLW,
	"PSUBB",	LTYPE3,	APSUBB,
	"PSUBL",	LTYPE3,	APSUBL,
	"PSUBQ",	LTYPE3,	APSUBQ,
	"PSUBSB",	LTYPE3,	APSUBSB,
	"PSUBSW",	LTYPE3,	APSUBSW,
	"PSUBUSB",	LTYPE3,	APSUBUSB,
	"PSUBUSW",	LTYPE3,	APSUBUSW,
	"PSUBW",	LTYPE3,	APSUBW,
	"PUNPCKHQDQ",	LTYPE3,	APUNPCKHQDQ,
	"PUNPCKLQDQ",	LTYPE3,	APUNPCKLQDQ,
	"PXOR",		LTYPE3, APXOR,
	"RCPPS",	LTYPE3,	ARCPPS,
	"RCPSS",	LTYPE3,	ARCPSS,
	"RSQRTPS",	LTYPE3,	ARSQRTPS,
	"RSQRTSS",	LTYPE3,	ARSQRTSS,
	"SQRTPD",	LTYPE3,	ASQRTPD,
	"SQRTPS",	LTYPE3,	ASQRTPS,
	"SQRTSD",	LTYPE3,	ASQRTSD,
	"SQRTSS",	LTYPE3,	ASQRTSS,
	"SUBPD",	LTYPE3,	ASUBPD,
	"SUBPS",	LTYPE3,	ASUBPS,
	"SUBSD",	LTYPE3,	ASUBSD,
	"SUBSS",	LTYPE3,	ASUBSS,
	"UCOMISD",	LTYPE3,	AUCOMISD,
	"UCOMISS",	LTYPE3,	AUCOMISS,
	"UNPCKHPD",	LTYPE3,	AUNPCKHPD,
	"UNPCKHPS",	LTYPE3,	AUNPCKHPS,
	"UNPCKLPD",	LTYPE3,	AUNPCKLPD,
	"UNPCKLPS",	LTYPE3,	AUNPCKLPS,
	"XORPD",	LTYPE3,	AXORPD,
	"XORPS",	LTYPE3,	AXORPS,
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
	nullgen.index = D_NONE;

	nerrors = 0;
	iostack = I;
	iofree = I;
	peekc = IGN;
	nhunk = 0;
	for(i=0; i<NHASH; i++)
		hash[i] = S;
	for(i=0; itab[i].name; i++) {
		s = slookup(itab[i].name);
		if(s->type != LNAME)
			yyerror("double initialization %s", itab[i].name);
		s->type = itab[i].type;
		s->value = itab[i].value;
	}
}

void
checkscale(int scale)
{

	switch(scale) {
	case 1:
	case 2:
	case 4:
	case 8:
		return;
	}
	yyerror("scale must be 1248: %d", scale);
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
	Addr2 g2;

	g2.from = nullgen;
	g2.to = nullgen;
	outcode(AEND, &g2);
}

static Prog *lastpc;

void
outcode(int a, Addr2 *g2)
{
	Prog *p;
	Plist *pl;
	
	if(pass == 1)
		goto out;

	p = malloc(sizeof *p);
	memset(p, 0, sizeof *p);
	p->as = a;
	p->lineno = stmtline;
	p->from = g2->from;
	p->to = g2->to;
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
