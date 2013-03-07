// Inferno utils/6a/lex.c
// http://code.google.com/p/inferno-os/source/browse/utils/6a/lex.c
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

void
main(int argc, char *argv[])
{
	char *p;
	int c;

	thechar = '6';
	thestring = "amd64";

	ensuresymb(NSYMB);
	memset(debug, 0, sizeof(debug));
	cinit();
	outfile = 0;
	setinclude(".");
	ARGBEGIN {
	default:
		c = ARGC();
		if(c >= 0 || c < sizeof(debug))
			debug[c] = 1;
		break;

	case 'o':
		outfile = ARGF();
		break;

	case 'D':
		p = ARGF();
		if(p) {
			if (nDlist%8 == 0)
				Dlist = allocn(Dlist, nDlist*sizeof(char *), 
					8*sizeof(char *));
			Dlist[nDlist++] = p;
		}
		break;

	case 'I':
		p = ARGF();
		setinclude(p);
		break;
	} ARGEND
	if(*argv == 0) {
		print("usage: %ca [-options] file.s\n", thechar);
		errorexit();
	}
	if(argc > 1){
		print("can't assemble multiple files\n");
		errorexit();
	}
	if(assemble(argv[0]))
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

	pass = 1;
	pinit(file);

	Bprint(&obuf, "go object %s %s %s\n", getgoos(), thestring, getgoversion());

	for(i=0; i<nDlist; i++)
		dodefine(Dlist[i]);
	yyparse();
	if(nerrors) {
		cclean();
		return nerrors;
	}

	Bprint(&obuf, "\n!\n");

	pass = 2;
	outhist();
	pinit(file);
	for(i=0; i<nDlist; i++)
		dodefine(Dlist[i]);
	yyparse();
	cclean();
	return nerrors;
}

struct
{
	char	*name;
	/*
	 * type is the lexical type to return.  It dictates what kind of
	 * operands 6a allows to follow it (in a.y) as the possible operand
	 * types are handled by a grammar.  How do you know which LTYPE?
	 * Either read a.y or think of an instruction that has the same
	 * possible operands and look up what it takes.
	 */
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
/*	"SPB",		LBREG,	D_SPB,	*/
	"SIB",		LBREG,	D_SIB,
	"DIB",		LBREG,	D_DIB,
	"BPB",		LBREG,	D_BPB,
	"R8B",		LBREG,	D_R8B,
	"R9B",		LBREG,	D_R9B,
	"R10B",		LBREG,	D_R10B,
	"R11B",		LBREG,	D_R11B,
	"R12B",		LBREG,	D_R12B,
	"R13B",		LBREG,	D_R13B,
	"R14B",		LBREG,	D_R14B,
	"R15B",		LBREG,	D_R15B,

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
	"R8",		LLREG,	D_R8,
	"R9",		LLREG,	D_R9,
	"R10",		LLREG,	D_R10,
	"R11",		LLREG,	D_R11,
	"R12",		LLREG,	D_R12,
	"R13",		LLREG,	D_R13,
	"R14",		LLREG,	D_R14,
	"R15",		LLREG,	D_R15,

	"RARG",		LLREG,	REGARG,

	"F0",		LFREG,	D_F0+0,
	"F1",		LFREG,	D_F0+1,
	"F2",		LFREG,	D_F0+2,
	"F3",		LFREG,	D_F0+3,
	"F4",		LFREG,	D_F0+4,
	"F5",		LFREG,	D_F0+5,
	"F6",		LFREG,	D_F0+6,
	"F7",		LFREG,	D_F0+7,

	"M0",		LMREG,	D_M0+0,
	"M1",		LMREG,	D_M0+1,
	"M2",		LMREG,	D_M0+2,
	"M3",		LMREG,	D_M0+3,
	"M4",		LMREG,	D_M0+4,
	"M5",		LMREG,	D_M0+5,
	"M6",		LMREG,	D_M0+6,
	"M7",		LMREG,	D_M0+7,

	"X0",		LXREG,	D_X0+0,
	"X1",		LXREG,	D_X0+1,
	"X2",		LXREG,	D_X0+2,
	"X3",		LXREG,	D_X0+3,
	"X4",		LXREG,	D_X0+4,
	"X5",		LXREG,	D_X0+5,
	"X6",		LXREG,	D_X0+6,
	"X7",		LXREG,	D_X0+7,
	"X8",		LXREG,	D_X0+8,
	"X9",		LXREG,	D_X0+9,
	"X10",		LXREG,	D_X0+10,
	"X11",		LXREG,	D_X0+11,
	"X12",		LXREG,	D_X0+12,
	"X13",		LXREG,	D_X0+13,
	"X14",		LXREG,	D_X0+14,
	"X15",		LXREG,	D_X0+15,

	"CS",		LSREG,	D_CS,
	"SS",		LSREG,	D_SS,
	"DS",		LSREG,	D_DS,
	"ES",		LSREG,	D_ES,
	"FS",		LSREG,	D_FS,
	"GS",		LSREG,	D_GS,

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
	"CR8",		LBREG,	D_CR+8,
	"CR9",		LBREG,	D_CR+9,
	"CR10",		LBREG,	D_CR+10,
	"CR11",		LBREG,	D_CR+11,
	"CR12",		LBREG,	D_CR+12,
	"CR13",		LBREG,	D_CR+13,
	"CR14",		LBREG,	D_CR+14,
	"CR15",		LBREG,	D_CR+15,

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
	"ADCQ",		LTYPE3,	AADCQ,
	"ADCW",		LTYPE3,	AADCW,
	"ADDB",		LTYPE3,	AADDB,
	"ADDL",		LTYPE3,	AADDL,
	"ADDQ",		LTYPE3,	AADDQ,
	"ADDW",		LTYPE3,	AADDW,
	"ADJSP",	LTYPE2,	AADJSP,
	"ANDB",		LTYPE3,	AANDB,
	"ANDL",		LTYPE3,	AANDL,
	"ANDQ",		LTYPE3,	AANDQ,
	"ANDW",		LTYPE3,	AANDW,
	"ARPL",		LTYPE3,	AARPL,
	"BOUNDL",	LTYPE3,	ABOUNDL,
	"BOUNDW",	LTYPE3,	ABOUNDW,
	"BSFL",		LTYPE3,	ABSFL,
	"BSFQ",		LTYPE3,	ABSFQ,
	"BSFW",		LTYPE3,	ABSFW,
	"BSRL",		LTYPE3,	ABSRL,
	"BSRQ",		LTYPE3,	ABSRQ,
	"BSRW",		LTYPE3,	ABSRW,
	"BSWAPL",	LTYPE1,	ABSWAPL,
	"BSWAPQ",	LTYPE1,	ABSWAPQ,
	"BTCL",		LTYPE3,	ABTCL,
	"BTCQ",		LTYPE3,	ABTCQ,
	"BTCW",		LTYPE3,	ABTCW,
	"BTL",		LTYPE3,	ABTL,
	"BTQ",		LTYPE3,	ABTQ,
	"BTRL",		LTYPE3,	ABTRL,
	"BTRQ",		LTYPE3,	ABTRQ,
	"BTRW",		LTYPE3,	ABTRW,
	"BTSL",		LTYPE3,	ABTSL,
	"BTSQ",		LTYPE3,	ABTSQ,
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
	"CMPQ",		LTYPE4,	ACMPQ,
	"CMPW",		LTYPE4,	ACMPW,
	"CMPSB",	LTYPE0,	ACMPSB,
	"CMPSL",	LTYPE0,	ACMPSL,
	"CMPSQ",	LTYPE0,	ACMPSQ,
	"CMPSW",	LTYPE0,	ACMPSW,
	"CMPXCHG8B",	LTYPE1,	ACMPXCHG8B,
	"CMPXCHGB",	LTYPE3,	ACMPXCHGB,	/* LTYPE3? */
	"CMPXCHGL",	LTYPE3,	ACMPXCHGL,
	"CMPXCHGQ",	LTYPE3,	ACMPXCHGQ,
	"CMPXCHGW",	LTYPE3,	ACMPXCHGW,
	"CPUID",	LTYPE0,	ACPUID,
	"DAA",		LTYPE0,	ADAA,
	"DAS",		LTYPE0,	ADAS,
	"DATA",		LTYPED,	ADATA,
	"DECB",		LTYPE1,	ADECB,
	"DECL",		LTYPE1,	ADECL,
	"DECQ",		LTYPE1,	ADECQ,
	"DECW",		LTYPE1,	ADECW,
	"DIVB",		LTYPE2,	ADIVB,
	"DIVL",		LTYPE2,	ADIVL,
	"DIVQ",		LTYPE2,	ADIVQ,
	"DIVW",		LTYPE2,	ADIVW,
	"EMMS",		LTYPE0,	AEMMS,
	"END",		LTYPE0,	AEND,
	"ENTER",	LTYPE2,	AENTER,
	"GLOBL",	LTYPEG,	AGLOBL,
	"HLT",		LTYPE0,	AHLT,
	"IDIVB",	LTYPE2,	AIDIVB,
	"IDIVL",	LTYPE2,	AIDIVL,
	"IDIVQ",	LTYPE2,	AIDIVQ,
	"IDIVW",	LTYPE2,	AIDIVW,
	"IMULB",	LTYPEI,	AIMULB,
	"IMULL",	LTYPEI,	AIMULL,
	"IMULQ",	LTYPEI,	AIMULQ,
	"IMUL3Q",	LTYPEX,	AIMUL3Q,
	"IMULW",	LTYPEI,	AIMULW,
	"INB",		LTYPE0,	AINB,
	"INL",		LTYPE0,	AINL,
	"INW",		LTYPE0,	AINW,
	"INCB",		LTYPE1,	AINCB,
	"INCL",		LTYPE1,	AINCL,
	"INCQ",		LTYPE1,	AINCQ,
	"INCW",		LTYPE1,	AINCW,
	"INSB",		LTYPE0,	AINSB,
	"INSL",		LTYPE0,	AINSL,
	"INSW",		LTYPE0,	AINSW,
	"INT",		LTYPE2,	AINT,
	"INTO",		LTYPE0,	AINTO,
	"INVD",		LTYPE0,	AINVD,
	"INVLPG",	LTYPE2,	AINVLPG,
	"IRETL",	LTYPE0,	AIRETL,
	"IRETQ",	LTYPE0,	AIRETQ,
	"IRETW",	LTYPE0,	AIRETW,

	"JOS",		LTYPER,	AJOS,
	"JO",		LTYPER,	AJOS,	/* alternate */
	"JOC",		LTYPER,	AJOC,
	"JNO",		LTYPER,	AJOC,	/* alternate */
	"JCS",		LTYPER,	AJCS,
	"JB",		LTYPER,	AJCS,	/* alternate */
	"JC",		LTYPER,	AJCS,	/* alternate */
	"JNAE",		LTYPER,	AJCS,	/* alternate */
	"JLO",		LTYPER,	AJCS,	/* alternate */
	"JCC",		LTYPER,	AJCC,
	"JAE",		LTYPER,	AJCC,	/* alternate */
	"JNB",		LTYPER,	AJCC,	/* alternate */
	"JNC",		LTYPER,	AJCC,	/* alternate */
	"JHS",		LTYPER,	AJCC,	/* alternate */
	"JEQ",		LTYPER,	AJEQ,
	"JE",		LTYPER,	AJEQ,	/* alternate */
	"JZ",		LTYPER,	AJEQ,	/* alternate */
	"JNE",		LTYPER,	AJNE,
	"JNZ",		LTYPER,	AJNE,	/* alternate */
	"JLS",		LTYPER,	AJLS,
	"JBE",		LTYPER,	AJLS,	/* alternate */
	"JNA",		LTYPER,	AJLS,	/* alternate */
	"JHI",		LTYPER,	AJHI,
	"JA",		LTYPER,	AJHI,	/* alternate */
	"JNBE",		LTYPER,	AJHI,	/* alternate */
	"JMI",		LTYPER,	AJMI,
	"JS",		LTYPER,	AJMI,	/* alternate */
	"JPL",		LTYPER,	AJPL,
	"JNS",		LTYPER,	AJPL,	/* alternate */
	"JPS",		LTYPER,	AJPS,
	"JP",		LTYPER,	AJPS,	/* alternate */
	"JPE",		LTYPER,	AJPS,	/* alternate */
	"JPC",		LTYPER,	AJPC,
	"JNP",		LTYPER,	AJPC,	/* alternate */
	"JPO",		LTYPER,	AJPC,	/* alternate */
	"JLT",		LTYPER,	AJLT,
	"JL",		LTYPER,	AJLT,	/* alternate */
	"JNGE",		LTYPER,	AJLT,	/* alternate */
	"JGE",		LTYPER,	AJGE,
	"JNL",		LTYPER,	AJGE,	/* alternate */
	"JLE",		LTYPER,	AJLE,
	"JNG",		LTYPER,	AJLE,	/* alternate */
	"JGT",		LTYPER,	AJGT,
	"JG",		LTYPER,	AJGT,	/* alternate */
	"JNLE",		LTYPER,	AJGT,	/* alternate */
	"JCXZL",	LTYPER,	AJCXZL,
	"JCXZQ",	LTYPER,	AJCXZQ,
	"JMP",		LTYPEC,	AJMP,
	"LAHF",		LTYPE0,	ALAHF,
	"LARL",		LTYPE3,	ALARL,
	"LARW",		LTYPE3,	ALARW,
	"LEAL",		LTYPE3,	ALEAL,
	"LEAQ",		LTYPE3,	ALEAQ,
	"LEAW",		LTYPE3,	ALEAW,
	"LEAVEL",	LTYPE0,	ALEAVEL,
	"LEAVEQ",	LTYPE0,	ALEAVEQ,
	"LEAVEW",	LTYPE0,	ALEAVEW,
	"LFENCE",	LTYPE0,	ALFENCE,
	"LOCK",		LTYPE0,	ALOCK,
	"LODSB",	LTYPE0,	ALODSB,
	"LODSL",	LTYPE0,	ALODSL,
	"LODSQ",	LTYPE0,	ALODSQ,
	"LODSW",	LTYPE0,	ALODSW,
	"LONG",		LTYPE2,	ALONG,
	"LOOP",		LTYPER,	ALOOP,
	"LOOPEQ",	LTYPER,	ALOOPEQ,
	"LOOPNE",	LTYPER,	ALOOPNE,
	"LSLL",		LTYPE3,	ALSLL,
	"LSLW",		LTYPE3,	ALSLW,
	"MFENCE",	LTYPE0,	AMFENCE,
	"MODE",		LTYPE2,	AMODE,
	"MOVB",		LTYPE3,	AMOVB,
	"MOVL",		LTYPEM,	AMOVL,
	"MOVQ",		LTYPEM,	AMOVQ,
	"MOVW",		LTYPEM,	AMOVW,
	"MOVBLSX",	LTYPE3, AMOVBLSX,
	"MOVBLZX",	LTYPE3, AMOVBLZX,
	"MOVBQSX",	LTYPE3,	AMOVBQSX,
	"MOVBQZX",	LTYPE3,	AMOVBQZX,
	"MOVBWSX",	LTYPE3, AMOVBWSX,
	"MOVBWZX",	LTYPE3, AMOVBWZX,
	"MOVLQSX",	LTYPE3, AMOVLQSX,
	"MOVLQZX",	LTYPE3, AMOVLQZX,
	"MOVNTIL",	LTYPE3,	AMOVNTIL,
	"MOVNTIQ",	LTYPE3,	AMOVNTIQ,
	"MOVWLSX",	LTYPE3, AMOVWLSX,
	"MOVWLZX",	LTYPE3, AMOVWLZX,
	"MOVWQSX",	LTYPE3,	AMOVWQSX,
	"MOVWQZX",	LTYPE3,	AMOVWQZX,
	"MOVSB",	LTYPE0,	AMOVSB,
	"MOVSL",	LTYPE0,	AMOVSL,
	"MOVSQ",	LTYPE0,	AMOVSQ,
	"MOVSW",	LTYPE0,	AMOVSW,
	"MULB",		LTYPE2,	AMULB,
	"MULL",		LTYPE2,	AMULL,
	"MULQ",		LTYPE2,	AMULQ,
	"MULW",		LTYPE2,	AMULW,
	"NEGB",		LTYPE1,	ANEGB,
	"NEGL",		LTYPE1,	ANEGL,
	"NEGQ",		LTYPE1,	ANEGQ,
	"NEGW",		LTYPE1,	ANEGW,
	"NOP",		LTYPEN,	ANOP,
	"NOTB",		LTYPE1,	ANOTB,
	"NOTL",		LTYPE1,	ANOTL,
	"NOTQ",		LTYPE1,	ANOTQ,
	"NOTW",		LTYPE1,	ANOTW,
	"ORB",		LTYPE3,	AORB,
	"ORL",		LTYPE3,	AORL,
	"ORQ",		LTYPE3,	AORQ,
	"ORW",		LTYPE3,	AORW,
	"OUTB",		LTYPE0,	AOUTB,
	"OUTL",		LTYPE0,	AOUTL,
	"OUTW",		LTYPE0,	AOUTW,
	"OUTSB",	LTYPE0,	AOUTSB,
	"OUTSL",	LTYPE0,	AOUTSL,
	"OUTSW",	LTYPE0,	AOUTSW,
	"PAUSE",	LTYPEN,	APAUSE,
	"POPAL",	LTYPE0,	APOPAL,
	"POPAW",	LTYPE0,	APOPAW,
	"POPFL",	LTYPE0,	APOPFL,
	"POPFQ",	LTYPE0,	APOPFQ,
	"POPFW",	LTYPE0,	APOPFW,
	"POPL",		LTYPE1,	APOPL,
	"POPQ",		LTYPE1,	APOPQ,
	"POPW",		LTYPE1,	APOPW,
	"PUSHAL",	LTYPE0,	APUSHAL,
	"PUSHAW",	LTYPE0,	APUSHAW,
	"PUSHFL",	LTYPE0,	APUSHFL,
	"PUSHFQ",	LTYPE0,	APUSHFQ,
	"PUSHFW",	LTYPE0,	APUSHFW,
	"PUSHL",	LTYPE2,	APUSHL,
	"PUSHQ",	LTYPE2,	APUSHQ,
	"PUSHW",	LTYPE2,	APUSHW,
	"RCLB",		LTYPE3,	ARCLB,
	"RCLL",		LTYPE3,	ARCLL,
	"RCLQ",		LTYPE3,	ARCLQ,
	"RCLW",		LTYPE3,	ARCLW,
	"RCRB",		LTYPE3,	ARCRB,
	"RCRL",		LTYPE3,	ARCRL,
	"RCRQ",		LTYPE3,	ARCRQ,
	"RCRW",		LTYPE3,	ARCRW,
	"RDMSR",	LTYPE0,	ARDMSR,
	"RDPMC",	LTYPE0,	ARDPMC,
	"RDTSC",	LTYPE0,	ARDTSC,
	"REP",		LTYPE0,	AREP,
	"REPN",		LTYPE0,	AREPN,
	"RET",		LTYPE0,	ARET,
	"RETFL",	LTYPERT,ARETFL,
	"RETFW",	LTYPERT,ARETFW,
	"RETFQ",	LTYPERT,ARETFQ,
	"ROLB",		LTYPE3,	AROLB,
	"ROLL",		LTYPE3,	AROLL,
	"ROLQ",		LTYPE3,	AROLQ,
	"ROLW",		LTYPE3,	AROLW,
	"RORB",		LTYPE3,	ARORB,
	"RORL",		LTYPE3,	ARORL,
	"RORQ",		LTYPE3,	ARORQ,
	"RORW",		LTYPE3,	ARORW,
	"RSM",		LTYPE0,	ARSM,
	"SAHF",		LTYPE0,	ASAHF,
	"SALB",		LTYPE3,	ASALB,
	"SALL",		LTYPE3,	ASALL,
	"SALQ",		LTYPE3,	ASALQ,
	"SALW",		LTYPE3,	ASALW,
	"SARB",		LTYPE3,	ASARB,
	"SARL",		LTYPE3,	ASARL,
	"SARQ",		LTYPE3,	ASARQ,
	"SARW",		LTYPE3,	ASARW,
	"SBBB",		LTYPE3,	ASBBB,
	"SBBL",		LTYPE3,	ASBBL,
	"SBBQ",		LTYPE3,	ASBBQ,
	"SBBW",		LTYPE3,	ASBBW,
	"SCASB",	LTYPE0,	ASCASB,
	"SCASL",	LTYPE0,	ASCASL,
	"SCASQ",	LTYPE0,	ASCASQ,
	"SCASW",	LTYPE0,	ASCASW,
	"SETCC",	LTYPE1,	ASETCC,
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
	"SFENCE",	LTYPE0,	ASFENCE,
	"CDQ",		LTYPE0,	ACDQ,
	"CWD",		LTYPE0,	ACWD,
	"CQO",		LTYPE0,	ACQO,
	"SHLB",		LTYPE3,	ASHLB,
	"SHLL",		LTYPES,	ASHLL,
	"SHLQ",		LTYPES,	ASHLQ,
	"SHLW",		LTYPES,	ASHLW,
	"SHRB",		LTYPE3,	ASHRB,
	"SHRL",		LTYPES,	ASHRL,
	"SHRQ",		LTYPES,	ASHRQ,
	"SHRW",		LTYPES,	ASHRW,
	"STC",		LTYPE0,	ASTC,
	"STD",		LTYPE0,	ASTD,
	"STI",		LTYPE0,	ASTI,
	"STOSB",	LTYPE0,	ASTOSB,
	"STOSL",	LTYPE0,	ASTOSL,
	"STOSQ",	LTYPE0,	ASTOSQ,
	"STOSW",	LTYPE0,	ASTOSW,
	"SUBB",		LTYPE3,	ASUBB,
	"SUBL",		LTYPE3,	ASUBL,
	"SUBQ",		LTYPE3,	ASUBQ,
	"SUBW",		LTYPE3,	ASUBW,
	"SYSCALL",	LTYPE0,	ASYSCALL,
	"SYSRET",	LTYPE0,	ASYSRET,
	"SWAPGS",	LTYPE0,	ASWAPGS,
	"TESTB",	LTYPE3,	ATESTB,
	"TESTL",	LTYPE3,	ATESTL,
	"TESTQ",	LTYPE3,	ATESTQ,
	"TESTW",	LTYPE3,	ATESTW,
	"TEXT",		LTYPET,	ATEXT,
	"VERR",		LTYPE2,	AVERR,
	"VERW",		LTYPE2,	AVERW,
	"QUAD",		LTYPE2,	AQUAD,
	"WAIT",		LTYPE0,	AWAIT,
	"WBINVD",	LTYPE0,	AWBINVD,
	"WRMSR",	LTYPE0,	AWRMSR,
	"WORD",		LTYPE2,	AWORD,
	"XADDB",	LTYPE3,	AXADDB,
	"XADDL",	LTYPE3,	AXADDL,
	"XADDQ",	LTYPE3,	AXADDQ,
	"XADDW",	LTYPE3,	AXADDW,
	"XCHGB",	LTYPE3,	AXCHGB,
	"XCHGL",	LTYPE3,	AXCHGL,
	"XCHGQ",	LTYPE3,	AXCHGQ,
	"XCHGW",	LTYPE3,	AXCHGW,
	"XLAT",		LTYPE2,	AXLAT,
	"XORB",		LTYPE3,	AXORB,
	"XORL",		LTYPE3,	AXORL,
	"XORQ",		LTYPE3,	AXORQ,
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
	"CMOVQCC",	LTYPE3,	ACMOVQCC,
	"CMOVQCS",	LTYPE3,	ACMOVQCS,
	"CMOVQEQ",	LTYPE3,	ACMOVQEQ,
	"CMOVQGE",	LTYPE3,	ACMOVQGE,
	"CMOVQGT",	LTYPE3,	ACMOVQGT,
	"CMOVQHI",	LTYPE3,	ACMOVQHI,
	"CMOVQLE",	LTYPE3,	ACMOVQLE,
	"CMOVQLS",	LTYPE3,	ACMOVQLS,
	"CMOVQLT",	LTYPE3,	ACMOVQLT,
	"CMOVQMI",	LTYPE3,	ACMOVQMI,
	"CMOVQNE",	LTYPE3,	ACMOVQNE,
	"CMOVQOC",	LTYPE3,	ACMOVQOC,
	"CMOVQOS",	LTYPE3,	ACMOVQOS,
	"CMOVQPC",	LTYPE3,	ACMOVQPC,
	"CMOVQPL",	LTYPE3,	ACMOVQPL,
	"CMOVQPS",	LTYPE3,	ACMOVQPS,
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
	"FCOMB",	LTYPE3, AFCOMB,
	"FCOMBP",	LTYPE3, AFCOMBP,
	"FCOMD",	LTYPE3, AFCOMD,
	"FCOMDP",	LTYPE3, AFCOMDP,
	"FCOMDPP",	LTYPE3, AFCOMDPP,
	"FCOMF",	LTYPE3, AFCOMF,
	"FCOMFP",	LTYPE3, AFCOMFP,
	"FCOML",	LTYPE3, AFCOML,
	"FCOMLP",	LTYPE3, AFCOMLP,
	"FCOMW",	LTYPE3, AFCOMW,
	"FCOMWP",	LTYPE3, AFCOMWP,
	"FUCOM",	LTYPE3, AFUCOM,
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

	"ADDPD",	LTYPE3,	AADDPD,
	"ADDPS",	LTYPE3,	AADDPS,
	"ADDSD",	LTYPE3,	AADDSD,
	"ADDSS",	LTYPE3,	AADDSS,
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
	"PF2IW",	LTYPE3,	APF2IW,
	"PF2IL",	LTYPE3,	APF2IL,
	"PF2ID",	LTYPE3,	APF2IL,	/* syn */
	"PI2FL",	LTYPE3,	API2FL,
	"PI2FD",	LTYPE3,	API2FL,	/* syn */
	"PI2FW",	LTYPE3,	API2FW,
	"CVTPS2PD",	LTYPE3,	ACVTPS2PD,
	"CVTSD2SL",	LTYPE3,	ACVTSD2SL,
	"CVTSD2SQ",	LTYPE3,	ACVTSD2SQ,
	"CVTSD2SS",	LTYPE3,	ACVTSD2SS,
	"CVTSL2SD",	LTYPE3,	ACVTSL2SD,
	"CVTSQ2SD",	LTYPE3,	ACVTSQ2SD,
	"CVTSL2SS",	LTYPE3,	ACVTSL2SS,
	"CVTSQ2SS",	LTYPE3,	ACVTSQ2SS,
	"CVTSS2SD",	LTYPE3,	ACVTSS2SD,
	"CVTSS2SL",	LTYPE3,	ACVTSS2SL,
	"CVTSS2SQ",	LTYPE3,	ACVTSS2SQ,
	"CVTTPD2PL",	LTYPE3,	ACVTTPD2PL,
	"CVTTPS2PL",	LTYPE3,	ACVTTPS2PL,
	"CVTTSD2SL",	LTYPE3,	ACVTTSD2SL,
	"CVTTSD2SQ",	LTYPE3,	ACVTTSD2SQ,
	"CVTTSS2SL",	LTYPE3,	ACVTTSS2SL,
	"CVTTSS2SQ",	LTYPE3,	ACVTTSS2SQ,
	"DIVPD",	LTYPE3,	ADIVPD,
	"DIVPS",	LTYPE3,	ADIVPS,
	"DIVSD",	LTYPE3,	ADIVSD,
	"DIVSS",	LTYPE3,	ADIVSS,
	"FXRSTOR",	LTYPE2,	AFXRSTOR,
	"FXRSTOR64",	LTYPE2,	AFXRSTOR64,
	"FXSAVE",	LTYPE1,	AFXSAVE,
	"FXSAVE64",	LTYPE1,	AFXSAVE64,
	"LDMXCSR",	LTYPE2,	ALDMXCSR,
	"MASKMOVOU",	LTYPE3,	AMASKMOVOU,
	"MASKMOVDQU",	LTYPE3,	AMASKMOVOU,	/* syn */
	"MASKMOVQ",	LTYPE3,	AMASKMOVQ,
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
	"MOVD",		LTYPE3,	AMOVQ,	/* syn */
	"MOVDQ2Q",	LTYPE3,	AMOVQ,	/* syn */
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
	"MOVNTQ",	LTYPE3,	AMOVNTQ,
	"MOVQOZX",	LTYPE3,	AMOVQOZX,
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
	"PACKSSLW",	LTYPE3,	APACKSSLW,
	"PACKSSWB",	LTYPE3,	APACKSSWB,
	"PACKUSWB",	LTYPE3,	APACKUSWB,
	"PADDB",	LTYPE3,	APADDB,
	"PADDL",	LTYPE3,	APADDL,
	"PADDQ",	LTYPE3,	APADDQ,
	"PADDSB",	LTYPE3,	APADDSB,
	"PADDSW",	LTYPE3,	APADDSW,
	"PADDUSB",	LTYPE3,	APADDUSB,
	"PADDUSW",	LTYPE3,	APADDUSW,
	"PADDW",	LTYPE3,	APADDW,
	"PAND",		LTYPE3, APAND,
	"PANDB",	LTYPE3,	APANDB,
	"PANDL",	LTYPE3,	APANDL,
	"PANDSB",	LTYPE3,	APANDSB,
	"PANDSW",	LTYPE3,	APANDSW,
	"PANDUSB",	LTYPE3,	APANDUSB,
	"PANDUSW",	LTYPE3,	APANDUSW,
	"PANDW",	LTYPE3,	APANDW,
	"PANDN",	LTYPE3, APANDN,
	"PAVGB",	LTYPE3,	APAVGB,
	"PAVGW",	LTYPE3,	APAVGW,
	"PCMPEQB",	LTYPE3,	APCMPEQB,
	"PCMPEQL",	LTYPE3,	APCMPEQL,
	"PCMPEQW",	LTYPE3,	APCMPEQW,
	"PCMPGTB",	LTYPE3,	APCMPGTB,
	"PCMPGTL",	LTYPE3,	APCMPGTL,	
	"PCMPGTW",	LTYPE3,	APCMPGTW,
	"PEXTRW",	LTYPEX,	APEXTRW,
	"PINSRW",	LTYPEX,	APINSRW,
	"PINSRD",	LTYPEX,	APINSRD,
	"PINSRQ",	LTYPEX,	APINSRQ,
	"PMADDWL",	LTYPE3,	APMADDWL,
	"PMAXSW",	LTYPE3,	APMAXSW,
	"PMAXUB",	LTYPE3,	APMAXUB,
	"PMINSW",	LTYPE3,	APMINSW,
	"PMINUB",	LTYPE3,	APMINUB,
	"PMOVMSKB",	LTYPE3,	APMOVMSKB,
	"PMULHRW",	LTYPE3,	APMULHRW,
	"PMULHUW",	LTYPE3,	APMULHUW,
	"PMULHW",	LTYPE3,	APMULHW,
	"PMULLW",	LTYPE3,	APMULLW,
	"PMULULQ",	LTYPE3,	APMULULQ,
	"POR",		LTYPE3,	APOR,
	"PSADBW",	LTYPE3,	APSADBW,
	"PSHUFHW",	LTYPEX,	APSHUFHW,
	"PSHUFL",	LTYPEX,	APSHUFL,
	"PSHUFLW",	LTYPEX,	APSHUFLW,
	"PSHUFW",	LTYPEX, APSHUFW,
	"PSHUFB",	LTYPEM, APSHUFB,
	"PSLLO",	LTYPE3,	APSLLO,
	"PSLLDQ",	LTYPE3,	APSLLO,	/* syn */
	"PSLLL",	LTYPE3,	APSLLL,
	"PSLLQ",	LTYPE3,	APSLLQ,
	"PSLLW",	LTYPE3,	APSLLW,
	"PSRAL",	LTYPE3,	APSRAL,
	"PSRAW",	LTYPE3,	APSRAW,
	"PSRLO",	LTYPE3,	APSRLO,
	"PSRLDQ",	LTYPE3,	APSRLO,	/* syn */
	"PSRLL",	LTYPE3,	APSRLL,
	"PSRLQ",	LTYPE3,	APSRLQ,
	"PSRLW",	LTYPE3,	APSRLW,
	"PSUBB",	LTYPE3,	APSUBB,
	"PSUBL",	LTYPE3,	APSUBL,
	"PSUBQ",	LTYPE3,	APSUBQ,
	"PSUBSB",	LTYPE3,	APSUBSB,
	"PSUBSW",	LTYPE3,	APSUBSW,
	"PSUBUSB",	LTYPE3,	APSUBUSB,
	"PSUBUSW",	LTYPE3,	APSUBUSW,
	"PSUBW",	LTYPE3,	APSUBW,
	"PUNPCKHBW",	LTYPE3,	APUNPCKHBW,
	"PUNPCKHLQ",	LTYPE3,	APUNPCKHLQ,
	"PUNPCKHQDQ",	LTYPE3,	APUNPCKHQDQ,
	"PUNPCKHWL",	LTYPE3,	APUNPCKHWL,
	"PUNPCKLBW",	LTYPE3,	APUNPCKLBW,
	"PUNPCKLLQ",	LTYPE3,	APUNPCKLLQ,
	"PUNPCKLQDQ",	LTYPE3,	APUNPCKLQDQ,
	"PUNPCKLWL",	LTYPE3,	APUNPCKLWL,
	"PXOR",		LTYPE3,	APXOR,
	"RCPPS",	LTYPE3,	ARCPPS,
	"RCPSS",	LTYPE3,	ARCPSS,
	"RSQRTPS",	LTYPE3,	ARSQRTPS,
	"RSQRTSS",	LTYPE3,	ARSQRTSS,
	"SHUFPD",	LTYPEX,	ASHUFPD,
	"SHUFPS",	LTYPEX,	ASHUFPS,
	"SQRTPD",	LTYPE3,	ASQRTPD,
	"SQRTPS",	LTYPE3,	ASQRTPS,
	"SQRTSD",	LTYPE3,	ASQRTSD,
	"SQRTSS",	LTYPE3,	ASQRTSS,
	"STMXCSR",	LTYPE1,	ASTMXCSR,
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
	"CRC32B",	LTYPE4, ACRC32B,
	"CRC32Q",	LTYPE4, ACRC32Q,
	"PREFETCHT0",		LTYPE2,	APREFETCHT0,
	"PREFETCHT1",		LTYPE2,	APREFETCHT1,
	"PREFETCHT2",		LTYPE2,	APREFETCHT2,
	"PREFETCHNTA",		LTYPE2,	APREFETCHNTA,
	"UNDEF",	LTYPE0,	AUNDEF,
	"AESENC",	LTYPE3,	AAESENC,
	"AESENCLAST",	LTYPE3, AAESENCLAST,
	"AESDEC",	LTYPE3, AAESDEC,
	"AESDECLAST",	LTYPE3, AAESDECLAST,
	"AESIMC",	LTYPE3, AAESIMC,
	"AESKEYGENASSIST", LTYPEX, AAESKEYGENASSIST,
	"PSHUFD",	LTYPEX, APSHUFD,
	"USEFIELD",	LTYPEN, AUSEFIELD,

	0
};

void
cinit(void)
{
	Sym *s;
	int i;

	nullgen.sym = S;
	nullgen.offset = 0;
	if(FPCHIP)
		nullgen.dval = 0;
	for(i=0; i<sizeof(nullgen.sval); i++)
		nullgen.sval[i] = 0;
	nullgen.type = D_NONE;
	nullgen.index = D_NONE;
	nullgen.scale = 0;

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

	pathname = allocn(pathname, 0, 100);
	if(getwd(pathname, 99) == 0) {
		pathname = allocn(pathname, 100, 900);
		if(getwd(pathname, 999) == 0)
			strcpy(pathname, "/???");
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
	Gen2 g2;

	g2.from = nullgen;
	g2.to = nullgen;
	outcode(AEND, &g2);
	Bflush(&obuf);
}

void
zname(char *n, int t, int s)
{

	Bputc(&obuf, ANAME);		/* as(2) */
	Bputc(&obuf, ANAME>>8);
	Bputc(&obuf, t);		/* type */
	Bputc(&obuf, s);		/* sym */
	while(*n) {
		Bputc(&obuf, *n);
		n++;
	}
	Bputc(&obuf, 0);
}

void
zaddr(Gen *a, int s)
{
	int32 l;
	int i, t;
	char *n;
	Ieee e;

	t = 0;
	if(a->index != D_NONE || a->scale != 0)
		t |= T_INDEX;
	if(a->offset != 0) {
		t |= T_OFFSET;
		l = a->offset;
		if((vlong)l != a->offset)
			t |= T_64;
	}
	if(s != 0)
		t |= T_SYM;

	switch(a->type) {
	default:
		t |= T_TYPE;
		break;
	case D_FCONST:
		t |= T_FCONST;
		break;
	case D_SCONST:
		t |= T_SCONST;
		break;
	case D_NONE:
		break;
	}
	Bputc(&obuf, t);

	if(t & T_INDEX) {	/* implies index, scale */
		Bputc(&obuf, a->index);
		Bputc(&obuf, a->scale);
	}
	if(t & T_OFFSET) {	/* implies offset */
		l = a->offset;
		Bputc(&obuf, l);
		Bputc(&obuf, l>>8);
		Bputc(&obuf, l>>16);
		Bputc(&obuf, l>>24);
		if(t & T_64) {
			l = a->offset>>32;
			Bputc(&obuf, l);
			Bputc(&obuf, l>>8);
			Bputc(&obuf, l>>16);
			Bputc(&obuf, l>>24);
		}
	}
	if(t & T_SYM)		/* implies sym */
		Bputc(&obuf, s);
	if(t & T_FCONST) {
		ieeedtod(&e, a->dval);
		l = e.l;
		Bputc(&obuf, l);
		Bputc(&obuf, l>>8);
		Bputc(&obuf, l>>16);
		Bputc(&obuf, l>>24);
		l = e.h;
		Bputc(&obuf, l);
		Bputc(&obuf, l>>8);
		Bputc(&obuf, l>>16);
		Bputc(&obuf, l>>24);
		return;
	}
	if(t & T_SCONST) {
		n = a->sval;
		for(i=0; i<NSNAME; i++) {
			Bputc(&obuf, *n);
			n++;
		}
		return;
	}
	if(t & T_TYPE)
		Bputc(&obuf, a->type);
}

void
outcode(int a, Gen2 *g2)
{
	int sf, st, t;
	Sym *s;

	if(pass == 1)
		goto out;

jackpot:
	sf = 0;
	s = g2->from.sym;
	while(s != S) {
		sf = s->sym;
		if(sf < 0 || sf >= NSYM)
			sf = 0;
		t = g2->from.type;
		if(t == D_ADDR)
			t = g2->from.index;
		if(h[sf].type == t)
		if(h[sf].sym == s)
			break;
		zname(s->name, t, sym);
		s->sym = sym;
		h[sym].sym = s;
		h[sym].type = t;
		sf = sym;
		sym++;
		if(sym >= NSYM)
			sym = 1;
		break;
	}
	st = 0;
	s = g2->to.sym;
	while(s != S) {
		st = s->sym;
		if(st < 0 || st >= NSYM)
			st = 0;
		t = g2->to.type;
		if(t == D_ADDR)
			t = g2->to.index;
		if(h[st].type == t)
		if(h[st].sym == s)
			break;
		zname(s->name, t, sym);
		s->sym = sym;
		h[sym].sym = s;
		h[sym].type = t;
		st = sym;
		sym++;
		if(sym >= NSYM)
			sym = 1;
		if(st == sf)
			goto jackpot;
		break;
	}
	Bputc(&obuf, a);
	Bputc(&obuf, a>>8);
	Bputc(&obuf, stmtline);
	Bputc(&obuf, stmtline>>8);
	Bputc(&obuf, stmtline>>16);
	Bputc(&obuf, stmtline>>24);
	zaddr(&g2->from, sf);
	zaddr(&g2->to, st);

out:
	if(a != AGLOBL && a != ADATA)
		pc++;
}

void
outhist(void)
{
	Gen g;
	Hist *h;
	char *p, *q, *op, c;
	int n;
	char *tofree;
	static int first = 1;
	static char *goroot, *goroot_final;

	if(first) {
		// Decide whether we need to rewrite paths from $GOROOT to $GOROOT_FINAL.
		first = 0;
		goroot = getenv("GOROOT");
		goroot_final = getenv("GOROOT_FINAL");
		if(goroot == nil)
			goroot = "";
		if(goroot_final == nil)
			goroot_final = goroot;
		if(strcmp(goroot, goroot_final) == 0) {
			goroot = nil;
			goroot_final = nil;
		}
	}

	tofree = nil;

	g = nullgen;
	c = pathchar();
	for(h = hist; h != H; h = h->link) {
		p = h->name;
		if(p != nil && goroot != nil) {
			n = strlen(goroot);
			if(strncmp(p, goroot, strlen(goroot)) == 0 && p[n] == '/') {
				tofree = smprint("%s%s", goroot_final, p+n);
				p = tofree;
			}
		}
		op = 0;
		if(systemtype(Windows) && p && p[1] == ':'){
			c = p[2];
		} else if(p && p[0] != c && h->offset == 0 && pathname){
			if(systemtype(Windows) && pathname[1] == ':') {
				op = p;
				p = pathname;
				c = p[2];
			} else if(pathname[0] == c){
				op = p;
				p = pathname;
			}
		}
		while(p) {
			q = strchr(p, c);
			if(q) {
				n = q-p;
				if(n == 0){
					n = 1;	/* leading "/" */
					*p = '/';	/* don't emit "\" on windows */
				}
				q++;
			} else {
				n = strlen(p);
				q = 0;
			}
			if(n) {
				Bputc(&obuf, ANAME);
				Bputc(&obuf, ANAME>>8);
				Bputc(&obuf, D_FILE);	/* type */
				Bputc(&obuf, 1);	/* sym */
				Bputc(&obuf, '<');
				Bwrite(&obuf, p, n);
				Bputc(&obuf, 0);
			}
			p = q;
			if(p == 0 && op) {
				p = op;
				op = 0;
			}
		}
		g.offset = h->offset;

		Bputc(&obuf, AHISTORY);
		Bputc(&obuf, AHISTORY>>8);
		Bputc(&obuf, h->line);
		Bputc(&obuf, h->line>>8);
		Bputc(&obuf, h->line>>16);
		Bputc(&obuf, h->line>>24);
		zaddr(&nullgen, 0);
		zaddr(&g, 0);

		if(tofree) {
			free(tofree);
			tofree = nil;
		}
	}
}

#include "../cc/lexbody"
#include "../cc/macbody"
