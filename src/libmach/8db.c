// Inferno libmach/8db.c
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/8db.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.
//	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
//	Portions Copyright © 1997-1999 Vita Nuova Limited.
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
//	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
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

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <mach.h>
#define Ureg UregAmd64
#include <ureg_amd64.h>
#undef Ureg
#define Ureg Ureg386
#include <ureg_x86.h>
#undef Ureg

typedef struct UregAmd64 UregAmd64;
typedef struct Ureg386 Ureg386;

/*
 * i386-specific debugger interface
 * also amd64 extensions
 */

static	char	*i386excep(Map*, Rgetter);

static	int	i386trace(Map*, uvlong, uvlong, uvlong, Tracer);
static	uvlong	i386frame(Map*, uvlong, uvlong, uvlong, uvlong);
static	int	i386foll(Map*, uvlong, Rgetter, uvlong*);
static	int	i386inst(Map*, uvlong, char, char*, int);
static	int	i386das(Map*, uvlong, char*, int);
static	int	i386instlen(Map*, uvlong);

static	char	STARTSYM[] =	"_main";
static	char	GOSTARTSYM[] =	"sys·goexit";
static	char	PROFSYM[] =	"_mainp";
static	char	FRAMENAME[] =	".frame";
static	char	LESSSTACK[] = "sys·lessstack";
static	char	MORESTACK[] = "sys·morestack";
static char *excname[] =
{
[0] =	"divide error",
[1] =	"debug exception",
[4] =	"overflow",
[5] =	"bounds check",
[6] =	"invalid opcode",
[7] =	"math coprocessor emulation",
[8] =	"double fault",
[9] =	"math coprocessor overrun",
[10] =	"invalid TSS",
[11] =	"segment not present",
[12] =	"stack exception",
[13] =	"general protection violation",
[14] =	"page fault",
[16] =	"math coprocessor error",
[17] =	"alignment check",
[18] =	"machine check",
[19] =	"floating-point exception",
[24] =	"clock",
[25] =	"keyboard",
[27] =	"modem status",
[28] =	"serial line status",
[30] =	"floppy disk",
[36] =	"mouse",
[37] =	"math coprocessor",
[38] =	"hard disk",
[64] =	"system call",
};

Machdata i386mach =
{
	{0xCC, 0, 0, 0},	/* break point: INT 3 */
	1,			/* break point size */

	leswab,			/* convert short to local byte order */
	leswal,			/* convert int32 to local byte order */
	leswav,			/* convert vlong to local byte order */
	i386trace,		/* C traceback */
	i386frame,		/* frame finder */
	i386excep,		/* print exception */
	0,			/* breakpoint fixup */
	leieeesftos,		/* single precision float printer */
	leieeedftos,		/* double precision float printer */
	i386foll,		/* following addresses */
	i386inst,		/* print instruction */
	i386das,		/* dissembler */
	i386instlen,		/* instruction size calculation */
};

static char*
i386excep(Map *map, Rgetter rget)
{
	uint32 c;
	uvlong pc;
	static char buf[16];

	c = (*rget)(map, "TRAP");
	if(c > 64 || excname[c] == 0) {
		if (c == 3) {
			pc = (*rget)(map, "PC");
			if (get1(map, pc, (uchar*)buf, machdata->bpsize) > 0)
			if (memcmp(buf, machdata->bpinst, machdata->bpsize) == 0)
				return "breakpoint";
		}
		snprint(buf, sizeof(buf), "exception %d", c);
		return buf;
	} else
		return excname[c];
}

static int
i386trace(Map *map, uvlong pc, uvlong sp, uvlong link, Tracer trace)
{
	int i;
	uvlong osp, pc1;
	Symbol s, f, s1;
	extern Mach mamd64;
	int isamd64;
	uvlong g, m, lessstack, morestack, stktop;

	isamd64 = (mach == &mamd64);

	// ../pkg/runtime/runtime.h
	// G is
	//	byte* stackguard
	//	byte* stackbase (= Stktop*)
	// TODO(rsc): Need some way to get at the g for other threads.
	// Probably need to pass it into the trace function.
	g = 0;
	if(isamd64)
		geta(map, offsetof(struct UregAmd64, r15), &g);
	else {
		// TODO(rsc): How to fetch g on 386?
	}
	stktop = 0;
	if(g != 0)
		geta(map, g+1*mach->szaddr, &stktop);

	lessstack = 0;
	if(lookup(0, LESSSTACK, &s))
		lessstack = s.value;
	morestack = 0;
	if(lookup(0, MORESTACK, &s))
		morestack = s.value;

	USED(link);
	osp = 0;
	i = 0;

	for(;;) {
		if(!findsym(pc, CTEXT, &s)) {
			// check for closure return sequence
			uchar buf[8], *p;
			if(get1(map, pc, buf, 8) < 0)
				break;
			// ADDQ $xxx, SP; RET
			p = buf;
			if(mach == &mamd64) {
				if(p[0] != 0x48)
					break;
				p++;
			}
			if(p[0] != 0x81 || p[1] != 0xc4 || p[6] != 0xc3)
				break;
			sp += p[2] | (p[3]<<8) | (p[4]<<16) | (p[5]<<24);
			if(geta(map, sp, &pc) < 0)
				break;
			sp += mach->szaddr;
			continue;
		}

		if (osp == sp)
			break;
		osp = sp;

		if(strcmp(STARTSYM, s.name) == 0 ||
		   strcmp(GOSTARTSYM, s.name) == 0 ||
		   strcmp(PROFSYM, s.name) == 0)
			break;

		if(s.value == morestack) {
			// In the middle of morestack.
			// Caller is m->morepc.
			// Caller's caller is in m->morearg.
			// TODO(rsc): 386
			geta(map, offsetof(struct UregAmd64, r14), &m);

			pc = 0;
			sp = 0;
			pc1 = 0;
			s1 = s;
			memset(&s, 0, sizeof s);
			geta(map, m+1*mach->szaddr, &pc1);	// m->morepc
			geta(map, m+2*mach->szaddr, &sp);	// m->morebuf.sp
			geta(map, m+3*mach->szaddr, &pc);	// m->morebuf.pc
			findsym(pc1, CTEXT, &s);
			(*trace)(map, pc1, sp-mach->szaddr, &s1);	// morestack symbol; caller's PC/SP

			// caller's caller
			s1 = s;
			findsym(pc, CTEXT, &s);
			(*trace)(map, pc, sp, &s1);		// morestack's caller; caller's caller's PC/SP
			continue;
		}

		if(pc == lessstack) {
			// ../pkg/runtime/runtime.h
			// Stktop is
			//	byte* stackguard
			//	byte* stackbase
			//	Gobuf gobuf
			//		byte* sp;
			//		byte* pc;
			//		G*	g;
			if(!isamd64)
				fprint(2, "warning: cannot unwind stack split on 386\n");
			if(stktop == 0)
				break;
			pc = 0;
			sp = 0;
			geta(map, stktop+2*mach->szaddr, &sp);
			geta(map, stktop+3*mach->szaddr, &pc);
			geta(map, stktop+1*mach->szaddr, &stktop);
			(*trace)(map, pc, sp, &s1);
			continue;
		}

		s1 = s;
		pc1 = 0;
		if(pc != s.value) {	/* not at first instruction */
			if(findlocal(&s, FRAMENAME, &f) == 0)
				break;
			geta(map, sp, &pc1);
			sp += f.value-mach->szaddr;
		}
		if(geta(map, sp, &pc) < 0)
			break;

		// If PC is not valid, assume we caught the function
		// before it moved the stack pointer down or perhaps
		// after it moved the stack pointer back up.
		// Try the PC we'd have gotten without the stack
		// pointer adjustment above (pc != s.value).
		// This only matters for the first frame, and it is only
		// a heuristic, but it does help.
		if(!findsym(pc, CTEXT, &s) || strcmp(s.name, "etext") == 0)
			pc = pc1;

		if(pc == 0)
			break;

		if(pc != lessstack)
			(*trace)(map, pc, sp, &s1);
		sp += mach->szaddr;

		if(++i > 1000)
			break;
	}
	return i;
}

static uvlong
i386frame(Map *map, uvlong addr, uvlong pc, uvlong sp, uvlong link)
{
	Symbol s, f;

	USED(link);
	while (findsym(pc, CTEXT, &s)) {
		if(strcmp(STARTSYM, s.name) == 0 || strcmp(PROFSYM, s.name) == 0)
			break;

		if(pc != s.value) {	/* not first instruction */
			if(findlocal(&s, FRAMENAME, &f) == 0)
				break;
			sp += f.value-mach->szaddr;
		}

		if (s.value == addr)
			return sp;

		if (geta(map, sp, &pc) < 0)
			break;
		sp += mach->szaddr;
	}
	return 0;
}

	/* I386/486 - Disassembler and related functions */

/*
 *  an instruction
 */
typedef struct Instr Instr;
struct	Instr
{
	uchar	mem[1+1+1+1+2+1+1+4+4];		/* raw instruction */
	uvlong	addr;		/* address of start of instruction */
	int	n;		/* number of bytes in instruction */
	char	*prefix;	/* instr prefix */
	char	*segment;	/* segment override */
	uchar	jumptype;	/* set to the operand type for jump/ret/call */
	uchar	amd64;
	uchar	rex;		/* REX prefix (or zero) */
	char	osize;		/* 'W' or 'L' (or 'Q' on amd64) */
	char	asize;		/* address size 'W' or 'L' (or 'Q' or amd64) */
	uchar	mod;		/* bits 6-7 of mod r/m field */
	uchar	reg;		/* bits 3-5 of mod r/m field */
	char	ss;		/* bits 6-7 of SIB */
	schar	index;		/* bits 3-5 of SIB */
	schar	base;		/* bits 0-2 of SIB */
	char	rip;		/* RIP-relative in amd64 mode */
	uchar	opre;		/* f2/f3 could introduce media */
	short	seg;		/* segment of far address */
	uint32	disp;		/* displacement */
	uint32 	imm;		/* immediate */
	uint32 	imm2;		/* second immediate operand */
	uvlong	imm64;		/* big immediate */
	char	*curr;		/* fill level in output buffer */
	char	*end;		/* end of output buffer */
	char	*err;		/* error message */
};

	/* 386 register (ha!) set */
enum{
	AX=0,
	CX,
	DX,
	BX,
	SP,
	BP,
	SI,
	DI,

	/* amd64 */
	/* be careful: some unix system headers #define R8, R9, etc */
	AMD64_R8,
	AMD64_R9,
	AMD64_R10,
	AMD64_R11,
	AMD64_R12,
	AMD64_R13,
	AMD64_R14,
	AMD64_R15
};

	/* amd64 rex extension byte */
enum{
	REXW		= 1<<3,	/* =1, 64-bit operand size */
	REXR		= 1<<2,	/* extend modrm reg */
	REXX		= 1<<1,	/* extend sib index */
	REXB		= 1<<0	/* extend modrm r/m, sib base, or opcode reg */
};

	/* Operand Format codes */
/*
%A	-	address size register modifier (!asize -> 'E')
%C	-	Control register CR0/CR1/CR2
%D	-	Debug register DR0/DR1/DR2/DR3/DR6/DR7
%I	-	second immediate operand
%O	-	Operand size register modifier (!osize -> 'E')
%T	-	Test register TR6/TR7
%S	-	size code ('W' or 'L')
%W	-	Weird opcode: OSIZE == 'W' => "CBW"; else => "CWDE"
%d	-	displacement 16-32 bits
%e	-	effective address - Mod R/M value
%f	-	floating point register F0-F7 - from Mod R/M register
%g	-	segment register
%i	-	immediate operand 8-32 bits
%p	-	PC-relative - signed displacement in immediate field
%r	-	Reg from Mod R/M
%w	-	Weird opcode: OSIZE == 'W' => "CWD"; else => "CDQ"
*/

typedef struct Optable Optable;
struct Optable
{
	char	operand[2];
	void	*proto;		/* actually either (char*) or (Optable*) */
};
	/* Operand decoding codes */
enum {
	Ib = 1,			/* 8-bit immediate - (no sign extension)*/
	Ibs,			/* 8-bit immediate (sign extended) */
	Jbs,			/* 8-bit sign-extended immediate in jump or call */
	Iw,			/* 16-bit immediate -> imm */
	Iw2,			/* 16-bit immediate -> imm2 */
	Iwd,			/* Operand-sized immediate (no sign extension)*/
	Iwdq,			/* Operand-sized immediate, possibly 64 bits */
	Awd,			/* Address offset */
	Iwds,			/* Operand-sized immediate (sign extended) */
	RM,			/* Word or int32 R/M field with register (/r) */
	RMB,			/* Byte R/M field with register (/r) */
	RMOP,			/* Word or int32 R/M field with op code (/digit) */
	RMOPB,			/* Byte R/M field with op code (/digit) */
	RMR,			/* R/M register only (mod = 11) */
	RMM,			/* R/M memory only (mod = 0/1/2) */
	Op_R0,			/* Base reg of Mod R/M is literal 0x00 */
	Op_R1,			/* Base reg of Mod R/M is literal 0x01 */
	FRMOP,			/* Floating point R/M field with opcode */
	FRMEX,			/* Extended floating point R/M field with opcode */
	JUMP,			/* Jump or Call flag - no operand */
	RET,			/* Return flag - no operand */
	OA,			/* literal 0x0a byte */
	PTR,			/* Seg:Displacement addr (ptr16:16 or ptr16:32) */
	AUX,			/* Multi-byte op code - Auxiliary table */
	AUXMM,			/* multi-byte op code - auxiliary table chosen by prefix */
	PRE,			/* Instr Prefix */
	OPRE,			/* Instr Prefix or media op extension */
	SEG,			/* Segment Prefix */
	OPOVER,			/* Operand size override */
	ADDOVER,		/* Address size override */
};

static Optable optab0F00[8]=
{
[0x00] =	{ 0,0,		"MOVW	LDT,%e" },
[0x01] =	{ 0,0,		"MOVW	TR,%e" },
[0x02] =	{ 0,0,		"MOVW	%e,LDT" },
[0x03] =	{ 0,0,		"MOVW	%e,TR" },
[0x04] =	{ 0,0,		"VERR	%e" },
[0x05] =	{ 0,0,		"VERW	%e" },
};

static Optable optab0F01[8]=
{
[0x00] =	{ 0,0,		"MOVL	GDTR,%e" },
[0x01] =	{ 0,0,		"MOVL	IDTR,%e" },
[0x02] =	{ 0,0,		"MOVL	%e,GDTR" },
[0x03] =	{ 0,0,		"MOVL	%e,IDTR" },
[0x04] =	{ 0,0,		"MOVW	MSW,%e" },	/* word */
[0x06] =	{ 0,0,		"MOVW	%e,MSW" },	/* word */
[0x07] =	{ 0,0,		"INVLPG	%e" },		/* or SWAPGS */
};

static Optable optab0F01F8[1]=
{
[0x00] =	{ 0,0,		"SWAPGS" },
};

/* 0F71 */
/* 0F72 */
/* 0F73 */

static Optable optab0FAE[8]=
{
[0x00] =	{ 0,0,		"FXSAVE	%e" },
[0x01] =	{ 0,0,		"FXRSTOR	%e" },
[0x02] =	{ 0,0,		"LDMXCSR	%e" },
[0x03] =	{ 0,0,		"STMXCSR	%e" },
[0x05] =	{ 0,0,		"LFENCE" },
[0x06] =	{ 0,0,		"MFENCE" },
[0x07] =	{ 0,0,		"SFENCE" },
};

/* 0F18 */
/* 0F0D */

static Optable optab0FBA[8]=
{
[0x04] =	{ Ib,0,		"BT%S	%i,%e" },
[0x05] =	{ Ib,0,		"BTS%S	%i,%e" },
[0x06] =	{ Ib,0,		"BTR%S	%i,%e" },
[0x07] =	{ Ib,0,		"BTC%S	%i,%e" },
};

static Optable optab0F0F[256]=
{
[0x0c] =	{ 0,0,		"PI2FW	%m,%M" },
[0x0d] =	{ 0,0,		"PI2L	%m,%M" },
[0x1c] =	{ 0,0,		"PF2IW	%m,%M" },
[0x1d] =	{ 0,0,		"PF2IL	%m,%M" },
[0x8a] =	{ 0,0,		"PFNACC	%m,%M" },
[0x8e] =	{ 0,0,		"PFPNACC	%m,%M" },
[0x90] =	{ 0,0,		"PFCMPGE	%m,%M" },
[0x94] =	{ 0,0,		"PFMIN	%m,%M" },
[0x96] =	{ 0,0,		"PFRCP	%m,%M" },
[0x97] =	{ 0,0,		"PFRSQRT	%m,%M" },
[0x9a] =	{ 0,0,		"PFSUB	%m,%M" },
[0x9e] =	{ 0,0,		"PFADD	%m,%M" },
[0xa0] =	{ 0,0,		"PFCMPGT	%m,%M" },
[0xa4] =	{ 0,0,		"PFMAX	%m,%M" },
[0xa6] =	{ 0,0,		"PFRCPIT1	%m,%M" },
[0xa7] =	{ 0,0,		"PFRSQIT1	%m,%M" },
[0xaa] =	{ 0,0,		"PFSUBR	%m,%M" },
[0xae] =	{ 0,0,		"PFACC	%m,%M" },
[0xb0] =	{ 0,0,		"PFCMPEQ	%m,%M" },
[0xb4] =	{ 0,0,		"PFMUL	%m,%M" },
[0xb6] =	{ 0,0,		"PFRCPI2T	%m,%M" },
[0xb7] =	{ 0,0,		"PMULHRW	%m,%M" },
[0xbb] =	{ 0,0,		"PSWAPL	%m,%M" },
};

static Optable optab0FC7[8]=
{
[0x01] =	{ 0,0,		"CMPXCHG8B	%e" },
};

static Optable optab660F71[8]=
{
[0x02] =	{ Ib,0,		"PSRLW	%i,%X" },
[0x04] =	{ Ib,0,		"PSRAW	%i,%X" },
[0x06] =	{ Ib,0,		"PSLLW	%i,%X" },
};

static Optable optab660F72[8]=
{
[0x02] =	{ Ib,0,		"PSRLL	%i,%X" },
[0x04] =	{ Ib,0,		"PSRAL	%i,%X" },
[0x06] =	{ Ib,0,		"PSLLL	%i,%X" },
};

static Optable optab660F73[8]=
{
[0x02] =	{ Ib,0,		"PSRLQ	%i,%X" },
[0x03] =	{ Ib,0,		"PSRLO	%i,%X" },
[0x06] =	{ Ib,0,		"PSLLQ	%i,%X" },
[0x07] =	{ Ib,0,		"PSLLO	%i,%X" },
};

static Optable optab660F[256]=
{
[0x2B] =	{ RM,0,		"MOVNTPD	%x,%e" },
[0x2E] =	{ RM,0,		"UCOMISD	%x,%X" },
[0x2F] =	{ RM,0,		"COMISD	%x,%X" },
[0x5A] =	{ RM,0,		"CVTPD2PS	%x,%X" },
[0x5B] =	{ RM,0,		"CVTPS2PL	%x,%X" },
[0x6A] =	{ RM,0,		"PUNPCKHLQ %x,%X" },
[0x6B] =	{ RM,0,		"PACKSSLW %x,%X" },
[0x6C] =	{ RM,0,		"PUNPCKLQDQ %x,%X" },
[0x6D] =	{ RM,0,		"PUNPCKHQDQ %x,%X" },
[0x6E] =	{ RM,0,		"MOV%S	%e,%X" },
[0x6F] =	{ RM,0,		"MOVO	%x,%X" },		/* MOVDQA */
[0x70] =	{ RM,Ib,		"PSHUFL	%i,%x,%X" },
[0x71] =	{ RMOP,0,		optab660F71 },
[0x72] =	{ RMOP,0,		optab660F72 },
[0x73] =	{ RMOP,0,		optab660F73 },
[0x7E] =	{ RM,0,		"MOV%S	%X,%e" },
[0x7F] =	{ RM,0,		"MOVO	%X,%x" },
[0xC4] =	{ RM,Ib,		"PINSRW	%i,%e,%X" },
[0xC5] =	{ RMR,Ib,		"PEXTRW	%i,%X,%e" },
[0xD4] =	{ RM,0,		"PADDQ	%x,%X" },
[0xD5] =	{ RM,0,		"PMULLW	%x,%X" },
[0xD6] =	{ RM,0,		"MOVQ	%X,%x" },
[0xE6] =	{ RM,0,		"CVTTPD2PL	%x,%X" },
[0xE7] =	{ RM,0,		"MOVNTO	%X,%e" },
[0xF7] =	{ RM,0,		"MASKMOVOU	%x,%X" },
};

static Optable optabF20F[256]=
{
[0x10] =	{ RM,0,		"MOVSD	%x,%X" },
[0x11] =	{ RM,0,		"MOVSD	%X,%x" },
[0x2A] =	{ RM,0,		"CVTS%S2SD	%e,%X" },
[0x2C] =	{ RM,0,		"CVTTSD2S%S	%x,%r" },
[0x2D] =	{ RM,0,		"CVTSD2S%S	%x,%r" },
[0x5A] =	{ RM,0,		"CVTSD2SS	%x,%X" },
[0x6F] =	{ RM,0,		"MOVOU	%x,%X" },
[0x70] =	{ RM,Ib,		"PSHUFLW	%i,%x,%X" },
[0x7F] =	{ RM,0,		"MOVOU	%X,%x" },
[0xD6] =	{ RM,0,		"MOVQOZX	%M,%X" },
[0xE6] =	{ RM,0,		"CVTPD2PL	%x,%X" },
};

static Optable optabF30F[256]=
{
[0x10] =	{ RM,0,		"MOVSS	%x,%X" },
[0x11] =	{ RM,0,		"MOVSS	%X,%x" },
[0x2A] =	{ RM,0,		"CVTS%S2SS	%e,%X" },
[0x2C] =	{ RM,0,		"CVTTSS2S%S	%x,%r" },
[0x2D] =	{ RM,0,		"CVTSS2S%S	%x,%r" },
[0x5A] =	{ RM,0,		"CVTSS2SD	%x,%X" },
[0x5B] =	{ RM,0,		"CVTTPS2PL	%x,%X" },
[0x6F] =	{ RM,0,		"MOVOU	%x,%X" },
[0x70] =	{ RM,Ib,		"PSHUFHW	%i,%x,%X" },
[0x7E] =	{ RM,0,		"MOVQOZX	%x,%X" },
[0x7F] =	{ RM,0,		"MOVOU	%X,%x" },
[0xD6] =	{ RM,0,		"MOVQOZX	%m*,%X" },
[0xE6] =	{ RM,0,		"CVTPL2PD	%x,%X" },
};

static Optable optab0F[256]=
{
[0x00] =	{ RMOP,0,		optab0F00 },
[0x01] =	{ RMOP,0,		optab0F01 },
[0x02] =	{ RM,0,		"LAR	%e,%r" },
[0x03] =	{ RM,0,		"LSL	%e,%r" },
[0x05] =	{ 0,0,		"SYSCALL" },
[0x06] =	{ 0,0,		"CLTS" },
[0x07] =	{ 0,0,		"SYSRET" },
[0x08] =	{ 0,0,		"INVD" },
[0x09] =	{ 0,0,		"WBINVD" },
[0x0B] =	{ 0,0,		"UD2" },
[0x0F] =	{ RM,AUX,		optab0F0F },		/* 3DNow! */
[0x10] =	{ RM,0,		"MOVU%s	%x,%X" },
[0x11] =	{ RM,0,		"MOVU%s	%X,%x" },
[0x12] =	{ RM,0,		"MOV[H]L%s	%x,%X" },	/* TO DO: H if source is XMM */
[0x13] =	{ RM,0,		"MOVL%s	%X,%e" },
[0x14] =	{ RM,0,		"UNPCKL%s	%x,%X" },
[0x15] =	{ RM,0,		"UNPCKH%s	%x,%X" },
[0x16] =	{ RM,0,		"MOV[L]H%s	%x,%X" },	/* TO DO: L if source is XMM */
[0x17] =	{ RM,0,		"MOVH%s	%X,%x" },
[0x1F] =	{ RM,0,		"NOP%S	%e" },
[0x20] =	{ RMR,0,		"MOVL	%C,%e" },
[0x21] =	{ RMR,0,		"MOVL	%D,%e" },
[0x22] =	{ RMR,0,		"MOVL	%e,%C" },
[0x23] =	{ RMR,0,		"MOVL	%e,%D" },
[0x24] =	{ RMR,0,		"MOVL	%T,%e" },
[0x26] =	{ RMR,0,		"MOVL	%e,%T" },
[0x28] =	{ RM,0,		"MOVA%s	%x,%X" },
[0x29] =	{ RM,0,		"MOVA%s	%X,%x" },
[0x2A] =	{ RM,0,		"CVTPL2%s	%m*,%X" },
[0x2B] =	{ RM,0,		"MOVNT%s	%X,%e" },
[0x2C] =	{ RM,0,		"CVTT%s2PL	%x,%M" },
[0x2D] =	{ RM,0,		"CVT%s2PL	%x,%M" },
[0x2E] =	{ RM,0,		"UCOMISS	%x,%X" },
[0x2F] =	{ RM,0,		"COMISS	%x,%X" },
[0x30] =	{ 0,0,		"WRMSR" },
[0x31] =	{ 0,0,		"RDTSC" },
[0x32] =	{ 0,0,		"RDMSR" },
[0x33] =	{ 0,0,		"RDPMC" },
[0x42] =	{ RM,0,		"CMOVC	%e,%r" },		/* CF */
[0x43] =	{ RM,0,		"CMOVNC	%e,%r" },		/* ¬ CF */
[0x44] =	{ RM,0,		"CMOVZ	%e,%r" },		/* ZF */
[0x45] =	{ RM,0,		"CMOVNZ	%e,%r" },		/* ¬ ZF */
[0x46] =	{ RM,0,		"CMOVBE	%e,%r" },		/* CF ∨ ZF */
[0x47] =	{ RM,0,		"CMOVA	%e,%r" },		/* ¬CF ∧ ¬ZF */
[0x48] =	{ RM,0,		"CMOVS	%e,%r" },		/* SF */
[0x49] =	{ RM,0,		"CMOVNS	%e,%r" },		/* ¬ SF */
[0x4A] =	{ RM,0,		"CMOVP	%e,%r" },		/* PF */
[0x4B] =	{ RM,0,		"CMOVNP	%e,%r" },		/* ¬ PF */
[0x4C] =	{ RM,0,		"CMOVLT	%e,%r" },		/* LT ≡ OF ≠ SF */
[0x4D] =	{ RM,0,		"CMOVGE	%e,%r" },		/* GE ≡ ZF ∨ SF */
[0x4E] =	{ RM,0,		"CMOVLE	%e,%r" },		/* LE ≡ ZF ∨ LT */
[0x4F] =	{ RM,0,		"CMOVGT	%e,%r" },		/* GT ≡ ¬ZF ∧ GE */
[0x50] =	{ RM,0,		"MOVMSK%s	%X,%r" },	/* TO DO: check */
[0x51] =	{ RM,0,		"SQRT%s	%x,%X" },
[0x52] =	{ RM,0,		"RSQRT%s	%x,%X" },
[0x53] =	{ RM,0,		"RCP%s	%x,%X" },
[0x54] =	{ RM,0,		"AND%s	%x,%X" },
[0x55] =	{ RM,0,		"ANDN%s	%x,%X" },
[0x56] =	{ RM,0,		"OR%s	%x,%X" },		/* TO DO: S/D */
[0x57] =	{ RM,0,		"XOR%s	%x,%X" },		/* S/D */
[0x58] =	{ RM,0,		"ADD%s	%x,%X" },		/* S/P S/D */
[0x59] =	{ RM,0,		"MUL%s	%x,%X" },
[0x5A] =	{ RM,0,		"CVTPS2PD	%x,%X" },
[0x5B] =	{ RM,0,		"CVTPL2PS	%x,%X" },
[0x5C] =	{ RM,0,		"SUB%s	%x,%X" },
[0x5D] =	{ RM,0,		"MIN%s	%x,%X" },
[0x5E] =	{ RM,0,		"DIV%s	%x,%X" },		/* TO DO: S/P S/D */
[0x5F] =	{ RM,0,		"MAX%s	%x,%X" },
[0x60] =	{ RM,0,		"PUNPCKLBW %m,%M" },
[0x61] =	{ RM,0,		"PUNPCKLWL %m,%M" },
[0x62] =	{ RM,0,		"PUNPCKLLQ %m,%M" },
[0x63] =	{ RM,0,		"PACKSSWB %m,%M" },
[0x64] =	{ RM,0,		"PCMPGTB %m,%M" },
[0x65] =	{ RM,0,		"PCMPGTW %m,%M" },
[0x66] =	{ RM,0,		"PCMPGTL %m,%M" },
[0x67] =	{ RM,0,		"PACKUSWB %m,%M" },
[0x68] =	{ RM,0,		"PUNPCKHBW %m,%M" },
[0x69] =	{ RM,0,		"PUNPCKHWL %m,%M" },
[0x6A] =	{ RM,0,		"PUNPCKHLQ %m,%M" },
[0x6B] =	{ RM,0,		"PACKSSLW %m,%M" },
[0x6E] =	{ RM,0,		"MOV%S %e,%M" },
[0x6F] =	{ RM,0,		"MOVQ %m,%M" },
[0x70] =	{ RM,Ib,		"PSHUFW	%i,%m,%M" },
[0x74] =	{ RM,0,		"PCMPEQB %m,%M" },
[0x75] =	{ RM,0,		"PCMPEQW %m,%M" },
[0x76] =	{ RM,0,		"PCMPEQL %m,%M" },
[0x77] =	{ 0,0,		"EMMS" },
[0x7E] =	{ RM,0,		"MOV%S %M,%e" },
[0x7F] =	{ RM,0,		"MOVQ %M,%m" },
[0xAE] =	{ RMOP,0,		optab0FAE },
[0xAA] =	{ 0,0,		"RSM" },
[0xB0] =	{ RM,0,		"CMPXCHGB	%r,%e" },
[0xB1] =	{ RM,0,		"CMPXCHG%S	%r,%e" },
[0xC0] =	{ RMB,0,		"XADDB	%r,%e" },
[0xC1] =	{ RM,0,		"XADD%S	%r,%e" },
[0xC2] =	{ RM,Ib,		"CMP%s	%x,%X,%#i" },
[0xC3] =	{ RM,0,		"MOVNTI%S	%r,%e" },
[0xC6] =	{ RM,Ib,		"SHUF%s	%i,%x,%X" },
[0xC8] =	{ 0,0,		"BSWAP	AX" },
[0xC9] =	{ 0,0,		"BSWAP	CX" },
[0xCA] =	{ 0,0,		"BSWAP	DX" },
[0xCB] =	{ 0,0,		"BSWAP	BX" },
[0xCC] =	{ 0,0,		"BSWAP	SP" },
[0xCD] =	{ 0,0,		"BSWAP	BP" },
[0xCE] =	{ 0,0,		"BSWAP	SI" },
[0xCF] =	{ 0,0,		"BSWAP	DI" },
[0xD1] =	{ RM,0,		"PSRLW %m,%M" },
[0xD2] =	{ RM,0,		"PSRLL %m,%M" },
[0xD3] =	{ RM,0,		"PSRLQ %m,%M" },
[0xD5] =	{ RM,0,		"PMULLW %m,%M" },
[0xD6] =	{ RM,0,		"MOVQOZX	%m*,%X" },
[0xD7] =	{ RM,0,		"PMOVMSKB %m,%r" },
[0xD8] =	{ RM,0,		"PSUBUSB %m,%M" },
[0xD9] =	{ RM,0,		"PSUBUSW %m,%M" },
[0xDA] =	{ RM,0,		"PMINUB %m,%M" },
[0xDB] =	{ RM,0,		"PAND %m,%M" },
[0xDC] =	{ RM,0,		"PADDUSB %m,%M" },
[0xDD] =	{ RM,0,		"PADDUSW %m,%M" },
[0xDE] =	{ RM,0,		"PMAXUB %m,%M" },
[0xDF] =	{ RM,0,		"PANDN %m,%M" },
[0xE0] =	{ RM,0,		"PAVGB %m,%M" },
[0xE1] =	{ RM,0,		"PSRAW %m,%M" },
[0xE2] =	{ RM,0,		"PSRAL %m,%M" },
[0xE3] =	{ RM,0,		"PAVGW %m,%M" },
[0xE4] =	{ RM,0,		"PMULHUW %m,%M" },
[0xE5] =	{ RM,0,		"PMULHW %m,%M" },
[0xE7] =	{ RM,0,		"MOVNTQ	%M,%e" },
[0xE8] =	{ RM,0,		"PSUBSB %m,%M" },
[0xE9] =	{ RM,0,		"PSUBSW %m,%M" },
[0xEA] =	{ RM,0,		"PMINSW %m,%M" },
[0xEB] =	{ RM,0,		"POR %m,%M" },
[0xEC] =	{ RM,0,		"PADDSB %m,%M" },
[0xED] =	{ RM,0,		"PADDSW %m,%M" },
[0xEE] =	{ RM,0,		"PMAXSW %m,%M" },
[0xEF] =	{ RM,0,		"PXOR %m,%M" },
[0xF1] =	{ RM,0,		"PSLLW %m,%M" },
[0xF2] =	{ RM,0,		"PSLLL %m,%M" },
[0xF3] =	{ RM,0,		"PSLLQ %m,%M" },
[0xF4] =	{ RM,0,		"PMULULQ	%m,%M" },
[0xF5] =	{ RM,0,		"PMADDWL %m,%M" },
[0xF6] =	{ RM,0,		"PSADBW %m,%M" },
[0xF7] =	{ RMR,0,		"MASKMOVQ	%m,%M" },
[0xF8] =	{ RM,0,		"PSUBB %m,%M" },
[0xF9] =	{ RM,0,		"PSUBW %m,%M" },
[0xFA] =	{ RM,0,		"PSUBL %m,%M" },
[0xFC] =	{ RM,0,		"PADDB %m,%M" },
[0xFD] =	{ RM,0,		"PADDW %m,%M" },
[0xFE] =	{ RM,0,		"PADDL %m,%M" },

[0x80] =	{ Iwds,0,		"JOS	%p" },
[0x81] =	{ Iwds,0,		"JOC	%p" },
[0x82] =	{ Iwds,0,		"JCS	%p" },
[0x83] =	{ Iwds,0,		"JCC	%p" },
[0x84] =	{ Iwds,0,		"JEQ	%p" },
[0x85] =	{ Iwds,0,		"JNE	%p" },
[0x86] =	{ Iwds,0,		"JLS	%p" },
[0x87] =	{ Iwds,0,		"JHI	%p" },
[0x88] =	{ Iwds,0,		"JMI	%p" },
[0x89] =	{ Iwds,0,		"JPL	%p" },
[0x8a] =	{ Iwds,0,		"JPS	%p" },
[0x8b] =	{ Iwds,0,		"JPC	%p" },
[0x8c] =	{ Iwds,0,		"JLT	%p" },
[0x8d] =	{ Iwds,0,		"JGE	%p" },
[0x8e] =	{ Iwds,0,		"JLE	%p" },
[0x8f] =	{ Iwds,0,		"JGT	%p" },
[0x90] =	{ RMB,0,		"SETOS	%e" },
[0x91] =	{ RMB,0,		"SETOC	%e" },
[0x92] =	{ RMB,0,		"SETCS	%e" },
[0x93] =	{ RMB,0,		"SETCC	%e" },
[0x94] =	{ RMB,0,		"SETEQ	%e" },
[0x95] =	{ RMB,0,		"SETNE	%e" },
[0x96] =	{ RMB,0,		"SETLS	%e" },
[0x97] =	{ RMB,0,		"SETHI	%e" },
[0x98] =	{ RMB,0,		"SETMI	%e" },
[0x99] =	{ RMB,0,		"SETPL	%e" },
[0x9a] =	{ RMB,0,		"SETPS	%e" },
[0x9b] =	{ RMB,0,		"SETPC	%e" },
[0x9c] =	{ RMB,0,		"SETLT	%e" },
[0x9d] =	{ RMB,0,		"SETGE	%e" },
[0x9e] =	{ RMB,0,		"SETLE	%e" },
[0x9f] =	{ RMB,0,		"SETGT	%e" },
[0xa0] =	{ 0,0,		"PUSHL	FS" },
[0xa1] =	{ 0,0,		"POPL	FS" },
[0xa2] =	{ 0,0,		"CPUID" },
[0xa3] =	{ RM,0,		"BT%S	%r,%e" },
[0xa4] =	{ RM,Ib,		"SHLD%S	%r,%i,%e" },
[0xa5] =	{ RM,0,		"SHLD%S	%r,CL,%e" },
[0xa8] =	{ 0,0,		"PUSHL	GS" },
[0xa9] =	{ 0,0,		"POPL	GS" },
[0xab] =	{ RM,0,		"BTS%S	%r,%e" },
[0xac] =	{ RM,Ib,		"SHRD%S	%r,%i,%e" },
[0xad] =	{ RM,0,		"SHRD%S	%r,CL,%e" },
[0xaf] =	{ RM,0,		"IMUL%S	%e,%r" },
[0xb2] =	{ RMM,0,		"LSS	%e,%r" },
[0xb3] =	{ RM,0,		"BTR%S	%r,%e" },
[0xb4] =	{ RMM,0,		"LFS	%e,%r" },
[0xb5] =	{ RMM,0,		"LGS	%e,%r" },
[0xb6] =	{ RMB,0,		"MOVBZX	%e,%R" },
[0xb7] =	{ RM,0,		"MOVWZX	%e,%R" },
[0xba] =	{ RMOP,0,		optab0FBA },
[0xbb] =	{ RM,0,		"BTC%S	%e,%r" },
[0xbc] =	{ RM,0,		"BSF%S	%e,%r" },
[0xbd] =	{ RM,0,		"BSR%S	%e,%r" },
[0xbe] =	{ RMB,0,		"MOVBSX	%e,%R" },
[0xbf] =	{ RM,0,		"MOVWSX	%e,%R" },
[0xc7] =	{ RMOP,0,		optab0FC7 },
};

static Optable optab80[8]=
{
[0x00] =	{ Ib,0,		"ADDB	%i,%e" },
[0x01] =	{ Ib,0,		"ORB	%i,%e" },
[0x02] =	{ Ib,0,		"ADCB	%i,%e" },
[0x03] =	{ Ib,0,		"SBBB	%i,%e" },
[0x04] =	{ Ib,0,		"ANDB	%i,%e" },
[0x05] =	{ Ib,0,		"SUBB	%i,%e" },
[0x06] =	{ Ib,0,		"XORB	%i,%e" },
[0x07] =	{ Ib,0,		"CMPB	%e,%i" },
};

static Optable optab81[8]=
{
[0x00] =	{ Iwd,0,		"ADD%S	%i,%e" },
[0x01] =	{ Iwd,0,		"OR%S	%i,%e" },
[0x02] =	{ Iwd,0,		"ADC%S	%i,%e" },
[0x03] =	{ Iwd,0,		"SBB%S	%i,%e" },
[0x04] =	{ Iwd,0,		"AND%S	%i,%e" },
[0x05] =	{ Iwd,0,		"SUB%S	%i,%e" },
[0x06] =	{ Iwd,0,		"XOR%S	%i,%e" },
[0x07] =	{ Iwd,0,		"CMP%S	%e,%i" },
};

static Optable optab83[8]=
{
[0x00] =	{ Ibs,0,		"ADD%S	%i,%e" },
[0x01] =	{ Ibs,0,		"OR%S	%i,%e" },
[0x02] =	{ Ibs,0,		"ADC%S	%i,%e" },
[0x03] =	{ Ibs,0,		"SBB%S	%i,%e" },
[0x04] =	{ Ibs,0,		"AND%S	%i,%e" },
[0x05] =	{ Ibs,0,		"SUB%S	%i,%e" },
[0x06] =	{ Ibs,0,		"XOR%S	%i,%e" },
[0x07] =	{ Ibs,0,		"CMP%S	%e,%i" },
};

static Optable optabC0[8] =
{
[0x00] =	{ Ib,0,		"ROLB	%i,%e" },
[0x01] =	{ Ib,0,		"RORB	%i,%e" },
[0x02] =	{ Ib,0,		"RCLB	%i,%e" },
[0x03] =	{ Ib,0,		"RCRB	%i,%e" },
[0x04] =	{ Ib,0,		"SHLB	%i,%e" },
[0x05] =	{ Ib,0,		"SHRB	%i,%e" },
[0x07] =	{ Ib,0,		"SARB	%i,%e" },
};

static Optable optabC1[8] =
{
[0x00] =	{ Ib,0,		"ROL%S	%i,%e" },
[0x01] =	{ Ib,0,		"ROR%S	%i,%e" },
[0x02] =	{ Ib,0,		"RCL%S	%i,%e" },
[0x03] =	{ Ib,0,		"RCR%S	%i,%e" },
[0x04] =	{ Ib,0,		"SHL%S	%i,%e" },
[0x05] =	{ Ib,0,		"SHR%S	%i,%e" },
[0x07] =	{ Ib,0,		"SAR%S	%i,%e" },
};

static Optable optabD0[8] =
{
[0x00] =	{ 0,0,		"ROLB	%e" },
[0x01] =	{ 0,0,		"RORB	%e" },
[0x02] =	{ 0,0,		"RCLB	%e" },
[0x03] =	{ 0,0,		"RCRB	%e" },
[0x04] =	{ 0,0,		"SHLB	%e" },
[0x05] =	{ 0,0,		"SHRB	%e" },
[0x07] =	{ 0,0,		"SARB	%e" },
};

static Optable optabD1[8] =
{
[0x00] =	{ 0,0,		"ROL%S	%e" },
[0x01] =	{ 0,0,		"ROR%S	%e" },
[0x02] =	{ 0,0,		"RCL%S	%e" },
[0x03] =	{ 0,0,		"RCR%S	%e" },
[0x04] =	{ 0,0,		"SHL%S	%e" },
[0x05] =	{ 0,0,		"SHR%S	%e" },
[0x07] =	{ 0,0,		"SAR%S	%e" },
};

static Optable optabD2[8] =
{
[0x00] =	{ 0,0,		"ROLB	CL,%e" },
[0x01] =	{ 0,0,		"RORB	CL,%e" },
[0x02] =	{ 0,0,		"RCLB	CL,%e" },
[0x03] =	{ 0,0,		"RCRB	CL,%e" },
[0x04] =	{ 0,0,		"SHLB	CL,%e" },
[0x05] =	{ 0,0,		"SHRB	CL,%e" },
[0x07] =	{ 0,0,		"SARB	CL,%e" },
};

static Optable optabD3[8] =
{
[0x00] =	{ 0,0,		"ROL%S	CL,%e" },
[0x01] =	{ 0,0,		"ROR%S	CL,%e" },
[0x02] =	{ 0,0,		"RCL%S	CL,%e" },
[0x03] =	{ 0,0,		"RCR%S	CL,%e" },
[0x04] =	{ 0,0,		"SHL%S	CL,%e" },
[0x05] =	{ 0,0,		"SHR%S	CL,%e" },
[0x07] =	{ 0,0,		"SAR%S	CL,%e" },
};

static Optable optabD8[8+8] =
{
[0x00] =	{ 0,0,		"FADDF	%e,F0" },
[0x01] =	{ 0,0,		"FMULF	%e,F0" },
[0x02] =	{ 0,0,		"FCOMF	%e,F0" },
[0x03] =	{ 0,0,		"FCOMFP	%e,F0" },
[0x04] =	{ 0,0,		"FSUBF	%e,F0" },
[0x05] =	{ 0,0,		"FSUBRF	%e,F0" },
[0x06] =	{ 0,0,		"FDIVF	%e,F0" },
[0x07] =	{ 0,0,		"FDIVRF	%e,F0" },
[0x08] =	{ 0,0,		"FADDD	%f,F0" },
[0x09] =	{ 0,0,		"FMULD	%f,F0" },
[0x0a] =	{ 0,0,		"FCOMD	%f,F0" },
[0x0b] =	{ 0,0,		"FCOMPD	%f,F0" },
[0x0c] =	{ 0,0,		"FSUBD	%f,F0" },
[0x0d] =	{ 0,0,		"FSUBRD	%f,F0" },
[0x0e] =	{ 0,0,		"FDIVD	%f,F0" },
[0x0f] =	{ 0,0,		"FDIVRD	%f,F0" },
};
/*
 *	optabD9 and optabDB use the following encoding:
 *	if (0 <= modrm <= 2) instruction = optabDx[modrm&0x07];
 *	else instruction = optabDx[(modrm&0x3f)+8];
 *
 *	the instructions for MOD == 3, follow the 8 instructions
 *	for the other MOD values stored at the front of the table.
 */
static Optable optabD9[64+8] =
{
[0x00] =	{ 0,0,		"FMOVF	%e,F0" },
[0x02] =	{ 0,0,		"FMOVF	F0,%e" },
[0x03] =	{ 0,0,		"FMOVFP	F0,%e" },
[0x04] =	{ 0,0,		"FLDENV%S %e" },
[0x05] =	{ 0,0,		"FLDCW	%e" },
[0x06] =	{ 0,0,		"FSTENV%S %e" },
[0x07] =	{ 0,0,		"FSTCW	%e" },
[0x08] =	{ 0,0,		"FMOVD	F0,F0" },		/* Mod R/M = 11xx xxxx*/
[0x09] =	{ 0,0,		"FMOVD	F1,F0" },
[0x0a] =	{ 0,0,		"FMOVD	F2,F0" },
[0x0b] =	{ 0,0,		"FMOVD	F3,F0" },
[0x0c] =	{ 0,0,		"FMOVD	F4,F0" },
[0x0d] =	{ 0,0,		"FMOVD	F5,F0" },
[0x0e] =	{ 0,0,		"FMOVD	F6,F0" },
[0x0f] =	{ 0,0,		"FMOVD	F7,F0" },
[0x10] =	{ 0,0,		"FXCHD	F0,F0" },
[0x11] =	{ 0,0,		"FXCHD	F1,F0" },
[0x12] =	{ 0,0,		"FXCHD	F2,F0" },
[0x13] =	{ 0,0,		"FXCHD	F3,F0" },
[0x14] =	{ 0,0,		"FXCHD	F4,F0" },
[0x15] =	{ 0,0,		"FXCHD	F5,F0" },
[0x16] =	{ 0,0,		"FXCHD	F6,F0" },
[0x17] =	{ 0,0,		"FXCHD	F7,F0" },
[0x18] =	{ 0,0,		"FNOP" },
[0x28] =	{ 0,0,		"FCHS" },
[0x29] =	{ 0,0,		"FABS" },
[0x2c] =	{ 0,0,		"FTST" },
[0x2d] =	{ 0,0,		"FXAM" },
[0x30] =	{ 0,0,		"FLD1" },
[0x31] =	{ 0,0,		"FLDL2T" },
[0x32] =	{ 0,0,		"FLDL2E" },
[0x33] =	{ 0,0,		"FLDPI" },
[0x34] =	{ 0,0,		"FLDLG2" },
[0x35] =	{ 0,0,		"FLDLN2" },
[0x36] =	{ 0,0,		"FLDZ" },
[0x38] =	{ 0,0,		"F2XM1" },
[0x39] =	{ 0,0,		"FYL2X" },
[0x3a] =	{ 0,0,		"FPTAN" },
[0x3b] =	{ 0,0,		"FPATAN" },
[0x3c] =	{ 0,0,		"FXTRACT" },
[0x3d] =	{ 0,0,		"FPREM1" },
[0x3e] =	{ 0,0,		"FDECSTP" },
[0x3f] =	{ 0,0,		"FNCSTP" },
[0x40] =	{ 0,0,		"FPREM" },
[0x41] =	{ 0,0,		"FYL2XP1" },
[0x42] =	{ 0,0,		"FSQRT" },
[0x43] =	{ 0,0,		"FSINCOS" },
[0x44] =	{ 0,0,		"FRNDINT" },
[0x45] =	{ 0,0,		"FSCALE" },
[0x46] =	{ 0,0,		"FSIN" },
[0x47] =	{ 0,0,		"FCOS" },
};

static Optable optabDA[8+8] =
{
[0x00] =	{ 0,0,		"FADDL	%e,F0" },
[0x01] =	{ 0,0,		"FMULL	%e,F0" },
[0x02] =	{ 0,0,		"FCOML	%e,F0" },
[0x03] =	{ 0,0,		"FCOMLP	%e,F0" },
[0x04] =	{ 0,0,		"FSUBL	%e,F0" },
[0x05] =	{ 0,0,		"FSUBRL	%e,F0" },
[0x06] =	{ 0,0,		"FDIVL	%e,F0" },
[0x07] =	{ 0,0,		"FDIVRL	%e,F0" },
[0x08] =	{ 0,0,		"FCMOVCS	%f,F0" },
[0x09] =	{ 0,0,		"FCMOVEQ	%f,F0" },
[0x0a] =	{ 0,0,		"FCMOVLS	%f,F0" },
[0x0b] =	{ 0,0,		"FCMOVUN	%f,F0" },
[0x0d] =	{ Op_R1,0,		"FUCOMPP" },
};

static Optable optabDB[8+64] =
{
[0x00] =	{ 0,0,		"FMOVL	%e,F0" },
[0x02] =	{ 0,0,		"FMOVL	F0,%e" },
[0x03] =	{ 0,0,		"FMOVLP	F0,%e" },
[0x05] =	{ 0,0,		"FMOVX	%e,F0" },
[0x07] =	{ 0,0,		"FMOVXP	F0,%e" },
[0x08] =	{ 0,0,		"FCMOVCC	F0,F0" },	/* Mod R/M = 11xx xxxx*/
[0x09] =	{ 0,0,		"FCMOVCC	F1,F0" },
[0x0a] =	{ 0,0,		"FCMOVCC	F2,F0" },
[0x0b] =	{ 0,0,		"FCMOVCC	F3,F0" },
[0x0c] =	{ 0,0,		"FCMOVCC	F4,F0" },
[0x0d] =	{ 0,0,		"FCMOVCC	F5,F0" },
[0x0e] =	{ 0,0,		"FCMOVCC	F6,F0" },
[0x0f] =	{ 0,0,		"FCMOVCC	F7,F0" },
[0x10] =	{ 0,0,		"FCMOVNE	F0,F0" },
[0x11] =	{ 0,0,		"FCMOVNE	F1,F0" },
[0x12] =	{ 0,0,		"FCMOVNE	F2,F0" },
[0x13] =	{ 0,0,		"FCMOVNE	F3,F0" },
[0x14] =	{ 0,0,		"FCMOVNE	F4,F0" },
[0x15] =	{ 0,0,		"FCMOVNE	F5,F0" },
[0x16] =	{ 0,0,		"FCMOVNE	F6,F0" },
[0x17] =	{ 0,0,		"FCMOVNE	F7,F0" },
[0x18] =	{ 0,0,		"FCMOVHI	F0,F0" },
[0x19] =	{ 0,0,		"FCMOVHI	F1,F0" },
[0x1a] =	{ 0,0,		"FCMOVHI	F2,F0" },
[0x1b] =	{ 0,0,		"FCMOVHI	F3,F0" },
[0x1c] =	{ 0,0,		"FCMOVHI	F4,F0" },
[0x1d] =	{ 0,0,		"FCMOVHI	F5,F0" },
[0x1e] =	{ 0,0,		"FCMOVHI	F6,F0" },
[0x1f] =	{ 0,0,		"FCMOVHI	F7,F0" },
[0x20] =	{ 0,0,		"FCMOVNU	F0,F0" },
[0x21] =	{ 0,0,		"FCMOVNU	F1,F0" },
[0x22] =	{ 0,0,		"FCMOVNU	F2,F0" },
[0x23] =	{ 0,0,		"FCMOVNU	F3,F0" },
[0x24] =	{ 0,0,		"FCMOVNU	F4,F0" },
[0x25] =	{ 0,0,		"FCMOVNU	F5,F0" },
[0x26] =	{ 0,0,		"FCMOVNU	F6,F0" },
[0x27] =	{ 0,0,		"FCMOVNU	F7,F0" },
[0x2a] =	{ 0,0,		"FCLEX" },
[0x2b] =	{ 0,0,		"FINIT" },
[0x30] =	{ 0,0,		"FUCOMI	F0,F0" },
[0x31] =	{ 0,0,		"FUCOMI	F1,F0" },
[0x32] =	{ 0,0,		"FUCOMI	F2,F0" },
[0x33] =	{ 0,0,		"FUCOMI	F3,F0" },
[0x34] =	{ 0,0,		"FUCOMI	F4,F0" },
[0x35] =	{ 0,0,		"FUCOMI	F5,F0" },
[0x36] =	{ 0,0,		"FUCOMI	F6,F0" },
[0x37] =	{ 0,0,		"FUCOMI	F7,F0" },
[0x38] =	{ 0,0,		"FCOMI	F0,F0" },
[0x39] =	{ 0,0,		"FCOMI	F1,F0" },
[0x3a] =	{ 0,0,		"FCOMI	F2,F0" },
[0x3b] =	{ 0,0,		"FCOMI	F3,F0" },
[0x3c] =	{ 0,0,		"FCOMI	F4,F0" },
[0x3d] =	{ 0,0,		"FCOMI	F5,F0" },
[0x3e] =	{ 0,0,		"FCOMI	F6,F0" },
[0x3f] =	{ 0,0,		"FCOMI	F7,F0" },
};

static Optable optabDC[8+8] =
{
[0x00] =	{ 0,0,		"FADDD	%e,F0" },
[0x01] =	{ 0,0,		"FMULD	%e,F0" },
[0x02] =	{ 0,0,		"FCOMD	%e,F0" },
[0x03] =	{ 0,0,		"FCOMDP	%e,F0" },
[0x04] =	{ 0,0,		"FSUBD	%e,F0" },
[0x05] =	{ 0,0,		"FSUBRD	%e,F0" },
[0x06] =	{ 0,0,		"FDIVD	%e,F0" },
[0x07] =	{ 0,0,		"FDIVRD	%e,F0" },
[0x08] =	{ 0,0,		"FADDD	F0,%f" },
[0x09] =	{ 0,0,		"FMULD	F0,%f" },
[0x0c] =	{ 0,0,		"FSUBRD	F0,%f" },
[0x0d] =	{ 0,0,		"FSUBD	F0,%f" },
[0x0e] =	{ 0,0,		"FDIVRD	F0,%f" },
[0x0f] =	{ 0,0,		"FDIVD	F0,%f" },
};

static Optable optabDD[8+8] =
{
[0x00] =	{ 0,0,		"FMOVD	%e,F0" },
[0x02] =	{ 0,0,		"FMOVD	F0,%e" },
[0x03] =	{ 0,0,		"FMOVDP	F0,%e" },
[0x04] =	{ 0,0,		"FRSTOR%S %e" },
[0x06] =	{ 0,0,		"FSAVE%S %e" },
[0x07] =	{ 0,0,		"FSTSW	%e" },
[0x08] =	{ 0,0,		"FFREED	%f" },
[0x0a] =	{ 0,0,		"FMOVD	%f,F0" },
[0x0b] =	{ 0,0,		"FMOVDP	%f,F0" },
[0x0c] =	{ 0,0,		"FUCOMD	%f,F0" },
[0x0d] =	{ 0,0,		"FUCOMDP %f,F0" },
};

static Optable optabDE[8+8] =
{
[0x00] =	{ 0,0,		"FADDW	%e,F0" },
[0x01] =	{ 0,0,		"FMULW	%e,F0" },
[0x02] =	{ 0,0,		"FCOMW	%e,F0" },
[0x03] =	{ 0,0,		"FCOMWP	%e,F0" },
[0x04] =	{ 0,0,		"FSUBW	%e,F0" },
[0x05] =	{ 0,0,		"FSUBRW	%e,F0" },
[0x06] =	{ 0,0,		"FDIVW	%e,F0" },
[0x07] =	{ 0,0,		"FDIVRW	%e,F0" },
[0x08] =	{ 0,0,		"FADDDP	F0,%f" },
[0x09] =	{ 0,0,		"FMULDP	F0,%f" },
[0x0b] =	{ Op_R1,0,		"FCOMPDP" },
[0x0c] =	{ 0,0,		"FSUBRDP F0,%f" },
[0x0d] =	{ 0,0,		"FSUBDP	F0,%f" },
[0x0e] =	{ 0,0,		"FDIVRDP F0,%f" },
[0x0f] =	{ 0,0,		"FDIVDP	F0,%f" },
};

static Optable optabDF[8+8] =
{
[0x00] =	{ 0,0,		"FMOVW	%e,F0" },
[0x02] =	{ 0,0,		"FMOVW	F0,%e" },
[0x03] =	{ 0,0,		"FMOVWP	F0,%e" },
[0x04] =	{ 0,0,		"FBLD	%e" },
[0x05] =	{ 0,0,		"FMOVL	%e,F0" },
[0x06] =	{ 0,0,		"FBSTP	%e" },
[0x07] =	{ 0,0,		"FMOVLP	F0,%e" },
[0x0c] =	{ Op_R0,0,		"FSTSW	%OAX" },
[0x0d] =	{ 0,0,		"FUCOMIP	F0,%f" },
[0x0e] =	{ 0,0,		"FCOMIP	F0,%f" },
};

static Optable optabF6[8] =
{
[0x00] =	{ Ib,0,		"TESTB	%i,%e" },
[0x02] =	{ 0,0,		"NOTB	%e" },
[0x03] =	{ 0,0,		"NEGB	%e" },
[0x04] =	{ 0,0,		"MULB	AL,%e" },
[0x05] =	{ 0,0,		"IMULB	AL,%e" },
[0x06] =	{ 0,0,		"DIVB	AL,%e" },
[0x07] =	{ 0,0,		"IDIVB	AL,%e" },
};

static Optable optabF7[8] =
{
[0x00] =	{ Iwd,0,		"TEST%S	%i,%e" },
[0x02] =	{ 0,0,		"NOT%S	%e" },
[0x03] =	{ 0,0,		"NEG%S	%e" },
[0x04] =	{ 0,0,		"MUL%S	%OAX,%e" },
[0x05] =	{ 0,0,		"IMUL%S	%OAX,%e" },
[0x06] =	{ 0,0,		"DIV%S	%OAX,%e" },
[0x07] =	{ 0,0,		"IDIV%S	%OAX,%e" },
};

static Optable optabFE[8] =
{
[0x00] =	{ 0,0,		"INCB	%e" },
[0x01] =	{ 0,0,		"DECB	%e" },
};

static Optable optabFF[8] =
{
[0x00] =	{ 0,0,		"INC%S	%e" },
[0x01] =	{ 0,0,		"DEC%S	%e" },
[0x02] =	{ JUMP,0,		"CALL*	%e" },
[0x03] =	{ JUMP,0,		"CALLF*	%e" },
[0x04] =	{ JUMP,0,		"JMP*	%e" },
[0x05] =	{ JUMP,0,		"JMPF*	%e" },
[0x06] =	{ 0,0,		"PUSHL	%e" },
};

static Optable optable[256+2] =
{
[0x00] =	{ RMB,0,		"ADDB	%r,%e" },
[0x01] =	{ RM,0,		"ADD%S	%r,%e" },
[0x02] =	{ RMB,0,		"ADDB	%e,%r" },
[0x03] =	{ RM,0,		"ADD%S	%e,%r" },
[0x04] =	{ Ib,0,		"ADDB	%i,AL" },
[0x05] =	{ Iwd,0,		"ADD%S	%i,%OAX" },
[0x06] =	{ 0,0,		"PUSHL	ES" },
[0x07] =	{ 0,0,		"POPL	ES" },
[0x08] =	{ RMB,0,		"ORB	%r,%e" },
[0x09] =	{ RM,0,		"OR%S	%r,%e" },
[0x0a] =	{ RMB,0,		"ORB	%e,%r" },
[0x0b] =	{ RM,0,		"OR%S	%e,%r" },
[0x0c] =	{ Ib,0,		"ORB	%i,AL" },
[0x0d] =	{ Iwd,0,		"OR%S	%i,%OAX" },
[0x0e] =	{ 0,0,		"PUSHL	CS" },
[0x0f] =	{ AUXMM,0,	optab0F },
[0x10] =	{ RMB,0,		"ADCB	%r,%e" },
[0x11] =	{ RM,0,		"ADC%S	%r,%e" },
[0x12] =	{ RMB,0,		"ADCB	%e,%r" },
[0x13] =	{ RM,0,		"ADC%S	%e,%r" },
[0x14] =	{ Ib,0,		"ADCB	%i,AL" },
[0x15] =	{ Iwd,0,		"ADC%S	%i,%OAX" },
[0x16] =	{ 0,0,		"PUSHL	SS" },
[0x17] =	{ 0,0,		"POPL	SS" },
[0x18] =	{ RMB,0,		"SBBB	%r,%e" },
[0x19] =	{ RM,0,		"SBB%S	%r,%e" },
[0x1a] =	{ RMB,0,		"SBBB	%e,%r" },
[0x1b] =	{ RM,0,		"SBB%S	%e,%r" },
[0x1c] =	{ Ib,0,		"SBBB	%i,AL" },
[0x1d] =	{ Iwd,0,		"SBB%S	%i,%OAX" },
[0x1e] =	{ 0,0,		"PUSHL	DS" },
[0x1f] =	{ 0,0,		"POPL	DS" },
[0x20] =	{ RMB,0,		"ANDB	%r,%e" },
[0x21] =	{ RM,0,		"AND%S	%r,%e" },
[0x22] =	{ RMB,0,		"ANDB	%e,%r" },
[0x23] =	{ RM,0,		"AND%S	%e,%r" },
[0x24] =	{ Ib,0,		"ANDB	%i,AL" },
[0x25] =	{ Iwd,0,		"AND%S	%i,%OAX" },
[0x26] =	{ SEG,0,		"ES:" },
[0x27] =	{ 0,0,		"DAA" },
[0x28] =	{ RMB,0,		"SUBB	%r,%e" },
[0x29] =	{ RM,0,		"SUB%S	%r,%e" },
[0x2a] =	{ RMB,0,		"SUBB	%e,%r" },
[0x2b] =	{ RM,0,		"SUB%S	%e,%r" },
[0x2c] =	{ Ib,0,		"SUBB	%i,AL" },
[0x2d] =	{ Iwd,0,		"SUB%S	%i,%OAX" },
[0x2e] =	{ SEG,0,		"CS:" },
[0x2f] =	{ 0,0,		"DAS" },
[0x30] =	{ RMB,0,		"XORB	%r,%e" },
[0x31] =	{ RM,0,		"XOR%S	%r,%e" },
[0x32] =	{ RMB,0,		"XORB	%e,%r" },
[0x33] =	{ RM,0,		"XOR%S	%e,%r" },
[0x34] =	{ Ib,0,		"XORB	%i,AL" },
[0x35] =	{ Iwd,0,		"XOR%S	%i,%OAX" },
[0x36] =	{ SEG,0,		"SS:" },
[0x37] =	{ 0,0,		"AAA" },
[0x38] =	{ RMB,0,		"CMPB	%r,%e" },
[0x39] =	{ RM,0,		"CMP%S	%r,%e" },
[0x3a] =	{ RMB,0,		"CMPB	%e,%r" },
[0x3b] =	{ RM,0,		"CMP%S	%e,%r" },
[0x3c] =	{ Ib,0,		"CMPB	%i,AL" },
[0x3d] =	{ Iwd,0,		"CMP%S	%i,%OAX" },
[0x3e] =	{ SEG,0,		"DS:" },
[0x3f] =	{ 0,0,		"AAS" },
[0x40] =	{ 0,0,		"INC%S	%OAX" },
[0x41] =	{ 0,0,		"INC%S	%OCX" },
[0x42] =	{ 0,0,		"INC%S	%ODX" },
[0x43] =	{ 0,0,		"INC%S	%OBX" },
[0x44] =	{ 0,0,		"INC%S	%OSP" },
[0x45] =	{ 0,0,		"INC%S	%OBP" },
[0x46] =	{ 0,0,		"INC%S	%OSI" },
[0x47] =	{ 0,0,		"INC%S	%ODI" },
[0x48] =	{ 0,0,		"DEC%S	%OAX" },
[0x49] =	{ 0,0,		"DEC%S	%OCX" },
[0x4a] =	{ 0,0,		"DEC%S	%ODX" },
[0x4b] =	{ 0,0,		"DEC%S	%OBX" },
[0x4c] =	{ 0,0,		"DEC%S	%OSP" },
[0x4d] =	{ 0,0,		"DEC%S	%OBP" },
[0x4e] =	{ 0,0,		"DEC%S	%OSI" },
[0x4f] =	{ 0,0,		"DEC%S	%ODI" },
[0x50] =	{ 0,0,		"PUSH%S	%OAX" },
[0x51] =	{ 0,0,		"PUSH%S	%OCX" },
[0x52] =	{ 0,0,		"PUSH%S	%ODX" },
[0x53] =	{ 0,0,		"PUSH%S	%OBX" },
[0x54] =	{ 0,0,		"PUSH%S	%OSP" },
[0x55] =	{ 0,0,		"PUSH%S	%OBP" },
[0x56] =	{ 0,0,		"PUSH%S	%OSI" },
[0x57] =	{ 0,0,		"PUSH%S	%ODI" },
[0x58] =	{ 0,0,		"POP%S	%OAX" },
[0x59] =	{ 0,0,		"POP%S	%OCX" },
[0x5a] =	{ 0,0,		"POP%S	%ODX" },
[0x5b] =	{ 0,0,		"POP%S	%OBX" },
[0x5c] =	{ 0,0,		"POP%S	%OSP" },
[0x5d] =	{ 0,0,		"POP%S	%OBP" },
[0x5e] =	{ 0,0,		"POP%S	%OSI" },
[0x5f] =	{ 0,0,		"POP%S	%ODI" },
[0x60] =	{ 0,0,		"PUSHA%S" },
[0x61] =	{ 0,0,		"POPA%S" },
[0x62] =	{ RMM,0,		"BOUND	%e,%r" },
[0x63] =	{ RM,0,		"ARPL	%r,%e" },
[0x64] =	{ SEG,0,		"FS:" },
[0x65] =	{ SEG,0,		"GS:" },
[0x66] =	{ OPOVER,0,	"" },
[0x67] =	{ ADDOVER,0,	"" },
[0x68] =	{ Iwd,0,		"PUSH%S	%i" },
[0x69] =	{ RM,Iwd,		"IMUL%S	%e,%i,%r" },
[0x6a] =	{ Ib,0,		"PUSH%S	%i" },
[0x6b] =	{ RM,Ibs,		"IMUL%S	%e,%i,%r" },
[0x6c] =	{ 0,0,		"INSB	DX,(%ODI)" },
[0x6d] =	{ 0,0,		"INS%S	DX,(%ODI)" },
[0x6e] =	{ 0,0,		"OUTSB	(%ASI),DX" },
[0x6f] =	{ 0,0,		"OUTS%S	(%ASI),DX" },
[0x70] =	{ Jbs,0,		"JOS	%p" },
[0x71] =	{ Jbs,0,		"JOC	%p" },
[0x72] =	{ Jbs,0,		"JCS	%p" },
[0x73] =	{ Jbs,0,		"JCC	%p" },
[0x74] =	{ Jbs,0,		"JEQ	%p" },
[0x75] =	{ Jbs,0,		"JNE	%p" },
[0x76] =	{ Jbs,0,		"JLS	%p" },
[0x77] =	{ Jbs,0,		"JHI	%p" },
[0x78] =	{ Jbs,0,		"JMI	%p" },
[0x79] =	{ Jbs,0,		"JPL	%p" },
[0x7a] =	{ Jbs,0,		"JPS	%p" },
[0x7b] =	{ Jbs,0,		"JPC	%p" },
[0x7c] =	{ Jbs,0,		"JLT	%p" },
[0x7d] =	{ Jbs,0,		"JGE	%p" },
[0x7e] =	{ Jbs,0,		"JLE	%p" },
[0x7f] =	{ Jbs,0,		"JGT	%p" },
[0x80] =	{ RMOPB,0,	optab80 },
[0x81] =	{ RMOP,0,		optab81 },
[0x83] =	{ RMOP,0,		optab83 },
[0x84] =	{ RMB,0,		"TESTB	%r,%e" },
[0x85] =	{ RM,0,		"TEST%S	%r,%e" },
[0x86] =	{ RMB,0,		"XCHGB	%r,%e" },
[0x87] =	{ RM,0,		"XCHG%S	%r,%e" },
[0x88] =	{ RMB,0,		"MOVB	%r,%e" },
[0x89] =	{ RM,0,		"MOV%S	%r,%e" },
[0x8a] =	{ RMB,0,		"MOVB	%e,%r" },
[0x8b] =	{ RM,0,		"MOV%S	%e,%r" },
[0x8c] =	{ RM,0,		"MOVW	%g,%e" },
[0x8d] =	{ RM,0,		"LEA%S	%e,%r" },
[0x8e] =	{ RM,0,		"MOVW	%e,%g" },
[0x8f] =	{ RM,0,		"POP%S	%e" },
[0x90] =	{ 0,0,		"NOP" },
[0x91] =	{ 0,0,		"XCHG	%OCX,%OAX" },
[0x92] =	{ 0,0,		"XCHG	%ODX,%OAX" },
[0x93] =	{ 0,0,		"XCHG	%OBX,%OAX" },
[0x94] =	{ 0,0,		"XCHG	%OSP,%OAX" },
[0x95] =	{ 0,0,		"XCHG	%OBP,%OAX" },
[0x96] =	{ 0,0,		"XCHG	%OSI,%OAX" },
[0x97] =	{ 0,0,		"XCHG	%ODI,%OAX" },
[0x98] =	{ 0,0,		"%W" },			/* miserable CBW or CWDE */
[0x99] =	{ 0,0,		"%w" },			/* idiotic CWD or CDQ */
[0x9a] =	{ PTR,0,		"CALL%S	%d" },
[0x9b] =	{ 0,0,		"WAIT" },
[0x9c] =	{ 0,0,		"PUSHF" },
[0x9d] =	{ 0,0,		"POPF" },
[0x9e] =	{ 0,0,		"SAHF" },
[0x9f] =	{ 0,0,		"LAHF" },
[0xa0] =	{ Awd,0,		"MOVB	%i,AL" },
[0xa1] =	{ Awd,0,		"MOV%S	%i,%OAX" },
[0xa2] =	{ Awd,0,		"MOVB	AL,%i" },
[0xa3] =	{ Awd,0,		"MOV%S	%OAX,%i" },
[0xa4] =	{ 0,0,		"MOVSB	(%ASI),(%ADI)" },
[0xa5] =	{ 0,0,		"MOVS%S	(%ASI),(%ADI)" },
[0xa6] =	{ 0,0,		"CMPSB	(%ASI),(%ADI)" },
[0xa7] =	{ 0,0,		"CMPS%S	(%ASI),(%ADI)" },
[0xa8] =	{ Ib,0,		"TESTB	%i,AL" },
[0xa9] =	{ Iwd,0,		"TEST%S	%i,%OAX" },
[0xaa] =	{ 0,0,		"STOSB	AL,(%ADI)" },
[0xab] =	{ 0,0,		"STOS%S	%OAX,(%ADI)" },
[0xac] =	{ 0,0,		"LODSB	(%ASI),AL" },
[0xad] =	{ 0,0,		"LODS%S	(%ASI),%OAX" },
[0xae] =	{ 0,0,		"SCASB	(%ADI),AL" },
[0xaf] =	{ 0,0,		"SCAS%S	(%ADI),%OAX" },
[0xb0] =	{ Ib,0,		"MOVB	%i,AL" },
[0xb1] =	{ Ib,0,		"MOVB	%i,CL" },
[0xb2] =	{ Ib,0,		"MOVB	%i,DL" },
[0xb3] =	{ Ib,0,		"MOVB	%i,BL" },
[0xb4] =	{ Ib,0,		"MOVB	%i,AH" },
[0xb5] =	{ Ib,0,		"MOVB	%i,CH" },
[0xb6] =	{ Ib,0,		"MOVB	%i,DH" },
[0xb7] =	{ Ib,0,		"MOVB	%i,BH" },
[0xb8] =	{ Iwdq,0,		"MOV%S	%i,%OAX" },
[0xb9] =	{ Iwdq,0,		"MOV%S	%i,%OCX" },
[0xba] =	{ Iwdq,0,		"MOV%S	%i,%ODX" },
[0xbb] =	{ Iwdq,0,		"MOV%S	%i,%OBX" },
[0xbc] =	{ Iwdq,0,		"MOV%S	%i,%OSP" },
[0xbd] =	{ Iwdq,0,		"MOV%S	%i,%OBP" },
[0xbe] =	{ Iwdq,0,		"MOV%S	%i,%OSI" },
[0xbf] =	{ Iwdq,0,		"MOV%S	%i,%ODI" },
[0xc0] =	{ RMOPB,0,	optabC0 },
[0xc1] =	{ RMOP,0,		optabC1 },
[0xc2] =	{ Iw,0,		"RET	%i" },
[0xc3] =	{ RET,0,		"RET" },
[0xc4] =	{ RM,0,		"LES	%e,%r" },
[0xc5] =	{ RM,0,		"LDS	%e,%r" },
[0xc6] =	{ RMB,Ib,		"MOVB	%i,%e" },
[0xc7] =	{ RM,Iwd,		"MOV%S	%i,%e" },
[0xc8] =	{ Iw2,Ib,		"ENTER	%i,%I" },		/* loony ENTER */
[0xc9] =	{ RET,0,		"LEAVE" },		/* bizarre LEAVE */
[0xca] =	{ Iw,0,		"RETF	%i" },
[0xcb] =	{ RET,0,		"RETF" },
[0xcc] =	{ 0,0,		"INT	3" },
[0xcd] =	{ Ib,0,		"INTB	%i" },
[0xce] =	{ 0,0,		"INTO" },
[0xcf] =	{ 0,0,		"IRET" },
[0xd0] =	{ RMOPB,0,	optabD0 },
[0xd1] =	{ RMOP,0,		optabD1 },
[0xd2] =	{ RMOPB,0,	optabD2 },
[0xd3] =	{ RMOP,0,		optabD3 },
[0xd4] =	{ OA,0,		"AAM" },
[0xd5] =	{ OA,0,		"AAD" },
[0xd7] =	{ 0,0,		"XLAT" },
[0xd8] =	{ FRMOP,0,	optabD8 },
[0xd9] =	{ FRMEX,0,	optabD9 },
[0xda] =	{ FRMOP,0,	optabDA },
[0xdb] =	{ FRMEX,0,	optabDB },
[0xdc] =	{ FRMOP,0,	optabDC },
[0xdd] =	{ FRMOP,0,	optabDD },
[0xde] =	{ FRMOP,0,	optabDE },
[0xdf] =	{ FRMOP,0,	optabDF },
[0xe0] =	{ Jbs,0,		"LOOPNE	%p" },
[0xe1] =	{ Jbs,0,		"LOOPE	%p" },
[0xe2] =	{ Jbs,0,		"LOOP	%p" },
[0xe3] =	{ Jbs,0,		"JCXZ	%p" },
[0xe4] =	{ Ib,0,		"INB	%i,AL" },
[0xe5] =	{ Ib,0,		"IN%S	%i,%OAX" },
[0xe6] =	{ Ib,0,		"OUTB	AL,%i" },
[0xe7] =	{ Ib,0,		"OUT%S	%OAX,%i" },
[0xe8] =	{ Iwds,0,		"CALL	%p" },
[0xe9] =	{ Iwds,0,		"JMP	%p" },
[0xea] =	{ PTR,0,		"JMP	%d" },
[0xeb] =	{ Jbs,0,		"JMP	%p" },
[0xec] =	{ 0,0,		"INB	DX,AL" },
[0xed] =	{ 0,0,		"IN%S	DX,%OAX" },
[0xee] =	{ 0,0,		"OUTB	AL,DX" },
[0xef] =	{ 0,0,		"OUT%S	%OAX,DX" },
[0xf0] =	{ PRE,0,		"LOCK" },
[0xf2] =	{ OPRE,0,		"REPNE" },
[0xf3] =	{ OPRE,0,		"REP" },
[0xf4] =	{ 0,0,		"HLT" },
[0xf5] =	{ 0,0,		"CMC" },
[0xf6] =	{ RMOPB,0,	optabF6 },
[0xf7] =	{ RMOP,0,		optabF7 },
[0xf8] =	{ 0,0,		"CLC" },
[0xf9] =	{ 0,0,		"STC" },
[0xfa] =	{ 0,0,		"CLI" },
[0xfb] =	{ 0,0,		"STI" },
[0xfc] =	{ 0,0,		"CLD" },
[0xfd] =	{ 0,0,		"STD" },
[0xfe] =	{ RMOPB,0,	optabFE },
[0xff] =	{ RMOP,0,		optabFF },
[0x100] =	{ RM,0,		"MOVLQSX	%e,%r" },
[0x101] =	{ RM,0,		"MOVLQZX	%e,%r" },
};

/*
 *  get a byte of the instruction
 */
static int
igetc(Map *map, Instr *ip, uchar *c)
{
	if(ip->n+1 > sizeof(ip->mem)){
		werrstr("instruction too long");
		return -1;
	}
	if (get1(map, ip->addr+ip->n, c, 1) < 0) {
		werrstr("can't read instruction: %r");
		return -1;
	}
	ip->mem[ip->n++] = *c;
	return 1;
}

/*
 *  get two bytes of the instruction
 */
static int
igets(Map *map, Instr *ip, ushort *sp)
{
	uchar c;
	ushort s;

	if (igetc(map, ip, &c) < 0)
		return -1;
	s = c;
	if (igetc(map, ip, &c) < 0)
		return -1;
	s |= (c<<8);
	*sp = s;
	return 1;
}

/*
 *  get 4 bytes of the instruction
 */
static int
igetl(Map *map, Instr *ip, uint32 *lp)
{
	ushort s;
	int32	l;

	if (igets(map, ip, &s) < 0)
		return -1;
	l = s;
	if (igets(map, ip, &s) < 0)
		return -1;
	l |= (s<<16);
	*lp = l;
	return 1;
}

/*
 *  get 8 bytes of the instruction
 *
static int
igetq(Map *map, Instr *ip, vlong *qp)
{
	uint32	l;
	uvlong q;

	if (igetl(map, ip, &l) < 0)
		return -1;
	q = l;
	if (igetl(map, ip, &l) < 0)
		return -1;
	q |= ((uvlong)l<<32);
	*qp = q;
	return 1;
}
 */

static int
getdisp(Map *map, Instr *ip, int mod, int rm, int code, int pcrel)
{
	uchar c;
	ushort s;

	if (mod > 2)
		return 1;
	if (mod == 1) {
		if (igetc(map, ip, &c) < 0)
			return -1;
		if (c&0x80)
			ip->disp = c|0xffffff00;
		else
			ip->disp = c&0xff;
	} else if (mod == 2 || rm == code) {
		if (ip->asize == 'E') {
			if (igetl(map, ip, &ip->disp) < 0)
				return -1;
			if (mod == 0)
				ip->rip = pcrel;
		} else {
			if (igets(map, ip, &s) < 0)
				return -1;
			if (s&0x8000)
				ip->disp = s|0xffff0000;
			else
				ip->disp = s;
		}
		if (mod == 0)
			ip->base = -1;
	}
	return 1;
}

static int
modrm(Map *map, Instr *ip, uchar c)
{
	uchar rm, mod;

	mod = (c>>6)&3;
	rm = c&7;
	ip->mod = mod;
	ip->base = rm;
	ip->reg = (c>>3)&7;
	ip->rip = 0;
	if (mod == 3)			/* register */
		return 1;
	if (ip->asize == 0) {		/* 16-bit mode */
		switch(rm) {
		case 0:
			ip->base = BX; ip->index = SI;
			break;
		case 1:
			ip->base = BX; ip->index = DI;
			break;
		case 2:
			ip->base = BP; ip->index = SI;
			break;
		case 3:
			ip->base = BP; ip->index = DI;
			break;
		case 4:
			ip->base = SI;
			break;
		case 5:
			ip->base = DI;
			break;
		case 6:
			ip->base = BP;
			break;
		case 7:
			ip->base = BX;
			break;
		default:
			break;
		}
		return getdisp(map, ip, mod, rm, 6, 0);
	}
	if (rm == 4) {	/* scummy sib byte */
		if (igetc(map, ip, &c) < 0)
			return -1;
		ip->ss = (c>>6)&0x03;
		ip->index = (c>>3)&0x07;
		if (ip->index == 4)
			ip->index = -1;
		ip->base = c&0x07;
		return getdisp(map, ip, mod, ip->base, 5, 0);
	}
	return getdisp(map, ip, mod, rm, 5, ip->amd64);
}

static Optable *
mkinstr(Map *map, Instr *ip, uvlong pc)
{
	int i, n, norex;
	uchar c;
	ushort s;
	Optable *op, *obase;
	char buf[128];

	memset(ip, 0, sizeof(*ip));
	norex = 1;
	ip->base = -1;
	ip->index = -1;
	if(asstype == AI8086)
		ip->osize = 'W';
	else {
		ip->osize = 'L';
		ip->asize = 'E';
		ip->amd64 = asstype != AI386;
		norex = 0;
	}
	ip->addr = pc;
	if (igetc(map, ip, &c) < 0)
		return 0;
	obase = optable;
newop:
	if(ip->amd64 && !norex){
		if(c >= 0x40 && c <= 0x4f) {
			ip->rex = c;
			if(igetc(map, ip, &c) < 0)
				return 0;
		}
		if(c == 0x63){
			if(ip->rex&REXW)
				op = &obase[0x100];	/* MOVLQSX */
			else
				op = &obase[0x101];	/* MOVLQZX */
			goto hack;
		}
	}
	op = &obase[c];
hack:
	if (op->proto == 0) {
badop:
		n = snprint(buf, sizeof(buf), "opcode: ??");
		for (i = 0; i < ip->n && n < sizeof(buf)-3; i++, n+=2)
			_hexify(buf+n, ip->mem[i], 1);
		strcpy(buf+n, "??");
		werrstr(buf);
		return 0;
	}
	for(i = 0; i < 2 && op->operand[i]; i++) {
		switch(op->operand[i]) {
		case Ib:	/* 8-bit immediate - (no sign extension)*/
			if (igetc(map, ip, &c) < 0)
				return 0;
			ip->imm = c&0xff;
			ip->imm64 = ip->imm;
			break;
		case Jbs:	/* 8-bit jump immediate (sign extended) */
			if (igetc(map, ip, &c) < 0)
				return 0;
			if (c&0x80)
				ip->imm = c|0xffffff00;
			else
				ip->imm = c&0xff;
			ip->imm64 = (int32)ip->imm;
			ip->jumptype = Jbs;
			break;
		case Ibs:	/* 8-bit immediate (sign extended) */
			if (igetc(map, ip, &c) < 0)
				return 0;
			if (c&0x80)
				if (ip->osize == 'L')
					ip->imm = c|0xffffff00;
				else
					ip->imm = c|0xff00;
			else
				ip->imm = c&0xff;
			ip->imm64 = (int32)ip->imm;
			break;
		case Iw:	/* 16-bit immediate -> imm */
			if (igets(map, ip, &s) < 0)
				return 0;
			ip->imm = s&0xffff;
			ip->imm64 = ip->imm;
			ip->jumptype = Iw;
			break;
		case Iw2:	/* 16-bit immediate -> in imm2*/
			if (igets(map, ip, &s) < 0)
				return 0;
			ip->imm2 = s&0xffff;
			break;
		case Iwd:	/* Operand-sized immediate (no sign extension unless 64 bits)*/
			if (ip->osize == 'L') {
				if (igetl(map, ip, &ip->imm) < 0)
					return 0;
				ip->imm64 = ip->imm;
				if(ip->rex&REXW && (ip->imm & (1<<31)) != 0)
					ip->imm64 |= (vlong)~0 << 32;
			} else {
				if (igets(map, ip, &s)< 0)
					return 0;
				ip->imm = s&0xffff;
				ip->imm64 = ip->imm;
			}
			break;
		case Iwdq:	/* Operand-sized immediate, possibly big */
			if (ip->osize == 'L') {
				if (igetl(map, ip, &ip->imm) < 0)
					return 0;
				ip->imm64 = ip->imm;
				if (ip->rex & REXW) {
					uint32 l;
					if (igetl(map, ip, &l) < 0)
						return 0;
					ip->imm64 |= (uvlong)l << 32;
				}
			} else {
				if (igets(map, ip, &s)< 0)
					return 0;
				ip->imm = s&0xffff;
			}
			break;
		case Awd:	/* Address-sized immediate (no sign extension)*/
			if (ip->asize == 'E') {
				if (igetl(map, ip, &ip->imm) < 0)
					return 0;
				/* TO DO: REX */
			} else {
				if (igets(map, ip, &s)< 0)
					return 0;
				ip->imm = s&0xffff;
			}
			break;
		case Iwds:	/* Operand-sized immediate (sign extended) */
			if (ip->osize == 'L') {
				if (igetl(map, ip, &ip->imm) < 0)
					return 0;
			} else {
				if (igets(map, ip, &s)< 0)
					return 0;
				if (s&0x8000)
					ip->imm = s|0xffff0000;
				else
					ip->imm = s&0xffff;
			}
			ip->jumptype = Iwds;
			break;
		case OA:	/* literal 0x0a byte */
			if (igetc(map, ip, &c) < 0)
				return 0;
			if (c != 0x0a)
				goto badop;
			break;
		case Op_R0:	/* base register must be R0 */
			if (ip->base != 0)
				goto badop;
			break;
		case Op_R1:	/* base register must be R1 */
			if (ip->base != 1)
				goto badop;
			break;
		case RMB:	/* R/M field with byte register (/r)*/
			if (igetc(map, ip, &c) < 0)
				return 0;
			if (modrm(map, ip, c) < 0)
				return 0;
			ip->osize = 'B';
			break;
		case RM:	/* R/M field with register (/r) */
			if (igetc(map, ip, &c) < 0)
				return 0;
			if (modrm(map, ip, c) < 0)
				return 0;
			break;
		case RMOPB:	/* R/M field with op code (/digit) */
			if (igetc(map, ip, &c) < 0)
				return 0;
			if (modrm(map, ip, c) < 0)
				return 0;
			c = ip->reg;		/* secondary op code */
			obase = (Optable*)op->proto;
			ip->osize = 'B';
			goto newop;
		case RMOP:	/* R/M field with op code (/digit) */
			if (igetc(map, ip, &c) < 0)
				return 0;
			if (modrm(map, ip, c) < 0)
				return 0;
			obase = (Optable*)op->proto;
			if(ip->amd64 && obase == optab0F01 && c == 0xF8)
				return optab0F01F8;
			c = ip->reg;
			goto newop;
		case FRMOP:	/* FP R/M field with op code (/digit) */
			if (igetc(map, ip, &c) < 0)
				return 0;
			if (modrm(map, ip, c) < 0)
				return 0;
			if ((c&0xc0) == 0xc0)
				c = ip->reg+8;		/* 16 entry table */
			else
				c = ip->reg;
			obase = (Optable*)op->proto;
			goto newop;
		case FRMEX:	/* Extended FP R/M field with op code (/digit) */
			if (igetc(map, ip, &c) < 0)
				return 0;
			if (modrm(map, ip, c) < 0)
				return 0;
			if ((c&0xc0) == 0xc0)
				c = (c&0x3f)+8;		/* 64-entry table */
			else
				c = ip->reg;
			obase = (Optable*)op->proto;
			goto newop;
		case RMR:	/* R/M register only (mod = 11) */
			if (igetc(map, ip, &c) < 0)
				return 0;
			if ((c&0xc0) != 0xc0) {
				werrstr("invalid R/M register: %x", c);
				return 0;
			}
			if (modrm(map, ip, c) < 0)
				return 0;
			break;
		case RMM:	/* R/M register only (mod = 11) */
			if (igetc(map, ip, &c) < 0)
				return 0;
			if ((c&0xc0) == 0xc0) {
				werrstr("invalid R/M memory mode: %x", c);
				return 0;
			}
			if (modrm(map, ip, c) < 0)
				return 0;
			break;
		case PTR:	/* Seg:Displacement addr (ptr16:16 or ptr16:32) */
			if (ip->osize == 'L') {
				if (igetl(map, ip, &ip->disp) < 0)
					return 0;
			} else {
				if (igets(map, ip, &s)< 0)
					return 0;
				ip->disp = s&0xffff;
			}
			if (igets(map, ip, (ushort*)&ip->seg) < 0)
				return 0;
			ip->jumptype = PTR;
			break;
		case AUXMM:	/* Multi-byte op code; prefix determines table selection */
			if (igetc(map, ip, &c) < 0)
				return 0;
			obase = (Optable*)op->proto;
			switch (ip->opre) {
			case 0x66:	op = optab660F; break;
			case 0xF2:	op = optabF20F; break;
			case 0xF3:	op = optabF30F; break;
			default:	op = nil; break;
			}
			if(op != nil && op[c].proto != nil)
				obase = op;
			norex = 1;	/* no more rex prefixes */
			/* otherwise the optab entry captures it */
			goto newop;
		case AUX:	/* Multi-byte op code - Auxiliary table */
			obase = (Optable*)op->proto;
			if (igetc(map, ip, &c) < 0)
				return 0;
			goto newop;
		case OPRE:	/* Instr Prefix or media op */
			ip->opre = c;
			/* fall through */
		case PRE:	/* Instr Prefix */
			ip->prefix = (char*)op->proto;
			if (igetc(map, ip, &c) < 0)
				return 0;
			if (ip->opre && c == 0x0F)
				ip->prefix = 0;
			goto newop;
		case SEG:	/* Segment Prefix */
			ip->segment = (char*)op->proto;
			if (igetc(map, ip, &c) < 0)
				return 0;
			goto newop;
		case OPOVER:	/* Operand size override */
			ip->opre = c;
			ip->osize = 'W';
			if (igetc(map, ip, &c) < 0)
				return 0;
			if (c == 0x0F)
				ip->osize = 'L';
			else if (ip->amd64 && (c&0xF0) == 0x40)
				ip->osize = 'Q';
			goto newop;
		case ADDOVER:	/* Address size override */
			ip->asize = 0;
			if (igetc(map, ip, &c) < 0)
				return 0;
			goto newop;
		case JUMP:	/* mark instruction as JUMP or RET */
		case RET:
			ip->jumptype = op->operand[i];
			break;
		default:
			werrstr("bad operand type %d", op->operand[i]);
			return 0;
		}
	}
	return op;
}

#pragma	varargck	argpos	bprint		2

static void
bprint(Instr *ip, char *fmt, ...)
{
	va_list arg;

	va_start(arg, fmt);
	ip->curr = vseprint(ip->curr, ip->end, fmt, arg);
	va_end(arg);
}

/*
 *  if we want to call 16 bit regs AX,BX,CX,...
 *  and 32 bit regs EAX,EBX,ECX,... then
 *  change the defs of ANAME and ONAME to:
 *  #define	ANAME(ip)	((ip->asize == 'E' ? "E" : "")
 *  #define	ONAME(ip)	((ip)->osize == 'L' ? "E" : "")
 */
#define	ANAME(ip)	""
#define	ONAME(ip)	""

static char *reg[] =  {
[AX] =	"AX",
[CX] =	"CX",
[DX] =	"DX",
[BX] =	"BX",
[SP] =	"SP",
[BP] =	"BP",
[SI] =	"SI",
[DI] =	"DI",

	/* amd64 */
[AMD64_R8] =	"R8",
[AMD64_R9] =	"R9",
[AMD64_R10] =	"R10",
[AMD64_R11] =	"R11",
[AMD64_R12] =	"R12",
[AMD64_R13] =	"R13",
[AMD64_R14] =	"R14",
[AMD64_R15] =	"R15",
};

static char *breg[] = { "AL", "CL", "DL", "BL", "AH", "CH", "DH", "BH" };
static char *breg64[] = { "AL", "CL", "DL", "BL", "SPB", "BPB", "SIB", "DIB",
	"R8B", "R9B", "R10B", "R11B", "R12B", "R13B", "R14B", "R15B" };
static char *sreg[] = { "ES", "CS", "SS", "DS", "FS", "GS" };

static void
plocal(Instr *ip)
{
	int ret;
	int32 offset;
	Symbol s;
	char *reg;

	offset = ip->disp;
	if (!findsym(ip->addr, CTEXT, &s) || !findlocal(&s, FRAMENAME, &s)) {
		bprint(ip, "%ux(SP)", offset);
		return;
	}

	if (s.value > ip->disp) {
		ret = getauto(&s, s.value-ip->disp-mach->szaddr, CAUTO, &s);
		reg = "(SP)";
	} else {
		offset -= s.value;
		ret = getauto(&s, offset, CPARAM, &s);
		reg = "(FP)";
	}
	if (ret)
		bprint(ip, "%s+", s.name);
	else
		offset = ip->disp;
	bprint(ip, "%ux%s", offset, reg);
}

static int
isjmp(Instr *ip)
{
	switch(ip->jumptype){
	case Iwds:
	case Jbs:
	case JUMP:
		return 1;
	default:
		return 0;
	}
}

/*
 * This is too smart for its own good, but it really is nice
 * to have accurate translations when debugging, and it
 * helps us identify which code is different in binaries that
 * are changed on sources.
 */
static int
issymref(Instr *ip, Symbol *s, int32 w, int32 val)
{
	Symbol next, tmp;
	int32 isstring, size;

	if (isjmp(ip))
		return 1;
	if (s->class==CTEXT && w==0)
		return 1;
	if (s->class==CDATA) {
		/* use first bss symbol (or "end") rather than edata */
		if (s->name[0]=='e' && strcmp(s->name, "edata") == 0){
			if((s ->index >= 0 && globalsym(&tmp, s->index+1) && tmp.value==s->value)
			|| (s->index > 0 && globalsym(&tmp, s->index-1) && tmp.value==s->value))
				*s = tmp;
		}
		if (w == 0)
			return 1;
		for (next=*s; next.value==s->value; next=tmp)
			if (!globalsym(&tmp, next.index+1))
				break;
		size = next.value - s->value;
		if (w >= size)
			return 0;
		if (w > size-w)
			w = size-w;
		/* huge distances are usually wrong except in .string */
		isstring = (s->name[0]=='.' && strcmp(s->name, ".string") == 0);
		if (w > 8192 && !isstring)
			return 0;
		/* medium distances are tricky - look for constants */
		/* near powers of two */
		if ((val&(val-1)) == 0 || (val&(val+1)) == 0)
			return 0;
		return 1;
	}
	return 0;
}

static void
immediate(Instr *ip, vlong val)
{
	Symbol s;
	int32 w;

	if (findsym(val, CANY, &s)) {		/* TO DO */
		w = val - s.value;
		if (w < 0)
			w = -w;
		if (issymref(ip, &s, w, val)) {
			if (w)
				bprint(ip, "%s+%#ux(SB)", s.name, w);
			else
				bprint(ip, "%s(SB)", s.name);
			return;
		}
/*
		if (s.class==CDATA && globalsym(&s, s.index+1)) {
			w = s.value - val;
			if (w < 0)
				w = -w;
			if (w < 4096) {
				bprint(ip, "%s-%#lux(SB)", s.name, w);
				return;
			}
		}
*/
	}
	if((ip->rex & REXW) == 0)
		bprint(ip, "%lux", (long)val);
	else
		bprint(ip, "%llux", val);
}

static void
pea(Instr *ip)
{
	int base;

	base = ip->base;
	if(base >= 0 && (ip->rex & REXB))
		base += 8;

	if (ip->mod == 3) {
		if (ip->osize == 'B')
			bprint(ip, (ip->rex & REXB? breg64: breg)[(uchar)ip->base]);
		else
			bprint(ip, "%s%s", ANAME(ip), reg[base]);
		return;
	}

	if (ip->segment)
		bprint(ip, ip->segment);
	if (ip->asize == 'E' && base == SP)
		plocal(ip);
	else {
		if (ip->base < 0)
			immediate(ip, ip->disp);
		else {
			bprint(ip, "%ux", ip->disp);
			if(ip->rip)
				bprint(ip, "(RIP)");
			bprint(ip,"(%s%s)", ANAME(ip), reg[ip->rex&REXB? ip->base+8: ip->base]);
		}
	}
	if (ip->index >= 0)
		bprint(ip,"(%s%s*%d)", ANAME(ip), reg[ip->rex&REXX? ip->index+8: ip->index], 1<<ip->ss);
}

static void
prinstr(Instr *ip, char *fmt)
{
	int sharp;
	vlong v;

	if (ip->prefix)
		bprint(ip, "%s ", ip->prefix);
	for (; *fmt && ip->curr < ip->end; fmt++) {
		if (*fmt != '%'){
			*ip->curr++ = *fmt;
			continue;
		}
		sharp = 0;
		if(*++fmt == '#') {
			sharp = 1;
			++fmt;
		}
		switch(*fmt){
		case '%':
			*ip->curr++ = '%';
			break;
		case 'A':
			bprint(ip, "%s", ANAME(ip));
			break;
		case 'C':
			bprint(ip, "CR%d", ip->reg);
			break;
		case 'D':
			if (ip->reg < 4 || ip->reg == 6 || ip->reg == 7)
				bprint(ip, "DR%d",ip->reg);
			else
				bprint(ip, "???");
			break;
		case 'I':
			bprint(ip, "$");
			immediate(ip, ip->imm2);
			break;
		case 'O':
			bprint(ip,"%s", ONAME(ip));
			break;
		case 'i':
			if(!sharp)
				bprint(ip, "$");
			v = ip->imm;
			if(ip->rex & REXW)
				v = ip->imm64;
			immediate(ip, v);
			break;
		case 'R':
			bprint(ip, "%s%s", ONAME(ip), reg[ip->rex&REXR? ip->reg+8: ip->reg]);
			break;
		case 'S':
			if(ip->osize == 'Q' || ip->osize == 'L' && ip->rex & REXW)
				bprint(ip, "Q");
			else
				bprint(ip, "%c", ip->osize);
			break;
		case 's':
			if(ip->opre == 0 || ip->opre == 0x66)
				bprint(ip, "P");
			else
				bprint(ip, "S");
			if(ip->opre == 0xf2 || ip->opre == 0x66)
				bprint(ip, "D");
			else
				bprint(ip, "S");
			break;
		case 'T':
			if (ip->reg == 6 || ip->reg == 7)
				bprint(ip, "TR%d",ip->reg);
			else
				bprint(ip, "???");
			break;
		case 'W':
			if (ip->osize == 'Q' || ip->osize == 'L' && ip->rex & REXW)
				bprint(ip, "CDQE");
			else if (ip->osize == 'L')
				bprint(ip,"CWDE");
			else
				bprint(ip, "CBW");
			break;
		case 'd':
			bprint(ip,"%ux:%ux", ip->seg, ip->disp);
			break;
		case 'm':
			if (ip->mod == 3 && ip->osize != 'B') {
				if(fmt[1] != '*'){
					if(ip->opre != 0) {
						bprint(ip, "X%d", ip->rex&REXB? ip->base+8: ip->base);
						break;
					}
				} else
					fmt++;
				bprint(ip, "M%d", ip->base);
				break;
			}
			pea(ip);
			break;
		case 'e':
			pea(ip);
			break;
		case 'f':
			bprint(ip, "F%d", ip->base);
			break;
		case 'g':
			if (ip->reg < 6)
				bprint(ip,"%s",sreg[ip->reg]);
			else
				bprint(ip,"???");
			break;
		case 'p':
			/*
			 * signed immediate in the uint32 ip->imm.
			 */
			v = (int32)ip->imm;
			immediate(ip, v+ip->addr+ip->n);
			break;
		case 'r':
			if (ip->osize == 'B')
				bprint(ip,"%s", (ip->rex? breg64: breg)[ip->rex&REXR? ip->reg+8: ip->reg]);
			else
				bprint(ip, reg[ip->rex&REXR? ip->reg+8: ip->reg]);
			break;
		case 'w':
			if (ip->osize == 'Q' || ip->rex & REXW)
				bprint(ip, "CQO");
			else if (ip->osize == 'L')
				bprint(ip,"CDQ");
			else
				bprint(ip, "CWD");
			break;
		case 'M':
			if(ip->opre != 0)
				bprint(ip, "X%d", ip->rex&REXR? ip->reg+8: ip->reg);
			else
				bprint(ip, "M%d", ip->reg);
			break;
		case 'x':
			if (ip->mod == 3 && ip->osize != 'B') {
				bprint(ip, "X%d", ip->rex&REXB? ip->base+8: ip->base);
				break;
			}
			pea(ip);
			break;
		case 'X':
			bprint(ip, "X%d", ip->rex&REXR? ip->reg+8: ip->reg);
			break;
		default:
			bprint(ip, "%%%c", *fmt);
			break;
		}
	}
	*ip->curr = 0;		/* there's always room for 1 byte */
}

static int
i386inst(Map *map, uvlong pc, char modifier, char *buf, int n)
{
	Instr instr;
	Optable *op;

	USED(modifier);
	op = mkinstr(map, &instr, pc);
	if (op == 0) {
		errstr(buf, n);
		return -1;
	}
	instr.curr = buf;
	instr.end = buf+n-1;
	prinstr(&instr, op->proto);
	return instr.n;
}

static int
i386das(Map *map, uvlong pc, char *buf, int n)
{
	Instr instr;
	int i;

	if (mkinstr(map, &instr, pc) == 0) {
		errstr(buf, n);
		return -1;
	}
	for(i = 0; i < instr.n && n > 2; i++) {
		_hexify(buf, instr.mem[i], 1);
		buf += 2;
		n -= 2;
	}
	*buf = 0;
	return instr.n;
}

static int
i386instlen(Map *map, uvlong pc)
{
	Instr i;

	if (mkinstr(map, &i, pc))
		return i.n;
	return -1;
}

static int
i386foll(Map *map, uvlong pc, Rgetter rget, uvlong *foll)
{
	Instr i;
	Optable *op;
	ushort s;
	uvlong l, addr;
	vlong v;
	int n;

	op = mkinstr(map, &i, pc);
	if (!op)
		return -1;

	n = 0;

	switch(i.jumptype) {
	case RET:		/* RETURN or LEAVE */
	case Iw:		/* RETURN */
		if (strcmp(op->proto, "LEAVE") == 0) {
			if (geta(map, (*rget)(map, "BP"), &l) < 0)
				return -1;
		} else if (geta(map, (*rget)(map, mach->sp), &l) < 0)
			return -1;
		foll[0] = l;
		return 1;
	case Iwds:		/* pc relative JUMP or CALL*/
	case Jbs:		/* pc relative JUMP or CALL */
		v = (int32)i.imm;
		foll[0] = pc+v+i.n;
		n = 1;
		break;
	case PTR:		/* seg:displacement JUMP or CALL */
		foll[0] = (i.seg<<4)+i.disp;
		return 1;
	case JUMP:		/* JUMP or CALL EA */

		if(i.mod == 3) {
			foll[0] = (*rget)(map, reg[i.rex&REXB? i.base+8: i.base]);
			return 1;
		}
			/* calculate the effective address */
		addr = i.disp;
		if (i.base >= 0) {
			if (geta(map, (*rget)(map, reg[i.rex&REXB? i.base+8: i.base]), &l) < 0)
				return -1;
			addr += l;
		}
		if (i.index >= 0) {
			if (geta(map, (*rget)(map, reg[i.rex&REXX? i.index+8: i.index]), &l) < 0)
				return -1;
			addr += l*(1<<i.ss);
		}
			/* now retrieve a seg:disp value at that address */
		if (get2(map, addr, &s) < 0)			/* seg */
			return -1;
		foll[0] = s<<4;
		addr += 2;
		if (i.asize == 'L') {
			if (geta(map, addr, &l) < 0)		/* disp32 */
				return -1;
			foll[0] += l;
		} else {					/* disp16 */
			if (get2(map, addr, &s) < 0)
				return -1;
			foll[0] += s;
		}
		return 1;
	default:
		break;
	}
	if (strncmp(op->proto,"JMP", 3) == 0 || strncmp(op->proto,"CALL", 4) == 0)
		return 1;
	foll[n++] = pc+i.n;
	return n;
}
