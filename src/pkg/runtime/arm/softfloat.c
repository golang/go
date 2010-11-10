// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Software floating point interpretaton of ARM 7500 FP instructions.
// The interpretation is not bit compatible with the 7500.
// It uses true little-endian doubles, while the 7500 used mixed-endian.

#include "runtime.h"

void	abort(void);

static void
fabort(void)
{
	if (1) {
		runtime·printf("Unsupported floating point instruction\n");
		runtime·abort();
	}
}

static uint32 doabort = 0;
static uint32 trace = 0;

static const int8* opnames[] = {
	// binary
	"adf",
	"muf",
	"suf",
	"rsf",
	"dvf",
	"rdf",
	"pow",
	"rpw",
	"rmf",
	"fml",
	"fdv",
	"frd",
	"pol",
	"UNDEFINED",
	"UNDEFINED",
	"UNDEFINED",

	// unary
	"mvf",
	"mnf",
	"abs",
	"rnd",
	"sqt",
	"log",
	"lgn",
	"exp",
	"sin",
	"cos",
	"tan",
	"asn",
	"acs",
	"atn",
	"urd",
	"nrm"
};

static const int8* fpconst[] = {
	"0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "0.5", "10.0",
};

static const uint64 fpdconst[] = {
	0x0000000000000000ll,
	0x3ff0000000000000ll,
	0x4000000000000000ll,
	0x4008000000000000ll,
	0x4010000000000000ll,
	0x4014000000000000ll,
	0x3fe0000000000000ll,
	0x4024000000000000ll
};

static const int8* fpprec[] = {
	"s", "d", "e", "?"
};

static uint32
precision(uint32 i)
{
	switch (i&0x00080080) {
	case 0:
		return 0;
	case 0x80:
		return 1;
	default:
		fabort();
	}
	return 0;
}

static uint64
frhs(uint32 rhs)
{
	if (rhs & 0x8) {
		return  fpdconst[rhs&0x7];
	} else {
		return m->freg[rhs&0x7];
	}
}

static void
fprint(void)
{
	uint32 i;
	for (i = 0; i < 8; i++) {
		runtime·printf("\tf%d:\t%X\n", i, m->freg[i]);
	}
}

static uint32
d2s(uint64 d)
{
	uint32 x;
	
	runtime·f64to32c(d, &x);
	return x;
}

static uint64
s2d(uint32 s)
{
	uint64 x;
	
	runtime·f32to64c(s, &x);
	return x;
}

// cdp, data processing instructions
static void
dataprocess(uint32* pc)
{
	uint32 i, opcode, unary, dest, lhs, rhs, prec;
	uint64 l, r;
	uint64 fd;
	i = *pc;

	// data processing
	opcode = i>>20 & 15;
	unary = i>>15 & 1;

	dest = i>>12 & 7;		
	lhs = i>>16 & 7;
	rhs = i & 15;

	prec = precision(i);
//	if (prec != 1)
//		goto undef;

	if (unary) {
		switch (opcode) {
		case 0: // mvf
			fd = frhs(rhs);
			if(prec == 0)
				fd = s2d(d2s(fd));
			m->freg[dest] = fd;
			goto ret;
		default:
			goto undef;
		}
	} else {
		l = m->freg[lhs];
		r = frhs(rhs);
		switch (opcode) {
		default:
			goto undef;
		case 0:
			runtime·fadd64c(l, r, &m->freg[dest]);
			break;
		case 1:
			runtime·fmul64c(l, r, &m->freg[dest]);
			break;
		case 2:
			runtime·fsub64c(l, r, &m->freg[dest]);
			break;
		case 4:
			runtime·fdiv64c(l, r, &m->freg[dest]);
			break;
		}
		goto ret;
	}


undef:
	doabort = 1;

ret:
	if (trace || doabort) {
		runtime·printf(" %p %x\t%s%s\tf%d, ", pc, *pc, opnames[opcode | unary<<4],
			fpprec[prec], dest);
		if (!unary)
			runtime·printf("f%d, ", lhs);
		if (rhs & 0x8)
			runtime·printf("#%s\n", fpconst[rhs&0x7]);
		else
			runtime·printf("f%d\n", rhs&0x7);
		fprint();
	}
	if (doabort)
		fabort();
}

#define CPSR 14
#define FLAGS_N (1 << 31)
#define FLAGS_Z (1 << 30)
#define FLAGS_C (1 << 29)
#define FLAGS_V (1 << 28)

// cmf, compare floating point
static void
compare(uint32 *pc, uint32 *regs)
{
	uint32 i, flags, lhs, rhs;
	uint64 l, r;
	int32 cmp;
	bool nan;

	i = *pc;
	flags = 0;
	lhs = i>>16 & 0x7;
	rhs = i & 0xf;

	l = m->freg[lhs];
	r = frhs(rhs);
	runtime·fcmp64c(l, r, &cmp, &nan);
	if (nan)
		flags = FLAGS_C | FLAGS_V;
	else if (cmp == 0)
		flags = FLAGS_Z | FLAGS_C;
	else if (cmp < 0)
		flags = FLAGS_N;
	else
		flags = FLAGS_C;

	if (trace) {
		runtime·printf(" %p %x\tcmf\tf%d, ", pc, *pc, lhs);
		if (rhs & 0x8)
			runtime·printf("#%s\n", fpconst[rhs&0x7]);
		else
			runtime·printf("f%d\n", rhs&0x7);
	}
	regs[CPSR] = regs[CPSR] & 0x0fffffff | flags;
}

// ldf/stf, load/store floating
static void
loadstore(uint32 *pc, uint32 *regs)
{
	uint32 i, isload, coproc, ud, wb, tlen, p, reg, freg, offset;
	uint32 addr;

	i = *pc;
	coproc = i>>8&0xf;
	isload = i>>20&1;
	p = i>>24&1;
	ud = i>>23&1;
	tlen = i>>(22 - 1)&2 | i>>15&1;
	wb = i>>21&1;
	reg = i>>16 &0xf;
	freg = i>>12 &0x7;
	offset = (i&0xff) << 2;
	
	if (coproc != 1 || p != 1 || wb != 0 || tlen > 1)
		goto undef;
	if (reg > 13)
		goto undef;

	if (ud)
		addr = regs[reg] + offset;
	else
		addr = regs[reg] - offset;

	if (isload)
		if (tlen)
			m->freg[freg] = *((uint64*)addr);
		else
			m->freg[freg] = s2d(*((uint32*)addr));
	else
		if (tlen)
			*((uint64*)addr) = m->freg[freg];
		else
			*((uint32*)addr) = d2s(m->freg[freg]);
	goto ret;

undef:
	doabort = 1;

ret:
	if (trace || doabort) {
		if (isload)
			runtime·printf(" %p %x\tldf", pc, *pc);
		else
			runtime·printf(" %p %x\tstf", pc, *pc);
		runtime·printf("%s\t\tf%d, %s%d(r%d)", fpprec[tlen], freg, ud ? "" : "-", offset, reg);
		runtime·printf("\t\t// %p", regs[reg] + (ud ? offset : -offset));
		if (coproc != 1 || p != 1 || wb != 0)
			runtime·printf(" coproc: %d pre: %d wb %d", coproc, p, wb);
		runtime·printf("\n");
		fprint();
	}
	if (doabort)
		fabort();
}

static void
fltfix(uint32 *pc, uint32 *regs)
{
	uint32 i, toarm, freg, reg, prec;
	int64 val;
	uint64 f0;
	bool ok;
	
	i = *pc;
	toarm = i>>20 & 0x1;
	freg = i>>16 & 0x7;
	reg = i>>12 & 0xf;
	prec = precision(i);

	if (toarm) { // fix
		f0 = m->freg[freg];
		runtime·f64tointc(f0, &val, &ok);
		if (!ok || (int32)val != val)
			val = 0;
		regs[reg] = val;
	} else { // flt
		runtime·fintto64c((int32)regs[reg], &f0);
		m->freg[freg] = f0;
	}
	goto ret;
	
ret:
	if (trace || doabort) {
		if (toarm)
			runtime·printf(" %p %x\tfix%s\t\tr%d, f%d\n", pc, *pc, fpprec[prec], reg, freg);
		else
			runtime·printf(" %p %x\tflt%s\t\tf%d, r%d\n", pc, *pc, fpprec[prec], freg, reg);
		fprint();
	}
	if (doabort)
		fabort();
}

// returns number of words that the fp instruction is occupying, 0 if next instruction isn't float.
// TODO(kaib): insert sanity checks for coproc 1
static uint32
stepflt(uint32 *pc, uint32 *regs)
{
	uint32 i, c;

//printf("stepflt %p %p\n", pc, *pc);

	i = *pc;

	// unconditional forward branches.
	// inserted by linker after we instrument the code.
	if ((i & 0xff000000) == 0xea000000) {
		if (i & 0x00800000) {
			return 0;
		}
		return (i & 0x007fffff) + 2;
	}
	
	c = i >> 25 & 7;
	switch(c) {
	case 6: // 110
		loadstore(pc, regs);
		return 1;
	case 7: // 111
		if (i>>24 & 1)
			return 0; // ignore swi

		if (i>>4 & 1) { //data transfer
			if ((i&0x00f0ff00) == 0x0090f100) {
				compare(pc, regs);
			} else if ((i&0x00e00f10) == 0x00000110) {
				fltfix(pc, regs);
			} else {
				runtime·printf(" %p %x\t// case 7 fail\n", pc, i);
				fabort();
			}
		} else {
			dataprocess(pc);
		}
		return 1;
	}

	if((i&0xfffff000) == 0xe59fb000) {
		// load r11 from pc-relative address.
		// might be part of a floating point move
		// (or might not, but no harm in simulating
		// one instruction too many).
		regs[11] = *(uint32*)((uint8*)pc + (i&0xfff) + 8);
		return 1;
	}
	
	if(i == 0xe08bb00d) {
		// add sp to 11.
		// might be part of a large stack offset address
		// (or might not, but again no harm done).
		regs[11] += regs[13];
		return 1;
	}

	return 0;
}

#pragma textflag 7
uint32*
runtime·_sfloat2(uint32 *lr, uint32 r0)
{
	uint32 skip;

	while(skip = stepflt(lr, &r0))
		lr += skip;
	return lr;
}

