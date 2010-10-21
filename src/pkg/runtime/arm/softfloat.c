// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

void	abort(void);

static void
fabort(void)
{
	if (1) {
		printf("Unsupported floating point instruction\n");
		abort();
	}
}

static uint32 doabort = 0;
static uint32 trace = 0;

#define DOUBLE_EXPBIAS 1023
#define DOUBLE_MANT_MASK 0xfffffffffffffull
#define DOUBLE_MANT_TOP_BIT 0x10000000000000ull
#define DZERO 0x0000000000000000ull
#define DNZERO 0x8000000000000000ull
#define DONE 0x3ff0000000000000ull
#define DINF 0x7ff0000000000000ull
#define DNINF 0xfff0000000000000ull
#define DNAN 0x7FF0000000000001ull

#define SINGLE_EXPBIAS 127
#define FINF 0x7f800000ul
#define FNINF 0xff800000ul
#define FNAN 0x7f800000ul


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

static int32
fexp(uint64 f)
{
	return (int32)((uint32)(f >> 52) & 0x7ff) - DOUBLE_EXPBIAS;
}

static uint32
fsign(uint64 f)
{
	return (uint32)(f >> 63) & 0x1;
}

static uint64
fmantissa(uint64 f)
{
	return f &0x000fffffffffffffll;
}

static void
fprint(void)
{
	uint32 i;
	for (i = 0; i < 8; i++) {
		printf("\tf%d:\t%X\n", i, m->freg[i]);
	}
}

static uint32
d2s(uint64 d)
{
	if ((d & ~(1ull << 63)) == 0)
		return (uint32)(d>>32);
	if (d == DINF)
		return FINF;
	if (d == DNINF)
		return FNINF;
	if ((d & ~(1ull << 63)) == DNAN)
		return FNAN;
	return (d>>32 & 0x80000000) |	//sign
		((uint32)(fexp(d) + SINGLE_EXPBIAS) & 0xff) << 23 |	// exponent
		(d >> 29 & 0x7fffff);	// mantissa
}

static uint64
s2d(uint32 s)
{
	if ((s & ~(1ul << 31)) == 0)
		return (uint64)(s) << 32;
	if (s == FINF)
		return DINF;
	if (s == FNINF)
		return DNINF;
	if ((s & ~(1ul << 31)) == FNAN)
		return DNAN;
	return (uint64)(s & 0x80000000) << 32 |	// sign
		(uint64)((s >> 23 &0xff) + (DOUBLE_EXPBIAS - SINGLE_EXPBIAS)) << 52  |	// exponent
		(uint64)(s & 0x7fffff) << 29;	// mantissa
}

static int64
rsh(int64 f, int32 s)
{
	if (s >= 0)
		return f>>s;
	else
		return f<<-s;
}

// cdp, data processing instructions
static void
dataprocess(uint32* pc)
{
	uint32 i, opcode, unary, dest, lhs, rhs, prec;
	uint32 high;
	int32 expd, exp0, exp1;
	uint64 fraw0, fraw1, exp, sign;
	uint64 fd, f0, f1;
	int64 fsd, fs0, fs1;

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
		fraw0 = m->freg[lhs];
		fraw1 = frhs(rhs);
		if (isNaN(float64frombits(fraw0)) || isNaN(float64frombits(fraw1))) {
			m->freg[dest] = DNAN;
			goto ret;
		}
		switch (opcode) {
		case 2: // suf
			fraw1 ^= 0x1ll << 63;
			// fallthrough
		case 0: // adf
			if (fraw0 == DZERO || fraw0 == DNZERO) {
				m->freg[dest] = fraw1;
				goto ret;
			}
			if (fraw1 == DZERO || fraw1 == DNZERO) {
				m->freg[dest] = fraw0;
				goto ret;
			}
			fs0 = fraw0 & DOUBLE_MANT_MASK | DOUBLE_MANT_TOP_BIT;
			fs1 = fraw1 & DOUBLE_MANT_MASK | DOUBLE_MANT_TOP_BIT;
			exp0 = fexp(fraw0);
			exp1 = fexp(fraw1);
			if (exp0 > exp1)
				fs1 = rsh(fs1, exp0-exp1);
			else
				fs0 = rsh(fs0, exp1-exp0);
			if (fraw0 & 0x1ll<<63)
				fs0 = -fs0;
			if (fraw1 & 0x1ll<<63)
				fs1 = -fs1;
			fsd = fs0 + fs1;
			if (fsd == 0) {
				m->freg[dest] = DZERO;
				goto ret;
			}
			sign = (uint64)fsd & 0x1ll<<63;
			if (fsd < 0)
				fsd = -fsd;
			for (expd = 55; expd > 0; expd--) {
				if (0x1ll<<expd & fsd)
					break;
			}
			if (expd - 52 < 0)
				fsd <<= -(expd - 52);
			else
				fsd >>= expd - 52;
			if (exp0 > exp1)
				exp = expd + exp0 - 52;
			else
				exp = expd + exp1 - 52;
			// too small value, can't represent
			if (1<<31 & expd) {
				m->freg[dest] = DZERO;
				goto ret;
			}
			// infinity
			if (expd > 1<<12) {
				m->freg[dest] = DINF;
				goto ret;
			}
			fd = sign | (exp + DOUBLE_EXPBIAS)<<52 | (uint64)fsd & DOUBLE_MANT_MASK;
			m->freg[dest] = fd;
			goto ret;

		case 4: //dvf
			if ((fraw1 & ~(1ull<<63)) == 0) {
				if ((fraw0 & ~(1ull<<63)) == 0) {
					m->freg[dest] = DNAN;
				} else {
					sign = fraw0 & 1ull<<63 ^ fraw1 & 1ull<<63;
					m->freg[dest] = sign | DINF;
				}
				goto ret;
			}
			// reciprocal for fraw1
			if (fraw1 == DONE)
				goto muf;
			f0 = 0x1ll << 63;
			f1 = fraw1 & DOUBLE_MANT_MASK | DOUBLE_MANT_TOP_BIT;
			f1 >>= 21;
			fd = f0/f1;
			fd <<= 21;
			fd &= DOUBLE_MANT_MASK;
			exp1 = -fexp(fraw1) - 1;
			sign = fraw1 & 0x1ll<<63;
			fraw1 = sign | (uint64)(exp1 + DOUBLE_EXPBIAS)<<52 | fd;
			// fallthrough
		case 1: // muf
muf:			
			if (fraw0 == DNZERO || fraw1 == DNZERO) {
				m->freg[dest] = DNZERO;
				goto ret;
			}
			if (fraw0 == DZERO || fraw1 == DZERO) {
				m->freg[dest] = DZERO;
				goto ret;
			}
			if (fraw0 == DONE) {
				m->freg[dest] = fraw1;
				goto ret;
			}
			if (fraw1 == DONE) {
				m->freg[dest] = fraw0;
				goto ret;
			}
			f0 = fraw0>>21 & 0x7fffffff | 0x1ll<<31;
			f1 = fraw1>>21 & 0x7fffffff | 0x1ll<<31;
			fd = f0*f1;
			high = fd >> 63;
			if (high)
				fd = fd >> 11 & DOUBLE_MANT_MASK;
			else
				fd = fd >> 10 & DOUBLE_MANT_MASK;
			exp = (uint64)(fexp(fraw0) + fexp(fraw1) + !!high + DOUBLE_EXPBIAS) & 0x7ff;
			sign = fraw0 >> 63 ^ fraw1 >> 63;
			fd = sign<<63 | exp<<52 | fd;
			m->freg[dest] = fd;
			goto ret;

		default:
			goto undef;
		}
	}


undef:
	doabort = 1;

ret:
	if (trace || doabort) {
		printf(" %p %x\t%s%s\tf%d, ", pc, *pc, opnames[opcode | unary<<4],
			fpprec[prec], dest);
		if (!unary)
			printf("f%d, ", lhs);
		if (rhs & 0x8)
			printf("#%s\n", fpconst[rhs&0x7]);
		else
			printf("f%d\n", rhs&0x7);
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
	uint32 i, flags, lhs, rhs, sign0, sign1;
	uint64 f0, f1, mant0, mant1;
	int32 exp0, exp1;

	i = *pc;
	flags = 0;
	lhs = i>>16 & 0x7;
	rhs = i & 0xf;

	f0 = m->freg[lhs];
	f1 = frhs(rhs);
	if (isNaN(float64frombits(f0)) || isNaN(float64frombits(f1))) {
		flags = FLAGS_C | FLAGS_V;
		goto ret;
	}
	if (f0 == f1) {
		flags = FLAGS_Z | FLAGS_C;
		goto ret;
	}

	sign0 = fsign(f0);
	sign1 = fsign(f1);
	if (sign0 == 1 && sign1 == 0) {
		flags = FLAGS_N;
		goto ret;
	}
	if (sign0 == 0 && sign1 == 1) {
		flags = FLAGS_C;
		goto ret;
	}

	if (sign0 == 0) {
		exp0 = fexp(f0);
		exp1 = fexp(f1);
		mant0 = fmantissa(f0);
		mant1 = fmantissa(f1);
	} else {
		exp0 = fexp(f1);
		exp1 = fexp(f0);
		mant0 = fmantissa(f1);
		mant1 = fmantissa(f0);
	}

	if (exp0 > exp1) {
		flags = FLAGS_C;
	} else if (exp0 < exp1) {
		flags = FLAGS_N;
	} else {
		if (mant0 > mant1)
			flags = FLAGS_C;
		else
			flags = FLAGS_N;
	}

ret:
	if (trace) {
		printf(" %p %x\tcmf\tf%d, ", pc, *pc, lhs);
		if (rhs & 0x8)
			printf("#%s\n", fpconst[rhs&0x7]);
		else
			printf("f%d\n", rhs&0x7);
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
			printf(" %p %x\tldf", pc, *pc);
		else
			printf(" %p %x\tstf", pc, *pc);
		printf("%s\t\tf%d, %s%d(r%d)", fpprec[tlen], freg, ud ? "" : "-", offset, reg);
		printf("\t\t// %p", regs[reg] + (ud ? offset : -offset));
		if (coproc != 1 || p != 1 || wb != 0)
			printf(" coproc: %d pre: %d wb %d", coproc, p, wb);
		printf("\n");
		fprint();
	}
	if (doabort)
		fabort();
}

static void
fltfix(uint32 *pc, uint32 *regs)
{
	uint32 i, toarm, freg, reg, sign, val, prec;
	int32 rd, exp;
	uint64 fd, f0;
	
	i = *pc;
	toarm = i>>20 & 0x1;
	freg = i>>16 & 0x7;
	reg = i>>12 & 0xf;
	prec = precision(i);

	if (toarm) { //fix
		f0 = m->freg[freg];
		fd = f0 & DOUBLE_MANT_MASK | DOUBLE_MANT_TOP_BIT;
		exp = fexp(f0) - 52;
		if (exp < 0)
			fd = fd>>(-exp);
		else
			fd = fd<<exp;
		rd = ((int32)fd & 0x7fffffff);
		if (f0 & 0x1ll<<63)
			rd = -rd;
		regs[reg] = (uint32)rd;
	} else { // flt
		if (regs[reg] == 0) {
			m->freg[freg] = DZERO;
			goto ret;
		}
		sign = regs[reg] >> 31 & 0x1;
		val = regs[reg];
		if (sign) val = -val;
		for (exp = 31; exp >= 0; exp--) {
			if (1<<(exp) & val)
				break;
		}
		fd = (uint64)val<<(52-exp) & DOUBLE_MANT_MASK;
		m->freg[freg] = (uint64)(sign) << 63 | 
			(uint64)(exp + DOUBLE_EXPBIAS) << 52 | fd;
	}
	goto ret;
	
ret:
	if (trace || doabort) {
		if (toarm)
			printf(" %p %x\tfix%s\t\tr%d, f%d\n", pc, *pc, fpprec[prec], reg, freg);
		else
			printf(" %p %x\tflt%s\t\tf%d, r%d\n", pc, *pc, fpprec[prec], freg, reg);
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
		if (i>>24 & 1) return 0; // ignore swi

		if (i>>4 & 1) { //data transfer
			if ((i&0x00f0ff00) == 0x0090f100) {
				compare(pc, regs);
			} else if ((i&0x00e00f10) == 0x00000110) {
				fltfix(pc, regs);
			} else {
				printf(" %p %x\t// case 7 fail\n", pc, i);
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

	return 0;
}

#pragma textflag 7
uint32*
_sfloat2(uint32 *lr, uint32 r0)
{
	uint32 skip;
//	uint32 cpsr;

	while(skip = stepflt(lr, &r0)) {
		lr += skip;
	}
	return lr;
}

