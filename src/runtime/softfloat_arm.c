// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Software floating point interpretaton of ARM 7500 FP instructions.
// The interpretation is not bit compatible with the 7500.
// It uses true little-endian doubles, while the 7500 used mixed-endian.

#include "runtime.h"
#include "textflag.h"

#define CPSR 14
#define FLAGS_N (1U << 31)
#define FLAGS_Z (1U << 30)
#define FLAGS_C (1U << 29)
#define FLAGS_V (1U << 28)

void	runtime·abort(void);
void	runtime·sqrtC(uint64, uint64*);

static	uint32	trace = 0;

static void
fabort(void)
{
	if (1) {
		runtime·printf("Unsupported floating point instruction\n");
		runtime·abort();
	}
}

static void
putf(uint32 reg, uint32 val)
{
	g->m->freglo[reg] = val;
}

static void
putd(uint32 reg, uint64 val)
{
	g->m->freglo[reg] = (uint32)val;
	g->m->freghi[reg] = (uint32)(val>>32);
}

static uint64
getd(uint32 reg)
{
	return (uint64)g->m->freglo[reg] | ((uint64)g->m->freghi[reg]<<32);
}

static void
fprint(void)
{
	uint32 i;
	for (i = 0; i < 16; i++) {
		runtime·printf("\tf%d:\t%X %X\n", i, g->m->freghi[i], g->m->freglo[i]);
	}
}

static uint32
d2f(uint64 d)
{
	uint32 x;

	runtime·f64to32c(d, &x);
	return x;
}

static uint64
f2d(uint32 f)
{
	uint64 x;

	runtime·f32to64c(f, &x);
	return x;
}

static uint32
fstatus(bool nan, int32 cmp)
{
	if(nan)
		return FLAGS_C | FLAGS_V;
	if(cmp == 0)
		return FLAGS_Z | FLAGS_C;
	if(cmp < 0)
		return FLAGS_N;
	return FLAGS_C;
}

// conditions array record the required CPSR cond field for the
// first 5 pairs of conditional execution opcodes
// higher 4 bits are must set, lower 4 bits are must clear
#pragma dataflag NOPTR
static const uint8 conditions[10/2] = {
	[0/2] = (FLAGS_Z >> 24) | 0, // 0: EQ (Z set), 1: NE (Z clear)
	[2/2] = (FLAGS_C >> 24) | 0, // 2: CS/HS (C set), 3: CC/LO (C clear)
	[4/2] = (FLAGS_N >> 24) | 0, // 4: MI (N set), 5: PL (N clear)
	[6/2] = (FLAGS_V >> 24) | 0, // 6: VS (V set), 7: VC (V clear)
	[8/2] = (FLAGS_C >> 24) | 
	        (FLAGS_Z >> 28),     // 8: HI (C set and Z clear), 9: LS (C clear and Z set)
};

#define FAULT (0x80000000U) // impossible PC offset

// returns number of words that the fp instruction
// is occupying, 0 if next instruction isn't float.
static uint32
stepflt(uint32 *pc, uint32 *regs)
{
	uint32 i, opc, regd, regm, regn, cpsr;
	int32 delta;
	uint32 *addr;
	uint64 uval;
	int64 sval;
	bool nan, ok;
	int32 cmp;
	M *m;

	// m is locked in vlop_arm.s, so g->m cannot change during this function call,
	// so caching it in a local variable is safe.
	m = g->m;
	i = *pc;

	if(trace)
		runtime·printf("stepflt %p %x (cpsr %x)\n", pc, i, regs[CPSR] >> 28);

	opc = i >> 28;
	if(opc == 14) // common case first
		goto execute;
	cpsr = regs[CPSR] >> 28;
	switch(opc) {
	case 0: case 1: case 2: case 3: case 4: 
	case 5: case 6: case 7: case 8: case 9:
		if(((cpsr & (conditions[opc/2] >> 4)) == (conditions[opc/2] >> 4)) &&
		   ((cpsr & (conditions[opc/2] & 0xf)) == 0)) {
			if(opc & 1) return 1;
		} else {
			if(!(opc & 1)) return 1;
		}
		break;
	case 10: // GE (N == V)
	case 11: // LT (N != V)
		if((cpsr & (FLAGS_N >> 28)) == (cpsr & (FLAGS_V >> 28))) {
			if(opc & 1) return 1;
		} else {
			if(!(opc & 1)) return 1;
		}
		break;
	case 12: // GT (N == V and Z == 0)
	case 13: // LE (N != V or Z == 1)
		if((cpsr & (FLAGS_N >> 28)) == (cpsr & (FLAGS_V >> 28)) &&
		   (cpsr & (FLAGS_Z >> 28)) == 0) {
			if(opc & 1) return 1;
		} else {
			if(!(opc & 1)) return 1;
		}
		break;
	case 14: // AL
		break;
	case 15: // shouldn't happen
		return 0;
	}
	if(trace)
		runtime·printf("conditional %x (cpsr %x) pass\n", opc, cpsr);
	i = (0xeU << 28) | (i & 0xfffffff);

execute:
	// special cases
	if((i&0xfffff000) == 0xe59fb000) {
		// load r11 from pc-relative address.
		// might be part of a floating point move
		// (or might not, but no harm in simulating
		// one instruction too many).
		addr = (uint32*)((uint8*)pc + (i&0xfff) + 8);
		regs[11] = addr[0];

		if(trace)
			runtime·printf("*** cpu R[%d] = *(%p) %x\n",
				11, addr, regs[11]);
		return 1;
	}
	if(i == 0xe08bb00d) {
		// add sp to r11.
		// might be part of a large stack offset address
		// (or might not, but again no harm done).
		regs[11] += regs[13];

		if(trace)
			runtime·printf("*** cpu R[%d] += R[%d] %x\n",
				11, 13, regs[11]);
		return 1;
	}
	if(i == 0xeef1fa10) {
		regs[CPSR] = (regs[CPSR]&0x0fffffff) | m->fflag;

		if(trace)
			runtime·printf("*** fpsr R[CPSR] = F[CPSR] %x\n", regs[CPSR]);
		return 1;
	}
	if((i&0xff000000) == 0xea000000) {
		// unconditional branch
		// can happen in the middle of floating point
		// if the linker decides it is time to lay down
		// a sequence of instruction stream constants.
		delta = i&0xffffff;
		delta = (delta<<8) >> 8;	// sign extend

		if(trace)
			runtime·printf("*** cpu PC += %x\n", (delta+2)*4);
		return delta+2;
	}

	goto stage1;

stage1:	// load/store regn is cpureg, regm is 8bit offset
	regd = i>>12 & 0xf;
	regn = i>>16 & 0xf;
	regm = (i & 0xff) << 2;	// PLUS or MINUS ??

	switch(i & 0xfff00f00) {
	default:
		goto stage2;

	case 0xed900a00:	// single load
		addr = (uint32*)(regs[regn] + regm);
		if((uintptr)addr < 4096) {
			if(trace)
				runtime·printf("*** load @%p => fault\n", addr);
			return FAULT;
		}
		m->freglo[regd] = addr[0];

		if(trace)
			runtime·printf("*** load F[%d] = %x\n",
				regd, m->freglo[regd]);
		break;

	case 0xed900b00:	// double load
		addr = (uint32*)(regs[regn] + regm);
		if((uintptr)addr < 4096) {
			if(trace)
				runtime·printf("*** double load @%p => fault\n", addr);
			return FAULT;
		}
		m->freglo[regd] = addr[0];
		m->freghi[regd] = addr[1];

		if(trace)
			runtime·printf("*** load D[%d] = %x-%x\n",
				regd, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xed800a00:	// single store
		addr = (uint32*)(regs[regn] + regm);
		if((uintptr)addr < 4096) {
			if(trace)
				runtime·printf("*** store @%p => fault\n", addr);
			return FAULT;
		}
		addr[0] = m->freglo[regd];

		if(trace)
			runtime·printf("*** *(%p) = %x\n",
				addr, addr[0]);
		break;

	case 0xed800b00:	// double store
		addr = (uint32*)(regs[regn] + regm);
		if((uintptr)addr < 4096) {
			if(trace)
				runtime·printf("*** double store @%p => fault\n", addr);
			return FAULT;
		}
		addr[0] = m->freglo[regd];
		addr[1] = m->freghi[regd];

		if(trace)
			runtime·printf("*** *(%p) = %x-%x\n",
				addr, addr[1], addr[0]);
		break;
	}
	return 1;

stage2:	// regd, regm, regn are 4bit variables
	regm = i>>0 & 0xf;
	switch(i & 0xfff00ff0) {
	default:
		goto stage3;

	case 0xf3000110:	// veor
		m->freglo[regd] = m->freglo[regm]^m->freglo[regn];
		m->freghi[regd] = m->freghi[regm]^m->freghi[regn];

		if(trace)
			runtime·printf("*** veor D[%d] = %x-%x\n",
				regd, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xeeb00b00:	// D[regd] = const(regn,regm)
		regn = (regn<<4) | regm;
		regm = 0x40000000UL;
		if(regn & 0x80)
			regm |= 0x80000000UL;
		if(regn & 0x40)
			regm ^= 0x7fc00000UL;
		regm |= (regn & 0x3f) << 16;
		m->freglo[regd] = 0;
		m->freghi[regd] = regm;

		if(trace)
			runtime·printf("*** immed D[%d] = %x-%x\n",
				regd, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xeeb00a00:	// F[regd] = const(regn,regm)
		regn = (regn<<4) | regm;
		regm = 0x40000000UL;
		if(regn & 0x80)
			regm |= 0x80000000UL;
		if(regn & 0x40)
			regm ^= 0x7e000000UL;
		regm |= (regn & 0x3f) << 19;
		m->freglo[regd] = regm;

		if(trace)
			runtime·printf("*** immed D[%d] = %x\n",
				regd, m->freglo[regd]);
		break;

	case 0xee300b00:	// D[regd] = D[regn]+D[regm]
		runtime·fadd64c(getd(regn), getd(regm), &uval);
		putd(regd, uval);

		if(trace)
			runtime·printf("*** add D[%d] = D[%d]+D[%d] %x-%x\n",
				regd, regn, regm, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xee300a00:	// F[regd] = F[regn]+F[regm]
		runtime·fadd64c(f2d(m->freglo[regn]), f2d(m->freglo[regm]), &uval);
		m->freglo[regd] = d2f(uval);

		if(trace)
			runtime·printf("*** add F[%d] = F[%d]+F[%d] %x\n",
				regd, regn, regm, m->freglo[regd]);
		break;

	case 0xee300b40:	// D[regd] = D[regn]-D[regm]
		runtime·fsub64c(getd(regn), getd(regm), &uval);
		putd(regd, uval);

		if(trace)
			runtime·printf("*** sub D[%d] = D[%d]-D[%d] %x-%x\n",
				regd, regn, regm, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xee300a40:	// F[regd] = F[regn]-F[regm]
		runtime·fsub64c(f2d(m->freglo[regn]), f2d(m->freglo[regm]), &uval);
		m->freglo[regd] = d2f(uval);

		if(trace)
			runtime·printf("*** sub F[%d] = F[%d]-F[%d] %x\n",
				regd, regn, regm, m->freglo[regd]);
		break;

	case 0xee200b00:	// D[regd] = D[regn]*D[regm]
		runtime·fmul64c(getd(regn), getd(regm), &uval);
		putd(regd, uval);

		if(trace)
			runtime·printf("*** mul D[%d] = D[%d]*D[%d] %x-%x\n",
				regd, regn, regm, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xee200a00:	// F[regd] = F[regn]*F[regm]
		runtime·fmul64c(f2d(m->freglo[regn]), f2d(m->freglo[regm]), &uval);
		m->freglo[regd] = d2f(uval);

		if(trace)
			runtime·printf("*** mul F[%d] = F[%d]*F[%d] %x\n",
				regd, regn, regm, m->freglo[regd]);
		break;

	case 0xee800b00:	// D[regd] = D[regn]/D[regm]
		runtime·fdiv64c(getd(regn), getd(regm), &uval);
		putd(regd, uval);

		if(trace)
			runtime·printf("*** div D[%d] = D[%d]/D[%d] %x-%x\n",
				regd, regn, regm, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xee800a00:	// F[regd] = F[regn]/F[regm]
		runtime·fdiv64c(f2d(m->freglo[regn]), f2d(m->freglo[regm]), &uval);
		m->freglo[regd] = d2f(uval);

		if(trace)
			runtime·printf("*** div F[%d] = F[%d]/F[%d] %x\n",
				regd, regn, regm, m->freglo[regd]);
		break;

	case 0xee000b10:	// S[regn] = R[regd] (MOVW) (regm ignored)
		m->freglo[regn] = regs[regd];

		if(trace)
			runtime·printf("*** cpy S[%d] = R[%d] %x\n",
				regn, regd, m->freglo[regn]);
		break;

	case 0xee100b10:	// R[regd] = S[regn] (MOVW) (regm ignored)
		regs[regd] = m->freglo[regn];

		if(trace)
			runtime·printf("*** cpy R[%d] = S[%d] %x\n",
				regd, regn, regs[regd]);
		break;
	}
	return 1;

stage3:	// regd, regm are 4bit variables
	switch(i & 0xffff0ff0) {
	default:
		goto done;

	case 0xeeb00a40:	// F[regd] = F[regm] (MOVF)
		m->freglo[regd] = m->freglo[regm];

		if(trace)
			runtime·printf("*** F[%d] = F[%d] %x\n",
				regd, regm, m->freglo[regd]);
		break;

	case 0xeeb00b40:	// D[regd] = D[regm] (MOVD)
		m->freglo[regd] = m->freglo[regm];
		m->freghi[regd] = m->freghi[regm];

		if(trace)
			runtime·printf("*** D[%d] = D[%d] %x-%x\n",
				regd, regm, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xeeb10bc0:	// D[regd] = sqrt D[regm]
		runtime·sqrtC(getd(regm), &uval);
		putd(regd, uval);

		if(trace)
			runtime·printf("*** D[%d] = sqrt D[%d] %x-%x\n",
				regd, regm, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xeeb00bc0:	// D[regd] = abs D[regm]
		m->freglo[regd] = m->freglo[regm];
		m->freghi[regd] = m->freghi[regm] & ((1<<31)-1);

		if(trace)
			runtime·printf("*** D[%d] = abs D[%d] %x-%x\n",
					regd, regm, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xeeb00ac0:	// F[regd] = abs F[regm]
		m->freglo[regd] = m->freglo[regm] & ((1<<31)-1);

		if(trace)
			runtime·printf("*** F[%d] = abs F[%d] %x\n",
					regd, regm, m->freglo[regd]);
		break;

	case 0xeeb40bc0:	// D[regd] :: D[regm] (CMPD)
		runtime·fcmp64c(getd(regd), getd(regm), &cmp, &nan);
		m->fflag = fstatus(nan, cmp);

		if(trace)
			runtime·printf("*** cmp D[%d]::D[%d] %x\n",
				regd, regm, m->fflag);
		break;

	case 0xeeb40ac0:	// F[regd] :: F[regm] (CMPF)
		runtime·fcmp64c(f2d(m->freglo[regd]), f2d(m->freglo[regm]), &cmp, &nan);
		m->fflag = fstatus(nan, cmp);

		if(trace)
			runtime·printf("*** cmp F[%d]::F[%d] %x\n",
				regd, regm, m->fflag);
		break;

	case 0xeeb70ac0:	// D[regd] = F[regm] (MOVFD)
		putd(regd, f2d(m->freglo[regm]));

		if(trace)
			runtime·printf("*** f2d D[%d]=F[%d] %x-%x\n",
				regd, regm, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xeeb70bc0:	// F[regd] = D[regm] (MOVDF)
		m->freglo[regd] = d2f(getd(regm));

		if(trace)
			runtime·printf("*** d2f F[%d]=D[%d] %x-%x\n",
				regd, regm, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xeebd0ac0:	// S[regd] = F[regm] (MOVFW)
		runtime·f64tointc(f2d(m->freglo[regm]), &sval, &ok);
		if(!ok || (int32)sval != sval)
			sval = 0;
		m->freglo[regd] = sval;

		if(trace)
			runtime·printf("*** fix S[%d]=F[%d] %x\n",
				regd, regm, m->freglo[regd]);
		break;

	case 0xeebc0ac0:	// S[regd] = F[regm] (MOVFW.U)
		runtime·f64tointc(f2d(m->freglo[regm]), &sval, &ok);
		if(!ok || (uint32)sval != sval)
			sval = 0;
		m->freglo[regd] = sval;

		if(trace)
			runtime·printf("*** fix unsigned S[%d]=F[%d] %x\n",
				regd, regm, m->freglo[regd]);
		break;

	case 0xeebd0bc0:	// S[regd] = D[regm] (MOVDW)
		runtime·f64tointc(getd(regm), &sval, &ok);
		if(!ok || (int32)sval != sval)
			sval = 0;
		m->freglo[regd] = sval;

		if(trace)
			runtime·printf("*** fix S[%d]=D[%d] %x\n",
				regd, regm, m->freglo[regd]);
		break;

	case 0xeebc0bc0:	// S[regd] = D[regm] (MOVDW.U)
		runtime·f64tointc(getd(regm), &sval, &ok);
		if(!ok || (uint32)sval != sval)
			sval = 0;
		m->freglo[regd] = sval;

		if(trace)
			runtime·printf("*** fix unsigned S[%d]=D[%d] %x\n",
				regd, regm, m->freglo[regd]);
		break;

	case 0xeeb80ac0:	// D[regd] = S[regm] (MOVWF)
		cmp = m->freglo[regm];
		if(cmp < 0) {
			runtime·fintto64c(-cmp, &uval);
			putf(regd, d2f(uval));
			m->freglo[regd] ^= 0x80000000;
		} else {
			runtime·fintto64c(cmp, &uval);
			putf(regd, d2f(uval));
		}

		if(trace)
			runtime·printf("*** float D[%d]=S[%d] %x-%x\n",
				regd, regm, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xeeb80a40:	// D[regd] = S[regm] (MOVWF.U)
		runtime·fintto64c(m->freglo[regm], &uval);
		putf(regd, d2f(uval));

		if(trace)
			runtime·printf("*** float unsigned D[%d]=S[%d] %x-%x\n",
				regd, regm, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xeeb80bc0:	// D[regd] = S[regm] (MOVWD)
		cmp = m->freglo[regm];
		if(cmp < 0) {
			runtime·fintto64c(-cmp, &uval);
			putd(regd, uval);
			m->freghi[regd] ^= 0x80000000;
		} else {
			runtime·fintto64c(cmp, &uval);
			putd(regd, uval);
		}

		if(trace)
			runtime·printf("*** float D[%d]=S[%d] %x-%x\n",
				regd, regm, m->freghi[regd], m->freglo[regd]);
		break;

	case 0xeeb80b40:	// D[regd] = S[regm] (MOVWD.U)
		runtime·fintto64c(m->freglo[regm], &uval);
		putd(regd, uval);

		if(trace)
			runtime·printf("*** float unsigned D[%d]=S[%d] %x-%x\n",
				regd, regm, m->freghi[regd], m->freglo[regd]);
		break;
	}
	return 1;

done:
	if((i&0xff000000) == 0xee000000 ||
	   (i&0xff000000) == 0xed000000) {
		runtime·printf("stepflt %p %x\n", pc, i);
		fabort();
	}
	return 0;
}

typedef struct Sfregs Sfregs;

// NOTE: These are all recorded as pointers because they are possibly live registers,
// and we don't know what they contain. Recording them as pointers should be
// safer than not.
struct Sfregs
{
	uint32 *r0;
	uint32 *r1;
	uint32 *r2;
	uint32 *r3;
	uint32 *r4;
	uint32 *r5;
	uint32 *r6;
	uint32 *r7;
	uint32 *r8;
	uint32 *r9;
	uint32 *r10;
	uint32 *r11;
	uint32 *r12;
	uint32 *r13;
	uint32 cspr;
};

static void sfloat2(void);
void _sfloatpanic(void);

#pragma textflag NOSPLIT
uint32*
runtime·_sfloat2(uint32 *pc, Sfregs regs)
{
	void (*fn)(void);
	
	g->m->ptrarg[0] = pc;
	g->m->ptrarg[1] = &regs;
	fn = sfloat2;
	runtime·onM(&fn);
	pc = g->m->ptrarg[0];
	g->m->ptrarg[0] = nil;
	return pc;
}

static void
sfloat2(void)
{
	uint32 *pc;
	G *curg;
	Sfregs *regs;
	int32 skip;
	bool first;
	
	pc = g->m->ptrarg[0];
	regs = g->m->ptrarg[1];
	g->m->ptrarg[0] = nil;
	g->m->ptrarg[1] = nil;

	first = true;
	while(skip = stepflt(pc, (uint32*)&regs->r0)) {
		first = false;
		if(skip == FAULT) {
			// Encountered bad address in store/load.
			// Record signal information and return to assembly
			// trampoline that fakes the call.
			enum { SIGSEGV = 11 };
			curg = g->m->curg;
			curg->sig = SIGSEGV;
			curg->sigcode0 = 0;
			curg->sigcode1 = 0;
			curg->sigpc = (uint32)pc;
			pc = (uint32*)_sfloatpanic;
			break;
		}
		pc += skip;
	}
	if(first) {
		runtime·printf("sfloat2 %p %x\n", pc, *pc);
		fabort(); // not ok to fail first instruction
	}
		
	g->m->ptrarg[0] = pc;
}
