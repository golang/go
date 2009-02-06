// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

extern int32	debug;

extern uint8 end;

void
traceback(byte *pc0, byte *sp, G *g)
{
	Stktop *stk;
	uint64 pc;
	int32 i, n;
	Func *f;
	byte *p;

	pc = (uint64)pc0;

	// If the PC is zero, it's likely a nil function call.
	// Start in the caller's frame.
	if(pc == 0) {
		pc = *(uint64*)sp;
		sp += 8;
	}

	stk = (Stktop*)g->stackbase;
	for(n=0; n<100; n++) {
		while(pc == (uint64)retfromnewstack) {
			// pop to earlier stack block
			sp = stk->oldsp;
			stk = (Stktop*)stk->oldbase;
			pc = *(uint64*)(sp+8);
			sp += 16;	// two irrelevant calls on stack: morestack plus its call
		}
		f = findfunc(pc);
		if(f == nil) {
			// dangerous, but poke around to see if it is a closure
			p = (byte*)pc;
			// ADDQ $xxx, SP; RET
			if(p[0] == 0x48 && p[1] == 0x81 && p[2] == 0xc4 && p[7] == 0xc3) {
				sp += *(uint32*)(p+3) + 8;
				pc = *(uint64*)(sp - 8);
				if(pc <= 0x1000)
					return;
				continue;
			}
			printf("%p unknown pc\n", pc);
			return;
		}
		if(f->frame < 8)	// assembly funcs say 0 but lie
			sp += 8;
		else
			sp += f->frame;

		// print this frame
		//	main+0xf /home/rsc/go/src/runtime/x.go:23
		//		main(0x1, 0x2, 0x3)
		printf("%S", f->name);
		if(pc > f->entry)
			printf("+%X", pc - f->entry);
		printf(" %S:%d\n", f->src, funcline(f, pc-1));	// -1 to get to CALL instr.
		printf("\t%S(", f->name);
		for(i = 0; i < f->args; i++) {
			if(i != 0)
				prints(", ");
			sys·printhex(((uint32*)sp)[i]);
			if(i >= 4) {
				prints(", ...");
				break;
			}
		}
		prints(")\n");

		pc = *(uint64*)(sp-8);
		if(pc <= 0x1000)
			return;
	}
	prints("...\n");
}

// func caller(n int) (pc uint64, file string, line int, ok bool)
void
sys·Caller(int32 n, uint64 retpc, string retfile, int32 retline, bool retbool)
{
	uint64 pc;
	byte *sp;
	byte *p;
	Stktop *stk;
	Func *f;

	// our caller's pc, sp.
	sp = (byte*)&n;
	pc = *(uint64*)(sp-8);
	if((f = findfunc(pc)) == nil) {
	error:
		retpc = 0;
		retline = 0;
		retfile = nil;
		retbool = false;
		FLUSH(&retpc);
		FLUSH(&retfile);
		FLUSH(&retline);
		FLUSH(&retbool);
		return;
	}

	// now unwind n levels
	stk = (Stktop*)g->stackbase;
	while(n-- > 0) {
		while(pc == (uint64)retfromnewstack) {
			sp = stk->oldsp;
			stk = (Stktop*)stk->oldbase;
			pc = *(uint64*)(sp+8);
			sp += 16;
		}

		if(f->frame < 8)	// assembly functions lie
			sp += 8;
		else
			sp += f->frame;

	loop:
		pc = *(uint64*)(sp-8);
		if(pc <= 0x1000 || (f = findfunc(pc)) == nil) {
			// dangerous, but let's try this.
			// see if it is a closure.
			p = (byte*)pc;
			// ADDQ $xxx, SP; RET
			if(p[0] == 0x48 && p[1] == 0x81 && p[2] == 0xc4 && p[7] == 0xc3) {
				sp += *(uint32*)(p+3) + 8;
				goto loop;
			}
			goto error;
		}
	}

	retpc = pc;
	retfile = f->src;
	retline = funcline(f, pc-1);
	retbool = true;
	FLUSH(&retpc);
	FLUSH(&retfile);
	FLUSH(&retline);
	FLUSH(&retbool);
}

#pragma textflag 7
// func closure(siz int32,
//	fn func(arg0, arg1, arg2 *ptr, callerpc uintptr, xxx) yyy,
//	arg0, arg1, arg2 *ptr) (func(xxx) yyy)
void
sys·closure(int32 siz, byte *fn, byte *arg0)
{
	byte *p, *q, **ret;
	int32 i, n;
	int64 pcrel;

	if(siz < 0 || siz%8 != 0)
		throw("bad closure size");

	ret = (byte**)((byte*)&arg0 + siz);

	if(siz > 100) {
		// TODO(rsc): implement stack growth preamble?
		throw("closure too big");
	}

	// compute size of new fn.
	// must match code laid out below.
	n = 7+10+3;	// SUBQ MOVQ MOVQ
	if(siz <= 4*8)
		n += 2*siz/8;	// MOVSQ MOVSQ...
	else
		n += 7+3;	// MOVQ REP MOVSQ
	n += 12;	// CALL worst case; sometimes only 5
	n += 7+1;	// ADDQ RET

	// store args aligned after code, so gc can find them.
	n += siz;
	if(n%8)
		n += 8 - n%8;

	p = mal(n);
	*ret = p;
	q = p + n - siz;
	mcpy(q, (byte*)&arg0, siz);

	// SUBQ $siz, SP
	*p++ = 0x48;
	*p++ = 0x81;
	*p++ = 0xec;
	*(uint32*)p = siz;
	p += 4;

	// MOVQ $q, SI
	*p++ = 0x48;
	*p++ = 0xbe;
	*(byte**)p = q;
	p += 8;

	// MOVQ SP, DI
	*p++ = 0x48;
	*p++ = 0x89;
	*p++ = 0xe7;

	if(siz <= 4*8) {
		for(i=0; i<siz; i+=8) {
			// MOVSQ
			*p++ = 0x48;
			*p++ = 0xa5;
		}
	} else {
		// MOVQ $(siz/8), CX  [32-bit immediate siz/8]
		*p++ = 0x48;
		*p++ = 0xc7;
		*p++ = 0xc1;
		*(uint32*)p = siz/8;
		p += 4;

		// REP; MOVSQ
		*p++ = 0xf3;
		*p++ = 0x48;
		*p++ = 0xa5;
	}


	// call fn
	pcrel = fn - (p+5);
	if((int32)pcrel == pcrel) {
		// can use direct call with pc-relative offset
		// CALL fn
		*p++ = 0xe8;
		*(int32*)p = pcrel;
		p += 4;
	} else {
		// MOVQ $fn, CX  [64-bit immediate fn]
		*p++ = 0x48;
		*p++ = 0xb9;
		*(byte**)p = fn;
		p += 8;

		// CALL *CX
		*p++ = 0xff;
		*p++ = 0xd1;
	}

	// ADDQ $siz, SP
	*p++ = 0x48;
	*p++ = 0x81;
	*p++ = 0xc4;
	*(uint32*)p = siz;
	p += 4;

	// RET
	*p++ = 0xc3;

	if(p > q)
		throw("bad math in sys.closure");
}


