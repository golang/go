// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

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
runtime·Caller(int32 n, uint64 retpc, String retfile, int32 retline, bool retbool)
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
		retfile = emptystring;
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


