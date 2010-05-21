// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "malloc.h"

static uintptr isclosureentry(uintptr);

// This code is also used for the 386 tracebacks.
// Use uintptr for an appropriate word-sized integer.

// Generic traceback.  Handles runtime stack prints (pcbuf == nil)
// as well as the runtime.Callers function (pcbuf != nil).
// A little clunky to merge the two but avoids duplicating
// the code and all its subtlety.
static int32
gentraceback(byte *pc0, byte *sp, G *g, int32 skip, uintptr *pcbuf, int32 m)
{
	byte *p;
	int32 i, n, iter, nascent;
	uintptr pc, tracepc;
	Stktop *stk;
	Func *f;
	
	pc = (uintptr)pc0;

	// If the PC is zero, it's likely a nil function call.
	// Start in the caller's frame.
	if(pc == 0) {
		pc = *(uintptr*)sp;
		sp += sizeof(uintptr);
	}
	
	nascent = 0;
	if(pc0 == g->sched.pc && sp == g->sched.sp && pc0 == (byte*)goexit) {
		// Hasn't started yet.  g->sched is set up for goexit
		// but goroutine will start at g->entry.
		nascent = 1;
		pc = (uintptr)g->entry;
	}
	
	n = 0;
	stk = (Stktop*)g->stackbase;
	for(iter = 0; iter < 100 && n < m; iter++) {	// iter avoids looping forever
		if(pc == (uintptr)·lessstack) {
			// Hit top of stack segment.  Unwind to next segment.
			pc = (uintptr)stk->gobuf.pc;
			sp = stk->gobuf.sp;
			stk = (Stktop*)stk->stackbase;
			continue;
		}

		if(pc <= 0x1000 || (f = findfunc(pc)) == nil) {
			// Dangerous, but worthwhile: see if this is a closure:
			//	ADDQ $wwxxyyzz, SP; RET
			//	[48] 81 c4 zz yy xx ww c3
			// The 0x48 byte is only on amd64.
			p = (byte*)pc;
			if(mheap.min < p && p+8 < mheap.max &&  // pointer in allocated memory
			   (sizeof(uintptr) != 8 || *p++ == 0x48) &&  // skip 0x48 byte on amd64
			   p[0] == 0x81 && p[1] == 0xc4 && p[6] == 0xc3) {
				sp += *(uint32*)(p+2);
				pc = *(uintptr*)sp;
				sp += sizeof(uintptr);
				continue;
			}
			
			if(nascent && (pc = isclosureentry(pc)) != 0)
				continue;

			// Unknown pc; stop.
			break;
		}

		// Found an actual function worth reporting.
		if(skip > 0)
			skip--;
		else if(pcbuf != nil)
			pcbuf[n++] = pc;
		else {
			// Print during crash.
			//	main+0xf /home/rsc/go/src/runtime/x.go:23
			//		main(0x1, 0x2, 0x3)
			printf("%S", f->name);
			if(pc > f->entry)
				printf("+%p", (uintptr)(pc - f->entry));
			tracepc = pc;	// back up to CALL instruction for funcline.
			if(n > 0 && pc > f->entry)
				tracepc--;
			printf(" %S:%d\n", f->src, funcline(f, tracepc));
			printf("\t%S(", f->name);
			for(i = 0; i < f->args; i++) {
				if(i != 0)
					prints(", ");
				·printhex(((uintptr*)sp)[i]);
				if(i >= 4) {
					prints(", ...");
					break;
				}
			}
			prints(")\n");
			n++;
		}
		
		if(nascent) {
			pc = (uintptr)g->sched.pc;
			sp = g->sched.sp;
			nascent = 0;
			continue;
		}

		if(f->frame < sizeof(uintptr))	// assembly functions lie
			sp += sizeof(uintptr);
		else
			sp += f->frame;
		pc = *((uintptr*)sp - 1);
	}
	return n;
}

void
traceback(byte *pc0, byte *sp, byte*, G *g)
{
	gentraceback(pc0, sp, g, 0, nil, 100);
}

int32
callers(int32 skip, uintptr *pcbuf, int32 m)
{
	byte *pc, *sp;

	// our caller's pc, sp.
	sp = (byte*)&skip;
	pc = ·getcallerpc(&skip);

	return gentraceback(pc, sp, g, skip, pcbuf, m);
}

static uintptr
isclosureentry(uintptr pc)
{
	byte *p;
	int32 i, siz;
	
	p = (byte*)pc;
	if(p < mheap.min || p+32 > mheap.max)
		return 0;
	
	// SUBQ $siz, SP
	if((sizeof(uintptr) == 8 && *p++ != 0x48) || *p++ != 0x81 || *p++ != 0xec)
		return 0;
	siz = *(uint32*)p;
	p += 4;
	
	// MOVQ $q, SI
	if((sizeof(uintptr) == 8 && *p++ != 0x48) || *p++ != 0xbe)
		return 0;
	p += sizeof(uintptr);

	// MOVQ SP, DI
	if((sizeof(uintptr) == 8 && *p++ != 0x48) || *p++ != 0x89 || *p++ != 0xe7)
		return 0;

	// CLD on 32-bit
	if(sizeof(uintptr) == 4 && *p++ != 0xfc)
		return 0;

	if(siz <= 4*sizeof(uintptr)) {
		// MOVSQ...
		for(i=0; i<siz; i+=sizeof(uintptr))
			if((sizeof(uintptr) == 8 && *p++ != 0x48) || *p++ != 0xa5)
				return 0;
	} else {
		// MOVQ $(siz/8), CX  [32-bit immediate siz/8]
		if((sizeof(uintptr) == 8 && *p++ != 0x48) || *p++ != 0xc7 || *p++ != 0xc1)
			return 0;
		p += 4;
		
		// REP MOVSQ
		if(*p++ != 0xf3 || (sizeof(uintptr) == 8 && *p++ != 0x48) || *p++ != 0xa5)
			return 0;
	}
	
	// CALL fn
	if(*p == 0xe8) {
		p++;
		return (uintptr)p+4 + *(int32*)p;
	}
	
	// MOVQ $fn, CX; CALL *CX
	if(sizeof(uintptr) != 8 || *p++ != 0x48 || *p++ != 0xb9)
		return 0;

	pc = *(uintptr*)p;
	p += 8;
	
	if(*p++ != 0xff || *p != 0xd1)
		return 0;

	return pc;
}
