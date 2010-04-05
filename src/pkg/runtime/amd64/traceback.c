// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "malloc.h"

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
	int32 i, n, iter;
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
