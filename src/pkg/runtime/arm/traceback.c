// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static int32
gentraceback(byte *pc0, byte *sp, byte *lr0, G *g, int32 skip, uintptr *pcbuf, int32 m)
{
	int32 i, n, iter;
	uintptr pc, lr, tracepc;
	Stktop *stk;
	Func *f;
	
	pc = (uintptr)pc0;
	lr = (uintptr)lr0;
	
	// If the PC is goexit, it hasn't started yet.
	if(pc == (uintptr)runtime·goexit) {
		pc = (uintptr)g->entry;
		lr = (uintptr)runtime·goexit;
	}

	// If the PC is zero, it's likely a nil function call.
	// Start in the caller's frame.
	if(pc == 0) {
		pc = lr;
		lr = 0;
	}

	n = 0;
	stk = (Stktop*)g->stackbase;
	for(iter = 0; iter < 100 && n < m; iter++) {	// iter avoids looping forever
		if(pc == (uintptr)runtime·lessstack) {
			// Hit top of stack segment.  Unwind to next segment.
			pc = (uintptr)stk->gobuf.pc;
			sp = stk->gobuf.sp;
			lr = *(uintptr*)sp;
			stk = (Stktop*)stk->stackbase;
			continue;
		}
		if(pc <= 0x1000 || (f = runtime·findfunc(pc-4)) == nil) {
			// TODO: Check for closure.
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
			runtime·printf("%S", f->name);
			if(pc > f->entry)
				runtime·printf("+%p", (uintptr)(pc - f->entry));
			tracepc = pc;	// back up to CALL instruction for funcline.
			if(n > 0 && pc > f->entry)
				tracepc -= sizeof(uintptr);
			runtime·printf(" %S:%d\n", f->src, runtime·funcline(f, tracepc));
			runtime·printf("\t%S(", f->name);
			for(i = 0; i < f->args; i++) {
				if(i != 0)
					runtime·prints(", ");
				runtime·printhex(((uintptr*)sp)[1+i]);
				if(i >= 4) {
					runtime·prints(", ...");
					break;
				}
			}
			runtime·prints(")\n");
			n++;
		}
		
		if(lr == 0)
			lr = *(uintptr*)sp;
		pc = lr;
		lr = 0;
		if(f->frame >= 0)
			sp += f->frame;
	}
	return n;		
}

void
runtime·traceback(byte *pc0, byte *sp, byte *lr, G *g)
{
	gentraceback(pc0, sp, lr, g, 0, nil, 100);
}

// func caller(n int) (pc uintptr, file string, line int, ok bool)
int32
runtime·callers(int32 skip, uintptr *pcbuf, int32 m)
{
	byte *pc, *sp;
	
	sp = runtime·getcallersp(&skip);
	pc = runtime·getcallerpc(&skip);

	return gentraceback(pc, sp, 0, g, skip, pcbuf, m);
}
