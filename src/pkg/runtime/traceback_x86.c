// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 386

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

void runtime·deferproc(void);
void runtime·newproc(void);
void runtime·newstack(void);
void runtime·morestack(void);
void runtime·sigpanic(void);

// This code is also used for the 386 tracebacks.
// Use uintptr for an appropriate word-sized integer.

// Generic traceback.  Handles runtime stack prints (pcbuf == nil),
// the runtime.Callers function (pcbuf != nil), as well as the garbage
// collector (fn != nil).  A little clunky to merge the two but avoids
// duplicating the code and all its subtlety.
int32
runtime·gentraceback(byte *pc0, byte *sp, byte *lr0, G *gp, int32 skip, uintptr *pcbuf, int32 max, void (*fn)(Func*, byte*, byte*, void*), void *arg)
{
	int32 i, n, sawnewstack;
	uintptr pc, lr, tracepc;
	byte *fp;
	Stktop *stk;
	Func *f;
	bool waspanic;

	USED(lr0);
	pc = (uintptr)pc0;
	lr = 0;
	fp = nil;
	waspanic = false;
	
	// If the PC is goexit, the goroutine hasn't started yet.
	if(pc0 == gp->sched.pc && sp == (byte*)gp->sched.sp && pc0 == (byte*)runtime·goexit && gp->fnstart != nil) {
		fp = sp;
		lr = pc;
		pc = (uintptr)gp->fnstart->fn;
	}
	
	// If the PC is zero, it's likely a nil function call.
	// Start in the caller's frame.
	if(pc == 0) {
		pc = *(uintptr*)sp;
		sp += sizeof(uintptr);
	}

	n = 0;
	sawnewstack = 0;
	stk = (Stktop*)gp->stackbase;
	while(n < max) {
		// Typically:
		//	pc is the PC of the running function.
		//	sp is the stack pointer at that program counter.
		//	fp is the frame pointer (caller's stack pointer) at that program counter, or nil if unknown.
		//	stk is the stack containing sp.
		//	The caller's program counter is lr, unless lr is zero, in which case it is *(uintptr*)sp.
	
		if(pc == (uintptr)runtime·lessstack) {
			// Hit top of stack segment.  Unwind to next segment.
			pc = (uintptr)stk->gobuf.pc;
			sp = (byte*)stk->gobuf.sp;
			lr = 0;
			fp = nil;
			if(pcbuf == nil && fn == nil && runtime·showframe(nil, gp == m->curg))
				runtime·printf("----- stack segment boundary -----\n");
			stk = (Stktop*)stk->stackbase;
			continue;
		}
		if(pc <= 0x1000 || (f = runtime·findfunc(pc)) == nil) {
			if(fn != nil)
				runtime·throw("unknown pc");
			break;
		}

		// Found an actual function.
		if(fp == nil) {
			fp = sp;
			if(pc > f->entry && f->frame >= sizeof(uintptr))
				fp += f->frame - sizeof(uintptr);
			if(lr == 0)
				lr = *(uintptr*)fp;
			fp += sizeof(uintptr);
		} else if(lr == 0)
			lr = *(uintptr*)fp;

		if(skip > 0)
			skip--;
		else if(pcbuf != nil)
			pcbuf[n++] = pc;
		else if(fn != nil)
			(*fn)(f, (byte*)pc, sp, arg);
		else {
			if(runtime·showframe(f, gp == m->curg)) {
				// Print during crash.
				//	main(0x1, 0x2, 0x3)
				//		/home/rsc/go/src/runtime/x.go:23 +0xf
				//		
				tracepc = pc;	// back up to CALL instruction for funcline.
				if(n > 0 && pc > f->entry && !waspanic)
					tracepc--;
				if(m->throwing && gp == m->curg)
					runtime·printf("[fp=%p] ", fp);
				runtime·printf("%S(", f->name);
				for(i = 0; i < f->args/sizeof(uintptr); i++) {
					if(i != 0)
						runtime·prints(", ");
					runtime·printhex(((uintptr*)fp)[i]);
					if(i >= 4) {
						runtime·prints(", ...");
						break;
					}
				}
				runtime·prints(")\n");
				runtime·printf("\t%S:%d", f->src, runtime·funcline(f, tracepc));
				if(pc > f->entry)
					runtime·printf(" +%p", (uintptr)(pc - f->entry));
				runtime·printf("\n");
			}
			n++;
		}
		
		waspanic = f->entry == (uintptr)runtime·sigpanic;

		if(f->entry == (uintptr)runtime·deferproc || f->entry == (uintptr)runtime·newproc)
			fp += 2*sizeof(uintptr);

		if(f->entry == (uintptr)runtime·newstack)
			sawnewstack = 1;

		if(pcbuf == nil && fn == nil && f->entry == (uintptr)runtime·morestack && gp == m->g0 && sawnewstack) {
			// The fact that we saw newstack means that morestack
			// has managed to record its information in m, so we can
			// use it to keep unwinding the stack.
			runtime·printf("----- morestack called from goroutine %D -----\n", m->curg->goid);
			pc = (uintptr)m->morepc;
			sp = (byte*)m->morebuf.sp - sizeof(void*);
			lr = (uintptr)m->morebuf.pc;
			fp = (byte*)m->morebuf.sp;
			sawnewstack = 0;
			gp = m->curg;
			stk = (Stktop*)gp->stackbase;
			continue;
		}

		if(pcbuf == nil && fn == nil && f->entry == (uintptr)runtime·lessstack && gp == m->g0) {
			// Lessstack is running on scheduler stack.  Switch to original goroutine.
			runtime·printf("----- lessstack called from goroutine %D -----\n", m->curg->goid);
			gp = m->curg;
			stk = (Stktop*)gp->stackbase;
			sp = (byte*)stk->gobuf.sp;
			pc = (uintptr)stk->gobuf.pc;
			fp = nil;
			lr = 0;
			continue;
		}

		// Do not unwind past the bottom of the stack.
		if(pc == (uintptr)runtime·goexit)
			break;

		// Unwind to next frame.
		pc = lr;
		lr = 0;
		sp = fp;
		fp = nil;
	}
	
	// Show what created goroutine, except main goroutine (goid 1).
	if(pcbuf == nil && fn == nil && (pc = gp->gopc) != 0 && (f = runtime·findfunc(pc)) != nil
			&& runtime·showframe(f, gp == m->curg) && gp->goid != 1) {
		runtime·printf("created by %S\n", f->name);
		tracepc = pc;	// back up to CALL instruction for funcline.
		if(n > 0 && pc > f->entry)
			tracepc--;
		runtime·printf("\t%S:%d", f->src, runtime·funcline(f, tracepc));
		if(pc > f->entry)
			runtime·printf(" +%p", (uintptr)(pc - f->entry));
		runtime·printf("\n");
	}
		
	return n;
}

void
runtime·traceback(byte *pc0, byte *sp, byte*, G *gp)
{
	if(gp->status == Gsyscall) {
		// Override signal registers if blocked in system call.
		pc0 = gp->sched.pc;
		sp = (byte*)gp->sched.sp;
	}
	runtime·gentraceback(pc0, sp, nil, gp, 0, nil, 100, nil, nil);
}

int32
runtime·callers(int32 skip, uintptr *pcbuf, int32 m)
{
	byte *pc, *sp;

	// our caller's pc, sp.
	sp = (byte*)&skip;
	pc = runtime·getcallerpc(&skip);

	return runtime·gentraceback(pc, sp, nil, g, skip, pcbuf, m, nil, nil);
}
