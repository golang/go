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
// collector (callback != nil).  A little clunky to merge these, but avoids
// duplicating the code and all its subtlety.
int32
runtime·gentraceback(uintptr pc0, uintptr sp0, uintptr lr0, G *gp, int32 skip, uintptr *pcbuf, int32 max, void (*callback)(Stkframe*, void*), void *v)
{
	int32 i, n, sawnewstack;
	uintptr tracepc;
	bool waspanic, printing;
	Func *f, *f2;
	Stkframe frame;
	Stktop *stk;

	USED(lr0);

	runtime·memclr((byte*)&frame, sizeof frame);
	frame.pc = pc0;
	frame.sp = sp0;
	waspanic = false;
	printing = pcbuf==nil && callback==nil;

	// If the PC is goexit, the goroutine hasn't started yet.
	if(frame.pc == gp->sched.pc && frame.sp == gp->sched.sp && frame.pc == (uintptr)runtime·goexit && gp->fnstart != nil) {
		frame.fp = frame.sp;
		frame.lr = frame.pc;
		frame.pc = (uintptr)gp->fnstart->fn;
	}
	
	// If the PC is zero, it's likely a nil function call.
	// Start in the caller's frame.
	if(frame.pc == 0) {
		frame.pc = *(uintptr*)frame.sp;
		frame.sp += sizeof(uintptr);
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
	
		if(frame.pc == (uintptr)runtime·lessstack) {
			// Hit top of stack segment.  Unwind to next segment.
			frame.pc = stk->gobuf.pc;
			frame.sp = stk->gobuf.sp;
			frame.lr = 0;
			frame.fp = 0;
			if(printing && runtime·showframe(nil, gp == m->curg))
				runtime·printf("----- stack segment boundary -----\n");
			stk = (Stktop*)stk->stackbase;
			continue;
		}
		if(frame.pc <= 0x1000 || (frame.fn = f = runtime·findfunc(frame.pc)) == nil) {
			if(callback != nil)
				runtime·throw("unknown pc");
			break;
		}

		// Found an actual function.
		// Derive frame pointer and link register.
		if(frame.fp == 0) {
			frame.fp = frame.sp;
			if(frame.pc > f->entry && f->frame >= sizeof(uintptr))
				frame.fp += f->frame;
			else
				frame.fp += sizeof(uintptr);
		}
		if(frame.lr == 0)
			frame.lr = ((uintptr*)frame.fp)[-1];

		// Derive size of arguments.
		frame.argp = (byte*)frame.fp;
		frame.arglen = 0;
		if(f->args != ArgsSizeUnknown)
			frame.arglen = f->args;
		else if(frame.pc == (uintptr)runtime·goexit || f->entry == (uintptr)runtime·mcall || f->entry == (uintptr)runtime·mstart || f->entry == (uintptr)_rt0_go)
			frame.arglen = 0;
		else if(frame.lr == (uintptr)runtime·lessstack)
			frame.arglen = stk->argsize;
		else if(f->entry == (uintptr)runtime·deferproc || f->entry == (uintptr)runtime·newproc)
			frame.arglen = 2*sizeof(uintptr) + ((uintptr*)frame.argp)[1];
		else if((f2 = runtime·findfunc(frame.lr)) != nil && f2->frame >= sizeof(uintptr))
			frame.arglen = f2->frame; // conservative overestimate
		else {
			runtime·printf("runtime: unknown argument frame size for %S\n", f->name);
			runtime·throw("invalid stack");
		}

		// Derive location and size of local variables.
		if(frame.fp == frame.sp) {
			// Function has not created a frame for itself yet.
			frame.varp = nil;
			frame.varlen = 0;
		} else if(f->locals == 0) {
			// Assume no information, so use whole frame.
			// TODO: Distinguish local==0 from local==unknown.
			frame.varp = (byte*)frame.sp;
			frame.varlen = frame.fp - sizeof(uintptr) - frame.sp;
		} else {
			if(f->locals > frame.fp - sizeof(uintptr) - frame.sp) {
				runtime·printf("runtime: inconsistent locals=%p frame=%p fp=%p sp=%p for %S\n", (uintptr)f->locals, (uintptr)f->frame, frame.fp, frame.sp, f->name);
				runtime·throw("invalid stack");
			}
			frame.varp = (byte*)frame.fp - sizeof(uintptr) - f->locals;
			frame.varlen = f->locals;
		}

		if(skip > 0) {
			skip--;
			goto skipped;
		}

		if(pcbuf != nil)
			pcbuf[n] = frame.pc;
		if(callback != nil)
			callback(&frame, v);
		if(printing) {
			if(runtime·showframe(f, gp == m->curg)) {
				// Print during crash.
				//	main(0x1, 0x2, 0x3)
				//		/home/rsc/go/src/runtime/x.go:23 +0xf
				//		
				tracepc = frame.pc;	// back up to CALL instruction for funcline.
				if(n > 0 && frame.pc > f->entry && !waspanic)
					tracepc--;
				if(m->throwing && gp == m->curg)
					runtime·printf("[fp=%p] ", frame.fp);
				runtime·printf("%S(", f->name);
				for(i = 0; i < f->args/sizeof(uintptr); i++) {
					if(i != 0)
						runtime·prints(", ");
					runtime·printhex(((uintptr*)frame.argp)[i]);
					if(i >= 4) {
						runtime·prints(", ...");
						break;
					}
				}
				runtime·prints(")\n");
				runtime·printf("\t%S:%d", f->src, runtime·funcline(f, tracepc));
				if(frame.pc > f->entry)
					runtime·printf(" +%p", (uintptr)(frame.pc - f->entry));
				runtime·printf("\n");
			}
		}
		n++;
	
	skipped:
		waspanic = f->entry == (uintptr)runtime·sigpanic;

		if(f->entry == (uintptr)runtime·deferproc || f->entry == (uintptr)runtime·newproc)
			frame.fp += 2*sizeof(uintptr);

		if(f->entry == (uintptr)runtime·newstack)
			sawnewstack = 1;

		if(printing && f->entry == (uintptr)runtime·morestack && gp == m->g0 && sawnewstack) {
			// The fact that we saw newstack means that morestack
			// has managed to record its information in m, so we can
			// use it to keep unwinding the stack.
			runtime·printf("----- morestack called from goroutine %D -----\n", m->curg->goid);
			frame.pc = (uintptr)m->morepc;
			frame.sp = m->morebuf.sp - sizeof(void*);
			frame.lr = m->morebuf.pc;
			frame.fp = m->morebuf.sp;
			sawnewstack = 0;
			gp = m->curg;
			stk = (Stktop*)gp->stackbase;
			continue;
		}

		if(printing && f->entry == (uintptr)runtime·lessstack && gp == m->g0) {
			// Lessstack is running on scheduler stack.  Switch to original goroutine.
			runtime·printf("----- lessstack called from goroutine %D -----\n", m->curg->goid);
			gp = m->curg;
			stk = (Stktop*)gp->stackbase;
			frame.sp = stk->gobuf.sp;
			frame.pc = stk->gobuf.pc;
			frame.fp = 0;
			frame.lr = 0;
			continue;
		}

		// Do not unwind past the bottom of the stack.
		if(frame.pc == (uintptr)runtime·goexit || f->entry == (uintptr)runtime·mstart || f->entry == (uintptr)_rt0_go)
			break;

		// Unwind to next frame.
		frame.pc = frame.lr;
		frame.lr = 0;
		frame.sp = frame.fp;
		frame.fp = 0;
	}
	
	// Show what created goroutine, except main goroutine (goid 1).
	if(printing && (frame.pc = gp->gopc) != 0 && (f = runtime·findfunc(frame.pc)) != nil
			&& runtime·showframe(f, gp == m->curg) && gp->goid != 1) {
		runtime·printf("created by %S\n", f->name);
		tracepc = frame.pc;	// back up to CALL instruction for funcline.
		if(n > 0 && frame.pc > f->entry)
			tracepc--;
		runtime·printf("\t%S:%d", f->src, runtime·funcline(f, tracepc));
		if(frame.pc > f->entry)
			runtime·printf(" +%p", (uintptr)(frame.pc - f->entry));
		runtime·printf("\n");
	}
	
	return n;
}

void
runtime·traceback(uintptr pc, uintptr sp, uintptr lr, G *gp)
{
	USED(lr);

	if(gp->status == Gsyscall) {
		// Override signal registers if blocked in system call.
		pc = gp->sched.pc;
		sp = gp->sched.sp;
	}
	runtime·gentraceback(pc, sp, 0, gp, 0, nil, 100, nil, nil);
}

int32
runtime·callers(int32 skip, uintptr *pcbuf, int32 m)
{
	uintptr pc, sp;

	sp = runtime·getcallersp(&skip);
	pc = (uintptr)runtime·getcallerpc(&skip);

	return runtime·gentraceback(pc, sp, 0, g, skip, pcbuf, m, nil, nil);
}
