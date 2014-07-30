// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "funcdata.h"

void runtime·sigpanic(void);
void runtime·newproc(void);
void runtime·deferproc(void);

int32
runtime·gentraceback(uintptr pc0, uintptr sp0, uintptr lr0, G *gp, int32 skip, uintptr *pcbuf, int32 max, bool (*callback)(Stkframe*, void*), void *v, bool printall)
{
	int32 i, n, nprint, line, gotraceback;
	uintptr x, tracepc, sparg;
	bool waspanic, wasnewproc, printing;
	Func *f, *flr;
	Stkframe frame;
	Stktop *stk;
	String file;
	Panic *panic;
	Defer *defer;

	gotraceback = runtime·gotraceback(nil);

	if(pc0 == ~(uintptr)0 && sp0 == ~(uintptr)0) { // Signal to fetch saved values from gp.
		if(gp->syscallstack != (uintptr)nil) {
			pc0 = gp->syscallpc;
			sp0 = gp->syscallsp;
			lr0 = 0;
		} else {
			pc0 = gp->sched.pc;
			sp0 = gp->sched.sp;
			lr0 = gp->sched.lr;
		}
	}

	nprint = 0;
	runtime·memclr((byte*)&frame, sizeof frame);
	frame.pc = pc0;
	frame.lr = lr0;
	frame.sp = sp0;
	waspanic = false;
	wasnewproc = false;
	printing = pcbuf==nil && callback==nil;

	panic = gp->panic;
	defer = gp->defer;

	while(defer != nil && defer->argp == NoArgs)
		defer = defer->link;	
	while(panic != nil && panic->defer == nil)
		panic = panic->link;

	// If the PC is zero, it's likely a nil function call.
	// Start in the caller's frame.
	if(frame.pc == 0) {
		frame.pc = frame.lr;
		frame.lr = 0;
	}
	
	f = runtime·findfunc(frame.pc);
	if(f == nil) {
		if(callback != nil) {
			runtime·printf("runtime: unknown pc %p\n", frame.pc);
			runtime·throw("unknown pc");
		}
		return 0;
	}
	frame.fn = f;

	n = 0;
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
			if(printing && runtime·showframe(nil, gp))
				runtime·printf("----- stack segment boundary -----\n");
			stk = (Stktop*)stk->stackbase;
			
			f = runtime·findfunc(frame.pc);
			if(f == nil) {
				runtime·printf("runtime: unknown pc %p after stack split\n", frame.pc);
				if(callback != nil)
					runtime·throw("unknown pc");
			}
			frame.fn = f;
			continue;
		}
		f = frame.fn;
		
		// Found an actual function.
		// Derive frame pointer and link register.
		if(frame.fp == 0)
			frame.fp = frame.sp + runtime·funcspdelta(f, frame.pc);
		if(runtime·topofstack(f)) {
			frame.lr = 0;
			flr = nil;
		} else if(f->entry == (uintptr)runtime·jmpdefer) {
			// jmpdefer modifies SP/LR/PC non-atomically.
			// If a profiling interrupt arrives during jmpdefer,
			// the stack unwind may see a mismatched register set
			// and get confused. Stop if we see PC within jmpdefer
			// to avoid that confusion.
			// See golang.org/issue/8153.
			// This check can be deleted if jmpdefer is changed
			// to restore all three atomically using pop.
			if(callback != nil)
				runtime·throw("traceback_arm: found jmpdefer when tracing with callback");
			frame.lr = 0;
			flr = nil;
		} else {
			if((n == 0 && frame.sp < frame.fp) || frame.lr == 0)
				frame.lr = *(uintptr*)frame.sp;
			flr = runtime·findfunc(frame.lr);
			if(flr == nil) {
				runtime·printf("runtime: unexpected return pc for %s called from %p\n", runtime·funcname(f), frame.lr);
				if(callback != nil)
					runtime·throw("unknown caller pc");
			}
		}

		frame.varp = (byte*)frame.fp;

		// Derive size of arguments.
		// Most functions have a fixed-size argument block,
		// so we can use metadata about the function f.
		// Not all, though: there are some variadic functions
		// in package runtime and reflect, and for those we use call-specific
		// metadata recorded by f's caller.
		if(callback != nil || printing) {
			frame.argp = (byte*)frame.fp + sizeof(uintptr);
			if(f->args != ArgsSizeUnknown)
				frame.arglen = f->args;
			else if(flr == nil)
				frame.arglen = 0;
			else if(frame.lr == (uintptr)runtime·lessstack)
				frame.arglen = stk->argsize;
			else if((i = runtime·funcarglen(flr, frame.lr)) >= 0)
				frame.arglen = i;
			else {
				runtime·printf("runtime: unknown argument frame size for %s called from %p [%s]\n",
					runtime·funcname(f), frame.lr, flr ? runtime·funcname(flr) : "?");
				if(callback != nil)
					runtime·throw("invalid stack");
				frame.arglen = 0;
			}
		}

		// Determine function SP where deferproc would find its arguments.
		// On ARM that's just the standard bottom-of-stack plus 1 word for
		// the saved LR. If the previous frame was a direct call to newproc/deferproc,
		// however, the SP is three words lower than normal.
		// If the function has no frame at all - perhaps it just started, or perhaps
		// it is a leaf with no local variables - then we cannot possibly find its
		// SP in a defer, and we might confuse its SP for its caller's SP, so
		// set sparg=0 in that case.
		sparg = 0;
		if(frame.fp != frame.sp) {
			sparg = frame.sp + sizeof(uintreg);
			if(wasnewproc)
				sparg += 3*sizeof(uintreg);
		}

		// Determine frame's 'continuation PC', where it can continue.
		// Normally this is the return address on the stack, but if sigpanic
		// is immediately below this function on the stack, then the frame
		// stopped executing due to a trap, and frame.pc is probably not
		// a safe point for looking up liveness information. In this panicking case,
		// the function either doesn't return at all (if it has no defers or if the
		// defers do not recover) or it returns from one of the calls to 
		// deferproc a second time (if the corresponding deferred func recovers).
		// It suffices to assume that the most recent deferproc is the one that
		// returns; everything live at earlier deferprocs is still live at that one.
		frame.continpc = frame.pc;
		if(waspanic) {
			if(panic != nil && panic->defer->argp == (byte*)sparg)
				frame.continpc = (uintptr)panic->defer->pc;
			else if(defer != nil && defer->argp == (byte*)sparg)
				frame.continpc = (uintptr)defer->pc;
			else
				frame.continpc = 0;
		}

		// Unwind our local panic & defer stacks past this frame.
		while(panic != nil && (panic->defer == nil || panic->defer->argp == (byte*)sparg || panic->defer->argp == NoArgs))
			panic = panic->link;
		while(defer != nil && (defer->argp == (byte*)sparg || defer->argp == NoArgs))
			defer = defer->link;	

		if(skip > 0) {
			skip--;
			goto skipped;
		}

		if(pcbuf != nil)
			pcbuf[n] = frame.pc;
		if(callback != nil) {
			if(!callback(&frame, v))
				return n;
		}
		if(printing) {
			if(printall || runtime·showframe(f, gp)) {
				// Print during crash.
				//	main(0x1, 0x2, 0x3)
				//		/home/rsc/go/src/runtime/x.go:23 +0xf
				tracepc = frame.pc;	// back up to CALL instruction for funcline.
				if(n > 0 && frame.pc > f->entry && !waspanic)
					tracepc -= sizeof(uintptr);
				runtime·printf("%s(", runtime·funcname(f));
				for(i = 0; i < frame.arglen/sizeof(uintptr); i++) {
					if(i >= 10) {
						runtime·prints(", ...");
						break;
					}
					if(i != 0)
						runtime·prints(", ");
					runtime·printhex(((uintptr*)frame.argp)[i]);
				}
				runtime·prints(")\n");
				line = runtime·funcline(f, tracepc, &file);
				runtime·printf("\t%S:%d", file, line);
				if(frame.pc > f->entry)
					runtime·printf(" +%p", (uintptr)(frame.pc - f->entry));
				if(g->m->throwing > 0 && gp == g->m->curg || gotraceback >= 2)
					runtime·printf(" fp=%p sp=%p", frame.fp, frame.sp);
				runtime·printf("\n");
				nprint++;
			}
		}
		n++;
		
	skipped:
		waspanic = f->entry == (uintptr)runtime·sigpanic;
		wasnewproc = f->entry == (uintptr)runtime·newproc || f->entry == (uintptr)runtime·deferproc;

		// Do not unwind past the bottom of the stack.
		if(flr == nil)
			break;

		// Unwind to next frame.
		frame.pc = frame.lr;
		frame.fn = flr;
		frame.lr = 0;
		frame.sp = frame.fp;
		frame.fp = 0;
	
		// sighandler saves the lr on stack before faking a call to sigpanic
		if(waspanic) {
			x = *(uintptr*)frame.sp;
			frame.sp += 4;
			frame.fn = f = runtime·findfunc(frame.pc);
			if(f == nil)
				frame.pc = x;
			else if(f->frame == 0)
				frame.lr = x;
		}
	}
	
	if(pcbuf == nil && callback == nil)
		n = nprint;

	// For rationale, see long comment in traceback_x86.c.
	if(callback != nil && n < max && defer != nil) {
		if(defer != nil)
			runtime·printf("runtime: g%D: leftover defer argp=%p pc=%p\n", gp->goid, defer->argp, defer->pc);
		if(panic != nil)
			runtime·printf("runtime: g%D: leftover panic argp=%p pc=%p\n", gp->goid, panic->defer->argp, panic->defer->pc);
		for(defer = gp->defer; defer != nil; defer = defer->link)
			runtime·printf("\tdefer %p argp=%p pc=%p\n", defer, defer->argp, defer->pc);
		for(panic = gp->panic; panic != nil; panic = panic->link) {
			runtime·printf("\tpanic %p defer %p", panic, panic->defer);
			if(panic->defer != nil)
				runtime·printf(" argp=%p pc=%p", panic->defer->argp, panic->defer->pc);
			runtime·printf("\n");
		}
		runtime·throw("traceback has leftover defers or panics");
	}

	return n;		
}

void
runtime·printcreatedby(G *gp)
{
	int32 line;
	uintptr pc, tracepc;
	Func *f;
	String file;

	// Show what created goroutine, except main goroutine (goid 1).
	if((pc = gp->gopc) != 0 && (f = runtime·findfunc(pc)) != nil &&
		runtime·showframe(f, gp) && gp->goid != 1) {
		runtime·printf("created by %s\n", runtime·funcname(f));
		tracepc = pc;	// back up to CALL instruction for funcline.
		if(pc > f->entry)
			tracepc -= PCQuantum;
		line = runtime·funcline(f, tracepc, &file);
		runtime·printf("\t%S:%d", file, line);
		if(pc > f->entry)
			runtime·printf(" +%p", (uintptr)(pc - f->entry));
		runtime·printf("\n");
	}
}

void
runtime·traceback(uintptr pc, uintptr sp, uintptr lr, G *gp)
{
	int32 n;

	if(gp->status == Gsyscall) {
		// Override signal registers if blocked in system call.
		pc = gp->syscallpc;
		sp = gp->syscallsp;
		lr = 0;
	}

	// Print traceback. By default, omits runtime frames.
	// If that means we print nothing at all, repeat forcing all frames printed.
	n = runtime·gentraceback(pc, sp, lr, gp, 0, nil, TracebackMaxFrames, nil, nil, false);
	if(n == 0)
		runtime·gentraceback(pc, sp, lr, gp, 0, nil, TracebackMaxFrames, nil, nil, true);
	if(n == TracebackMaxFrames)
		runtime·printf("...additional frames elided...\n");
	runtime·printcreatedby(gp);
}

// func caller(n int) (pc uintptr, file string, line int, ok bool)
int32
runtime·callers(int32 skip, uintptr *pcbuf, int32 m)
{
	uintptr pc, sp;
	
	sp = runtime·getcallersp(&skip);
	pc = (uintptr)runtime·getcallerpc(&skip);

	return runtime·gentraceback(pc, sp, 0, g, skip, pcbuf, m, nil, nil, false);
}

int32
runtime·gcallers(G *gp, int32 skip, uintptr *pcbuf, int32 m)
{
	return runtime·gentraceback(~(uintptr)0, ~(uintptr)0, 0, gp, skip, pcbuf, m, nil, nil, false);
}
