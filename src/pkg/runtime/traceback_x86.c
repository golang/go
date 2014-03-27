// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 amd64p32 386

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "funcdata.h"
#ifdef GOOS_windows
#include "defs_GOOS_GOARCH.h"
#endif

void runtime·sigpanic(void);

#ifdef GOOS_windows
void runtime·sigtramp(void);
#endif

// This code is also used for the 386 tracebacks.
// Use uintptr for an appropriate word-sized integer.

// Generic traceback.  Handles runtime stack prints (pcbuf == nil),
// the runtime.Callers function (pcbuf != nil), as well as the garbage
// collector (callback != nil).  A little clunky to merge these, but avoids
// duplicating the code and all its subtlety.
int32
runtime·gentraceback(uintptr pc0, uintptr sp0, uintptr lr0, G *gp, int32 skip, uintptr *pcbuf, int32 max, bool (*callback)(Stkframe*, void*), void *v, bool printall)
{
	int32 i, n, nprint, line, gotraceback;
	uintptr tracepc;
	bool waspanic, printing;
	Func *f, *flr;
	Stkframe frame;
	Stktop *stk;
	String file;

	USED(lr0);
	
	gotraceback = runtime·gotraceback(nil);
	
	if(pc0 == ~(uintptr)0 && sp0 == ~(uintptr)0) { // Signal to fetch saved values from gp.
		if(gp->syscallstack != (uintptr)nil) {
			pc0 = gp->syscallpc;
			sp0 = gp->syscallsp;
		} else {
			pc0 = gp->sched.pc;
			sp0 = gp->sched.sp;
		}
	}

	nprint = 0;
	runtime·memclr((byte*)&frame, sizeof frame);
	frame.pc = pc0;
	frame.sp = sp0;
	waspanic = false;
	printing = pcbuf==nil && callback==nil;
	
	// If the PC is zero, it's likely a nil function call.
	// Start in the caller's frame.
	if(frame.pc == 0) {
		frame.pc = *(uintptr*)frame.sp;
		frame.sp += sizeof(uintreg);
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
			frame.fn = nil;
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

#ifdef GOOS_windows
		// Windows exception handlers run on the actual g stack (there is room
		// dedicated to this below the usual "bottom of stack"), not on a separate
		// stack. As a result, we have to be able to unwind past the exception
		// handler when called to unwind during stack growth inside the handler.
		// Recognize the frame at the call to sighandler in sigtramp and unwind
		// using the context argument passed to the call. This is awful.
		if(f != nil && f->entry == (uintptr)runtime·sigtramp && frame.pc > f->entry) {
			Context *r;
			
			// Invoke callback so that stack copier sees an uncopyable frame.
			if(callback != nil) {
				frame.argp = nil;
				frame.arglen = 0;
				if(!callback(&frame, v))
					return n;
			}
			r = (Context*)((uintptr*)frame.sp)[1];
#ifdef GOARCH_amd64
			frame.pc = r->Rip;
			frame.sp = r->Rsp;
#else
			frame.pc = r->Eip;
			frame.sp = r->Esp;
#endif
			frame.lr = 0;
			frame.fp = 0;
			frame.fn = nil;
			if(printing && runtime·showframe(nil, gp))
				runtime·printf("----- exception handler -----\n");
			f = runtime·findfunc(frame.pc);
			if(f == nil) {
				runtime·printf("runtime: unknown pc %p after exception handler\n", frame.pc);
				if(callback != nil)
					runtime·throw("unknown pc");
			}
			frame.fn = f;
			continue;
		}
#endif

		// Found an actual function.
		// Derive frame pointer and link register.
		if(frame.fp == 0) {
			frame.fp = frame.sp + runtime·funcspdelta(f, frame.pc);
			frame.fp += sizeof(uintreg); // caller PC
		}
		if(runtime·topofstack(f)) {
			frame.lr = 0;
			flr = nil;
		} else {
			if(frame.lr == 0)
				frame.lr = ((uintreg*)frame.fp)[-1];
			flr = runtime·findfunc(frame.lr);
			if(flr == nil) {
				runtime·printf("runtime: unexpected return pc for %s called from %p\n", runtime·funcname(f), frame.lr);
				if(callback != nil)
					runtime·throw("unknown caller pc");
			}
		}
		
		frame.varp = (byte*)frame.fp - sizeof(uintreg);

		// Derive size of arguments.
		// Most functions have a fixed-size argument block,
		// so we can use metadata about the function f.
		// Not all, though: there are some variadic functions
		// in package runtime and reflect, and for those we use call-specific
		// metadata recorded by f's caller.
		if(callback != nil || printing) {
			frame.argp = (byte*)frame.fp;
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
				//		
				tracepc = frame.pc;	// back up to CALL instruction for funcline.
				if(n > 0 && frame.pc > f->entry && !waspanic)
					tracepc--;
				runtime·printf("%s(", runtime·funcname(f));
				for(i = 0; i < frame.arglen/sizeof(uintptr); i++) {
					if(i >= 5) {
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
				if(m->throwing > 0 && gp == m->curg || gotraceback >= 2)
					runtime·printf(" fp=%p", frame.fp);
				runtime·printf("\n");
				nprint++;
			}
		}
		n++;
	
	skipped:
		waspanic = f->entry == (uintptr)runtime·sigpanic;

		// Do not unwind past the bottom of the stack.
		if(flr == nil)
			break;

		// Unwind to next frame.
		frame.fn = flr;
		frame.pc = frame.lr;
		frame.lr = 0;
		frame.sp = frame.fp;
		frame.fp = 0;
	}
	
	if(pcbuf == nil && callback == nil)
		n = nprint;
	
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

	USED(lr);

	if(gp->status == Gsyscall) {
		// Override signal registers if blocked in system call.
		pc = gp->syscallpc;
		sp = gp->syscallsp;
	}
	
	// Print traceback. By default, omits runtime frames.
	// If that means we print nothing at all, repeat forcing all frames printed.
	n = runtime·gentraceback(pc, sp, 0, gp, 0, nil, TracebackMaxFrames, nil, nil, false);
	if(n == 0)
		n = runtime·gentraceback(pc, sp, 0, gp, 0, nil, TracebackMaxFrames, nil, nil, true);
	if(n == TracebackMaxFrames)
		runtime·printf("...additional frames elided...\n");
	runtime·printcreatedby(gp);
}

int32
runtime·callers(int32 skip, uintptr *pcbuf, int32 m)
{
	uintptr pc, sp;

	sp = runtime·getcallersp(&skip);
	pc = (uintptr)runtime·getcallerpc(&skip);

	return runtime·gentraceback(pc, sp, 0, g, skip, pcbuf, m, nil, nil, false);
}
