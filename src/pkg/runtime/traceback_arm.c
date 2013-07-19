// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "funcdata.h"

void runtime·sigpanic(void);

static String unknown = { (uint8*)"?", 1 };

int32
runtime·gentraceback(uintptr pc0, uintptr sp0, uintptr lr0, G *gp, int32 skip, uintptr *pcbuf, int32 max, void (*callback)(Stkframe*, void*), void *v, bool printall)
{
	int32 i, n, nprint, line;
	uintptr x, tracepc;
	bool waspanic, printing;
	Func *f, *flr;
	Stkframe frame;
	Stktop *stk;
	String file;

	nprint = 0;
	runtime·memclr((byte*)&frame, sizeof frame);
	frame.pc = pc0;
	frame.lr = lr0;
	frame.sp = sp0;
	waspanic = false;
	printing = pcbuf==nil && callback==nil;

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
		} else {
			if(frame.lr == 0)
				frame.lr = *(uintptr*)frame.sp;
			flr = runtime·findfunc(frame.lr);
			if(flr == nil) {
				runtime·printf("runtime: unexpected return pc for %s called from %p\n", runtime·funcname(f), frame.lr);
				if(callback != nil)
					runtime·throw("unknown caller pc");
			}
		}
			
		// Derive size of arguments.
		// Most functions have a fixed-size argument block,
		// so we can use metadata about the function f.
		// Not all, though: there are some variadic functions
		// in package runtime, and for those we use call-specific
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

		// Derive location and size of local variables.
		if(frame.fp == frame.sp) {
			// Function has not created a frame for itself yet.
			frame.varp = nil;
			frame.varlen = 0;
		} else if(f->locals == 0) {
			// Assume no information, so use whole frame.
			// TODO: Distinguish local==0 from local==unknown.
			frame.varp = (byte*)frame.sp;
			frame.varlen = frame.fp - frame.sp;
		} else {
			if(f->locals > frame.fp - frame.sp) {
				runtime·printf("runtime: inconsistent locals=%p frame=%p fp=%p sp=%p for %s\n", (uintptr)f->locals, (uintptr)f->frame, frame.fp, frame.sp, runtime·funcname(f));
				if(callback != nil)
					runtime·throw("invalid stack");
			}
			frame.varp = (byte*)frame.fp - f->locals;
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
			if(printall || runtime·showframe(f, gp)) {
				// Print during crash.
				//	main(0x1, 0x2, 0x3)
				//		/home/rsc/go/src/runtime/x.go:23 +0xf
				tracepc = frame.pc;	// back up to CALL instruction for funcline.
				if(n > 0 && frame.pc > f->entry && !waspanic)
					tracepc -= sizeof(uintptr);
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
				if(m->throwing && gp == m->curg)
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
			else if (f->frame == 0)
				frame.lr = x;
		}
	}
	
	if(pcbuf == nil && callback == nil)
		n = nprint;

	return n;		
}

static void
printcreatedby(G *gp)
{
	int32 line;
	uintptr pc, tracepc;
	Func *f;
	String file;

	if((pc = gp->gopc) != 0 && (f = runtime·findfunc(pc)) != nil
		&& runtime·showframe(f, gp) && gp->goid != 1) {
		runtime·printf("created by %s\n", runtime·funcname(f));
		tracepc = pc;	// back up to CALL instruction for funcline.
		if(pc > f->entry)
			tracepc -= sizeof(uintptr);
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
	if(gp->status == Gsyscall) {
		// Override signal registers if blocked in system call.
		pc = gp->sched.pc;
		sp = gp->sched.sp;
		lr = 0;
	}

	// Print traceback. By default, omits runtime frames.
	// If that means we print nothing at all, repeat forcing all frames printed.
	if(runtime·gentraceback(pc, sp, lr, gp, 0, nil, 100, nil, nil, false) == 0)
		runtime·gentraceback(pc, sp, lr, gp, 0, nil, 100, nil, nil, true);
	printcreatedby(gp);
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
