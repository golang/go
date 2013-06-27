// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 386

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

void runtime·deferproc(void);
void runtime·newproc(void);
void runtime·morestack(void);
void runtime·sigpanic(void);

// This code is also used for the 386 tracebacks.
// Use uintptr for an appropriate word-sized integer.

// Generic traceback.  Handles runtime stack prints (pcbuf == nil),
// the runtime.Callers function (pcbuf != nil), as well as the garbage
// collector (callback != nil).  A little clunky to merge these, but avoids
// duplicating the code and all its subtlety.
int32
runtime·gentraceback(uintptr pc0, uintptr sp0, uintptr lr0, G *gp, int32 skip, uintptr *pcbuf, int32 max, void (*callback)(Stkframe*, void*), void *v, bool printall)
{
	int32 i, n, nprint;
	uintptr tracepc;
	bool waspanic, printing;
	Func *f, *f2;
	Stkframe frame;
	Stktop *stk;

	USED(lr0);

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
		frame.sp += sizeof(uintptr);
	}

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
		else if(runtime·haszeroargs(f->entry))
			frame.arglen = 0;
		else if(frame.lr == (uintptr)runtime·lessstack)
			frame.arglen = stk->argsize;
		else if(f->entry == (uintptr)runtime·deferproc || f->entry == (uintptr)runtime·newproc)
			frame.arglen = 2*sizeof(uintptr) + *(int32*)frame.argp;
		else if((f2 = runtime·findfunc(frame.lr)) != nil && f2->frame >= sizeof(uintptr))
			frame.arglen = f2->frame; // conservative overestimate
		else {
			runtime·printf("runtime: unknown argument frame size for %S\n", f->name);
			if(!printing)
				runtime·throw("invalid stack");
		}

		// Derive location and size of local variables.
		if(frame.fp == frame.sp + sizeof(uintptr)) {
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
			if(printall || runtime·showframe(f, gp)) {
				// Print during crash.
				//	main(0x1, 0x2, 0x3)
				//		/home/rsc/go/src/runtime/x.go:23 +0xf
				//		
				tracepc = frame.pc;	// back up to CALL instruction for funcline.
				if(n > 0 && frame.pc > f->entry && !waspanic)
					tracepc--;
				runtime·printf("%S(", f->name);
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
				runtime·printf("\t%S:%d", f->src, runtime·funcline(f, tracepc));
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

		if(f->entry == (uintptr)runtime·deferproc || f->entry == (uintptr)runtime·newproc)
			frame.fp += 2*sizeof(uintptr);

		// Do not unwind past the bottom of the stack.
		if(frame.pc == (uintptr)runtime·goexit || f->entry == (uintptr)runtime·mstart || f->entry == (uintptr)_rt0_go)
			break;

		// Unwind to next frame.
		frame.pc = frame.lr;
		frame.lr = 0;
		frame.sp = frame.fp;
		frame.fp = 0;
	}
	
	if(pcbuf == nil && callback == nil)
		n = nprint;
	
	return n;
}

static void
printcreatedby(G *gp)
{
	uintptr pc, tracepc;
	Func *f;

	// Show what created goroutine, except main goroutine (goid 1).
	if((pc = gp->gopc) != 0 && (f = runtime·findfunc(pc)) != nil && gp->goid != 1) {
		runtime·printf("created by %S\n", f->name);
		tracepc = pc;	// back up to CALL instruction for funcline.
		if(pc > f->entry)
			tracepc--;
		runtime·printf("\t%S:%d", f->src, runtime·funcline(f, tracepc));
		if(pc > f->entry)
			runtime·printf(" +%p", (uintptr)(pc - f->entry));
		runtime·printf("\n");
	}
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
	
	// Print traceback. By default, omits runtime frames.
	// If that means we print nothing at all, repeat forcing all frames printed.
	if(runtime·gentraceback(pc, sp, 0, gp, 0, nil, 100, nil, nil, false) == 0)
		runtime·gentraceback(pc, sp, 0, gp, 0, nil, 100, nil, nil, true);
	printcreatedby(gp);
}

int32
runtime·callers(int32 skip, uintptr *pcbuf, int32 m)
{
	uintptr pc, sp;

	sp = runtime·getcallersp(&skip);
	pc = (uintptr)runtime·getcallerpc(&skip);

	return runtime·gentraceback(pc, sp, 0, g, skip, pcbuf, m, nil, nil, false);
}
