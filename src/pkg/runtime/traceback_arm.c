// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

void runtime·deferproc(void);
void runtime·newproc(void);
void runtime·morestack(void);
void runtime·sigpanic(void);
void _div(void);
void _mod(void);
void _divu(void);
void _modu(void);

int32
runtime·gentraceback(uintptr pc0, uintptr sp0, uintptr lr0, G *gp, int32 skip, uintptr *pcbuf, int32 max, void (*callback)(Stkframe*, void*), void *v, bool printall)
{
	int32 i, n, nprint, skip0;
	uintptr x, tracepc;
	bool waspanic, printing;
	Func *f, *f2;
	Stkframe frame;
	Stktop *stk;

	skip0 = skip;

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
			if(callback != nil) {
				runtime·printf("runtime: unknown pc %p at frame %d\n", frame.pc, skip0-skip+n);
				runtime·throw("invalid stack");
			}
			break;
		}
		
		// Found an actual function.
		// Derive frame pointer and link register.
		if(frame.lr == 0)
			frame.lr = *(uintptr*)frame.sp;
		if(frame.fp == 0) {
			frame.fp = frame.sp;
			if(frame.pc > f->entry && f->frame >= sizeof(uintptr))
				frame.fp += f->frame;
		}

		// Derive size of arguments.
		frame.argp = (byte*)frame.fp + sizeof(uintptr);
		frame.arglen = 0;
		if(f->args != ArgsSizeUnknown)
			frame.arglen = f->args;
		else if(runtime·haszeroargs(f->entry))
			frame.arglen = 0;
		else if(frame.lr == (uintptr)runtime·lessstack)
			frame.arglen = stk->argsize;
		else if(f->entry == (uintptr)runtime·deferproc || f->entry == (uintptr)runtime·newproc)
			frame.arglen = 3*sizeof(uintptr) + *(int32*)frame.argp;
		else if((f2 = runtime·findfunc(frame.lr)) != nil && f2->frame >= sizeof(uintptr))
			frame.arglen = f2->frame; // conservative overestimate
		else {
			runtime·printf("runtime: unknown argument frame size for %S\n", f->name);
			if(!printing)
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
			frame.varlen = frame.fp - frame.sp;
		} else {
			if(f->locals > frame.fp - frame.sp) {
				runtime·printf("runtime: inconsistent locals=%p frame=%p fp=%p sp=%p for %S\n", (uintptr)f->locals, (uintptr)f->frame, frame.fp, frame.sp, f->name);
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

		// Do not unwind past the bottom of the stack.
		if(frame.pc == (uintptr)runtime·goexit || f->entry == (uintptr)runtime·mstart || f->entry == (uintptr)_rt0_go)
			break;

		// Unwind to next frame.
		frame.pc = frame.lr;
		frame.lr = 0;
		frame.sp = frame.fp;
		frame.fp = 0;
		
		// If this was div or divu or mod or modu, the caller had
		// an extra 8 bytes on its stack.  Adjust sp.
		if(f->entry == (uintptr)_div || f->entry == (uintptr)_divu || f->entry == (uintptr)_mod || f->entry == (uintptr)_modu)
			frame.sp += 8;
		
		// If this was deferproc or newproc, the caller had an extra 12.
		if(f->entry == (uintptr)runtime·deferproc || f->entry == (uintptr)runtime·newproc)
			frame.sp += 12;

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
	uintptr pc, tracepc;
	Func *f;

	if((pc = gp->gopc) != 0 && (f = runtime·findfunc(pc)) != nil
		&& runtime·showframe(f, gp) && gp->goid != 1) {
		runtime·printf("created by %S\n", f->name);
		tracepc = pc;	// back up to CALL instruction for funcline.
		if(pc > f->entry)
			tracepc -= sizeof(uintptr);
		runtime·printf("\t%S:%d", f->src, runtime·funcline(f, tracepc));
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
