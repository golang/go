// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

void runtime·deferproc(void);
void runtime·newproc(void);
void runtime·newstack(void);
void runtime·morestack(void);
void runtime·sigpanic(void);
void _div(void);
void _mod(void);
void _divu(void);
void _modu(void);

int32
runtime·gentraceback(byte *pc0, byte *sp, byte *lr0, G *gp, int32 skip, uintptr *pcbuf, int32 max, void (*fn)(Func*, byte*, byte*, void*), void *arg)
{
	int32 i, n;
	uintptr pc, lr, tracepc, x;
	byte *fp;
	bool waspanic;
	Stktop *stk;
	Func *f;

	pc = (uintptr)pc0;
	lr = (uintptr)lr0;
	fp = nil;
	waspanic = false;

	// If the PC is goexit, the goroutine hasn't started yet.
	if(pc == (uintptr)runtime·goexit && gp->fnstart != nil) {
		pc = (uintptr)gp->fnstart->fn;
		lr = (uintptr)runtime·goexit;
	}

	// If the PC is zero, it's likely a nil function call.
	// Start in the caller's frame.
	if(pc == 0) {
		pc = lr;
		lr = 0;
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
		if(lr == 0)
			lr = *(uintptr*)sp;
		if(fp == nil) {
			fp = sp;
			if(pc > f->entry && f->frame >= sizeof(uintptr))
				fp += f->frame - sizeof(uintptr);
			fp += sizeof(uintptr);
		}

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
				tracepc = pc;	// back up to CALL instruction for funcline.
				if(n > 0 && pc > f->entry && !waspanic)
					tracepc -= sizeof(uintptr);
				if(m->throwing && gp == m->curg)
					runtime·printf("[fp=%p] ", fp);
				runtime·printf("%S(", f->name);
				for(i = 0; i < f->args/sizeof(uintptr); i++) {
					if(i != 0)
						runtime·prints(", ");
					runtime·printhex(((uintptr*)fp)[1+i]);
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

		if(pcbuf == nil && fn == nil && f->entry == (uintptr)runtime·newstack && gp == m->g0) {
			runtime·printf("----- newstack called from goroutine %D -----\n", m->curg->goid);
			pc = (uintptr)m->morepc;
			sp = (byte*)m->moreargp - sizeof(void*);
			lr = (uintptr)m->morebuf.pc;
			fp = (byte*)m->morebuf.sp;
			gp = m->curg;
			stk = (Stktop*)gp->stackbase;
			continue;
		}
		
		if(pcbuf == nil && fn == nil && f->entry == (uintptr)runtime·lessstack && gp == m->g0) {
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
		
		// If this was div or divu or mod or modu, the caller had
		// an extra 8 bytes on its stack.  Adjust sp.
		if(f->entry == (uintptr)_div || f->entry == (uintptr)_divu || f->entry == (uintptr)_mod || f->entry == (uintptr)_modu)
			sp += 8;
		
		// If this was deferproc or newproc, the caller had an extra 12.
		if(f->entry == (uintptr)runtime·deferproc || f->entry == (uintptr)runtime·newproc)
			sp += 12;

		// sighandler saves the lr on stack before faking a call to sigpanic
		if(waspanic) {
			x = *(uintptr *)sp;
			sp += 4;
			f = runtime·findfunc(pc);
			if (f == nil) {
				pc = x;
			} else if (f->frame == 0)
				lr = x;
		}
	}
	
	if(pcbuf == nil && fn == nil && (pc = gp->gopc) != 0 && (f = runtime·findfunc(pc)) != nil
			&& runtime·showframe(f, gp == m->curg) && gp->goid != 1) {
		runtime·printf("created by %S\n", f->name);
		tracepc = pc;	// back up to CALL instruction for funcline.
		if(n > 0 && pc > f->entry)
			tracepc -= sizeof(uintptr);
		runtime·printf("\t%S:%d", f->src, runtime·funcline(f, tracepc));
		if(pc > f->entry)
			runtime·printf(" +%p", (uintptr)(pc - f->entry));
		runtime·printf("\n");
	}

	return n;		
}

void
runtime·traceback(byte *pc0, byte *sp, byte *lr, G *gp)
{
	if(gp->status == Gsyscall) {
		// Override signal registers if blocked in system call.
		pc0 = gp->sched.pc;
		sp = (byte*)gp->sched.sp;
		lr = nil;
	}
	runtime·gentraceback(pc0, sp, lr, gp, 0, nil, 100, nil, nil);
}

// func caller(n int) (pc uintptr, file string, line int, ok bool)
int32
runtime·callers(int32 skip, uintptr *pcbuf, int32 m)
{
	byte *pc, *sp;
	
	sp = runtime·getcallersp(&skip);
	pc = runtime·getcallerpc(&skip);

	return runtime·gentraceback(pc, sp, 0, g, skip, pcbuf, m, nil, nil);
}
