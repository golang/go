// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 386

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

static uintptr isclosureentry(uintptr);
void runtime·deferproc(void);
void runtime·newproc(void);
void runtime·newstack(void);
void runtime·morestack(void);
void runtime·sigpanic(void);

// This code is also used for the 386 tracebacks.
// Use uintptr for an appropriate word-sized integer.

// Generic traceback.  Handles runtime stack prints (pcbuf == nil)
// as well as the runtime.Callers function (pcbuf != nil).
// A little clunky to merge the two but avoids duplicating
// the code and all its subtlety.
int32
runtime·gentraceback(byte *pc0, byte *sp, byte *lr0, G *g, int32 skip, uintptr *pcbuf, int32 max)
{
	byte *p;
	int32 i, n, iter, sawnewstack;
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
	if(pc0 == g->sched.pc && sp == (byte*)g->sched.sp && pc0 == (byte*)runtime·goexit) {
		fp = sp;
		lr = pc;
		pc = (uintptr)g->entry;
	}
	
	// If the PC is zero, it's likely a nil function call.
	// Start in the caller's frame.
	if(pc == 0) {
		pc = lr;
		lr = 0;
	}

	// If the PC is zero, it's likely a nil function call.
	// Start in the caller's frame.
	if(pc == 0) {
		pc = *(uintptr*)sp;
		sp += sizeof(uintptr);
	}

	n = 0;
	sawnewstack = 0;
	stk = (Stktop*)g->stackbase;
	for(iter = 0; iter < 100 && n < max; iter++) {	// iter avoids looping forever
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
			if(pcbuf == nil)
				runtime·printf("----- stack segment boundary -----\n");
			stk = (Stktop*)stk->stackbase;
			continue;
		}
		if(pc <= 0x1000 || (f = runtime·findfunc(pc)) == nil) {
			// Dangerous, but worthwhile: see if this is a closure:
			//	ADDQ $wwxxyyzz, SP; RET
			//	[48] 81 c4 zz yy xx ww c3
			// The 0x48 byte is only on amd64.
			p = (byte*)pc;
			// We check p < p+8 to avoid wrapping and faulting if we lose track.
			if(runtime·mheap.arena_start < p && p < p+8 && p+8 < runtime·mheap.arena_used &&  // pointer in allocated memory
			   (sizeof(uintptr) != 8 || *p++ == 0x48) &&  // skip 0x48 byte on amd64
			   p[0] == 0x81 && p[1] == 0xc4 && p[6] == 0xc3) {
				sp += *(uint32*)(p+2);
				pc = *(uintptr*)sp;
				sp += sizeof(uintptr);
				lr = 0;
				fp = nil;
				continue;
			}
			
			// Closure at top of stack, not yet started.
			if(lr == (uintptr)runtime·goexit && (pc = isclosureentry(pc)) != 0) {
				fp = sp;
				continue;
			}

			// Unknown pc: stop.
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
		else {
			if(runtime·showframe(f)) {
				// Print during crash.
				//	main(0x1, 0x2, 0x3)
				//		/home/rsc/go/src/runtime/x.go:23 +0xf
				//		
				tracepc = pc;	// back up to CALL instruction for funcline.
				if(n > 0 && pc > f->entry && !waspanic)
					tracepc--;
				runtime·printf("%S(", f->name);
				for(i = 0; i < f->args; i++) {
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

		if(pcbuf == nil && f->entry == (uintptr)runtime·morestack && g == m->g0 && sawnewstack) {
			// The fact that we saw newstack means that morestack
			// has managed to record its information in m, so we can
			// use it to keep unwinding the stack.
			runtime·printf("----- morestack called from goroutine %D -----\n", m->curg->goid);
			pc = (uintptr)m->morepc;
			sp = (byte*)m->morebuf.sp - sizeof(void*);
			lr = (uintptr)m->morebuf.pc;
			fp = (byte*)m->morebuf.sp;
			sawnewstack = 0;
			g = m->curg;
			stk = (Stktop*)g->stackbase;
			continue;
		}

		if(pcbuf == nil && f->entry == (uintptr)runtime·lessstack && g == m->g0) {
			// Lessstack is running on scheduler stack.  Switch to original goroutine.
			runtime·printf("----- lessstack called from goroutine %D -----\n", m->curg->goid);
			g = m->curg;
			stk = (Stktop*)g->stackbase;
			sp = (byte*)stk->gobuf.sp;
			pc = (uintptr)stk->gobuf.pc;
			fp = nil;
			lr = 0;
			continue;
		}

		// Unwind to next frame.
		pc = lr;
		lr = 0;
		sp = fp;
		fp = nil;
	}
	
	// Show what created goroutine, except main goroutine (goid 1).
	if(pcbuf == nil && (pc = g->gopc) != 0 && (f = runtime·findfunc(pc)) != nil && g->goid != 1) {
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
runtime·traceback(byte *pc0, byte *sp, byte*, G *g)
{
	runtime·gentraceback(pc0, sp, nil, g, 0, nil, 100);
}

int32
runtime·callers(int32 skip, uintptr *pcbuf, int32 m)
{
	byte *pc, *sp;

	// our caller's pc, sp.
	sp = (byte*)&skip;
	pc = runtime·getcallerpc(&skip);

	return runtime·gentraceback(pc, sp, nil, g, skip, pcbuf, m);
}

static uintptr
isclosureentry(uintptr pc)
{
	byte *p;
	int32 i, siz;
	
	p = (byte*)pc;
	if(p < runtime·mheap.arena_start || p+32 > runtime·mheap.arena_used)
		return 0;

	if(*p == 0xe8) {
		// CALL fn
		return pc+5+*(int32*)(p+1);
	}
	
	if(sizeof(uintptr) == 8 && p[0] == 0x48 && p[1] == 0xb9 && p[10] == 0xff && p[11] == 0xd1) {
		// MOVQ $fn, CX; CALL *CX
		return *(uintptr*)(p+2);
	}

	// SUBQ $siz, SP
	if((sizeof(uintptr) == 8 && *p++ != 0x48) || *p++ != 0x81 || *p++ != 0xec)
		return 0;
	siz = *(uint32*)p;
	p += 4;
	
	// MOVQ $q, SI
	if((sizeof(uintptr) == 8 && *p++ != 0x48) || *p++ != 0xbe)
		return 0;
	p += sizeof(uintptr);

	// MOVQ SP, DI
	if((sizeof(uintptr) == 8 && *p++ != 0x48) || *p++ != 0x89 || *p++ != 0xe7)
		return 0;

	// CLD on 32-bit
	if(sizeof(uintptr) == 4 && *p++ != 0xfc)
		return 0;

	if(siz <= 4*sizeof(uintptr)) {
		// MOVSQ...
		for(i=0; i<siz; i+=sizeof(uintptr))
			if((sizeof(uintptr) == 8 && *p++ != 0x48) || *p++ != 0xa5)
				return 0;
	} else {
		// MOVQ $(siz/8), CX  [32-bit immediate siz/8]
		if((sizeof(uintptr) == 8 && *p++ != 0x48) || *p++ != 0xc7 || *p++ != 0xc1)
			return 0;
		p += 4;
		
		// REP MOVSQ
		if(*p++ != 0xf3 || (sizeof(uintptr) == 8 && *p++ != 0x48) || *p++ != 0xa5)
			return 0;
	}
	
	// CALL fn
	if(*p == 0xe8) {
		p++;
		return (uintptr)p+4 + *(int32*)p;
	}
	
	// MOVQ $fn, CX; CALL *CX
	if(sizeof(uintptr) != 8 || *p++ != 0x48 || *p++ != 0xb9)
		return 0;

	pc = *(uintptr*)p;
	p += 8;
	
	if(*p++ != 0xff || *p != 0xd1)
		return 0;

	return pc;
}
