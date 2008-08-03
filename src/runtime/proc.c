// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	int32	debug	= 0;

void
sys·goexit(void)
{
//prints("goexit goid=");
//sys·printint(g->goid);
//prints("\n");
	g->status = Gdead;
	sys·gosched();
}

void
sys·newproc(int32 siz, byte* fn, byte* arg0)
{
	byte *stk, *sp;
	G *newg;

//prints("newproc siz=");
//sys·printint(siz);
//prints(" fn=");
//sys·printpointer(fn);

	siz = (siz+7) & ~7;
	if(siz > 1024)
		throw("sys·newproc: too many args");

	// try to rip off an old goroutine
	for(newg=allg; newg!=nil; newg=newg->alllink)
		if(newg->status == Gdead)
			break;

	if(newg == nil) {
		newg = mal(sizeof(G));
		stk = mal(4096);
		newg->stack0 = stk;

		newg->status = Gwaiting;
		newg->alllink = allg;
		allg = newg;
	} else {
		stk = newg->stack0;
		newg->status = Gwaiting;
	}

	newg->stackguard = stk+160;

	sp = stk + 4096 - 4*8;
	newg->stackbase = sp;

	sp -= siz;
	mcpy(sp, (byte*)&arg0, siz);

	sp -= 8;
	*(byte**)sp = (byte*)sys·goexit;

	sp -= 8;	// retpc used by gogo
	newg->sched.SP = sp;
	newg->sched.PC = fn;

	goidgen++;
	newg->goid = goidgen;

	newg->status = Grunnable;

//prints(" goid=");
//sys·printint(newg->goid);
//prints("\n");
}

void
tracebackothers(G *me)
{
	G *g;

	for(g = allg; g != nil; g = g->alllink) {
		if(g == me)
			continue;
		prints("\ngoroutine ");
		sys·printint(g->goid);
		prints(":\n");
		traceback(g->sched.PC, g->sched.SP+8, g);  // gogo adjusts SP by 8 (not portable!)
	}
}

G*
nextgoroutine(void)
{
	G *gp;

	gp = m->lastg;
	if(gp == nil)
		gp = allg;

	for(gp=gp->alllink; gp!=nil; gp=gp->alllink) {
		if(gp->status == Grunnable) {
			m->lastg = gp;
			return gp;
		}
	}
	for(gp=allg; gp!=nil; gp=gp->alllink) {
		if(gp->status == Grunnable) {
			m->lastg = gp;
			return gp;
		}
	}
	return nil;
}

void
scheduler(void)
{
	G* gp;
	
	gosave(&m->sched);
	gp = nextgoroutine();
	if(gp == nil) {
//		prints("sched: no more work\n");
		sys·exit(0);
	}
	m->curg = gp;
	g = gp;
	gogo(&gp->sched);
}

void
gom0init(void)
{
	scheduler();
}

void
sys·gosched(void)
{
	if(gosave(&g->sched))
		return;
	g = m->g0;
	gogo(&m->sched);
}

//
// the calling sequence for a routine that
// needs N bytes stack, A args.
//
//	N1 = (N+160 > 4096)? N+160: 0
//	A1 = A
//
// if N <= 75
//	CMPQ	SP, 0(R15)
//	JHI	4(PC)
//	MOVQ	$(N1<<0) | (A1<<32)), AX
//	MOVQ	AX, 0(R14)
//	CALL	sys·morestack(SB)
//
// if N > 75
//	LEAQ	(-N-75)(SP), AX
//	CMPQ	AX, 0(R15)
//	JHI	4(PC)
//	MOVQ	$(N1<<0) | (A1<<32)), AX
//	MOVQ	AX, 0(R14)
//	CALL	sys·morestack(SB)
//

void
oldstack(void)
{
	Stktop *top;
	uint32 siz2;
	byte *sp;

// prints("oldstack m->cret = ");
// sys·printpointer((void*)m->cret);
// prints("\n");

	top = (Stktop*)m->curg->stackbase;

	m->curg->stackbase = top->oldbase;
	m->curg->stackguard = top->oldguard;
	siz2 = (top->magic>>32) & 0xffffLL;

	sp = (byte*)top;
	if(siz2 > 0) {
		siz2 = (siz2+7) & ~7;
		sp -= siz2;
		mcpy(top->oldsp+16, sp, siz2);
	}

	m->morestack.SP = top->oldsp+8;
	m->morestack.PC = (byte*)(*(uint64*)(top->oldsp+8));

// prints("oldstack sp=");
// sys·printpointer(m->morestack.SP);
// prints(" pc=");
// sys·printpointer(m->morestack.PC);
// prints("\n");

	gogoret(&m->morestack, m->cret);
}

void
newstack(void)
{
	int32 siz1, siz2;
	Stktop *top;
	byte *stk, *sp;
	void (*fn)(void);

	siz1 = m->morearg & 0xffffffffLL;
	siz2 = (m->morearg>>32) & 0xffffLL;

// prints("newstack siz1=");
// sys·printint(siz1);
// prints(" siz2=");
// sys·printint(siz2);
// prints(" moresp=");
// sys·printpointer(m->moresp);
// prints("\n");

	if(siz1 < 4096)
		siz1 = 4096;
	stk = mal(siz1 + 1024);
	stk += 512;

	top = (Stktop*)(stk+siz1-sizeof(*top));

	top->oldbase = m->curg->stackbase;
	top->oldguard = m->curg->stackguard;
	top->oldsp = m->moresp;
	top->magic = m->morearg;

	m->curg->stackbase = (byte*)top;
	m->curg->stackguard = stk + 160;

	sp = (byte*)top;
	
	if(siz2 > 0) {
		siz2 = (siz2+7) & ~7;
		sp -= siz2;
		mcpy(sp, m->moresp+16, siz2);
	}

	g = m->curg;
	fn = (void(*)(void))(*(uint64*)m->moresp);

// prints("fn=");
// sys·printpointer(fn);
// prints("\n");

	setspgoto(sp, fn, retfromnewstack);

	*(int32*)345 = 123;	// never return
}

void
sys·morestack(uint64 u)
{
	while(g == m->g0) {
		// very bad news
		*(int32*)123 = 123;
	}

	g = m->g0;
	m->moresp = (byte*)(&u-1);
	setspgoto(m->sched.SP, newstack, nil);

	*(int32*)234 = 123;	// never return
}
