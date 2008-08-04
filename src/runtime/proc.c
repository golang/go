// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

typedef struct Sched Sched;

M	m0;
G	g0;	// idle goroutine for m0

// Maximum number of os procs (M's) to kick off.
// Can override with $gomaxprocs environment variable.
// For now set to 1 (single-threaded), because not 
// everything is properly locked (e.g., chans) and because
// Darwin's multithreading code isn't implemented.
int32	gomaxprocs = 1;

static	int32	debug	= 0;

struct Sched {
	G *runhead;
	G *runtail;
	int32 nwait;
	int32 nready;
	int32 ng;
	int32 nm;
	M *wait;
	Lock;
};

Sched sched;

void
sys·goexit(void)
{
	if(debug){
		prints("goexit goid=");
		sys·printint(g->goid);
		prints("\n");
	}
	g->status = Gdead;
	sys·gosched();
}

void
schedinit(void)
{
	byte *p;
	extern int32 getenvc(void);
	
	p = getenv("gomaxprocs");
	if(p && '0' <= *p && *p <= '9')
		gomaxprocs = atoi(p);
	sched.nm = 1;
	sched.nwait = 1;
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

	lock(&sched);
	sched.ng++;
	goidgen++;
	newg->goid = goidgen;
	unlock(&sched);

	ready(newg);

//prints(" goid=");
//sys·printint(newg->goid);
//prints("\n");
}

void
tracebackothers(G *me)
{
	G *g;

	for(g = allg; g != nil; g = g->alllink) {
		if(g == me || g->status == Gdead)
			continue;
		prints("\ngoroutine ");
		sys·printint(g->goid);
		prints(":\n");
		traceback(g->sched.PC, g->sched.SP+8, g);  // gogo adjusts SP by 8 (not portable!)
	}
}

void newmach(void);

static void
readylocked(G *g)
{
	g->status = Grunnable;
	if(sched.runhead == nil)
		sched.runhead = g;
	else
		sched.runtail->runlink = g;
	sched.runtail = g;
	g->runlink = nil;
	sched.nready++;
	// Don't wake up another scheduler.
	// This only gets called when we're
	// about to reschedule anyway.
}

static Lock print;
	
void
ready(G *g)
{
	M *mm;

	// gp might be running on another scheduler.
	// (E.g., it queued and then we decided to wake it up
	// before it had a chance to sys·gosched().)
	// Grabbing the runlock ensures that it is not running elsewhere.
	// You can delete the if check, but don't delete the 
	// lock/unlock sequence (being able to grab the lock
	// means the proc has gone to sleep).
	lock(&g->runlock);
	if(g->status == Grunnable || g->status == Grunning)
		*(int32*)0x1023 = 0x1023;
	lock(&sched);
	g->status = Grunnable;
	if(sched.runhead == nil)
		sched.runhead = g;
	else
		sched.runtail->runlink = g;
	sched.runtail = g;
	g->runlink = nil;
	unlock(&g->runlock);
	sched.nready++;
	if(sched.nready > sched.nwait)
	if(gomaxprocs == 0 || sched.nm < gomaxprocs){
		if(debug){
			prints("new scheduler: ");
			sys·printint(sched.nready);
			prints(" > ");
			sys·printint(sched.nwait);
			prints("\n");
		}
		sched.nwait++;
		newmach();
	}
	if(sched.wait){
		mm = sched.wait;
		sched.wait = mm->waitlink;
		rwakeupandunlock(&mm->waitr);
	}else
		unlock(&sched);
}

extern void p0(void), p1(void);

G*
nextgoroutine(void)
{
	G *gp;

	while((gp = sched.runhead) == nil){
		if(debug){
			prints("nextgoroutine runhead=nil ng=");
			sys·printint(sched.ng);
			prints("\n");
		}
		if(sched.ng == 0)
			return nil;
		m->waitlink = sched.wait;
		m->waitr.l = &sched.Lock;
		sched.wait = m;
		sched.nwait++;
		if(sched.nm == sched.nwait)
			prints("all goroutines are asleep - deadlock!\n");
		rsleep(&m->waitr);
		sched.nwait--;
	}
	sched.nready--;
	sched.runhead = gp->runlink;
	return gp;
}

void
scheduler(void)
{
	G* gp;

	m->pid = getprocid();

	gosave(&m->sched);
	lock(&sched);

	if(m->curg == nil){
		// Brand new scheduler; nwait counts us.
		// Not anymore.
		sched.nwait--;
	}else{
		gp = m->curg;
		gp->m = nil;
		switch(gp->status){
		case Gdead:
			sched.ng--;
			if(debug){
				prints("sched: dead: ");
				sys·printint(sched.ng);
				prints("\n");
			}
			break;
		case Grunning:
			readylocked(gp);
			break;
		case Grunnable:
			// don't want to see this
			*(int32*)0x456 = 0x234;
			break;
		}
		unlock(&gp->runlock);
	}

	gp = nextgoroutine();
	if(gp == nil) {
//		prints("sched: no more work\n");
		sys·exit(0);
	}
	unlock(&sched);
	
	lock(&gp->runlock);
	gp->status = Grunning;
	m->curg = gp;
	gp->m = m;
	g = gp;
	gogo(&gp->sched);
}

void
newmach(void)
{
	M *mm;
	byte *stk, *stktop;
	int64 ret;
	
	sched.nm++;
	if(!(sched.nm&(sched.nm-1))){
		sys·printint(sched.nm);
		prints(" threads\n");
	}
	mm = mal(sizeof(M)+sizeof(G)+1024+104);
	sys·memclr((byte*)mm, sizeof(M));
	mm->g0 = (G*)(mm+1);
	sys·memclr((byte*)mm->g0, sizeof(G));
	stk = (byte*)mm->g0 + 104;
	stktop = stk + 1024;
	mm->g0->stackguard = stk;
	mm->g0->stackbase = stktop;
	newosproc(mm, mm->g0, stktop, (void(*)(void*))scheduler, nil);
}

void
gom0init(void)
{
	scheduler();
}

void
sys·gosched(void)
{
	if(gosave(&g->sched) == 0){
		// (rsc) signal race here?
		g = m->g0;
		gogo(&m->sched);
	}
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
