// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

typedef struct Sched Sched;

M	m0;
G	g0;	// idle goroutine for m0

static	int32	debug	= 0;

// Go scheduler
//
// The go scheduler's job is to match ready-to-run goroutines (`g's)
// with waiting-for-work schedulers (`m's).  If there are ready gs
// and no waiting ms, ready() will start a new m running in a new
// OS thread, so that all ready gs can run simultaneously, up to a limit.
// For now, ms never go away.
//
// The default maximum number of ms is one: go runs single-threaded.
// This is because some locking details have to be worked ou
// (select in particular is not locked properly) and because the low-level
// code hasn't been written yet for OS X.  Setting the environmen
// variable $gomaxprocs changes sched.mmax for now.
//
// Even a program that can run without deadlock in a single process
// might use more ms if given the chance.  For example, the prime
// sieve will use as many ms as there are primes (up to sched.mmax),
// allowing different stages of the pipeline to execute in parallel.
// We could revisit this choice, only kicking off new ms for blocking
// system calls, but that would limit the amount of parallel computation
// that go would try to do.
//
// In general, one could imagine all sorts of refinements to the
// scheduler, but the goal now is just to get something working on
// Linux and OS X.

struct Sched {
	Lock;

	G *gfree;	// available gs (status == Gdead)

	G *ghead;	// gs waiting to run
	G *gtail;
	int32 gwait;	// number of gs waiting to run
	int32 gcount;	// number of gs that are alive

	M *mhead;	// ms waiting for work
	int32 mwait;	// number of ms waiting for work
	int32 mcount;	// number of ms that are alive
	int32 mmax;	// max number of ms allowed

	int32 predawn;	// running initialization, don't run new gs.
};

Sched sched;

// Scheduling helpers.  Sched must be locked.
static void gput(G*);	// put/get on ghead/gtail
static G* gget(void);
static void mput(M*);	// put/get on mhead
static M* mget(void);
static void gfput(G*);	// put/get on gfree
static G* gfget(void);
static void mnew(void);	// kick off new m
static void readylocked(G*);	// ready, but sched is locked

// Scheduler loop.
static void scheduler(void);

// Called before main·init_function.
void
schedinit(void)
{
	int32 n;
	byte *p;

	sched.mmax = 1;
	p = getenv("gomaxprocs");
	if(p != nil && (n = atoi(p)) != 0)
		sched.mmax = n;
	sched.mcount = 1;
	sched.predawn = 1;
}

// Called after main·init_function; main·main is on ready queue.
void
m0init(void)
{
	int32 i;

	// Let's go.
	sched.predawn = 0;

	// There's already one m (us).
	// If main·init_function started other goroutines,
	// kick off new ms to handle them, like ready
	// would have, had it not been pre-dawn.
	for(i=1; i<sched.gcount && i<sched.mmax; i++)
		mnew();

	scheduler();
}

void
sys·goexit(void)
{
	if(debug){
		prints("goexit goid=");
		sys·printint(g->goid);
		prints("\n");
	}
	g->status = Gmoribund;
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

	lock(&sched);

	if((newg = gfget()) != nil){
		newg->status = Gwaiting;
		stk = newg->stack0;
	}else{
		newg = mal(sizeof(G));
		stk = mal(4096);
		newg->stack0 = stk;
		newg->status = Gwaiting;
		newg->alllink = allg;
		allg = newg;
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

	sched.gcount++;
	goidgen++;
	newg->goid = goidgen;

	readylocked(newg);
	unlock(&sched);

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

// Put on `g' queue.  Sched must be locked.
static void
gput(G *g)
{
	g->schedlink = nil;
	if(sched.ghead == nil)
		sched.ghead = g;
	else
		sched.gtail->schedlink = g;
	sched.gtail = g;
	sched.gwait++;
}

// Get from `g' queue.  Sched must be locked.
static G*
gget(void)
{
	G *g;

	g = sched.ghead;
	if(g){
		sched.ghead = g->schedlink;
		if(sched.ghead == nil)
			sched.gtail = nil;
		sched.gwait--;
	}
	return g;
}

// Put on `m' list.  Sched must be locked.
static void
mput(M *m)
{
	m->schedlink = sched.mhead;
	sched.mhead = m;
	sched.mwait++;
}

// Get from `m' list.  Sched must be locked.
static M*
mget(void)
{
	M *m;

	m = sched.mhead;
	if(m){
		sched.mhead = m->schedlink;
		sched.mwait--;
	}
	return m;
}

// Put on gfree list.  Sched must be locked.
static void
gfput(G *g)
{
	g->schedlink = sched.gfree;
	sched.gfree = g;
}

// Get from gfree list.  Sched must be locked.
static G*
gfget(void)
{
	G *g;

	g = sched.gfree;
	if(g)
		sched.gfree = g->schedlink;
	return g;
}

// Mark g ready to run.
void
ready(G *g)
{
	// Wait for g to stop running (for example, it migh
	// have queued itself on a channel but not yet gotten
	// a chance to call sys·gosched and actually go to sleep).
	notesleep(&g->stopped);

	lock(&sched);
	readylocked(g);
	unlock(&sched);
}

// Mark g ready to run.  Sched is already locked,
// and g is known not to be running right now
// (i.e., ready has slept on g->stopped or the g was
// just allocated in sys·newproc).
static void
readylocked(G *g)
{
	M *m;

	// Mark runnable.
	if(g->status == Grunnable || g->status == Grunning)
		throw("bad g->status in ready");
	g->status = Grunnable;

	// Before we've gotten to main·main,
	// only queue new gs, don't run them
	// or try to allocate new ms for them.
	// That includes main·main itself.
	if(sched.predawn){
		gput(g);
	}

	// Else if there's an m waiting, give it g.
	else if((m = mget()) != nil){
		m->nextg = g;
		notewakeup(&m->havenextg);
	}

	// Else put g on queue, kicking off new m if needed.
	else{
		gput(g);
		if(sched.mcount < sched.mmax)
			mnew();
	}
}

// Get the next goroutine that m should run.
// Sched must be locked on entry, is unlocked on exit.
static G*
nextgandunlock(void)
{
	G *gp;

	if((gp = gget()) != nil){
		unlock(&sched);
		return gp;
	}

	mput(m);
	if(sched.mcount == sched.mwait)
		prints("warning: all goroutines are asleep - deadlock!\n");
	m->nextg = nil;
	noteclear(&m->havenextg);
	unlock(&sched);

	notesleep(&m->havenextg);
	if((gp = m->nextg) == nil)
		throw("bad m->nextg in nextgoroutine");
	m->nextg = nil;
	return gp;
}

// Scheduler loop: find g to run, run it, repeat.
static void
scheduler(void)
{
	G* gp;

	// Initialization.
	m->procid = getprocid();
	lock(&sched);

	if(gosave(&m->sched)){
		// Jumped here via gosave/gogo, so didn'
		// execute lock(&sched) above.
		lock(&sched);

		// Just finished running m->curg.
		gp = m->curg;
		gp->m = nil;	// for debugger
		switch(gp->status){
		case Grunnable:
		case Gdead:
			// Shouldn't have been running!
			throw("bad gp->status in sched");
		case Grunning:
			gp->status = Grunnable;
			gput(gp);
			break;
		case Gmoribund:
			gp->status = Gdead;
			if(--sched.gcount == 0)
				sys·exit(0);
			break;
		}
		notewakeup(&gp->stopped);
	}

	// Find (or wait for) g to run.  Unlocks sched.
	gp = nextgandunlock();

	noteclear(&gp->stopped);
	gp->status = Grunning;
	m->curg = gp;
	gp->m = m;	// for debugger
	g = gp;
	gogo(&gp->sched);
}

// Enter scheduler.  If g->status is Grunning,
// re-queues g and runs everyone else who is waiting
// before running g again.  If g->status is Gmoribund,
// kills off g.
void
sys·gosched(void)
{
	if(gosave(&g->sched) == 0){
		// TODO(rsc) signal race here?
		// If a signal comes in between
		// changing g and changing SP,
		// growing the stack will fail.
		g = m->g0;
		gogo(&m->sched);
	}
}

// Fork off a new m.  Sched must be locked.
static void
mnew(void)
{
	M *m;
	G *g;
	byte *stk, *stktop;

	sched.mcount++;
	if(debug){
		sys·printint(sched.mcount);
		prints(" threads\n");
	}

	// Allocate m, g, stack in one chunk.
	// 1024 and 104 are the magic constants
	// use in rt0_amd64.s when setting up g0.
	m = mal(sizeof(M)+sizeof(G)+104+1024);
	g = (G*)(m+1);
	stk = (byte*)g + 104;
	stktop = stk + 1024;

	m->g0 = g;
	g->stackguard = stk;
	g->stackbase = stktop;
	newosproc(m, g, stktop, scheduler);
}

//
// the calling sequence for a routine tha
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

	siz2 = (top->magic>>32) & 0xffffLL;

	sp = (byte*)top;
	if(siz2 > 0) {
		siz2 = (siz2+7) & ~7;
		sp -= siz2;
		mcpy(top->oldsp+16, sp, siz2);
	}

	// call  no more functions after this point - limit register disagrees with R15
	m->curg->stackbase = top->oldbase;
	m->curg->stackguard = top->oldguard;
	m->morestack.SP = top->oldsp+8;
	m->morestack.PC = (byte*)(*(uint64*)(top->oldsp+8));

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
