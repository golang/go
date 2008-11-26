// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

typedef struct Sched Sched;

M	m0;
G	g0;	// idle goroutine for m0

static	int32	debug	= 0;
static	Lock	debuglock;

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
	int32 mcount;	// number of ms that have been created
	int32 mcpu;	// number of ms executing on cpu
	int32 mcpumax;	// max number of ms allowed on cpu
	int32 msyscall;	// number of ms in system calls

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
static void matchmg(void);	// match ms to gs
static void readylocked(G*);	// ready, but sched is locked

// Scheduler loop.
static void scheduler(void);

// The bootstrap sequence is:
//
//	call osinit
//	call schedinit
//	make & queue new G
//	call mstart
//
// The new G does:
//
//	call main·init_function
//	call initdone
//	call main·main
void
schedinit(void)
{
	int32 n;
	byte *p;

	sched.mcpumax = 1;
	p = getenv("GOMAXPROCS");
	if(p != nil && (n = atoi(p)) != 0)
		sched.mcpumax = n;
	sched.mcount = 1;
	sched.predawn = 1;
}

// Called after main·init_function; main·main will be called on return.
void
initdone(void)
{
	// Let's go.
	sched.predawn = 0;

	// If main·init_function started other goroutines,
	// kick off new ms to handle them, like ready
	// would have, had it not been pre-dawn.
	lock(&sched);
	matchmg();
	unlock(&sched);
}

void
sys·goexit(void)
{
	if(debug > 1){
		lock(&debuglock);
		printf("goexit goid=%d\n", g->goid);
		unlock(&debuglock);
	}
	g->status = Gmoribund;
	sys·gosched();
}

G*
malg(int32 stacksize)
{
	G *g;
	byte *stk;

	// 160 is the slop amount known to the stack growth code
	g = mal(sizeof(G));
	stk = mal(160 + stacksize);
	g->stack0 = stk;
	g->stackguard = stk + 160;
	g->stackbase = stk + 160 + stacksize;
	return g;
}

void
sys·newproc(int32 siz, byte* fn, byte* arg0)
{
	byte *stk, *sp;
	G *newg;

//printf("newproc siz=%d fn=%p", siz, fn);

	siz = (siz+7) & ~7;
	if(siz > 1024)
		throw("sys·newproc: too many args");

	lock(&sched);

	if((newg = gfget()) != nil){
		newg->status = Gwaiting;
	}else{
		newg = malg(4096);
		newg->status = Gwaiting;
		newg->alllink = allg;
		allg = newg;
	}
	stk = newg->stack0;

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

//printf(" goid=%d\n", newg->goid);
}

void
tracebackothers(G *me)
{
	G *g;

	for(g = allg; g != nil; g = g->alllink) {
		if(g == me || g->status == Gdead)
			continue;
		printf("\ngoroutine %d:\n", g->goid);
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
	lock(&sched);
	readylocked(g);
	unlock(&sched);
}

// Mark g ready to run.  Sched is already locked.
// G might be running already and about to stop.
// The sched lock protects g->status from changing underfoot.
static void
readylocked(G *g)
{
	if(g->m){
		// Running on another machine.
		// Ready it when it stops.
		g->readyonstop = 1;
		return;
	}

	// Mark runnable.
	if(g->status == Grunnable || g->status == Grunning)
		throw("bad g->status in ready");
	g->status = Grunnable;

	gput(g);
	if(!sched.predawn)
		matchmg();
}

// Get the next goroutine that m should run.
// Sched must be locked on entry, is unlocked on exit.
// Makes sure that at most $GOMAXPROCS gs are
// running on cpus (not in system calls) at any given time.
static G*
nextgandunlock(void)
{
	G *gp;

	// On startup, each m is assigned a nextg and
	// has already been accounted for in mcpu.
	if(m->nextg != nil) {
		gp = m->nextg;
		m->nextg = nil;
		unlock(&sched);
		if(debug > 1) {
			lock(&debuglock);
			printf("m%d nextg found g%d\n", m->id, gp->goid);
			unlock(&debuglock);
		}
		return gp;
	}

	// Otherwise, look for work.
	if(sched.mcpu < sched.mcpumax && (gp=gget()) != nil) {
		sched.mcpu++;
		unlock(&sched);
		if(debug > 1) {
			lock(&debuglock);
			printf("m%d nextg got g%d\n", m->id, gp->goid);
			unlock(&debuglock);
		}
		return gp;
	}

	// Otherwise, sleep.
	mput(m);
	if(sched.mcpu == 0 && sched.msyscall == 0)
		throw("all goroutines are asleep - deadlock!");
	m->nextg = nil;
	noteclear(&m->havenextg);
	unlock(&sched);

	notesleep(&m->havenextg);
	if((gp = m->nextg) == nil)
		throw("bad m->nextg in nextgoroutine");
	m->nextg = nil;
	if(debug > 1) {
		lock(&debuglock);
		printf("m%d nextg woke g%d\n", m->id, gp->goid);
		unlock(&debuglock);
	}
	return gp;
}

// Called to start an M.
void
mstart(void)
{
	minit();
	scheduler();
}

// Kick of new ms as needed (up to mcpumax).
// There are already `other' other cpus that will
// start looking for goroutines shortly.
// Sched is locked.
static void
matchmg(void)
{
	M *m;
	G *g;

	if(debug > 1 && sched.ghead != nil) {
		lock(&debuglock);
		printf("matchmg mcpu=%d mcpumax=%d gwait=%d\n", sched.mcpu, sched.mcpumax, sched.gwait);
		unlock(&debuglock);
	}

	while(sched.mcpu < sched.mcpumax && (g = gget()) != nil){
		sched.mcpu++;
		if((m = mget()) != nil){
			if(debug > 1) {
				lock(&debuglock);
				printf("wakeup m%d g%d\n", m->id, g->goid);
				unlock(&debuglock);
			}
			m->nextg = g;
			notewakeup(&m->havenextg);
		}else{
			m = mal(sizeof(M));
			m->g0 = malg(1024);
			m->nextg = g;
			m->id = sched.mcount++;
			if(debug) {
				lock(&debuglock);
				printf("alloc m%d g%d\n", m->id, g->goid);
				unlock(&debuglock);
			}
			newosproc(m, m->g0, m->g0->stackbase, mstart);
		}
	}
}

// Scheduler loop: find g to run, run it, repeat.
static void
scheduler(void)
{
	G* gp;

	lock(&sched);
	if(gosave(&m->sched)){
		// Jumped here via gosave/gogo, so didn't
		// execute lock(&sched) above.
		lock(&sched);

		if(sched.predawn)
			throw("init sleeping");

		// Just finished running m->curg.
		gp = m->curg;
		gp->m = nil;
		sched.mcpu--;
		if(debug > 1) {
			lock(&debuglock);
			printf("m%d sched g%d status %d\n", m->id, gp->goid, gp->status);
			unlock(&debuglock);
		}
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
		if(gp->readyonstop){
			gp->readyonstop = 0;
			readylocked(gp);
		}
	}

	// Find (or wait for) g to run.  Unlocks sched.
	gp = nextgandunlock();
	gp->readyonstop = 0;
	gp->status = Grunning;
	if(debug > 1) {
		lock(&debuglock);
		printf("m%d run g%d\n", m->id, gp->goid);
		unlock(&debuglock);
	}
	m->curg = gp;
	gp->m = m;
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
		g = m->g0;
		gogo(&m->sched);
	}
}

// The goroutine g is about to enter a system call.
// Record that it's not using the cpu anymore.
// This is called only from the go syscall library, not
// from the low-level system calls used by the runtime.
// The "arguments" are syscall.Syscall's stack frame
void
sys·entersyscall(uint64 callerpc, int64 trap)
{
	USED(callerpc);

	if(debug > 1) {
		lock(&debuglock);
		printf("m%d g%d enter syscall %D\n", m->id, g->goid, trap);
		unlock(&debuglock);
	}
	lock(&sched);
	sched.mcpu--;
	sched.msyscall++;
	if(sched.gwait != 0)
		matchmg();
	unlock(&sched);
}

// The goroutine g exited its system call.
// Arrange for it to run on a cpu again.
// This is called only from the go syscall library, not
// from the low-level system calls used by the runtime.
void
sys·exitsyscall(void)
{
	if(debug > 1) {
		lock(&debuglock);
		printf("m%d g%d exit syscall mcpu=%d mcpumax=%d\n", m->id, g->goid, sched.mcpu, sched.mcpumax);
		unlock(&debuglock);
	}

	lock(&sched);
	sched.msyscall--;
	sched.mcpu++;
	// Fast path - if there's room for this m, we're done.
	if(sched.mcpu <= sched.mcpumax) {
		unlock(&sched);
		return;
	}
	unlock(&sched);

	// Slow path - all the cpus are taken.
	// The scheduler will ready g and put this m to sleep.
	// When the scheduler takes g awa from m,
	// it will undo the sched.mcpu++ above.
	sys·gosched();
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

// printf("oldstack m->cret=%p\n", m->cret);

	top = (Stktop*)m->curg->stackbase;

	siz2 = (top->magic>>32) & 0xffffLL;

	sp = (byte*)top;
	if(siz2 > 0) {
		siz2 = (siz2+7) & ~7;
		sp -= siz2;
		mcpy(top->oldsp+16, sp, siz2);
	}

	// call  no more functions after this point - stackguard disagrees with SP
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
