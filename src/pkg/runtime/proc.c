// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "malloc.h"

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
	int32 gomaxprocs;
	int32 msyscall;	// number of ms in system calls

	int32 predawn;	// running initialization, don't run new gs.

	Note	stopped;	// one g can wait here for ms to stop
	int32 waitstop;	// after setting this flag
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

	mallocinit();
	goargs();

	// Allocate internal symbol table representation now,
	// so that we don't need to call malloc when we crash.
	findfunc(0);

	sched.gomaxprocs = 1;
	p = getenv("GOMAXPROCS");
	if(p != nil && (n = atoi(p)) != 0)
		sched.gomaxprocs = n;
	sched.mcpumax = sched.gomaxprocs;
	sched.mcount = 1;
	sched.predawn = 1;
}

// Called after main·init_function; main·main will be called on return.
void
initdone(void)
{
	// Let's go.
	sched.predawn = 0;
	mstats.enablegc = 1;

	// If main·init_function started other goroutines,
	// kick off new ms to handle them, like ready
	// would have, had it not been pre-dawn.
	lock(&sched);
	matchmg();
	unlock(&sched);
}

void
goexit(void)
{
	if(debug > 1){
		lock(&debuglock);
		printf("goexit goid=%d\n", g->goid);
		unlock(&debuglock);
	}
	g->status = Gmoribund;
	gosched();
}

void
tracebackothers(G *me)
{
	G *g;

	for(g = allg; g != nil; g = g->alllink) {
		if(g == me || g->status == Gdead)
			continue;
		printf("\ngoroutine %d:\n", g->goid);
		traceback(g->sched.PC, g->sched.SP+sizeof(uintptr), g);  // gogo adjusts SP by one word
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
	if(sched.waitstop && sched.mcpu <= sched.mcpumax) {
		sched.waitstop = 0;
		notewakeup(&sched.stopped);
	}
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

// TODO(rsc): Remove. This is only temporary,
// for the mark and sweep collector.
void
stoptheworld(void)
{
	lock(&sched);
	sched.mcpumax = 1;
	while(sched.mcpu > 1) {
		noteclear(&sched.stopped);
		sched.waitstop = 1;
		unlock(&sched);
		notesleep(&sched.stopped);
		lock(&sched);
	}
	unlock(&sched);
}

// TODO(rsc): Remove. This is only temporary,
// for the mark and sweep collector.
void
starttheworld(void)
{
	lock(&sched);
	sched.mcpumax = sched.gomaxprocs;
	matchmg();
	unlock(&sched);
}

// Called to start an M.
void
mstart(void)
{
	if(m->mcache == nil)
		m->mcache = allocmcache();
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
			m = malloc(sizeof(M));
			m->g0 = malg(8192);
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
				exit(0);
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
		printf("m%d run g%d at %p\n", m->id, gp->goid, gp->sched.PC);
		traceback(gp->sched.PC, gp->sched.SP+8, gp);
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
gosched(void)
{
	if(g == m->g0)
		throw("gosched of g0");
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
	g->status = Gsyscall;
	sched.mcpu--;
	sched.msyscall++;
	if(sched.gwait != 0)
		matchmg();
	if(sched.waitstop && sched.mcpu <= sched.mcpumax) {
		sched.waitstop = 0;
		notewakeup(&sched.stopped);
	}
	unlock(&sched);
	// leave SP around for gc and traceback
	gosave(&g->sched);
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
	g->status = Grunning;
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
	gosched();
}

/*
 * stack layout parameters.
 * known to linkers.
 *
 * g->stackguard is set to point StackGuard bytes
 * above the bottom of the stack.  each function
 * compares its stack pointer against g->stackguard
 * to check for overflow.  to cut one instruction from
 * the check sequence for functions with tiny frames,
 * the stack is allowed to protrude StackSmall bytes
 * below the stack guard.  functions with large frames
 * don't bother with the check and always call morestack.
 * the sequences are:
 *
 *	guard = g->stackguard
 *	frame = function's stack frame size
 *	argsize = size of function arguments (call + return)
 *
 *	stack frame size <= StackSmall:
 *		CMPQ guard, SP
 *		JHI 3(PC)
 *		MOVQ m->morearg, $(argsize << 32)
 *		CALL sys.morestack(SB)
 *
 *	stack frame size > StackSmall but < StackBig
 *		LEAQ (frame-StackSmall)(SP), R0
 *		CMPQ guard, R0
 *		JHI 3(PC)
 *		MOVQ m->morearg, $(argsize << 32)
 *		CALL sys.morestack(SB)
 *
 *	stack frame size >= StackBig:
 *		MOVQ m->morearg, $((argsize << 32) | frame)
 *		CALL sys.morestack(SB)
 *
 * the bottom StackGuard - StackSmall bytes are important:
 * there has to be enough room to execute functions that
 * refuse to check for stack overflow, either because they
 * need to be adjacent to the actual caller's frame (sys.deferproc)
 * or because they handle the imminent stack overflow (sys.morestack).
 *
 * for example, sys.deferproc might call malloc,
 * which does one of the above checks (without allocating a full frame),
 * which might trigger a call to sys.morestack.
 * this sequence needs to fit in the bottom section of the stack.
 * on amd64, sys.morestack's frame is 40 bytes, and
 * sys.deferproc's frame is 56 bytes.  that fits well within
 * the StackGuard - StackSmall = 128 bytes at the bottom.
 * there may be other sequences lurking or yet to be written
 * that require more stack.  sys.morestack checks to make sure
 * the stack has not completely overflowed and should
 * catch such sequences.
 */
enum
{
	// byte offset of stack guard (g->stackguard) above bottom of stack.
	StackGuard = 256,

	// checked frames are allowed to protrude below the guard by
	// this many bytes.  this saves an instruction in the checking
	// sequence when the stack frame is tiny.
	StackSmall = 128,

	// extra space in the frame (beyond the function for which
	// the frame is allocated) is assumed not to be much bigger
	// than this amount.  it may not be used efficiently if it is.
	StackBig = 4096,
};

void
oldstack(void)
{
	Stktop *top;
	uint32 args;
	byte *sp;
	uintptr oldsp, oldpc, oldbase, oldguard;

// printf("oldstack m->cret=%p\n", m->cret);

	top = (Stktop*)m->curg->stackbase;

	args = (top->magic>>32) & 0xffffLL;

	sp = (byte*)top;
	if(args > 0) {
		args = (args+7) & ~7;
		sp -= args;
		mcpy(top->oldsp+2*sizeof(uintptr), sp, args);
	}

	oldsp = (uintptr)top->oldsp + sizeof(uintptr);
	oldpc = *(uintptr*)oldsp;
	oldbase = (uintptr)top->oldbase;
	oldguard = (uintptr)top->oldguard;

	stackfree((byte*)m->curg->stackguard - StackGuard);

	m->curg->stackbase = (byte*)oldbase;
	m->curg->stackguard = (byte*)oldguard;
	m->morestack.SP = (byte*)oldsp;
	m->morestack.PC = (byte*)oldpc;

	// These two lines must happen in sequence;
	// once g has been changed, must switch to g's stack
	// before calling any non-assembly functions.
	// TODO(rsc): Perhaps make the new g a parameter
	// to gogoret and setspgoto, so that g is never
	// explicitly assigned to without also setting
	// the stack pointer.
	g = m->curg;
	gogoret(&m->morestack, m->cret);
}

#pragma textflag 7
void
lessstack(void)
{
	g = m->g0;
	setspgoto(m->sched.SP, oldstack, nil);
}

void
newstack(void)
{
	int32 frame, args;
	Stktop *top;
	byte *stk, *sp;
	void (*fn)(void);

	frame = m->morearg & 0xffffffffLL;
	args = (m->morearg>>32) & 0xffffLL;

// printf("newstack frame=%d args=%d moresp=%p morepc=%p\n", frame, args, m->moresp, *(uintptr*)m->moresp);

	if(frame < StackBig)
		frame = StackBig;
	frame += 1024;	// for more functions, Stktop.
	stk = stackalloc(frame);

	top = (Stktop*)(stk+frame-sizeof(*top));

	top->oldbase = m->curg->stackbase;
	top->oldguard = m->curg->stackguard;
	top->oldsp = m->moresp;
	top->magic = m->morearg;

	m->curg->stackbase = (byte*)top;
	m->curg->stackguard = stk + StackGuard;

	sp = (byte*)top;

	if(args > 0) {
		// Copy args.  There have been two function calls
		// since they got pushed, so skip over those return
		// addresses.
		args = (args+7) & ~7;
		sp -= args;
		mcpy(sp, m->moresp+2*sizeof(uintptr), args);
	}

	g = m->curg;

	// sys.morestack's return address
	fn = (void(*)(void))(*(uintptr*)m->moresp);

// printf("fn=%p\n", fn);

	setspgoto(sp, fn, retfromnewstack);

	*(int32*)345 = 123;	// never return
}

#pragma textflag 7
void
sys·morestack(uintptr u)
{
	while(g == m->g0) {
		// very bad news
		*(int32*)0x1001 = 123;
	}

	// Morestack's frame is about 0x30 bytes on amd64.
	// If that the frame ends below the stack bottom, we've already
	// overflowed.  Stop right now.
	while((byte*)&u - 0x30 < m->curg->stackguard - StackGuard) {
		// very bad news
		*(int32*)0x1002 = 123;
	}

	g = m->g0;
	m->moresp = (byte*)(&u-1);
	setspgoto(m->sched.SP, newstack, nil);

	*(int32*)0x1003 = 123;	// never return
}

G*
malg(int32 stacksize)
{
	G *g;
	byte *stk;

	g = malloc(sizeof(G));
	stk = stackalloc(stacksize + StackGuard);
	g->stack0 = stk;
	g->stackguard = stk + StackGuard;
	g->stackbase = stk + StackGuard + stacksize;
	return g;
}

/*
 * Newproc and deferproc need to be textflag 7
 * (no possible stack split when nearing overflow)
 * because they assume that the arguments to fn
 * are available sequentially beginning at &arg0.
 * If a stack split happened, only the one word
 * arg0 would be copied.  It's okay if any functions
 * they call split the stack below the newproc frame.
 */
#pragma textflag 7
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
	} else {
		newg = malg(4096);
		newg->status = Gwaiting;
		newg->alllink = allg;
		allg = newg;
	}
	stk = newg->stack0;

	newg->stackguard = stk+StackGuard;

	sp = stk + 4096 - 4*8;
	newg->stackbase = sp;

	sp -= siz;
	mcpy(sp, (byte*)&arg0, siz);

	sp -= sizeof(uintptr);
	*(byte**)sp = (byte*)goexit;

	sp -= sizeof(uintptr);	// retpc used by gogo
	newg->sched.SP = sp;
	newg->sched.PC = fn;

	sched.gcount++;
	goidgen++;
	newg->goid = goidgen;

	readylocked(newg);
	unlock(&sched);

//printf(" goid=%d\n", newg->goid);
}

#pragma textflag 7
void
sys·deferproc(int32 siz, byte* fn, byte* arg0)
{
	Defer *d;

	d = malloc(sizeof(*d) + siz - sizeof(d->args));
	d->fn = fn;
	d->sp = (byte*)&arg0;
	d->siz = siz;
	mcpy(d->args, d->sp, d->siz);

	d->link = g->defer;
	g->defer = d;
}

#pragma textflag 7
void
sys·deferreturn(uintptr arg0)
{
	Defer *d;
	byte *sp, *fn;
	uintptr *caller;

	d = g->defer;
	if(d == nil)
		return;
	sp = (byte*)&arg0;
	if(d->sp != sp)
		return;
	mcpy(d->sp, d->args, d->siz);
	g->defer = d->link;
	fn = d->fn;
	free(d);
	jmpdefer(fn, sp);
  }

void
runtime·Breakpoint(void)
{
	breakpoint();
}

void
runtime·Goexit(void)
{
	goexit();
}

void
runtime·Gosched(void)
{
	gosched();
}

