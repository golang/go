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
static M* mget(G*);
static void gfput(G*);	// put/get on gfree
static G* gfget(void);
static void matchmg(void);	// match ms to gs
static void readylocked(G*);	// ready, but sched is locked
static void mnextg(M*, G*);

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
		traceback(g->sched.pc, g->sched.sp, g);
	}
}

// Put on `g' queue.  Sched must be locked.
static void
gput(G *g)
{
	M *m;

	// If g is wired, hand it off directly.
	if((m = g->lockedm) != nil) {
		mnextg(m, g);
		return;
	}

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

// Get an `m' to run `g'.  Sched must be locked.
static M*
mget(G *g)
{
	M *m;

	// if g has its own m, use it.
	if((m = g->lockedm) != nil)
		return m;

	// otherwise use general m pool.
	if((m = sched.mhead) != nil){
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

// Pass g to m for running.
static void
mnextg(M *m, G *g)
{
	sched.mcpu++;
	m->nextg = g;
	if(m->waitnextg) {
		m->waitnextg = 0;
		notewakeup(&m->havenextg);
	}
}

// Get the next goroutine that m should run.
// Sched must be locked on entry, is unlocked on exit.
// Makes sure that at most $GOMAXPROCS gs are
// running on cpus (not in system calls) at any given time.
static G*
nextgandunlock(void)
{
	G *gp;

	if(sched.mcpu < 0)
		throw("negative sched.mcpu");

	// If there is a g waiting as m->nextg,
	// mnextg took care of the sched.mcpu++.
	if(m->nextg != nil) {
		gp = m->nextg;
		m->nextg = nil;
		unlock(&sched);
		return gp;
	}

	if(m->lockedg != nil) {
		// We can only run one g, and it's not available.
		// Make sure some other cpu is running to handle
		// the ordinary run queue.
		if(sched.gwait != 0)
			matchmg();
	} else {
		// Look for work on global queue.
		while(sched.mcpu < sched.mcpumax && (gp=gget()) != nil) {
			if(gp->lockedm) {
				mnextg(gp->lockedm, gp);
				continue;
			}
			sched.mcpu++;		// this m will run gp
			unlock(&sched);
			return gp;
		}
		// Otherwise, wait on global m queue.
		mput(m);
	}
	if(sched.mcpu == 0 && sched.msyscall == 0)
		throw("all goroutines are asleep - deadlock!");
	m->nextg = nil;
	m->waitnextg = 1;
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

	while(sched.mcpu < sched.mcpumax && (g = gget()) != nil){
		// Find the m that will run g.
		if((m = mget(g)) == nil){
			m = malloc(sizeof(M));
			m->g0 = malg(8192);
			m->id = sched.mcount++;
			newosproc(m, m->g0, m->g0->stackbase, mstart);
		}
		mnextg(m, g);
	}
}

// Scheduler loop: find g to run, run it, repeat.
static void
scheduler(void)
{
	G* gp;

	lock(&sched);
	if(gosave(&m->sched) != 0){
		// Jumped here via gosave/gogo, so didn't
		// execute lock(&sched) above.
		lock(&sched);

		if(sched.predawn)
			throw("init sleeping");

		// Just finished running m->curg.
		gp = m->curg;
		gp->m = nil;
		sched.mcpu--;

		if(sched.mcpu < 0)
			throw("sched.mcpu < 0 in scheduler");
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
			if(gp->lockedm) {
				gp->lockedm = nil;
				m->lockedg = nil;
			}
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
	m->curg = gp;
	gp->m = m;
	if(gp->sched.pc == (byte*)goexit)	// kickoff
		gogocall(&gp->sched, (void(*)(void))gp->entry);
	gogo(&gp->sched, 1);
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
	if(gosave(&g->sched) == 0)
		gogo(&m->sched, 1);
}

// The goroutine g is about to enter a system call.
// Record that it's not using the cpu anymore.
// This is called only from the go syscall library, not
// from the low-level system calls used by the runtime.
// The "arguments" are syscall.Syscall's stack frame
void
sys·entersyscall(uint64 callerpc, int64 trap)
{
	USED(callerpc, trap);

	lock(&sched);
	g->status = Gsyscall;
	// Leave SP around for gc and traceback.
	// Do before notewakeup so that gc
	// never sees Gsyscall with wrong stack.
	gosave(&g->sched);
	sched.mcpu--;
	sched.msyscall++;
	if(sched.gwait != 0)
		matchmg();
	if(sched.waitstop && sched.mcpu <= sched.mcpumax) {
		sched.waitstop = 0;
		notewakeup(&sched.stopped);
	}
	unlock(&sched);
}

// The goroutine g exited its system call.
// Arrange for it to run on a cpu again.
// This is called only from the go syscall library, not
// from the low-level system calls used by the runtime.
void
sys·exitsyscall(void)
{
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
	// When the scheduler takes g away from m,
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
	Stktop *top, old;
	uint32 args;
	byte *sp;
	G *g1;

//printf("oldstack m->cret=%p\n", m->cret);

	g1 = m->curg;
	top = (Stktop*)g1->stackbase;
	sp = (byte*)top;
	old = *top;
	args = old.args;
	if(args > 0) {
		sp -= args;
		mcpy(top->fp, sp, args);
	}

	stackfree((byte*)g1->stackguard - StackGuard);
	g1->stackbase = old.stackbase;
	g1->stackguard = old.stackguard;

	gogo(&old.gobuf, m->cret);
}

void
newstack(void)
{
	int32 frame, args;
	Stktop *top;
	byte *stk, *sp;
	G *g1;
	Gobuf label;

	frame = m->moreframe;
	args = m->moreargs;

	// Round up to align things nicely.
	// This is sufficient for both 32- and 64-bit machines.
	args = (args+7) & ~7;

	if(frame < StackBig)
		frame = StackBig;
	frame += 1024;	// for more functions, Stktop.
	stk = stackalloc(frame);

//printf("newstack frame=%d args=%d morepc=%p morefp=%p gobuf=%p, %p newstk=%p\n", frame, args, m->morepc, m->morefp, g->sched.pc, g->sched.sp, stk);

	g1 = m->curg;
	top = (Stktop*)(stk+frame-sizeof(*top));
	top->stackbase = g1->stackbase;
	top->stackguard = g1->stackguard;
	top->gobuf = m->morebuf;
	top->fp = m->morefp;
	top->args = args;

	g1->stackbase = (byte*)top;
	g1->stackguard = stk + StackGuard;

	sp = (byte*)top;
	if(args > 0) {
		sp -= args;
		mcpy(sp, m->morefp, args);
	}

	// Continue as if lessstack had just called m->morepc
	// (the PC that decided to grow the stack).
	label.sp = sp;
	label.pc = (byte*)sys·lessstack;
	label.g = m->curg;
	gogocall(&label, m->morepc);

	*(int32*)345 = 123;	// never return
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

	newg->sched.sp = sp;
	newg->sched.pc = (byte*)goexit;
	newg->sched.g = newg;
	newg->entry = fn;

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

void
runtime·LockOSThread(void)
{
	if(sched.predawn)
		throw("cannot wire during init");
	m->lockedg = g;
	g->lockedm = m;
}

void
runtime·UnlockOSThread(void)
{
	m->lockedg = nil;
	g->lockedm = nil;
}

// for testing of wire, unwire
void
runtime·mid(uint32 ret)
{
	ret = m->id;
	FLUSH(&ret);
}

