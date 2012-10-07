// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "defs_GOOS_GOARCH.h"
#include "malloc.h"
#include "os_GOOS.h"
#include "stack.h"
#include "race.h"

bool	runtime·iscgo;

static void schedule(G*);

typedef struct Sched Sched;

M	runtime·m0;
G	runtime·g0;	// idle goroutine for m0

static	int32	debug	= 0;

int32	runtime·gcwaiting;

// Go scheduler
//
// The go scheduler's job is to match ready-to-run goroutines (`g's)
// with waiting-for-work schedulers (`m's).  If there are ready g's
// and no waiting m's, ready() will start a new m running in a new
// OS thread, so that all ready g's can run simultaneously, up to a limit.
// For now, m's never go away.
//
// By default, Go keeps only one kernel thread (m) running user code
// at a single time; other threads may be blocked in the operating system.
// Setting the environment variable $GOMAXPROCS or calling
// runtime.GOMAXPROCS() will change the number of user threads
// allowed to execute simultaneously.  $GOMAXPROCS is thus an
// approximation of the maximum number of cores to use.
//
// Even a program that can run without deadlock in a single process
// might use more m's if given the chance.  For example, the prime
// sieve will use as many m's as there are primes (up to runtime·sched.mmax),
// allowing different stages of the pipeline to execute in parallel.
// We could revisit this choice, only kicking off new m's for blocking
// system calls, but that would limit the amount of parallel computation
// that go would try to do.
//
// In general, one could imagine all sorts of refinements to the
// scheduler, but the goal now is just to get something working on
// Linux and OS X.

struct Sched {
	Lock;

	G *gfree;	// available g's (status == Gdead)
	int32 goidgen;

	G *ghead;	// g's waiting to run
	G *gtail;
	int32 gwait;	// number of g's waiting to run
	int32 gcount;	// number of g's that are alive
	int32 grunning;	// number of g's running on cpu or in syscall

	M *mhead;	// m's waiting for work
	int32 mwait;	// number of m's waiting for work
	int32 mcount;	// number of m's that have been created

	volatile uint32 atomic;	// atomic scheduling word (see below)

	int32 profilehz;	// cpu profiling rate

	bool init;  // running initialization
	bool lockmain;  // init called runtime.LockOSThread

	Note	stopped;	// one g can set waitstop and wait here for m's to stop
};

// The atomic word in sched is an atomic uint32 that
// holds these fields.
//
//	[15 bits] mcpu		number of m's executing on cpu
//	[15 bits] mcpumax	max number of m's allowed on cpu
//	[1 bit] waitstop	some g is waiting on stopped
//	[1 bit] gwaiting	gwait != 0
//
// These fields are the information needed by entersyscall
// and exitsyscall to decide whether to coordinate with the
// scheduler.  Packing them into a single machine word lets
// them use a fast path with a single atomic read/write and
// no lock/unlock.  This greatly reduces contention in
// syscall- or cgo-heavy multithreaded programs.
//
// Except for entersyscall and exitsyscall, the manipulations
// to these fields only happen while holding the schedlock,
// so the routines holding schedlock only need to worry about
// what entersyscall and exitsyscall do, not the other routines
// (which also use the schedlock).
//
// In particular, entersyscall and exitsyscall only read mcpumax,
// waitstop, and gwaiting.  They never write them.  Thus, writes to those
// fields can be done (holding schedlock) without fear of write conflicts.
// There may still be logic conflicts: for example, the set of waitstop must
// be conditioned on mcpu >= mcpumax or else the wait may be a
// spurious sleep.  The Promela model in proc.p verifies these accesses.
enum {
	mcpuWidth = 15,
	mcpuMask = (1<<mcpuWidth) - 1,
	mcpuShift = 0,
	mcpumaxShift = mcpuShift + mcpuWidth,
	waitstopShift = mcpumaxShift + mcpuWidth,
	gwaitingShift = waitstopShift+1,

	// The max value of GOMAXPROCS is constrained
	// by the max value we can store in the bit fields
	// of the atomic word.  Reserve a few high values
	// so that we can detect accidental decrement
	// beyond zero.
	maxgomaxprocs = mcpuMask - 10,
};

#define atomic_mcpu(v)		(((v)>>mcpuShift)&mcpuMask)
#define atomic_mcpumax(v)	(((v)>>mcpumaxShift)&mcpuMask)
#define atomic_waitstop(v)	(((v)>>waitstopShift)&1)
#define atomic_gwaiting(v)	(((v)>>gwaitingShift)&1)

Sched runtime·sched;
int32 runtime·gomaxprocs;
bool runtime·singleproc;

static bool canaddmcpu(void);

// An m that is waiting for notewakeup(&m->havenextg).  This may
// only be accessed while the scheduler lock is held.  This is used to
// minimize the number of times we call notewakeup while the scheduler
// lock is held, since the m will normally move quickly to lock the
// scheduler itself, producing lock contention.
static M* mwakeup;

// Scheduling helpers.  Sched must be locked.
static void gput(G*);	// put/get on ghead/gtail
static G* gget(void);
static void mput(M*);	// put/get on mhead
static M* mget(G*);
static void gfput(G*);	// put/get on gfree
static G* gfget(void);
static void matchmg(void);	// match m's to g's
static void readylocked(G*);	// ready, but sched is locked
static void mnextg(M*, G*);
static void mcommoninit(M*);

void
setmcpumax(uint32 n)
{
	uint32 v, w;

	for(;;) {
		v = runtime·sched.atomic;
		w = v;
		w &= ~(mcpuMask<<mcpumaxShift);
		w |= n<<mcpumaxShift;
		if(runtime·cas(&runtime·sched.atomic, v, w))
			break;
	}
}

// Keep trace of scavenger's goroutine for deadlock detection.
static G *scvg;

// The bootstrap sequence is:
//
//	call osinit
//	call schedinit
//	make & queue new G
//	call runtime·mstart
//
// The new G calls runtime·main.
void
runtime·schedinit(void)
{
	int32 n;
	byte *p;

	m->nomemprof++;
	runtime·mallocinit();
	mcommoninit(m);

	runtime·goargs();
	runtime·goenvs();

	// For debugging:
	// Allocate internal symbol table representation now,
	// so that we don't need to call malloc when we crash.
	// runtime·findfunc(0);

	runtime·gomaxprocs = 1;
	p = runtime·getenv("GOMAXPROCS");
	if(p != nil && (n = runtime·atoi(p)) != 0) {
		if(n > maxgomaxprocs)
			n = maxgomaxprocs;
		runtime·gomaxprocs = n;
	}
	// wait for the main goroutine to start before taking
	// GOMAXPROCS into account.
	setmcpumax(1);
	runtime·singleproc = runtime·gomaxprocs == 1;

	canaddmcpu();	// mcpu++ to account for bootstrap m
	m->helpgc = 1;	// flag to tell schedule() to mcpu--
	runtime·sched.grunning++;

	mstats.enablegc = 1;
	m->nomemprof--;

	if(raceenabled)
		runtime·raceinit();
}

extern void main·init(void);
extern void main·main(void);

// The main goroutine.
void
runtime·main(void)
{
	// Lock the main goroutine onto this, the main OS thread,
	// during initialization.  Most programs won't care, but a few
	// do require certain calls to be made by the main thread.
	// Those can arrange for main.main to run in the main thread
	// by calling runtime.LockOSThread during initialization
	// to preserve the lock.
	runtime·LockOSThread();
	// From now on, newgoroutines may use non-main threads.
	setmcpumax(runtime·gomaxprocs);
	runtime·sched.init = true;
	scvg = runtime·newproc1((byte*)runtime·MHeap_Scavenger, nil, 0, 0, runtime·main);
	main·init();
	runtime·sched.init = false;
	if(!runtime·sched.lockmain)
		runtime·UnlockOSThread();

	// The deadlock detection has false negatives.
	// Let scvg start up, to eliminate the false negative
	// for the trivial program func main() { select{} }.
	runtime·gosched();

	main·main();
	if(raceenabled)
		runtime·racefini();
	runtime·exit(0);
	for(;;)
		*(int32*)runtime·main = 0;
}

// Lock the scheduler.
static void
schedlock(void)
{
	runtime·lock(&runtime·sched);
}

// Unlock the scheduler.
static void
schedunlock(void)
{
	M *m;

	m = mwakeup;
	mwakeup = nil;
	runtime·unlock(&runtime·sched);
	if(m != nil)
		runtime·notewakeup(&m->havenextg);
}

void
runtime·goexit(void)
{
	g->status = Gmoribund;
	runtime·gosched();
}

void
runtime·goroutineheader(G *gp)
{
	int8 *status;

	switch(gp->status) {
	case Gidle:
		status = "idle";
		break;
	case Grunnable:
		status = "runnable";
		break;
	case Grunning:
		status = "running";
		break;
	case Gsyscall:
		status = "syscall";
		break;
	case Gwaiting:
		if(gp->waitreason)
			status = gp->waitreason;
		else
			status = "waiting";
		break;
	case Gmoribund:
		status = "moribund";
		break;
	default:
		status = "???";
		break;
	}
	runtime·printf("goroutine %d [%s]:\n", gp->goid, status);
}

void
runtime·tracebackothers(G *me)
{
	G *gp;

	for(gp = runtime·allg; gp != nil; gp = gp->alllink) {
		if(gp == me || gp->status == Gdead)
			continue;
		runtime·printf("\n");
		runtime·goroutineheader(gp);
		runtime·traceback(gp->sched.pc, (byte*)gp->sched.sp, 0, gp);
	}
}

// Mark this g as m's idle goroutine.
// This functionality might be used in environments where programs
// are limited to a single thread, to simulate a select-driven
// network server.  It is not exposed via the standard runtime API.
void
runtime·idlegoroutine(void)
{
	if(g->idlem != nil)
		runtime·throw("g is already an idle goroutine");
	g->idlem = m;
}

static void
mcommoninit(M *mp)
{
	mp->id = runtime·sched.mcount++;
	mp->fastrand = 0x49f6428aUL + mp->id + runtime·cputicks();
	mp->stackalloc = runtime·malloc(sizeof(*mp->stackalloc));
	runtime·FixAlloc_Init(mp->stackalloc, FixedStack, runtime·SysAlloc, nil, nil);

	if(mp->mcache == nil)
		mp->mcache = runtime·allocmcache();

	runtime·callers(1, mp->createstack, nelem(mp->createstack));

	// Add to runtime·allm so garbage collector doesn't free m
	// when it is just in a register or thread-local storage.
	mp->alllink = runtime·allm;
	// runtime·NumCgoCall() iterates over allm w/o schedlock,
	// so we need to publish it safely.
	runtime·atomicstorep(&runtime·allm, mp);
}

// Try to increment mcpu.  Report whether succeeded.
static bool
canaddmcpu(void)
{
	uint32 v;

	for(;;) {
		v = runtime·sched.atomic;
		if(atomic_mcpu(v) >= atomic_mcpumax(v))
			return 0;
		if(runtime·cas(&runtime·sched.atomic, v, v+(1<<mcpuShift)))
			return 1;
	}
}

// Put on `g' queue.  Sched must be locked.
static void
gput(G *gp)
{
	M *mp;

	// If g is wired, hand it off directly.
	if((mp = gp->lockedm) != nil && canaddmcpu()) {
		mnextg(mp, gp);
		return;
	}

	// If g is the idle goroutine for an m, hand it off.
	if(gp->idlem != nil) {
		if(gp->idlem->idleg != nil) {
			runtime·printf("m%d idle out of sync: g%d g%d\n",
				gp->idlem->id,
				gp->idlem->idleg->goid, gp->goid);
			runtime·throw("runtime: double idle");
		}
		gp->idlem->idleg = gp;
		return;
	}

	gp->schedlink = nil;
	if(runtime·sched.ghead == nil)
		runtime·sched.ghead = gp;
	else
		runtime·sched.gtail->schedlink = gp;
	runtime·sched.gtail = gp;

	// increment gwait.
	// if it transitions to nonzero, set atomic gwaiting bit.
	if(runtime·sched.gwait++ == 0)
		runtime·xadd(&runtime·sched.atomic, 1<<gwaitingShift);
}

// Report whether gget would return something.
static bool
haveg(void)
{
	return runtime·sched.ghead != nil || m->idleg != nil;
}

// Get from `g' queue.  Sched must be locked.
static G*
gget(void)
{
	G *gp;

	gp = runtime·sched.ghead;
	if(gp) {
		runtime·sched.ghead = gp->schedlink;
		if(runtime·sched.ghead == nil)
			runtime·sched.gtail = nil;
		// decrement gwait.
		// if it transitions to zero, clear atomic gwaiting bit.
		if(--runtime·sched.gwait == 0)
			runtime·xadd(&runtime·sched.atomic, -1<<gwaitingShift);
	} else if(m->idleg != nil) {
		gp = m->idleg;
		m->idleg = nil;
	}
	return gp;
}

// Put on `m' list.  Sched must be locked.
static void
mput(M *mp)
{
	mp->schedlink = runtime·sched.mhead;
	runtime·sched.mhead = mp;
	runtime·sched.mwait++;
}

// Get an `m' to run `g'.  Sched must be locked.
static M*
mget(G *gp)
{
	M *mp;

	// if g has its own m, use it.
	if(gp && (mp = gp->lockedm) != nil)
		return mp;

	// otherwise use general m pool.
	if((mp = runtime·sched.mhead) != nil) {
		runtime·sched.mhead = mp->schedlink;
		runtime·sched.mwait--;
	}
	return mp;
}

// Mark g ready to run.
void
runtime·ready(G *gp)
{
	schedlock();
	readylocked(gp);
	schedunlock();
}

// Mark g ready to run.  Sched is already locked.
// G might be running already and about to stop.
// The sched lock protects g->status from changing underfoot.
static void
readylocked(G *gp)
{
	if(gp->m) {
		// Running on another machine.
		// Ready it when it stops.
		gp->readyonstop = 1;
		return;
	}

	// Mark runnable.
	if(gp->status == Grunnable || gp->status == Grunning) {
		runtime·printf("goroutine %d has status %d\n", gp->goid, gp->status);
		runtime·throw("bad g->status in ready");
	}
	gp->status = Grunnable;

	gput(gp);
	matchmg();
}

static void
nop(void)
{
}

// Same as readylocked but a different symbol so that
// debuggers can set a breakpoint here and catch all
// new goroutines.
static void
newprocreadylocked(G *gp)
{
	nop();	// avoid inlining in 6l
	readylocked(gp);
}

// Pass g to m for running.
// Caller has already incremented mcpu.
static void
mnextg(M *mp, G *gp)
{
	runtime·sched.grunning++;
	mp->nextg = gp;
	if(mp->waitnextg) {
		mp->waitnextg = 0;
		if(mwakeup != nil)
			runtime·notewakeup(&mwakeup->havenextg);
		mwakeup = mp;
	}
}

// Get the next goroutine that m should run.
// Sched must be locked on entry, is unlocked on exit.
// Makes sure that at most $GOMAXPROCS g's are
// running on cpus (not in system calls) at any given time.
static G*
nextgandunlock(void)
{
	G *gp;
	uint32 v;

top:
	if(atomic_mcpu(runtime·sched.atomic) >= maxgomaxprocs)
		runtime·throw("negative mcpu");

	// If there is a g waiting as m->nextg, the mcpu++
	// happened before it was passed to mnextg.
	if(m->nextg != nil) {
		gp = m->nextg;
		m->nextg = nil;
		schedunlock();
		return gp;
	}

	if(m->lockedg != nil) {
		// We can only run one g, and it's not available.
		// Make sure some other cpu is running to handle
		// the ordinary run queue.
		if(runtime·sched.gwait != 0) {
			matchmg();
			// m->lockedg might have been on the queue.
			if(m->nextg != nil) {
				gp = m->nextg;
				m->nextg = nil;
				schedunlock();
				return gp;
			}
		}
	} else {
		// Look for work on global queue.
		while(haveg() && canaddmcpu()) {
			gp = gget();
			if(gp == nil)
				runtime·throw("gget inconsistency");

			if(gp->lockedm) {
				mnextg(gp->lockedm, gp);
				continue;
			}
			runtime·sched.grunning++;
			schedunlock();
			return gp;
		}

		// The while loop ended either because the g queue is empty
		// or because we have maxed out our m procs running go
		// code (mcpu >= mcpumax).  We need to check that
		// concurrent actions by entersyscall/exitsyscall cannot
		// invalidate the decision to end the loop.
		//
		// We hold the sched lock, so no one else is manipulating the
		// g queue or changing mcpumax.  Entersyscall can decrement
		// mcpu, but if does so when there is something on the g queue,
		// the gwait bit will be set, so entersyscall will take the slow path
		// and use the sched lock.  So it cannot invalidate our decision.
		//
		// Wait on global m queue.
		mput(m);
	}

	// Look for deadlock situation.
	// There is a race with the scavenger that causes false negatives:
	// if the scavenger is just starting, then we have
	//	scvg != nil && grunning == 0 && gwait == 0
	// and we do not detect a deadlock.  It is possible that we should
	// add that case to the if statement here, but it is too close to Go 1
	// to make such a subtle change.  Instead, we work around the
	// false negative in trivial programs by calling runtime.gosched
	// from the main goroutine just before main.main.
	// See runtime·main above.
	//
	// On a related note, it is also possible that the scvg == nil case is
	// wrong and should include gwait, but that does not happen in
	// standard Go programs, which all start the scavenger.
	//
	if((scvg == nil && runtime·sched.grunning == 0) ||
	   (scvg != nil && runtime·sched.grunning == 1 && runtime·sched.gwait == 0 &&
	    (scvg->status == Grunning || scvg->status == Gsyscall))) {
		runtime·throw("all goroutines are asleep - deadlock!");
	}

	m->nextg = nil;
	m->waitnextg = 1;
	runtime·noteclear(&m->havenextg);

	// Stoptheworld is waiting for all but its cpu to go to stop.
	// Entersyscall might have decremented mcpu too, but if so
	// it will see the waitstop and take the slow path.
	// Exitsyscall never increments mcpu beyond mcpumax.
	v = runtime·atomicload(&runtime·sched.atomic);
	if(atomic_waitstop(v) && atomic_mcpu(v) <= atomic_mcpumax(v)) {
		// set waitstop = 0 (known to be 1)
		runtime·xadd(&runtime·sched.atomic, -1<<waitstopShift);
		runtime·notewakeup(&runtime·sched.stopped);
	}
	schedunlock();

	runtime·notesleep(&m->havenextg);
	if(m->helpgc) {
		runtime·gchelper();
		m->helpgc = 0;
		runtime·lock(&runtime·sched);
		goto top;
	}
	if((gp = m->nextg) == nil)
		runtime·throw("bad m->nextg in nextgoroutine");
	m->nextg = nil;
	return gp;
}

int32
runtime·gcprocs(void)
{
	int32 n;
	
	// Figure out how many CPUs to use during GC.
	// Limited by gomaxprocs, number of actual CPUs, and MaxGcproc.
	n = runtime·gomaxprocs;
	if(n > runtime·ncpu)
		n = runtime·ncpu;
	if(n > MaxGcproc)
		n = MaxGcproc;
	if(n > runtime·sched.mwait+1) // one M is currently running
		n = runtime·sched.mwait+1;
	return n;
}

void
runtime·helpgc(int32 nproc)
{
	M *mp;
	int32 n;

	runtime·lock(&runtime·sched);
	for(n = 1; n < nproc; n++) { // one M is currently running
		mp = mget(nil);
		if(mp == nil)
			runtime·throw("runtime·gcprocs inconsistency");
		mp->helpgc = 1;
		mp->waitnextg = 0;
		runtime·notewakeup(&mp->havenextg);
	}
	runtime·unlock(&runtime·sched);
}

void
runtime·stoptheworld(void)
{
	uint32 v;

	schedlock();
	runtime·gcwaiting = 1;

	setmcpumax(1);

	// while mcpu > 1
	for(;;) {
		v = runtime·sched.atomic;
		if(atomic_mcpu(v) <= 1)
			break;

		// It would be unsafe for multiple threads to be using
		// the stopped note at once, but there is only
		// ever one thread doing garbage collection.
		runtime·noteclear(&runtime·sched.stopped);
		if(atomic_waitstop(v))
			runtime·throw("invalid waitstop");

		// atomic { waitstop = 1 }, predicated on mcpu <= 1 check above
		// still being true.
		if(!runtime·cas(&runtime·sched.atomic, v, v+(1<<waitstopShift)))
			continue;

		schedunlock();
		runtime·notesleep(&runtime·sched.stopped);
		schedlock();
	}
	runtime·singleproc = runtime·gomaxprocs == 1;
	schedunlock();
}

void
runtime·starttheworld(void)
{
	M *mp;
	int32 max;
	
	// Figure out how many CPUs GC could possibly use.
	max = runtime·gomaxprocs;
	if(max > runtime·ncpu)
		max = runtime·ncpu;
	if(max > MaxGcproc)
		max = MaxGcproc;

	schedlock();
	runtime·gcwaiting = 0;
	setmcpumax(runtime·gomaxprocs);
	matchmg();
	if(runtime·gcprocs() < max && canaddmcpu()) {
		// If GC could have used another helper proc, start one now,
		// in the hope that it will be available next time.
		// It would have been even better to start it before the collection,
		// but doing so requires allocating memory, so it's tricky to
		// coordinate.  This lazy approach works out in practice:
		// we don't mind if the first couple gc rounds don't have quite
		// the maximum number of procs.
		// canaddmcpu above did mcpu++
		// (necessary, because m will be doing various
		// initialization work so is definitely running),
		// but m is not running a specific goroutine,
		// so set the helpgc flag as a signal to m's
		// first schedule(nil) to mcpu-- and grunning--.
		mp = runtime·newm();
		mp->helpgc = 1;
		runtime·sched.grunning++;
	}
	schedunlock();
}

// Called to start an M.
void
runtime·mstart(void)
{
	// It is used by windows-386 only. Unfortunately, seh needs
	// to be located on os stack, and mstart runs on os stack
	// for both m0 and m.
	SEH seh;

	if(g != m->g0)
		runtime·throw("bad runtime·mstart");

	// Record top of stack for use by mcall.
	// Once we call schedule we're never coming back,
	// so other calls can reuse this stack space.
	runtime·gosave(&m->g0->sched);
	m->g0->sched.pc = (void*)-1;  // make sure it is never used
	m->seh = &seh;
	runtime·asminit();
	runtime·minit();

	// Install signal handlers; after minit so that minit can
	// prepare the thread to be able to handle the signals.
	if(m == &runtime·m0)
		runtime·initsig();

	schedule(nil);

	// TODO(brainman): This point is never reached, because scheduler
	// does not release os threads at the moment. But once this path
	// is enabled, we must remove our seh here.
}

// When running with cgo, we call libcgo_thread_start
// to start threads for us so that we can play nicely with
// foreign code.
void (*libcgo_thread_start)(void*);

typedef struct CgoThreadStart CgoThreadStart;
struct CgoThreadStart
{
	M *m;
	G *g;
	void (*fn)(void);
};

// Kick off new m's as needed (up to mcpumax).
// Sched is locked.
static void
matchmg(void)
{
	G *gp;
	M *mp;

	if(m->mallocing || m->gcing)
		return;

	while(haveg() && canaddmcpu()) {
		gp = gget();
		if(gp == nil)
			runtime·throw("gget inconsistency");

		// Find the m that will run gp.
		if((mp = mget(gp)) == nil)
			mp = runtime·newm();
		mnextg(mp, gp);
	}
}

// Create a new m.  It will start off with a call to runtime·mstart.
M*
runtime·newm(void)
{
	M *mp;

	mp = runtime·malloc(sizeof(M));
	mcommoninit(mp);

	if(runtime·iscgo) {
		CgoThreadStart ts;

		if(libcgo_thread_start == nil)
			runtime·throw("libcgo_thread_start missing");
		// pthread_create will make us a stack.
		mp->g0 = runtime·malg(-1);
		ts.m = mp;
		ts.g = mp->g0;
		ts.fn = runtime·mstart;
		runtime·asmcgocall(libcgo_thread_start, &ts);
	} else {
		if(Windows)
			// windows will layout sched stack on os stack
			mp->g0 = runtime·malg(-1);
		else
			mp->g0 = runtime·malg(8192);
		runtime·newosproc(mp, mp->g0, (byte*)mp->g0->stackbase, runtime·mstart);
	}

	return mp;
}

// One round of scheduler: find a goroutine and run it.
// The argument is the goroutine that was running before
// schedule was called, or nil if this is the first call.
// Never returns.
static void
schedule(G *gp)
{
	int32 hz;
	uint32 v;

	schedlock();
	if(gp != nil) {
		// Just finished running gp.
		gp->m = nil;
		runtime·sched.grunning--;

		// atomic { mcpu-- }
		v = runtime·xadd(&runtime·sched.atomic, -1<<mcpuShift);
		if(atomic_mcpu(v) > maxgomaxprocs)
			runtime·throw("negative mcpu in scheduler");

		switch(gp->status) {
		case Grunnable:
		case Gdead:
			// Shouldn't have been running!
			runtime·throw("bad gp->status in sched");
		case Grunning:
			gp->status = Grunnable;
			gput(gp);
			break;
		case Gmoribund:
			if(raceenabled)
				runtime·racegoend(gp->goid);
			gp->status = Gdead;
			if(gp->lockedm) {
				gp->lockedm = nil;
				m->lockedg = nil;
			}
			gp->idlem = nil;
			runtime·unwindstack(gp, nil);
			gfput(gp);
			if(--runtime·sched.gcount == 0)
				runtime·exit(0);
			break;
		}
		if(gp->readyonstop) {
			gp->readyonstop = 0;
			readylocked(gp);
		}
	} else if(m->helpgc) {
		// Bootstrap m or new m started by starttheworld.
		// atomic { mcpu-- }
		v = runtime·xadd(&runtime·sched.atomic, -1<<mcpuShift);
		if(atomic_mcpu(v) > maxgomaxprocs)
			runtime·throw("negative mcpu in scheduler");
		// Compensate for increment in starttheworld().
		runtime·sched.grunning--;
		m->helpgc = 0;
	} else if(m->nextg != nil) {
		// New m started by matchmg.
	} else {
		runtime·throw("invalid m state in scheduler");
	}

	// Find (or wait for) g to run.  Unlocks runtime·sched.
	gp = nextgandunlock();
	gp->readyonstop = 0;
	gp->status = Grunning;
	m->curg = gp;
	gp->m = m;

	// Check whether the profiler needs to be turned on or off.
	hz = runtime·sched.profilehz;
	if(m->profilehz != hz)
		runtime·resetcpuprofiler(hz);

	if(gp->sched.pc == (byte*)runtime·goexit) {	// kickoff
		runtime·gogocall(&gp->sched, (void(*)(void))gp->entry);
	}
	runtime·gogo(&gp->sched, 0);
}

// Enter scheduler.  If g->status is Grunning,
// re-queues g and runs everyone else who is waiting
// before running g again.  If g->status is Gmoribund,
// kills off g.
// Cannot split stack because it is called from exitsyscall.
// See comment below.
#pragma textflag 7
void
runtime·gosched(void)
{
	if(m->locks != 0)
		runtime·throw("gosched holding locks");
	if(g == m->g0)
		runtime·throw("gosched of g0");
	runtime·mcall(schedule);
}

// Puts the current goroutine into a waiting state and unlocks the lock.
// The goroutine can be made runnable again by calling runtime·ready(gp).
void
runtime·park(void (*unlockf)(Lock*), Lock *lock, int8 *reason)
{
	g->status = Gwaiting;
	g->waitreason = reason;
	if(unlockf)
		unlockf(lock);
	runtime·gosched();
}

// The goroutine g is about to enter a system call.
// Record that it's not using the cpu anymore.
// This is called only from the go syscall library and cgocall,
// not from the low-level system calls used by the runtime.
//
// Entersyscall cannot split the stack: the runtime·gosave must
// make g->sched refer to the caller's stack segment, because
// entersyscall is going to return immediately after.
// It's okay to call matchmg and notewakeup even after
// decrementing mcpu, because we haven't released the
// sched lock yet, so the garbage collector cannot be running.
#pragma textflag 7
void
runtime·entersyscall(void)
{
	uint32 v;

	if(m->profilehz > 0)
		runtime·setprof(false);

	// Leave SP around for gc and traceback.
	runtime·gosave(&g->sched);
	g->gcsp = g->sched.sp;
	g->gcstack = g->stackbase;
	g->gcguard = g->stackguard;
	g->status = Gsyscall;
	if(g->gcsp < g->gcguard-StackGuard || g->gcstack < g->gcsp) {
		// runtime·printf("entersyscall inconsistent %p [%p,%p]\n",
		//	g->gcsp, g->gcguard-StackGuard, g->gcstack);
		runtime·throw("entersyscall");
	}

	// Fast path.
	// The slow path inside the schedlock/schedunlock will get
	// through without stopping if it does:
	//	mcpu--
	//	gwait not true
	//	waitstop && mcpu <= mcpumax not true
	// If we can do the same with a single atomic add,
	// then we can skip the locks.
	v = runtime·xadd(&runtime·sched.atomic, -1<<mcpuShift);
	if(!atomic_gwaiting(v) && (!atomic_waitstop(v) || atomic_mcpu(v) > atomic_mcpumax(v)))
		return;

	schedlock();
	v = runtime·atomicload(&runtime·sched.atomic);
	if(atomic_gwaiting(v)) {
		matchmg();
		v = runtime·atomicload(&runtime·sched.atomic);
	}
	if(atomic_waitstop(v) && atomic_mcpu(v) <= atomic_mcpumax(v)) {
		runtime·xadd(&runtime·sched.atomic, -1<<waitstopShift);
		runtime·notewakeup(&runtime·sched.stopped);
	}

	// Re-save sched in case one of the calls
	// (notewakeup, matchmg) triggered something using it.
	runtime·gosave(&g->sched);

	schedunlock();
}

// The goroutine g exited its system call.
// Arrange for it to run on a cpu again.
// This is called only from the go syscall library, not
// from the low-level system calls used by the runtime.
void
runtime·exitsyscall(void)
{
	uint32 v;

	// Fast path.
	// If we can do the mcpu++ bookkeeping and
	// find that we still have mcpu <= mcpumax, then we can
	// start executing Go code immediately, without having to
	// schedlock/schedunlock.
	v = runtime·xadd(&runtime·sched.atomic, (1<<mcpuShift));
	if(m->profilehz == runtime·sched.profilehz && atomic_mcpu(v) <= atomic_mcpumax(v)) {
		// There's a cpu for us, so we can run.
		g->status = Grunning;
		// Garbage collector isn't running (since we are),
		// so okay to clear gcstack.
		g->gcstack = (uintptr)nil;

		if(m->profilehz > 0)
			runtime·setprof(true);
		return;
	}

	// Tell scheduler to put g back on the run queue:
	// mostly equivalent to g->status = Grunning,
	// but keeps the garbage collector from thinking
	// that g is running right now, which it's not.
	g->readyonstop = 1;

	// All the cpus are taken.
	// The scheduler will ready g and put this m to sleep.
	// When the scheduler takes g away from m,
	// it will undo the runtime·sched.mcpu++ above.
	runtime·gosched();

	// Gosched returned, so we're allowed to run now.
	// Delete the gcstack information that we left for
	// the garbage collector during the system call.
	// Must wait until now because until gosched returns
	// we don't know for sure that the garbage collector
	// is not running.
	g->gcstack = (uintptr)nil;
}

// Called from runtime·lessstack when returning from a function which
// allocated a new stack segment.  The function's return value is in
// m->cret.
void
runtime·oldstack(void)
{
	Stktop *top, old;
	uint32 argsize;
	uintptr cret;
	byte *sp;
	G *g1;
	int32 goid;

//printf("oldstack m->cret=%p\n", m->cret);

	g1 = m->curg;
	top = (Stktop*)g1->stackbase;
	sp = (byte*)top;
	old = *top;
	argsize = old.argsize;
	if(argsize > 0) {
		sp -= argsize;
		runtime·memmove(top->argp, sp, argsize);
	}
	goid = old.gobuf.g->goid;	// fault if g is bad, before gogo
	USED(goid);

	if(old.free != 0)
		runtime·stackfree((byte*)g1->stackguard - StackGuard, old.free);
	g1->stackbase = (uintptr)old.stackbase;
	g1->stackguard = (uintptr)old.stackguard;

	cret = m->cret;
	m->cret = 0;  // drop reference
	runtime·gogo(&old.gobuf, cret);
}

// Called from reflect·call or from runtime·morestack when a new
// stack segment is needed.  Allocate a new stack big enough for
// m->moreframesize bytes, copy m->moreargsize bytes to the new frame,
// and then act as though runtime·lessstack called the function at
// m->morepc.
void
runtime·newstack(void)
{
	int32 framesize, argsize;
	Stktop *top;
	byte *stk, *sp;
	G *g1;
	Gobuf label;
	bool reflectcall;
	uintptr free;

	framesize = m->moreframesize;
	argsize = m->moreargsize;
	g1 = m->curg;

	if(m->morebuf.sp < g1->stackguard - StackGuard) {
		runtime·printf("runtime: split stack overflow: %p < %p\n", m->morebuf.sp, g1->stackguard - StackGuard);
		runtime·throw("runtime: split stack overflow");
	}
	if(argsize % sizeof(uintptr) != 0) {
		runtime·printf("runtime: stack split with misaligned argsize %d\n", argsize);
		runtime·throw("runtime: stack split argsize");
	}

	reflectcall = framesize==1;
	if(reflectcall)
		framesize = 0;

	if(reflectcall && m->morebuf.sp - sizeof(Stktop) - argsize - 32 > g1->stackguard) {
		// special case: called from reflect.call (framesize==1)
		// to call code with an arbitrary argument size,
		// and we have enough space on the current stack.
		// the new Stktop* is necessary to unwind, but
		// we don't need to create a new segment.
		top = (Stktop*)(m->morebuf.sp - sizeof(*top));
		stk = (byte*)g1->stackguard - StackGuard;
		free = 0;
	} else {
		// allocate new segment.
		framesize += argsize;
		framesize += StackExtra;	// room for more functions, Stktop.
		if(framesize < StackMin)
			framesize = StackMin;
		framesize += StackSystem;
		stk = runtime·stackalloc(framesize);
		top = (Stktop*)(stk+framesize-sizeof(*top));
		free = framesize;
	}

//runtime·printf("newstack framesize=%d argsize=%d morepc=%p moreargp=%p gobuf=%p, %p top=%p old=%p\n",
//framesize, argsize, m->morepc, m->moreargp, m->morebuf.pc, m->morebuf.sp, top, g1->stackbase);

	top->stackbase = (byte*)g1->stackbase;
	top->stackguard = (byte*)g1->stackguard;
	top->gobuf = m->morebuf;
	top->argp = m->moreargp;
	top->argsize = argsize;
	top->free = free;
	m->moreargp = nil;
	m->morebuf.pc = nil;
	m->morebuf.sp = (uintptr)nil;

	// copy flag from panic
	top->panic = g1->ispanic;
	g1->ispanic = false;

	g1->stackbase = (uintptr)top;
	g1->stackguard = (uintptr)stk + StackGuard;

	sp = (byte*)top;
	if(argsize > 0) {
		sp -= argsize;
		runtime·memmove(sp, top->argp, argsize);
	}
	if(thechar == '5') {
		// caller would have saved its LR below args.
		sp -= sizeof(void*);
		*(void**)sp = nil;
	}

	// Continue as if lessstack had just called m->morepc
	// (the PC that decided to grow the stack).
	label.sp = (uintptr)sp;
	label.pc = (byte*)runtime·lessstack;
	label.g = m->curg;
	runtime·gogocall(&label, m->morepc);

	*(int32*)345 = 123;	// never return
}

// Hook used by runtime·malg to call runtime·stackalloc on the
// scheduler stack.  This exists because runtime·stackalloc insists
// on being called on the scheduler stack, to avoid trying to grow
// the stack while allocating a new stack segment.
static void
mstackalloc(G *gp)
{
	gp->param = runtime·stackalloc((uintptr)gp->param);
	runtime·gogo(&gp->sched, 0);
}

// Allocate a new g, with a stack big enough for stacksize bytes.
G*
runtime·malg(int32 stacksize)
{
	G *newg;
	byte *stk;

	if(StackTop < sizeof(Stktop)) {
		runtime·printf("runtime: SizeofStktop=%d, should be >=%d\n", (int32)StackTop, (int32)sizeof(Stktop));
		runtime·throw("runtime: bad stack.h");
	}

	newg = runtime·malloc(sizeof(G));
	if(stacksize >= 0) {
		if(g == m->g0) {
			// running on scheduler stack already.
			stk = runtime·stackalloc(StackSystem + stacksize);
		} else {
			// have to call stackalloc on scheduler stack.
			g->param = (void*)(StackSystem + stacksize);
			runtime·mcall(mstackalloc);
			stk = g->param;
			g->param = nil;
		}
		newg->stack0 = (uintptr)stk;
		newg->stackguard = (uintptr)stk + StackGuard;
		newg->stackbase = (uintptr)stk + StackSystem + stacksize - sizeof(Stktop);
		runtime·memclr((byte*)newg->stackbase, sizeof(Stktop));
	}
	return newg;
}

// Create a new g running fn with siz bytes of arguments.
// Put it on the queue of g's waiting to run.
// The compiler turns a go statement into a call to this.
// Cannot split the stack because it assumes that the arguments
// are available sequentially after &fn; they would not be
// copied if a stack split occurred.  It's OK for this to call
// functions that split the stack.
#pragma textflag 7
void
runtime·newproc(int32 siz, byte* fn, ...)
{
	byte *argp;

	if(thechar == '5')
		argp = (byte*)(&fn+2);  // skip caller's saved LR
	else
		argp = (byte*)(&fn+1);
	runtime·newproc1(fn, argp, siz, 0, runtime·getcallerpc(&siz));
}

// Create a new g running fn with narg bytes of arguments starting
// at argp and returning nret bytes of results.  callerpc is the
// address of the go statement that created this.  The new g is put
// on the queue of g's waiting to run.
G*
runtime·newproc1(byte *fn, byte *argp, int32 narg, int32 nret, void *callerpc)
{
	byte *sp;
	G *newg;
	int32 siz;
	int32 goid;

//printf("newproc1 %p %p narg=%d nret=%d\n", fn, argp, narg, nret);
	siz = narg + nret;
	siz = (siz+7) & ~7;

	// We could instead create a secondary stack frame
	// and make it look like goexit was on the original but
	// the call to the actual goroutine function was split.
	// Not worth it: this is almost always an error.
	if(siz > StackMin - 1024)
		runtime·throw("runtime.newproc: function arguments too large for new goroutine");

	goid = runtime·xadd((uint32*)&runtime·sched.goidgen, 1);
	if(raceenabled)
		runtime·racegostart(goid, callerpc);

	schedlock();

	if((newg = gfget()) != nil) {
		if(newg->stackguard - StackGuard != newg->stack0)
			runtime·throw("invalid stack in newg");
	} else {
		newg = runtime·malg(StackMin);
		if(runtime·lastg == nil)
			runtime·allg = newg;
		else
			runtime·lastg->alllink = newg;
		runtime·lastg = newg;
	}
	newg->status = Gwaiting;
	newg->waitreason = "new goroutine";

	sp = (byte*)newg->stackbase;
	sp -= siz;
	runtime·memmove(sp, argp, narg);
	if(thechar == '5') {
		// caller's LR
		sp -= sizeof(void*);
		*(void**)sp = nil;
	}

	newg->sched.sp = (uintptr)sp;
	newg->sched.pc = (byte*)runtime·goexit;
	newg->sched.g = newg;
	newg->entry = fn;
	newg->gopc = (uintptr)callerpc;

	runtime·sched.gcount++;
	newg->goid = goid;

	newprocreadylocked(newg);
	schedunlock();

	return newg;
//printf(" goid=%d\n", newg->goid);
}

// Put on gfree list.  Sched must be locked.
static void
gfput(G *gp)
{
	if(gp->stackguard - StackGuard != gp->stack0)
		runtime·throw("invalid stack in gfput");
	gp->schedlink = runtime·sched.gfree;
	runtime·sched.gfree = gp;
}

// Get from gfree list.  Sched must be locked.
static G*
gfget(void)
{
	G *gp;

	gp = runtime·sched.gfree;
	if(gp)
		runtime·sched.gfree = gp->schedlink;
	return gp;
}

void
runtime·Breakpoint(void)
{
	runtime·breakpoint();
}

void
runtime·Gosched(void)
{
	runtime·gosched();
}

// Implementation of runtime.GOMAXPROCS.
// delete when scheduler is stronger
int32
runtime·gomaxprocsfunc(int32 n)
{
	int32 ret;
	uint32 v;

	schedlock();
	ret = runtime·gomaxprocs;
	if(n <= 0)
		n = ret;
	if(n > maxgomaxprocs)
		n = maxgomaxprocs;
	runtime·gomaxprocs = n;
	if(runtime·gomaxprocs > 1)
		runtime·singleproc = false;
 	if(runtime·gcwaiting != 0) {
 		if(atomic_mcpumax(runtime·sched.atomic) != 1)
 			runtime·throw("invalid mcpumax during gc");
		schedunlock();
		return ret;
	}

	setmcpumax(n);

	// If there are now fewer allowed procs
	// than procs running, stop.
	v = runtime·atomicload(&runtime·sched.atomic);
	if(atomic_mcpu(v) > n) {
		schedunlock();
		runtime·gosched();
		return ret;
	}
	// handle more procs
	matchmg();
	schedunlock();
	return ret;
}

void
runtime·LockOSThread(void)
{
	if(m == &runtime·m0 && runtime·sched.init) {
		runtime·sched.lockmain = true;
		return;
	}
	m->lockedg = g;
	g->lockedm = m;
}

void
runtime·UnlockOSThread(void)
{
	if(m == &runtime·m0 && runtime·sched.init) {
		runtime·sched.lockmain = false;
		return;
	}
	m->lockedg = nil;
	g->lockedm = nil;
}

bool
runtime·lockedOSThread(void)
{
	return g->lockedm != nil && m->lockedg != nil;
}

// for testing of callbacks
void
runtime·golockedOSThread(bool ret)
{
	ret = runtime·lockedOSThread();
	FLUSH(&ret);
}

// for testing of wire, unwire
void
runtime·mid(uint32 ret)
{
	ret = m->id;
	FLUSH(&ret);
}

void
runtime·NumGoroutine(intgo ret)
{
	ret = runtime·sched.gcount;
	FLUSH(&ret);
}

int32
runtime·gcount(void)
{
	return runtime·sched.gcount;
}

int32
runtime·mcount(void)
{
	return runtime·sched.mcount;
}

void
runtime·badmcall(void)  // called from assembly
{
	runtime·throw("runtime: mcall called on m->g0 stack");
}

void
runtime·badmcall2(void)  // called from assembly
{
	runtime·throw("runtime: mcall function returned");
}

static struct {
	Lock;
	void (*fn)(uintptr*, int32);
	int32 hz;
	uintptr pcbuf[100];
} prof;

// Called if we receive a SIGPROF signal.
void
runtime·sigprof(uint8 *pc, uint8 *sp, uint8 *lr, G *gp)
{
	int32 n;

	if(prof.fn == nil || prof.hz == 0)
		return;

	runtime·lock(&prof);
	if(prof.fn == nil) {
		runtime·unlock(&prof);
		return;
	}
	n = runtime·gentraceback(pc, sp, lr, gp, 0, prof.pcbuf, nelem(prof.pcbuf));
	if(n > 0)
		prof.fn(prof.pcbuf, n);
	runtime·unlock(&prof);
}

// Arrange to call fn with a traceback hz times a second.
void
runtime·setcpuprofilerate(void (*fn)(uintptr*, int32), int32 hz)
{
	// Force sane arguments.
	if(hz < 0)
		hz = 0;
	if(hz == 0)
		fn = nil;
	if(fn == nil)
		hz = 0;

	// Stop profiler on this cpu so that it is safe to lock prof.
	// if a profiling signal came in while we had prof locked,
	// it would deadlock.
	runtime·resetcpuprofiler(0);

	runtime·lock(&prof);
	prof.fn = fn;
	prof.hz = hz;
	runtime·unlock(&prof);
	runtime·lock(&runtime·sched);
	runtime·sched.profilehz = hz;
	runtime·unlock(&runtime·sched);

	if(hz != 0)
		runtime·resetcpuprofiler(hz);
}
