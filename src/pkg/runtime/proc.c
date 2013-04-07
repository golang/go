// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "stack.h"
#include "race.h"
#include "type.h"

// Goroutine scheduler
// The scheduler's job is to distribute ready-to-run goroutines over worker threads.
//
// The main concepts are:
// G - goroutine.
// M - worker thread, or machine.
// P - processor, a resource that is required to execute Go code.
//     M must have an associated P to execute Go code, however it can be
//     blocked or in a syscall w/o an associated P.
//
// Design doc at http://golang.org/s/go11sched.

typedef struct Sched Sched;
struct Sched {
	Lock;

	uint64	goidgen;

	M*	midle;	 // idle m's waiting for work
	int32	nmidle;	 // number of idle m's waiting for work
	int32	mlocked; // number of locked m's waiting for work
	int32	mcount;	 // number of m's that have been created

	P*	pidle;  // idle P's
	uint32	npidle;
	uint32	nmspinning;

	// Global runnable queue.
	G*	runqhead;
	G*	runqtail;
	int32	runqsize;

	// Global cache of dead G's.
	Lock	gflock;
	G*	gfree;

	int32	stopwait;
	Note	stopnote;
	uint32	sysmonwait;
	Note	sysmonnote;
	uint64	lastpoll;

	int32	profilehz;	// cpu profiling rate
};

// The max value of GOMAXPROCS.
// There are no fundamental restrictions on the value.
enum { MaxGomaxprocs = 1<<8 };

Sched	runtime·sched;
int32	runtime·gomaxprocs;
bool	runtime·singleproc;
bool	runtime·iscgo;
uint32	runtime·gcwaiting;
M	runtime·m0;
G	runtime·g0;	 // idle goroutine for m0
G*	runtime·allg;
G*	runtime·lastg;
M*	runtime·allm;
M*	runtime·extram;
int8*	runtime·goos;
int32	runtime·ncpu;
static int32	newprocs;

void runtime·mstart(void);
static void runqput(P*, G*);
static G* runqget(P*);
static void runqgrow(P*);
static G* runqsteal(P*, P*);
static void mput(M*);
static M* mget(void);
static void mcommoninit(M*);
static void schedule(void);
static void procresize(int32);
static void acquirep(P*);
static P* releasep(void);
static void newm(void(*)(void), P*);
static void goidle(void);
static void stopm(void);
static void startm(P*, bool);
static void handoffp(P*);
static void wakep(void);
static void stoplockedm(void);
static void startlockedm(G*);
static void sysmon(void);
static uint32 retake(uint32*);
static void inclocked(int32);
static void checkdead(void);
static void exitsyscall0(G*);
static void park0(G*);
static void gosched0(G*);
static void goexit0(G*);
static void gfput(P*, G*);
static G* gfget(P*);
static void gfpurge(P*);
static void globrunqput(G*);
static G* globrunqget(P*);
static P* pidleget(void);
static void pidleput(P*);
static void injectglist(G*);

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
	int32 n, procs;
	byte *p;

	m->nomemprof++;
	runtime·mprofinit();
	runtime·mallocinit();
	mcommoninit(m);

	runtime·goargs();
	runtime·goenvs();

	// For debugging:
	// Allocate internal symbol table representation now,
	// so that we don't need to call malloc when we crash.
	// runtime·findfunc(0);

	runtime·sched.lastpoll = runtime·nanotime();
	procs = 1;
	p = runtime·getenv("GOMAXPROCS");
	if(p != nil && (n = runtime·atoi(p)) > 0) {
		if(n > MaxGomaxprocs)
			n = MaxGomaxprocs;
		procs = n;
	}
	runtime·allp = runtime·malloc((MaxGomaxprocs+1)*sizeof(runtime·allp[0]));
	procresize(procs);

	mstats.enablegc = 1;
	m->nomemprof--;

	if(raceenabled)
		g->racectx = runtime·raceinit();
}

extern void main·init(void);
extern void main·main(void);

static FuncVal scavenger = {runtime·MHeap_Scavenger};

// The main goroutine.
void
runtime·main(void)
{
	newm(sysmon, nil);

	// Lock the main goroutine onto this, the main OS thread,
	// during initialization.  Most programs won't care, but a few
	// do require certain calls to be made by the main thread.
	// Those can arrange for main.main to run in the main thread
	// by calling runtime.LockOSThread during initialization
	// to preserve the lock.
	runtime·lockOSThread();
	if(m != &runtime·m0)
		runtime·throw("runtime·main not on m0");
	runtime·newproc1(&scavenger, nil, 0, 0, runtime·main);
	main·init();
	runtime·unlockOSThread();

	main·main();
	if(raceenabled)
		runtime·racefini();

	// Make racy client program work: if panicking on
	// another goroutine at the same time as main returns,
	// let the other goroutine finish printing the panic trace.
	// Once it does, it will exit. See issue 3934.
	if(runtime·panicking)
		runtime·park(nil, nil, "panicwait");

	runtime·exit(0);
	for(;;)
		*(int32*)runtime·main = 0;
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
	default:
		status = "???";
		break;
	}
	runtime·printf("goroutine %D [%s]:\n", gp->goid, status);
}

void
runtime·tracebackothers(G *me)
{
	G *gp;
	int32 traceback;

	traceback = runtime·gotraceback(nil);
	for(gp = runtime·allg; gp != nil; gp = gp->alllink) {
		if(gp == me || gp->status == Gdead)
			continue;
		if(gp->issystem && traceback < 2)
			continue;
		runtime·printf("\n");
		runtime·goroutineheader(gp);
		runtime·traceback(gp->sched.pc, (byte*)gp->sched.sp, 0, gp);
	}
}

static void
mcommoninit(M *mp)
{
	// If there is no mcache runtime·callers() will crash,
	// and we are most likely in sysmon thread so the stack is senseless anyway.
	if(m->mcache)
		runtime·callers(1, mp->createstack, nelem(mp->createstack));

	mp->fastrand = 0x49f6428aUL + mp->id + runtime·cputicks();

	runtime·lock(&runtime·sched);
	mp->id = runtime·sched.mcount++;

	runtime·mpreinit(mp);

	// Add to runtime·allm so garbage collector doesn't free m
	// when it is just in a register or thread-local storage.
	mp->alllink = runtime·allm;
	// runtime·NumCgoCall() iterates over allm w/o schedlock,
	// so we need to publish it safely.
	runtime·atomicstorep(&runtime·allm, mp);
	runtime·unlock(&runtime·sched);
}

// Mark gp ready to run.
void
runtime·ready(G *gp)
{
	// Mark runnable.
	if(gp->status != Gwaiting) {
		runtime·printf("goroutine %D has status %d\n", gp->goid, gp->status);
		runtime·throw("bad g->status in ready");
	}
	gp->status = Grunnable;
	runqput(m->p, gp);
	if(runtime·atomicload(&runtime·sched.npidle) != 0 && runtime·atomicload(&runtime·sched.nmspinning) == 0)  // TODO: fast atomic
		wakep();
}

int32
runtime·gcprocs(void)
{
	int32 n;

	// Figure out how many CPUs to use during GC.
	// Limited by gomaxprocs, number of actual CPUs, and MaxGcproc.
	runtime·lock(&runtime·sched);
	n = runtime·gomaxprocs;
	if(n > runtime·ncpu)
		n = runtime·ncpu;
	if(n > MaxGcproc)
		n = MaxGcproc;
	if(n > runtime·sched.nmidle+1) // one M is currently running
		n = runtime·sched.nmidle+1;
	runtime·unlock(&runtime·sched);
	return n;
}

static bool
needaddgcproc(void)
{
	int32 n;

	runtime·lock(&runtime·sched);
	n = runtime·gomaxprocs;
	if(n > runtime·ncpu)
		n = runtime·ncpu;
	if(n > MaxGcproc)
		n = MaxGcproc;
	n -= runtime·sched.nmidle+1; // one M is currently running
	runtime·unlock(&runtime·sched);
	return n > 0;
}

void
runtime·helpgc(int32 nproc)
{
	M *mp;
	int32 n, pos;

	runtime·lock(&runtime·sched);
	pos = 0;
	for(n = 1; n < nproc; n++) {  // one M is currently running
		if(runtime·allp[pos]->mcache == m->mcache)
			pos++;
		mp = mget();
		if(mp == nil)
			runtime·throw("runtime·gcprocs inconsistency");
		mp->helpgc = n;
		mp->mcache = runtime·allp[pos]->mcache;
		pos++;
		runtime·notewakeup(&mp->park);
	}
	runtime·unlock(&runtime·sched);
}

void
runtime·stoptheworld(void)
{
	int32 i;
	uint32 s;
	P *p;
	bool wait;

	runtime·lock(&runtime·sched);
	runtime·sched.stopwait = runtime·gomaxprocs;
	runtime·atomicstore((uint32*)&runtime·gcwaiting, 1);
	// stop current P
	m->p->status = Pgcstop;
	runtime·sched.stopwait--;
	// try to retake all P's in Psyscall status
	for(i = 0; i < runtime·gomaxprocs; i++) {
		p = runtime·allp[i];
		s = p->status;
		if(s == Psyscall && runtime·cas(&p->status, s, Pgcstop))
			runtime·sched.stopwait--;
	}
	// stop idle P's
	while(p = pidleget()) {
		p->status = Pgcstop;
		runtime·sched.stopwait--;
	}
	wait = runtime·sched.stopwait > 0;
	runtime·unlock(&runtime·sched);

	// wait for remaining P's to stop voluntary
	if(wait) {
		runtime·notesleep(&runtime·sched.stopnote);
		runtime·noteclear(&runtime·sched.stopnote);
	}
	if(runtime·sched.stopwait)
		runtime·throw("stoptheworld: not stopped");
	for(i = 0; i < runtime·gomaxprocs; i++) {
		p = runtime·allp[i];
		if(p->status != Pgcstop)
			runtime·throw("stoptheworld: not stopped");
	}
}

static void
mhelpgc(void)
{
	m->helpgc = -1;
}

void
runtime·starttheworld(void)
{
	P *p, *p1;
	M *mp;
	G *gp;
	bool add;

	gp = runtime·netpoll(false);  // non-blocking
	injectglist(gp);
	add = needaddgcproc();
	runtime·lock(&runtime·sched);
	if(newprocs) {
		procresize(newprocs);
		newprocs = 0;
	} else
		procresize(runtime·gomaxprocs);
	runtime·gcwaiting = 0;

	p1 = nil;
	while(p = pidleget()) {
		// procresize() puts p's with work at the beginning of the list.
		// Once we reach a p without a run queue, the rest don't have one either.
		if(p->runqhead == p->runqtail) {
			pidleput(p);
			break;
		}
		mp = mget();
		if(mp == nil) {
			p->link = p1;
			p1 = p;
			continue;
		}
		if(mp->nextp)
			runtime·throw("starttheworld: inconsistent mp->nextp");
		mp->nextp = p;
		runtime·notewakeup(&mp->park);
	}
	if(runtime·sched.sysmonwait) {
		runtime·sched.sysmonwait = false;
		runtime·notewakeup(&runtime·sched.sysmonnote);
	}
	runtime·unlock(&runtime·sched);

	while(p1) {
		p = p1;
		p1 = p1->link;
		add = false;
		newm(nil, p);
	}

	if(add) {
		// If GC could have used another helper proc, start one now,
		// in the hope that it will be available next time.
		// It would have been even better to start it before the collection,
		// but doing so requires allocating memory, so it's tricky to
		// coordinate.  This lazy approach works out in practice:
		// we don't mind if the first couple gc rounds don't have quite
		// the maximum number of procs.
		newm(mhelpgc, nil);
	}
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
	if(m == &runtime·m0) {
		runtime·initsig();
		if(runtime·iscgo)
			runtime·newextram();
	}
	
	if(m->mstartfn)
		m->mstartfn();

	if(m->helpgc) {
		m->helpgc = 0;
		stopm();
	} else if(m != &runtime·m0) {
		acquirep(m->nextp);
		m->nextp = nil;
	}
	schedule();

	// TODO(brainman): This point is never reached, because scheduler
	// does not release os threads at the moment. But once this path
	// is enabled, we must remove our seh here.
}

// When running with cgo, we call _cgo_thread_start
// to start threads for us so that we can play nicely with
// foreign code.
void (*_cgo_thread_start)(void*);

typedef struct CgoThreadStart CgoThreadStart;
struct CgoThreadStart
{
	M *m;
	G *g;
	void (*fn)(void);
};

// Allocate a new m unassociated with any thread.
// Can use p for allocation context if needed.
M*
runtime·allocm(P *p)
{
	M *mp;
	static Type *mtype;  // The Go type M

	m->locks++;  // disable GC because it can be called from sysmon
	if(m->p == nil)
		acquirep(p);  // temporarily borrow p for mallocs in this function
	if(mtype == nil) {
		Eface e;
		runtime·gc_m_ptr(&e);
		mtype = ((PtrType*)e.type)->elem;
	}

	mp = runtime·cnew(mtype);
	mcommoninit(mp);

	// In case of cgo, pthread_create will make us a stack.
	// Windows will layout sched stack on OS stack.
	if(runtime·iscgo || Windows)
		mp->g0 = runtime·malg(-1);
	else
		mp->g0 = runtime·malg(8192);

	if(p == m->p)
		releasep();
	m->locks--;

	return mp;
}

static M* lockextra(bool nilokay);
static void unlockextra(M*);

// needm is called when a cgo callback happens on a
// thread without an m (a thread not created by Go).
// In this case, needm is expected to find an m to use
// and return with m, g initialized correctly.
// Since m and g are not set now (likely nil, but see below)
// needm is limited in what routines it can call. In particular
// it can only call nosplit functions (textflag 7) and cannot
// do any scheduling that requires an m.
//
// In order to avoid needing heavy lifting here, we adopt
// the following strategy: there is a stack of available m's
// that can be stolen. Using compare-and-swap
// to pop from the stack has ABA races, so we simulate
// a lock by doing an exchange (via casp) to steal the stack
// head and replace the top pointer with MLOCKED (1).
// This serves as a simple spin lock that we can use even
// without an m. The thread that locks the stack in this way
// unlocks the stack by storing a valid stack head pointer.
//
// In order to make sure that there is always an m structure
// available to be stolen, we maintain the invariant that there
// is always one more than needed. At the beginning of the
// program (if cgo is in use) the list is seeded with a single m.
// If needm finds that it has taken the last m off the list, its job
// is - once it has installed its own m so that it can do things like
// allocate memory - to create a spare m and put it on the list.
//
// Each of these extra m's also has a g0 and a curg that are
// pressed into service as the scheduling stack and current
// goroutine for the duration of the cgo callback.
//
// When the callback is done with the m, it calls dropm to
// put the m back on the list.
#pragma textflag 7
void
runtime·needm(byte x)
{
	M *mp;

	// Lock extra list, take head, unlock popped list.
	// nilokay=false is safe here because of the invariant above,
	// that the extra list always contains or will soon contain
	// at least one m.
	mp = lockextra(false);

	// Set needextram when we've just emptied the list,
	// so that the eventual call into cgocallbackg will
	// allocate a new m for the extra list. We delay the
	// allocation until then so that it can be done
	// after exitsyscall makes sure it is okay to be
	// running at all (that is, there's no garbage collection
	// running right now).
	mp->needextram = mp->schedlink == nil;
	unlockextra(mp->schedlink);

	// Install m and g (= m->g0) and set the stack bounds
	// to match the current stack. We don't actually know
	// how big the stack is, like we don't know how big any
	// scheduling stack is, but we assume there's at least 32 kB,
	// which is more than enough for us.
	runtime·setmg(mp, mp->g0);
	g->stackbase = (uintptr)(&x + 1024);
	g->stackguard = (uintptr)(&x - 32*1024);

	// On windows/386, we need to put an SEH frame (two words)
	// somewhere on the current stack. We are called
	// from needm, and we know there is some available
	// space one word into the argument frame. Use that.
	m->seh = (SEH*)((uintptr*)&x + 1);

	// Initialize this thread to use the m.
	runtime·asminit();
	runtime·minit();
}

// newextram allocates an m and puts it on the extra list.
// It is called with a working local m, so that it can do things
// like call schedlock and allocate.
void
runtime·newextram(void)
{
	M *mp, *mnext;
	G *gp;

	// Create extra goroutine locked to extra m.
	// The goroutine is the context in which the cgo callback will run.
	// The sched.pc will never be returned to, but setting it to
	// runtime.goexit makes clear to the traceback routines where
	// the goroutine stack ends.
	mp = runtime·allocm(nil);
	gp = runtime·malg(4096);
	gp->sched.pc = (void*)runtime·goexit;
	gp->sched.sp = gp->stackbase;
	gp->sched.g = gp;
	gp->status = Gsyscall;
	mp->curg = gp;
	mp->locked = LockInternal;
	mp->lockedg = gp;
	gp->lockedm = mp;
	// put on allg for garbage collector
	runtime·lock(&runtime·sched);
	if(runtime·lastg == nil)
		runtime·allg = gp;
	else
		runtime·lastg->alllink = gp;
	runtime·lastg = gp;
	runtime·unlock(&runtime·sched);
	gp->goid = runtime·xadd64(&runtime·sched.goidgen, 1);
	if(raceenabled)
		gp->racectx = runtime·racegostart(runtime·newextram);

	// Add m to the extra list.
	mnext = lockextra(true);
	mp->schedlink = mnext;
	unlockextra(mp);
}

// dropm is called when a cgo callback has called needm but is now
// done with the callback and returning back into the non-Go thread.
// It puts the current m back onto the extra list.
//
// The main expense here is the call to signalstack to release the
// m's signal stack, and then the call to needm on the next callback
// from this thread. It is tempting to try to save the m for next time,
// which would eliminate both these costs, but there might not be
// a next time: the current thread (which Go does not control) might exit.
// If we saved the m for that thread, there would be an m leak each time
// such a thread exited. Instead, we acquire and release an m on each
// call. These should typically not be scheduling operations, just a few
// atomics, so the cost should be small.
//
// TODO(rsc): An alternative would be to allocate a dummy pthread per-thread
// variable using pthread_key_create. Unlike the pthread keys we already use
// on OS X, this dummy key would never be read by Go code. It would exist
// only so that we could register at thread-exit-time destructor.
// That destructor would put the m back onto the extra list.
// This is purely a performance optimization. The current version,
// in which dropm happens on each cgo call, is still correct too.
// We may have to keep the current version on systems with cgo
// but without pthreads, like Windows.
void
runtime·dropm(void)
{
	M *mp, *mnext;

	// Undo whatever initialization minit did during needm.
	runtime·unminit();
	m->seh = nil;  // reset dangling typed pointer

	// Clear m and g, and return m to the extra list.
	// After the call to setmg we can only call nosplit functions.
	mp = m;
	runtime·setmg(nil, nil);

	mnext = lockextra(true);
	mp->schedlink = mnext;
	unlockextra(mp);
}

#define MLOCKED ((M*)1)

// lockextra locks the extra list and returns the list head.
// The caller must unlock the list by storing a new list head
// to runtime.extram. If nilokay is true, then lockextra will
// return a nil list head if that's what it finds. If nilokay is false,
// lockextra will keep waiting until the list head is no longer nil.
#pragma textflag 7
static M*
lockextra(bool nilokay)
{
	M *mp;
	void (*yield)(void);

	for(;;) {
		mp = runtime·atomicloadp(&runtime·extram);
		if(mp == MLOCKED) {
			yield = runtime·osyield;
			yield();
			continue;
		}
		if(mp == nil && !nilokay) {
			runtime·usleep(1);
			continue;
		}
		if(!runtime·casp(&runtime·extram, mp, MLOCKED)) {
			yield = runtime·osyield;
			yield();
			continue;
		}
		break;
	}
	return mp;
}

#pragma textflag 7
static void
unlockextra(M *mp)
{
	runtime·atomicstorep(&runtime·extram, mp);
}


// Create a new m.  It will start off with a call to fn, or else the scheduler.
static void
newm(void(*fn)(void), P *p)
{
	M *mp;

	mp = runtime·allocm(p);
	mp->nextp = p;
	mp->mstartfn = fn;

	if(runtime·iscgo) {
		CgoThreadStart ts;

		if(_cgo_thread_start == nil)
			runtime·throw("_cgo_thread_start missing");
		ts.m = mp;
		ts.g = mp->g0;
		ts.fn = runtime·mstart;
		runtime·asmcgocall(_cgo_thread_start, &ts);
		return;
	}
	runtime·newosproc(mp, (byte*)mp->g0->stackbase);
}

// Stops execution of the current m until new work is available.
// Returns with acquired P.
static void
stopm(void)
{
	if(m->locks)
		runtime·throw("stopm holding locks");
	if(m->p)
		runtime·throw("stopm holding p");
	if(m->spinning) {
		m->spinning = false;
		runtime·xadd(&runtime·sched.nmspinning, -1);
	}

retry:
	runtime·lock(&runtime·sched);
	mput(m);
	runtime·unlock(&runtime·sched);
	runtime·notesleep(&m->park);
	runtime·noteclear(&m->park);
	if(m->helpgc) {
		runtime·gchelper();
		m->helpgc = 0;
		m->mcache = nil;
		goto retry;
	}
	acquirep(m->nextp);
	m->nextp = nil;
}

static void
mspinning(void)
{
	m->spinning = true;
}

// Schedules some M to run the p (creates an M if necessary).
// If p==nil, tries to get an idle P, if no idle P's returns false.
static void
startm(P *p, bool spinning)
{
	M *mp;
	void (*fn)(void);

	runtime·lock(&runtime·sched);
	if(p == nil) {
		p = pidleget();
		if(p == nil) {
			runtime·unlock(&runtime·sched);
			if(spinning)
				runtime·xadd(&runtime·sched.nmspinning, -1);
			return;
		}
	}
	mp = mget();
	runtime·unlock(&runtime·sched);
	if(mp == nil) {
		fn = nil;
		if(spinning)
			fn = mspinning;
		newm(fn, p);
		return;
	}
	if(mp->spinning)
		runtime·throw("startm: m is spinning");
	if(mp->nextp)
		runtime·throw("startm: m has p");
	mp->spinning = spinning;
	mp->nextp = p;
	runtime·notewakeup(&mp->park);
}

// Hands off P from syscall or locked M.
static void
handoffp(P *p)
{
	// if it has local work, start it straight away
	if(p->runqhead != p->runqtail || runtime·sched.runqsize) {
		startm(p, false);
		return;
	}
	// no local work, check that there are no spinning/idle M's,
	// otherwise our help is not required
	if(runtime·atomicload(&runtime·sched.nmspinning) + runtime·atomicload(&runtime·sched.npidle) == 0 &&  // TODO: fast atomic
		runtime·cas(&runtime·sched.nmspinning, 0, 1)) {
		startm(p, true);
		return;
	}
	runtime·lock(&runtime·sched);
	if(runtime·gcwaiting) {
		p->status = Pgcstop;
		if(--runtime·sched.stopwait == 0)
			runtime·notewakeup(&runtime·sched.stopnote);
		runtime·unlock(&runtime·sched);
		return;
	}
	if(runtime·sched.runqsize) {
		runtime·unlock(&runtime·sched);
		startm(p, false);
		return;
	}
	// If this is the last running P and nobody is polling network,
	// need to wakeup another M to poll network.
	if(runtime·sched.npidle == runtime·gomaxprocs-1 && runtime·atomicload64(&runtime·sched.lastpoll) != 0) {
		runtime·unlock(&runtime·sched);
		startm(p, false);
		return;
	}
	pidleput(p);
	runtime·unlock(&runtime·sched);
}

// Tries to add one more P to execute G's.
// Called when a G is made runnable (newproc, ready).
static void
wakep(void)
{
	// be conservative about spinning threads
	if(!runtime·cas(&runtime·sched.nmspinning, 0, 1))
		return;
	startm(nil, true);
}

// Stops execution of the current m that is locked to a g until the g is runnable again.
// Returns with acquired P.
static void
stoplockedm(void)
{
	P *p;

	if(m->lockedg == nil || m->lockedg->lockedm != m)
		runtime·throw("stoplockedm: inconsistent locking");
	if(m->p) {
		// Schedule another M to run this p.
		p = releasep();
		handoffp(p);
	}
	inclocked(1);
	// Wait until another thread schedules lockedg again.
	runtime·notesleep(&m->park);
	runtime·noteclear(&m->park);
	if(m->lockedg->status != Grunnable)
		runtime·throw("stoplockedm: not runnable");
	acquirep(m->nextp);
	m->nextp = nil;
}

// Schedules the locked m to run the locked gp.
static void
startlockedm(G *gp)
{
	M *mp;
	P *p;

	mp = gp->lockedm;
	if(mp == m)
		runtime·throw("startlockedm: locked to me");
	if(mp->nextp)
		runtime·throw("startlockedm: m has p");
	// directly handoff current P to the locked m
	inclocked(-1);
	p = releasep();
	mp->nextp = p;
	runtime·notewakeup(&mp->park);
	stopm();
}

// Stops the current m for stoptheworld.
// Returns when the world is restarted.
static void
gcstopm(void)
{
	P *p;

	if(!runtime·gcwaiting)
		runtime·throw("gcstopm: not waiting for gc");
	if(m->spinning) {
		m->spinning = false;
		runtime·xadd(&runtime·sched.nmspinning, -1);
	}
	p = releasep();
	runtime·lock(&runtime·sched);
	p->status = Pgcstop;
	if(--runtime·sched.stopwait == 0)
		runtime·notewakeup(&runtime·sched.stopnote);
	runtime·unlock(&runtime·sched);
	stopm();
}

// Schedules gp to run on the current M.
// Never returns.
static void
execute(G *gp)
{
	int32 hz;

	if(gp->status != Grunnable) {
		runtime·printf("execute: bad g status %d\n", gp->status);
		runtime·throw("execute: bad g status");
	}
	gp->status = Grunning;
	m->p->tick++;
	m->curg = gp;
	gp->m = m;

	// Check whether the profiler needs to be turned on or off.
	hz = runtime·sched.profilehz;
	if(m->profilehz != hz)
		runtime·resetcpuprofiler(hz);

	if(gp->sched.pc == (byte*)runtime·goexit)  // kickoff
		runtime·gogocallfn(&gp->sched, gp->fnstart);
	runtime·gogo(&gp->sched, 0);
}

// Finds a runnable goroutine to execute.
// Tries to steal from other P's, get g from global queue, poll network.
static G*
findrunnable(void)
{
	G *gp;
	P *p;
	int32 i;

top:
	if(runtime·gcwaiting) {
		gcstopm();
		goto top;
	}
	// local runq
	gp = runqget(m->p);
	if(gp)
		return gp;
	// global runq
	if(runtime·sched.runqsize) {
		runtime·lock(&runtime·sched);
		gp = globrunqget(m->p);
		runtime·unlock(&runtime·sched);
		if(gp)
			return gp;
	}
	// poll network
	gp = runtime·netpoll(false);  // non-blocking
	if(gp) {
		injectglist(gp->schedlink);
		gp->status = Grunnable;
		return gp;
	}
	// If number of spinning M's >= number of busy P's, block.
	// This is necessary to prevent excessive CPU consumption
	// when GOMAXPROCS>>1 but the program parallelism is low.
	if(!m->spinning && 2 * runtime·atomicload(&runtime·sched.nmspinning) >= runtime·gomaxprocs - runtime·atomicload(&runtime·sched.npidle))  // TODO: fast atomic
		goto stop;
	if(!m->spinning) {
		m->spinning = true;
		runtime·xadd(&runtime·sched.nmspinning, 1);
	}
	// random steal from other P's
	for(i = 0; i < 2*runtime·gomaxprocs; i++) {
		if(runtime·gcwaiting)
			goto top;
		p = runtime·allp[runtime·fastrand1()%runtime·gomaxprocs];
		if(p == m->p)
			gp = runqget(p);
		else
			gp = runqsteal(m->p, p);
		if(gp)
			return gp;
	}
stop:
	// return P and block
	runtime·lock(&runtime·sched);
	if(runtime·gcwaiting) {
		runtime·unlock(&runtime·sched);
		goto top;
	}
	if(runtime·sched.runqsize) {
		gp = globrunqget(m->p);
		runtime·unlock(&runtime·sched);
		return gp;
	}
	p = releasep();
	pidleput(p);
	runtime·unlock(&runtime·sched);
	if(m->spinning) {
		m->spinning = false;
		runtime·xadd(&runtime·sched.nmspinning, -1);
	}
	// check all runqueues once again
	for(i = 0; i < runtime·gomaxprocs; i++) {
		p = runtime·allp[i];
		if(p && p->runqhead != p->runqtail) {
			runtime·lock(&runtime·sched);
			p = pidleget();
			runtime·unlock(&runtime·sched);
			if(p) {
				acquirep(p);
				goto top;
			}
			break;
		}
	}
	// poll network
	if(runtime·xchg64(&runtime·sched.lastpoll, 0) != 0) {
		if(m->p)
			runtime·throw("findrunnable: netpoll with p");
		if(m->spinning)
			runtime·throw("findrunnable: netpoll with spinning");
		gp = runtime·netpoll(true);  // block until new work is available
		runtime·atomicstore64(&runtime·sched.lastpoll, runtime·nanotime());
		if(gp) {
			runtime·lock(&runtime·sched);
			p = pidleget();
			runtime·unlock(&runtime·sched);
			if(p) {
				acquirep(p);
				injectglist(gp->schedlink);
				gp->status = Grunnable;
				return gp;
			}
			injectglist(gp);
		}
	}
	stopm();
	goto top;
}

// Injects the list of runnable G's into the scheduler.
// Can run concurrently with GC.
static void
injectglist(G *glist)
{
	int32 n;
	G *gp;

	if(glist == nil)
		return;
	runtime·lock(&runtime·sched);
	for(n = 0; glist; n++) {
		gp = glist;
		glist = gp->schedlink;
		gp->status = Grunnable;
		globrunqput(gp);
	}
	runtime·unlock(&runtime·sched);

	for(; n && runtime·sched.npidle; n--)
		startm(nil, false);
}

// One round of scheduler: find a runnable goroutine and execute it.
// Never returns.
static void
schedule(void)
{
	G *gp;

	if(m->locks)
		runtime·throw("schedule: holding locks");

top:
	if(runtime·gcwaiting) {
		gcstopm();
		goto top;
	}

	gp = runqget(m->p);
	if(gp == nil)
		gp = findrunnable();

	if(m->spinning) {
		m->spinning = false;
		runtime·xadd(&runtime·sched.nmspinning, -1);
	}

	// M wakeup policy is deliberately somewhat conservative (see nmspinning handling),
	// so see if we need to wakeup another M here.
	if (m->p->runqhead != m->p->runqtail &&
		runtime·atomicload(&runtime·sched.nmspinning) == 0 &&
		runtime·atomicload(&runtime·sched.npidle) > 0)  // TODO: fast atomic
		wakep();

	if(gp->lockedm) {
		startlockedm(gp);
		goto top;
	}

	execute(gp);
}

// Puts the current goroutine into a waiting state and unlocks the lock.
// The goroutine can be made runnable again by calling runtime·ready(gp).
void
runtime·park(void(*unlockf)(Lock*), Lock *lock, int8 *reason)
{
	m->waitlock = lock;
	m->waitunlockf = unlockf;
	g->waitreason = reason;
	runtime·mcall(park0);
}

// runtime·park continuation on g0.
static void
park0(G *gp)
{
	gp->status = Gwaiting;
	gp->m = nil;
	m->curg = nil;
	if(m->waitunlockf) {
		m->waitunlockf(m->waitlock);
		m->waitunlockf = nil;
		m->waitlock = nil;
	}
	if(m->lockedg) {
		stoplockedm();
		execute(gp);  // Never returns.
	}
	schedule();
}

// Scheduler yield.
void
runtime·gosched(void)
{
	runtime·mcall(gosched0);
}

// runtime·gosched continuation on g0.
static void
gosched0(G *gp)
{
	gp->status = Grunnable;
	gp->m = nil;
	m->curg = nil;
	runtime·lock(&runtime·sched);
	globrunqput(gp);
	runtime·unlock(&runtime·sched);
	if(m->lockedg) {
		stoplockedm();
		execute(gp);  // Never returns.
	}
	schedule();
}

// Finishes execution of the current goroutine.
void
runtime·goexit(void)
{
	if(raceenabled)
		runtime·racegoend();
	runtime·mcall(goexit0);
}

// runtime·goexit continuation on g0.
static void
goexit0(G *gp)
{
	gp->status = Gdead;
	gp->m = nil;
	gp->lockedm = nil;
	m->curg = nil;
	m->lockedg = nil;
	if(m->locked & ~LockExternal) {
		runtime·printf("invalid m->locked = %d", m->locked);
		runtime·throw("internal lockOSThread error");
	}	
	m->locked = 0;
	runtime·unwindstack(gp, nil);
	gfput(m->p, gp);
	schedule();
}

// The goroutine g is about to enter a system call.
// Record that it's not using the cpu anymore.
// This is called only from the go syscall library and cgocall,
// not from the low-level system calls used by the runtime.
//
// Entersyscall cannot split the stack: the runtime·gosave must
// make g->sched refer to the caller's stack segment, because
// entersyscall is going to return immediately after.
#pragma textflag 7
void
·entersyscall(int32 dummy)
{
	if(m->profilehz > 0)
		runtime·setprof(false);

	// Leave SP around for gc and traceback.
	g->sched.sp = (uintptr)runtime·getcallersp(&dummy);
	g->sched.pc = runtime·getcallerpc(&dummy);
	g->sched.g = g;
	g->gcsp = g->sched.sp;
	g->gcpc = g->sched.pc;
	g->gcstack = g->stackbase;
	g->gcguard = g->stackguard;
	g->status = Gsyscall;
	if(g->gcsp < g->gcguard-StackGuard || g->gcstack < g->gcsp) {
		// runtime·printf("entersyscall inconsistent %p [%p,%p]\n",
		//	g->gcsp, g->gcguard-StackGuard, g->gcstack);
		runtime·throw("entersyscall");
	}

	if(runtime·atomicload(&runtime·sched.sysmonwait)) {  // TODO: fast atomic
		runtime·lock(&runtime·sched);
		if(runtime·atomicload(&runtime·sched.sysmonwait)) {
			runtime·atomicstore(&runtime·sched.sysmonwait, 0);
			runtime·notewakeup(&runtime·sched.sysmonnote);
		}
		runtime·unlock(&runtime·sched);
		runtime·gosave(&g->sched);  // re-save for traceback
	}

	m->mcache = nil;
	m->p->tick++;
	m->p->m = nil;
	runtime·atomicstore(&m->p->status, Psyscall);
	if(runtime·gcwaiting) {
		runtime·lock(&runtime·sched);
		if (runtime·sched.stopwait > 0 && runtime·cas(&m->p->status, Psyscall, Pgcstop)) {
			if(--runtime·sched.stopwait == 0)
				runtime·notewakeup(&runtime·sched.stopnote);
		}
		runtime·unlock(&runtime·sched);
		runtime·gosave(&g->sched);  // re-save for traceback
	}
}

// The same as runtime·entersyscall(), but with a hint that the syscall is blocking.
#pragma textflag 7
void
·entersyscallblock(int32 dummy)
{
	P *p;

	if(m->profilehz > 0)
		runtime·setprof(false);

	// Leave SP around for gc and traceback.
	g->sched.sp = (uintptr)runtime·getcallersp(&dummy);
	g->sched.pc = runtime·getcallerpc(&dummy);
	g->sched.g = g;
	g->gcsp = g->sched.sp;
	g->gcpc = g->sched.pc;
	g->gcstack = g->stackbase;
	g->gcguard = g->stackguard;
	g->status = Gsyscall;
	if(g->gcsp < g->gcguard-StackGuard || g->gcstack < g->gcsp) {
		// runtime·printf("entersyscallblock inconsistent %p [%p,%p]\n",
		//	g->gcsp, g->gcguard-StackGuard, g->gcstack);
		runtime·throw("entersyscallblock");
	}

	p = releasep();
	handoffp(p);
	if(g->isbackground)  // do not consider blocked scavenger for deadlock detection
		inclocked(1);
	runtime·gosave(&g->sched);  // re-save for traceback
}

// The goroutine g exited its system call.
// Arrange for it to run on a cpu again.
// This is called only from the go syscall library, not
// from the low-level system calls used by the runtime.
void
runtime·exitsyscall(void)
{
	P *p;

	// Check whether the profiler needs to be turned on.
	if(m->profilehz > 0)
		runtime·setprof(true);

	// Try to re-acquire the last P.
	if(m->p && m->p->status == Psyscall && runtime·cas(&m->p->status, Psyscall, Prunning)) {
		// There's a cpu for us, so we can run.
		m->mcache = m->p->mcache;
		m->p->m = m;
		m->p->tick++;
		g->status = Grunning;
		// Garbage collector isn't running (since we are),
		// so okay to clear gcstack and gcsp.
		g->gcstack = (uintptr)nil;
		g->gcsp = (uintptr)nil;
		return;
	}

	if(g->isbackground)  // do not consider blocked scavenger for deadlock detection
		inclocked(-1);
	// Try to get any other idle P.
	m->p = nil;
	if(runtime·sched.pidle) {
		runtime·lock(&runtime·sched);
		p = pidleget();
		runtime·unlock(&runtime·sched);
		if(p) {
			acquirep(p);
			g->gcstack = (uintptr)nil;
			g->gcsp = (uintptr)nil;
			return;
		}
	}

	// Call the scheduler.
	runtime·mcall(exitsyscall0);

	// Scheduler returned, so we're allowed to run now.
	// Delete the gcstack information that we left for
	// the garbage collector during the system call.
	// Must wait until now because until gosched returns
	// we don't know for sure that the garbage collector
	// is not running.
	g->gcstack = (uintptr)nil;
	g->gcsp = (uintptr)nil;
}

// runtime·exitsyscall slow path on g0.
// Failed to acquire P, enqueue gp as runnable.
static void
exitsyscall0(G *gp)
{
	P *p;

	gp->status = Grunnable;
	gp->m = nil;
	m->curg = nil;
	runtime·lock(&runtime·sched);
	p = pidleget();
	if(p == nil)
		globrunqput(gp);
	runtime·unlock(&runtime·sched);
	if(p) {
		acquirep(p);
		execute(gp);  // Never returns.
	}
	if(m->lockedg) {
		// Wait until another thread schedules gp and so m again.
		stoplockedm();
		execute(gp);  // Never returns.
	}
	stopm();
	schedule();  // Never returns.
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
runtime·newproc(int32 siz, FuncVal* fn, ...)
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
runtime·newproc1(FuncVal *fn, byte *argp, int32 narg, int32 nret, void *callerpc)
{
	byte *sp;
	G *newg;
	int32 siz;

//printf("newproc1 %p %p narg=%d nret=%d\n", fn, argp, narg, nret);
	siz = narg + nret;
	siz = (siz+7) & ~7;

	// We could instead create a secondary stack frame
	// and make it look like goexit was on the original but
	// the call to the actual goroutine function was split.
	// Not worth it: this is almost always an error.
	if(siz > StackMin - 1024)
		runtime·throw("runtime.newproc: function arguments too large for new goroutine");

	if((newg = gfget(m->p)) != nil) {
		if(newg->stackguard - StackGuard != newg->stack0)
			runtime·throw("invalid stack in newg");
	} else {
		newg = runtime·malg(StackMin);
		runtime·lock(&runtime·sched);
		if(runtime·lastg == nil)
			runtime·allg = newg;
		else
			runtime·lastg->alllink = newg;
		runtime·lastg = newg;
		runtime·unlock(&runtime·sched);
	}

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
	newg->fnstart = fn;
	newg->gopc = (uintptr)callerpc;
	newg->status = Grunnable;
	newg->goid = runtime·xadd64(&runtime·sched.goidgen, 1);
	if(raceenabled)
		newg->racectx = runtime·racegostart(callerpc);
	runqput(m->p, newg);

	if(runtime·atomicload(&runtime·sched.npidle) != 0 && runtime·atomicload(&runtime·sched.nmspinning) == 0 && fn->fn != runtime·main)  // TODO: fast atomic
		wakep();
	return newg;
}

// Put on gfree list.
// If local list is too long, transfer a batch to the global list.
static void
gfput(P *p, G *gp)
{
	if(gp->stackguard - StackGuard != gp->stack0)
		runtime·throw("invalid stack in gfput");
	gp->schedlink = p->gfree;
	p->gfree = gp;
	p->gfreecnt++;
	if(p->gfreecnt >= 64) {
		runtime·lock(&runtime·sched.gflock);
		while(p->gfreecnt >= 32) {
			p->gfreecnt--;
			gp = p->gfree;
			p->gfree = gp->schedlink;
			gp->schedlink = runtime·sched.gfree;
			runtime·sched.gfree = gp;
		}
		runtime·unlock(&runtime·sched.gflock);
	}
}

// Get from gfree list.
// If local list is empty, grab a batch from global list.
static G*
gfget(P *p)
{
	G *gp;

retry:
	gp = p->gfree;
	if(gp == nil && runtime·sched.gfree) {
		runtime·lock(&runtime·sched.gflock);
		while(p->gfreecnt < 32 && runtime·sched.gfree) {
			p->gfreecnt++;
			gp = runtime·sched.gfree;
			runtime·sched.gfree = gp->schedlink;
			gp->schedlink = p->gfree;
			p->gfree = gp;
		}
		runtime·unlock(&runtime·sched.gflock);
		goto retry;
	}
	if(gp) {
		p->gfree = gp->schedlink;
		p->gfreecnt--;
	}
	return gp;
}

// Purge all cached G's from gfree list to the global list.
static void
gfpurge(P *p)
{
	G *gp;

	runtime·lock(&runtime·sched.gflock);
	while(p->gfreecnt) {
		p->gfreecnt--;
		gp = p->gfree;
		p->gfree = gp->schedlink;
		gp->schedlink = runtime·sched.gfree;
		runtime·sched.gfree = gp;
	}
	runtime·unlock(&runtime·sched.gflock);
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
// delete when scheduler is even stronger
int32
runtime·gomaxprocsfunc(int32 n)
{
	int32 ret;

	if(n > MaxGomaxprocs)
		n = MaxGomaxprocs;
	runtime·lock(&runtime·sched);
	ret = runtime·gomaxprocs;
	if(n <= 0 || n == ret) {
		runtime·unlock(&runtime·sched);
		return ret;
	}
	runtime·unlock(&runtime·sched);

	runtime·semacquire(&runtime·worldsema);
	m->gcing = 1;
	runtime·stoptheworld();
	newprocs = n;
	m->gcing = 0;
	runtime·semrelease(&runtime·worldsema);
	runtime·starttheworld();

	return ret;
}

static void
LockOSThread(void)
{
	m->lockedg = g;
	g->lockedm = m;
}

void
runtime·LockOSThread(void)
{
	m->locked |= LockExternal;
	LockOSThread();
}

void
runtime·lockOSThread(void)
{
	m->locked += LockInternal;
	LockOSThread();
}

static void
UnlockOSThread(void)
{
	if(m->locked != 0)
		return;
	m->lockedg = nil;
	g->lockedm = nil;
}

void
runtime·UnlockOSThread(void)
{
	m->locked &= ~LockExternal;
	UnlockOSThread();
}

void
runtime·unlockOSThread(void)
{
	if(m->locked < LockInternal)
		runtime·throw("runtime: internal error: misuse of lockOSThread/unlockOSThread");
	m->locked -= LockInternal;
	UnlockOSThread();
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
	ret = runtime·gcount();
	FLUSH(&ret);
}

int32
runtime·gcount(void)
{
	G *gp;
	int32 n, s;

	n = 0;
	runtime·lock(&runtime·sched);
	// TODO(dvyukov): runtime.NumGoroutine() is O(N).
	// We do not want to increment/decrement centralized counter in newproc/goexit,
	// just to make runtime.NumGoroutine() faster.
	// Compromise solution is to introduce per-P counters of active goroutines.
	for(gp = runtime·allg; gp; gp = gp->alllink) {
		s = gp->status;
		if(s == Grunnable || s == Grunning || s == Gsyscall || s == Gwaiting)
			n++;
	}
	runtime·unlock(&runtime·sched);
	return n;
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

	// Windows does profiling in a dedicated thread w/o m.
	if(!Windows && (m == nil || m->mcache == nil))
		return;
	if(prof.fn == nil || prof.hz == 0)
		return;

	runtime·lock(&prof);
	if(prof.fn == nil) {
		runtime·unlock(&prof);
		return;
	}
	n = runtime·gentraceback(pc, sp, lr, gp, 0, prof.pcbuf, nelem(prof.pcbuf), nil, nil);
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

// Change number of processors.  The world is stopped, sched is locked.
static void
procresize(int32 new)
{
	int32 i, old;
	G *gp;
	P *p;

	old = runtime·gomaxprocs;
	if(old < 0 || old > MaxGomaxprocs || new <= 0 || new >MaxGomaxprocs)
		runtime·throw("procresize: invalid arg");
	// initialize new P's
	for(i = 0; i < new; i++) {
		p = runtime·allp[i];
		if(p == nil) {
			p = (P*)runtime·mallocgc(sizeof(*p), 0, 0, 1);
			p->status = Pgcstop;
			runtime·atomicstorep(&runtime·allp[i], p);
		}
		if(p->mcache == nil) {
			if(old==0 && i==0)
				p->mcache = m->mcache;  // bootstrap
			else
				p->mcache = runtime·allocmcache();
		}
		if(p->runq == nil) {
			p->runqsize = 128;
			p->runq = (G**)runtime·mallocgc(p->runqsize*sizeof(G*), 0, 0, 1);
		}
	}

	// redistribute runnable G's evenly
	for(i = 0; i < old; i++) {
		p = runtime·allp[i];
		while(gp = runqget(p))
			globrunqput(gp);
	}
	// start at 1 because current M already executes some G and will acquire allp[0] below,
	// so if we have a spare G we want to put it into allp[1].
	for(i = 1; runtime·sched.runqhead; i++) {
		gp = runtime·sched.runqhead;
		runtime·sched.runqhead = gp->schedlink;
		runqput(runtime·allp[i%new], gp);
	}
	runtime·sched.runqtail = nil;
	runtime·sched.runqsize = 0;

	// free unused P's
	for(i = new; i < old; i++) {
		p = runtime·allp[i];
		runtime·freemcache(p->mcache);
		p->mcache = nil;
		gfpurge(p);
		p->status = Pdead;
		// can't free P itself because it can be referenced by an M in syscall
	}

	if(m->p)
		m->p->m = nil;
	m->p = nil;
	m->mcache = nil;
	p = runtime·allp[0];
	p->m = nil;
	p->status = Pidle;
	acquirep(p);
	for(i = new-1; i > 0; i--) {
		p = runtime·allp[i];
		p->status = Pidle;
		pidleput(p);
	}
	runtime·singleproc = new == 1;
	runtime·atomicstore((uint32*)&runtime·gomaxprocs, new);
}

// Associate p and the current m.
static void
acquirep(P *p)
{
	if(m->p || m->mcache)
		runtime·throw("acquirep: already in go");
	if(p->m || p->status != Pidle) {
		runtime·printf("acquirep: p->m=%p(%d) p->status=%d\n", p->m, p->m ? p->m->id : 0, p->status);
		runtime·throw("acquirep: invalid p state");
	}
	m->mcache = p->mcache;
	m->p = p;
	p->m = m;
	p->status = Prunning;
}

// Disassociate p and the current m.
static P*
releasep(void)
{
	P *p;

	if(m->p == nil || m->mcache == nil)
		runtime·throw("releasep: invalid arg");
	p = m->p;
	if(p->m != m || p->mcache != m->mcache || p->status != Prunning) {
		runtime·printf("releasep: m=%p m->p=%p p->m=%p m->mcache=%p p->mcache=%p p->status=%d\n",
			m, m->p, p->m, m->mcache, p->mcache, p->status);
		runtime·throw("releasep: invalid p state");
	}
	m->p = nil;
	m->mcache = nil;
	p->m = nil;
	p->status = Pidle;
	return p;
}

static void
inclocked(int32 v)
{
	runtime·lock(&runtime·sched);
	runtime·sched.mlocked += v;
	if(v > 0)
		checkdead();
	runtime·unlock(&runtime·sched);
}

// Check for deadlock situation.
// The check is based on number of running M's, if 0 -> deadlock.
static void
checkdead(void)
{
	G *gp;
	int32 run, grunning, s;

	// -1 for sysmon
	run = runtime·sched.mcount - runtime·sched.nmidle - runtime·sched.mlocked - 1;
	if(run > 0)
		return;
	if(run < 0) {
		runtime·printf("checkdead: nmidle=%d mlocked=%d mcount=%d\n",
			runtime·sched.nmidle, runtime·sched.mlocked, runtime·sched.mcount);
		runtime·throw("checkdead: inconsistent counts");
	}
	grunning = 0;
	for(gp = runtime·allg; gp; gp = gp->alllink) {
		if(gp->isbackground)
			continue;
		s = gp->status;
		if(s == Gwaiting)
			grunning++;
		else if(s == Grunnable || s == Grunning || s == Gsyscall) {
			runtime·printf("checkdead: find g %D in status %d\n", gp->goid, s);
			runtime·throw("checkdead: runnable g");
		}
	}
	if(grunning == 0)  // possible if main goroutine calls runtime·Goexit()
		runtime·exit(0);
	m->throwing = -1;  // do not dump full stacks
	runtime·throw("all goroutines are asleep - deadlock!");
}

static void
sysmon(void)
{
	uint32 idle, delay;
	int64 now, lastpoll;
	G *gp;
	uint32 ticks[MaxGomaxprocs];

	idle = 0;  // how many cycles in succession we had not wokeup somebody
	delay = 0;
	for(;;) {
		if(idle == 0)  // start with 20us sleep...
			delay = 20;
		else if(idle > 50)  // start doubling the sleep after 1ms...
			delay *= 2;
		if(delay > 10*1000)  // up to 10ms
			delay = 10*1000;
		runtime·usleep(delay);
		if(runtime·gcwaiting || runtime·atomicload(&runtime·sched.npidle) == runtime·gomaxprocs) {  // TODO: fast atomic
			runtime·lock(&runtime·sched);
			if(runtime·atomicload(&runtime·gcwaiting) || runtime·atomicload(&runtime·sched.npidle) == runtime·gomaxprocs) {
				runtime·atomicstore(&runtime·sched.sysmonwait, 1);
				runtime·unlock(&runtime·sched);
				runtime·notesleep(&runtime·sched.sysmonnote);
				runtime·noteclear(&runtime·sched.sysmonnote);
				idle = 0;
				delay = 20;
			} else
				runtime·unlock(&runtime·sched);
		}
		// poll network if not polled for more than 10ms
		lastpoll = runtime·atomicload64(&runtime·sched.lastpoll);
		now = runtime·nanotime();
		if(lastpoll != 0 && lastpoll + 10*1000*1000 > now) {
			gp = runtime·netpoll(false);  // non-blocking
			injectglist(gp);
		}
		// retake P's blocked in syscalls
		if(retake(ticks))
			idle = 0;
		else
			idle++;
	}
}

static uint32
retake(uint32 *ticks)
{
	uint32 i, s, n;
	int64 t;
	P *p;

	n = 0;
	for(i = 0; i < runtime·gomaxprocs; i++) {
		p = runtime·allp[i];
		if(p==nil)
			continue;
		t = p->tick;
		if(ticks[i] != t) {
			ticks[i] = t;
			continue;
		}
		s = p->status;
		if(s != Psyscall)
			continue;
		if(p->runqhead == p->runqtail && runtime·atomicload(&runtime·sched.nmspinning) + runtime·atomicload(&runtime·sched.npidle) > 0)  // TODO: fast atomic
			continue;
		// Need to increment number of locked M's before the CAS.
		// Otherwise the M from which we retake can exit the syscall,
		// increment nmidle and report deadlock.
		inclocked(-1);
		if(runtime·cas(&p->status, s, Pidle)) {
			n++;
			handoffp(p);
		}
		inclocked(1);
	}
	return n;
}

// Put mp on midle list.
// Sched must be locked.
static void
mput(M *mp)
{
	mp->schedlink = runtime·sched.midle;
	runtime·sched.midle = mp;
	runtime·sched.nmidle++;
	checkdead();
}

// Try to get an m from midle list.
// Sched must be locked.
static M*
mget(void)
{
	M *mp;

	if((mp = runtime·sched.midle) != nil){
		runtime·sched.midle = mp->schedlink;
		runtime·sched.nmidle--;
	}
	return mp;
}

// Put gp on the global runnable queue.
// Sched must be locked.
static void
globrunqput(G *gp)
{
	gp->schedlink = nil;
	if(runtime·sched.runqtail)
		runtime·sched.runqtail->schedlink = gp;
	else
		runtime·sched.runqhead = gp;
	runtime·sched.runqtail = gp;
	runtime·sched.runqsize++;
}

// Try get a batch of G's from the global runnable queue.
// Sched must be locked.
static G*
globrunqget(P *p)
{
	G *gp, *gp1;
	int32 n;

	if(runtime·sched.runqsize == 0)
		return nil;
	n = runtime·sched.runqsize/runtime·gomaxprocs+1;
	if(n > runtime·sched.runqsize)
		n = runtime·sched.runqsize;
	runtime·sched.runqsize -= n;
	if(runtime·sched.runqsize == 0)
		runtime·sched.runqtail = nil;
	gp = runtime·sched.runqhead;
	runtime·sched.runqhead = gp->schedlink;
	n--;
	while(n--) {
		gp1 = runtime·sched.runqhead;
		runtime·sched.runqhead = gp1->schedlink;
		runqput(p, gp1);
	}
	return gp;
}

// Put p to on pidle list.
// Sched must be locked.
static void
pidleput(P *p)
{
	p->link = runtime·sched.pidle;
	runtime·sched.pidle = p;
	runtime·xadd(&runtime·sched.npidle, 1);  // TODO: fast atomic
}

// Try get a p from pidle list.
// Sched must be locked.
static P*
pidleget(void)
{
	P *p;

	p = runtime·sched.pidle;
	if(p) {
		runtime·sched.pidle = p->link;
		runtime·xadd(&runtime·sched.npidle, -1);  // TODO: fast atomic
	}
	return p;
}

// Put g on local runnable queue.
// TODO(dvyukov): consider using lock-free queue.
static void
runqput(P *p, G *gp)
{
	int32 h, t, s;

	runtime·lock(p);
retry:
	h = p->runqhead;
	t = p->runqtail;
	s = p->runqsize;
	if(t == h-1 || (h == 0 && t == s-1)) {
		runqgrow(p);
		goto retry;
	}
	p->runq[t++] = gp;
	if(t == s)
		t = 0;
	p->runqtail = t;
	runtime·unlock(p);
}

// Get g from local runnable queue.
static G*
runqget(P *p)
{
	G *gp;
	int32 t, h, s;

	if(p->runqhead == p->runqtail)
		return nil;
	runtime·lock(p);
	h = p->runqhead;
	t = p->runqtail;
	s = p->runqsize;
	if(t == h) {
		runtime·unlock(p);
		return nil;
	}
	gp = p->runq[h++];
	if(h == s)
		h = 0;
	p->runqhead = h;
	runtime·unlock(p);
	return gp;
}

// Grow local runnable queue.
// TODO(dvyukov): consider using fixed-size array
// and transfer excess to the global list (local queue can grow way too big).
static void
runqgrow(P *p)
{
	G **q;
	int32 s, t, h, t2;

	h = p->runqhead;
	t = p->runqtail;
	s = p->runqsize;
	t2 = 0;
	q = runtime·malloc(2*s*sizeof(*q));
	while(t != h) {
		q[t2++] = p->runq[h++];
		if(h == s)
			h = 0;
	}
	runtime·free(p->runq);
	p->runq = q;
	p->runqhead = 0;
	p->runqtail = t2;
	p->runqsize = 2*s;
}

// Steal half of elements from local runnable queue of p2
// and put onto local runnable queue of p.
// Returns one of the stolen elements (or nil if failed).
static G*
runqsteal(P *p, P *p2)
{
	G *gp, *gp1;
	int32 t, h, s, t2, h2, s2, c, i;

	if(p2->runqhead == p2->runqtail)
		return nil;
	// sort locks to prevent deadlocks
	if(p < p2)
		runtime·lock(p);
	runtime·lock(p2);
	if(p2->runqhead == p2->runqtail) {
		runtime·unlock(p2);
		if(p < p2)
			runtime·unlock(p);
		return nil;
	}
	if(p >= p2)
		runtime·lock(p);
	// now we've locked both queues and know the victim is not empty
	h = p->runqhead;
	t = p->runqtail;
	s = p->runqsize;
	h2 = p2->runqhead;
	t2 = p2->runqtail;
	s2 = p2->runqsize;
	gp = p2->runq[h2++];  // return value
	if(h2 == s2)
		h2 = 0;
	// steal roughly half
	if(t2 > h2)
		c = (t2 - h2) / 2;
	else
		c = (s2 - h2 + t2) / 2;
	// copy
	for(i = 0; i != c; i++) {
		// the target queue is full?
		if(t == h-1 || (h == 0 && t == s-1))
			break;
		// the victim queue is empty?
		if(t2 == h2)
			break;
		gp1 = p2->runq[h2++];
		if(h2 == s2)
			h2 = 0;
		p->runq[t++] = gp1;
		if(t == s)
			t = 0;
	}
	p->runqtail = t;
	p2->runqhead = h2;
	runtime·unlock(p2);
	runtime·unlock(p);
	return gp;
}

void
runtime·testSchedLocalQueue(void)
{
	P p;
	G gs[1000];
	int32 i, j;

	runtime·memclr((byte*)&p, sizeof(p));
	p.runqsize = 1;
	p.runqhead = 0;
	p.runqtail = 0;
	p.runq = runtime·malloc(p.runqsize*sizeof(*p.runq));

	for(i = 0; i < nelem(gs); i++) {
		if(runqget(&p) != nil)
			runtime·throw("runq is not empty initially");
		for(j = 0; j < i; j++)
			runqput(&p, &gs[i]);
		for(j = 0; j < i; j++) {
			if(runqget(&p) != &gs[i]) {
				runtime·printf("bad element at iter %d/%d\n", i, j);
				runtime·throw("bad element");
			}
		}
		if(runqget(&p) != nil)
			runtime·throw("runq is not empty afterwards");
	}
}

void
runtime·testSchedLocalQueueSteal(void)
{
	P p1, p2;
	G gs[1000], *gp;
	int32 i, j, s;

	runtime·memclr((byte*)&p1, sizeof(p1));
	p1.runqsize = 1;
	p1.runqhead = 0;
	p1.runqtail = 0;
	p1.runq = runtime·malloc(p1.runqsize*sizeof(*p1.runq));

	runtime·memclr((byte*)&p2, sizeof(p2));
	p2.runqsize = nelem(gs);
	p2.runqhead = 0;
	p2.runqtail = 0;
	p2.runq = runtime·malloc(p2.runqsize*sizeof(*p2.runq));

	for(i = 0; i < nelem(gs); i++) {
		for(j = 0; j < i; j++) {
			gs[j].sig = 0;
			runqput(&p1, &gs[j]);
		}
		gp = runqsteal(&p2, &p1);
		s = 0;
		if(gp) {
			s++;
			gp->sig++;
		}
		while(gp = runqget(&p2)) {
			s++;
			gp->sig++;
		}
		while(gp = runqget(&p1))
			gp->sig++;
		for(j = 0; j < i; j++) {
			if(gs[j].sig != 1) {
				runtime·printf("bad element %d(%d) at iter %d\n", j, gs[j].sig, i);
				runtime·throw("bad element");
			}
		}
		if(s != i/2 && s != i/2+1) {
			runtime·printf("bad steal %d, want %d or %d, iter %d\n",
				s, i/2, i/2+1, i);
			runtime·throw("bad steal");
		}
	}
}

