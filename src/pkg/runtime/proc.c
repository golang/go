// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "zaexperiment.h"
#include "malloc.h"
#include "stack.h"
#include "race.h"
#include "type.h"
#include "../../cmd/ld/textflag.h"

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
	int32	nmidlelocked; // number of locked m's waiting for work
	int32	mcount;	 // number of m's that have been created
	int32	maxmcount;	// maximum number of m's allowed (or die)

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

	uint32	gcwaiting;	// gc is waiting to run
	int32	stopwait;
	Note	stopnote;
	uint32	sysmonwait;
	Note	sysmonnote;
	uint64	lastpoll;

	int32	profilehz;	// cpu profiling rate
};

enum
{
	// The max value of GOMAXPROCS.
	// There are no fundamental restrictions on the value.
	MaxGomaxprocs = 1<<8,

	// Number of goroutine ids to grab from runtime·sched.goidgen to local per-P cache at once.
	// 16 seems to provide enough amortization, but other than that it's mostly arbitrary number.
	GoidCacheBatch = 16,
};

Sched	runtime·sched;
int32	runtime·gomaxprocs;
uint32	runtime·needextram;
bool	runtime·iscgo;
M	runtime·m0;
G	runtime·g0;	// idle goroutine for m0
G*	runtime·lastg;
M*	runtime·allm;
M*	runtime·extram;
int8*	runtime·goos;
int32	runtime·ncpu;
static int32	newprocs;

static	Lock allglock;	// the following vars are protected by this lock or by stoptheworld
G**	runtime·allg;
uintptr runtime·allglen;
static	uintptr allgcap;

void runtime·mstart(void);
static void runqput(P*, G*);
static G* runqget(P*);
static bool runqputslow(P*, G*, uint32, uint32);
static G* runqsteal(P*, P*);
static void mput(M*);
static M* mget(void);
static void mcommoninit(M*);
static void schedule(void);
static void procresize(int32);
static void acquirep(P*);
static P* releasep(void);
static void newm(void(*)(void), P*);
static void stopm(void);
static void startm(P*, bool);
static void handoffp(P*);
static void wakep(void);
static void stoplockedm(void);
static void startlockedm(G*);
static void sysmon(void);
static uint32 retake(int64);
static void incidlelocked(int32);
static void checkdead(void);
static void exitsyscall0(G*);
static void park0(G*);
static void goexit0(G*);
static void gfput(P*, G*);
static G* gfget(P*);
static void gfpurge(P*);
static void globrunqput(G*);
static void globrunqputbatch(G*, G*, int32);
static G* globrunqget(P*, int32);
static P* pidleget(void);
static void pidleput(P*);
static void injectglist(G*);
static bool preemptall(void);
static bool preemptone(P*);
static bool exitsyscallfast(void);
static bool haveexperiment(int8*);
static void allgadd(G*);

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
	Eface i;

	runtime·sched.maxmcount = 10000;
	runtime·precisestack = haveexperiment("precisestack");

	runtime·mallocinit();
	mcommoninit(m);
	
	// Initialize the itable value for newErrorCString,
	// so that the next time it gets called, possibly
	// in a fault during a garbage collection, it will not
	// need to allocated memory.
	runtime·newErrorCString(0, &i);

	runtime·goargs();
	runtime·goenvs();
	runtime·parsedebugvars();

	// Allocate internal symbol table representation now, we need it for GC anyway.
	runtime·symtabinit();

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

	if(raceenabled)
		g->racectx = runtime·raceinit();
}

extern void main·init(void);
extern void main·main(void);

static FuncVal scavenger = {runtime·MHeap_Scavenger};

static FuncVal initDone = { runtime·unlockOSThread };

// The main goroutine.
void
runtime·main(void)
{
	Defer d;
	
	// Max stack size is 1 GB on 64-bit, 250 MB on 32-bit.
	// Using decimal instead of binary GB and MB because
	// they look nicer in the stack overflow failure message.
	if(sizeof(void*) == 8)
		runtime·maxstacksize = 1000000000;
	else
		runtime·maxstacksize = 250000000;

	newm(sysmon, nil);

	// Lock the main goroutine onto this, the main OS thread,
	// during initialization.  Most programs won't care, but a few
	// do require certain calls to be made by the main thread.
	// Those can arrange for main.main to run in the main thread
	// by calling runtime.LockOSThread during initialization
	// to preserve the lock.
	runtime·lockOSThread();
	
	// Defer unlock so that runtime.Goexit during init does the unlock too.
	d.fn = &initDone;
	d.siz = 0;
	d.link = g->defer;
	d.argp = (void*)-1;
	d.special = true;
	g->defer = &d;

	if(m != &runtime·m0)
		runtime·throw("runtime·main not on m0");
	runtime·newproc1(&scavenger, nil, 0, 0, runtime·main);
	main·init();

	if(g->defer != &d || d.fn != &initDone)
		runtime·throw("runtime: bad defer entry after init");
	g->defer = d.link;
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
	int64 waitfor;

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

	// approx time the G is blocked, in minutes
	waitfor = 0;
	if((gp->status == Gwaiting || gp->status == Gsyscall) && gp->waitsince != 0)
		waitfor = (runtime·nanotime() - gp->waitsince) / (60LL*1000*1000*1000);

	if(waitfor < 1)
		runtime·printf("goroutine %D [%s]:\n", gp->goid, status);
	else
		runtime·printf("goroutine %D [%s, %D minutes]:\n", gp->goid, status, waitfor);
}

void
runtime·tracebackothers(G *me)
{
	G *gp;
	int32 traceback;
	uintptr i;

	traceback = runtime·gotraceback(nil);
	
	// Show the current goroutine first, if we haven't already.
	if((gp = m->curg) != nil && gp != me) {
		runtime·printf("\n");
		runtime·goroutineheader(gp);
		runtime·traceback(~(uintptr)0, ~(uintptr)0, 0, gp);
	}

	runtime·lock(&allglock);
	for(i = 0; i < runtime·allglen; i++) {
		gp = runtime·allg[i];
		if(gp == me || gp == m->curg || gp->status == Gdead)
			continue;
		if(gp->issystem && traceback < 2)
			continue;
		runtime·printf("\n");
		runtime·goroutineheader(gp);
		if(gp->status == Grunning) {
			runtime·printf("\tgoroutine running on other thread; stack unavailable\n");
			runtime·printcreatedby(gp);
		} else
			runtime·traceback(~(uintptr)0, ~(uintptr)0, 0, gp);
	}
	runtime·unlock(&allglock);
}

static void
checkmcount(void)
{
	// sched lock is held
	if(runtime·sched.mcount > runtime·sched.maxmcount) {
		runtime·printf("runtime: program exceeds %d-thread limit\n", runtime·sched.maxmcount);
		runtime·throw("thread exhaustion");
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
	checkmcount();
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
	m->locks++;  // disable preemption because it can be holding p in a local var
	if(gp->status != Gwaiting) {
		runtime·printf("goroutine %D has status %d\n", gp->goid, gp->status);
		runtime·throw("bad g->status in ready");
	}
	gp->status = Grunnable;
	runqput(m->p, gp);
	if(runtime·atomicload(&runtime·sched.npidle) != 0 && runtime·atomicload(&runtime·sched.nmspinning) == 0)  // TODO: fast atomic
		wakep();
	m->locks--;
	if(m->locks == 0 && g->preempt)  // restore the preemption request in case we've cleared it in newstack
		g->stackguard0 = StackPreempt;
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

// Similar to stoptheworld but best-effort and can be called several times.
// There is no reverse operation, used during crashing.
// This function must not lock any mutexes.
void
runtime·freezetheworld(void)
{
	int32 i;

	if(runtime·gomaxprocs == 1)
		return;
	// stopwait and preemption requests can be lost
	// due to races with concurrently executing threads,
	// so try several times
	for(i = 0; i < 5; i++) {
		// this should tell the scheduler to not start any new goroutines
		runtime·sched.stopwait = 0x7fffffff;
		runtime·atomicstore((uint32*)&runtime·sched.gcwaiting, 1);
		// this should stop running goroutines
		if(!preemptall())
			break;  // no running goroutines
		runtime·usleep(1000);
	}
	// to be sure
	runtime·usleep(1000);
	preemptall();
	runtime·usleep(1000);
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
	runtime·atomicstore((uint32*)&runtime·sched.gcwaiting, 1);
	preemptall();
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

	// wait for remaining P's to stop voluntarily
	if(wait) {
		for(;;) {
			// wait for 100us, then try to re-preempt in case of any races
			if(runtime·notetsleep(&runtime·sched.stopnote, 100*1000)) {
				runtime·noteclear(&runtime·sched.stopnote);
				break;
			}
			preemptall();
		}
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

	m->locks++;  // disable preemption because it can be holding p in a local var
	gp = runtime·netpoll(false);  // non-blocking
	injectglist(gp);
	add = needaddgcproc();
	runtime·lock(&runtime·sched);
	if(newprocs) {
		procresize(newprocs);
		newprocs = 0;
	} else
		procresize(runtime·gomaxprocs);
	runtime·sched.gcwaiting = 0;

	p1 = nil;
	while(p = pidleget()) {
		// procresize() puts p's with work at the beginning of the list.
		// Once we reach a p without a run queue, the rest don't have one either.
		if(p->runqhead == p->runqtail) {
			pidleput(p);
			break;
		}
		p->m = mget();
		p->link = p1;
		p1 = p;
	}
	if(runtime·sched.sysmonwait) {
		runtime·sched.sysmonwait = false;
		runtime·notewakeup(&runtime·sched.sysmonnote);
	}
	runtime·unlock(&runtime·sched);

	while(p1) {
		p = p1;
		p1 = p1->link;
		if(p->m) {
			mp = p->m;
			p->m = nil;
			if(mp->nextp)
				runtime·throw("starttheworld: inconsistent mp->nextp");
			mp->nextp = p;
			runtime·notewakeup(&mp->park);
		} else {
			// Start M to run P.  Do not start another M below.
			newm(nil, p);
			add = false;
		}
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
	m->locks--;
	if(m->locks == 0 && g->preempt)  // restore the preemption request in case we've cleared it in newstack
		g->stackguard0 = StackPreempt;
}

// Called to start an M.
void
runtime·mstart(void)
{
#ifdef GOOS_windows
#ifdef GOARCH_386
	// It is used by windows-386 only. Unfortunately, seh needs
	// to be located on os stack, and mstart runs on os stack
	// for both m0 and m.
	SEH seh;
#endif
#endif

	if(g != m->g0)
		runtime·throw("bad runtime·mstart");

	// Record top of stack for use by mcall.
	// Once we call schedule we're never coming back,
	// so other calls can reuse this stack space.
	runtime·gosave(&m->g0->sched);
	m->g0->sched.pc = (uintptr)-1;  // make sure it is never used
	m->g0->stackguard = m->g0->stackguard0;  // cgo sets only stackguard0, copy it to stackguard
#ifdef GOOS_windows
#ifdef GOARCH_386
	m->seh = &seh;
#endif
#endif
	runtime·asminit();
	runtime·minit();

	// Install signal handlers; after minit so that minit can
	// prepare the thread to be able to handle the signals.
	if(m == &runtime·m0)
		runtime·initsig();
	
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
	uintptr *tls;
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

	// In case of cgo or Solaris, pthread_create will make us a stack.
	// Windows will layout sched stack on OS stack.
	if(runtime·iscgo || Solaris || Windows)
		mp->g0 = runtime·malg(-1);
	else
		mp->g0 = runtime·malg(8192);

	if(p == m->p)
		releasep();
	m->locks--;
	if(m->locks == 0 && g->preempt)  // restore the preemption request in case we've cleared it in newstack
		g->stackguard0 = StackPreempt;

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
#pragma textflag NOSPLIT
void
runtime·needm(byte x)
{
	M *mp;

	if(runtime·needextram) {
		// Can happen if C/C++ code calls Go from a global ctor.
		// Can not throw, because scheduler is not initialized yet.
		runtime·write(2, "fatal error: cgo callback before cgo call\n",
			sizeof("fatal error: cgo callback before cgo call\n")-1);
		runtime·exit(1);
	}

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
	g->stackguard0 = g->stackguard;

#ifdef GOOS_windows
#ifdef GOARCH_386
	// On windows/386, we need to put an SEH frame (two words)
	// somewhere on the current stack. We are called from cgocallback_gofunc
	// and we know that it will leave two unused words below m->curg->sched.sp.
	// Use those.
	m->seh = (SEH*)((uintptr*)&x + 1);
#endif
#endif

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
	gp->sched.pc = (uintptr)runtime·goexit;
	gp->sched.sp = gp->stackbase;
	gp->sched.lr = 0;
	gp->sched.g = gp;
	gp->syscallpc = gp->sched.pc;
	gp->syscallsp = gp->sched.sp;
	gp->syscallstack = gp->stackbase;
	gp->syscallguard = gp->stackguard;
	gp->status = Gsyscall;
	mp->curg = gp;
	mp->locked = LockInternal;
	mp->lockedg = gp;
	gp->lockedm = mp;
	gp->goid = runtime·xadd64(&runtime·sched.goidgen, 1);
	if(raceenabled)
		gp->racectx = runtime·racegostart(runtime·newextram);
	// put on allg for garbage collector
	allgadd(gp);

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

#ifdef GOOS_windows
#ifdef GOARCH_386
	m->seh = nil;  // reset dangling typed pointer
#endif
#endif

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
#pragma textflag NOSPLIT
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

#pragma textflag NOSPLIT
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
		ts.tls = mp->tls;
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
// If p==nil, tries to get an idle P, if no idle P's does nothing.
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
	if(runtime·sched.gcwaiting) {
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
	incidlelocked(1);
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
	incidlelocked(-1);
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

	if(!runtime·sched.gcwaiting)
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
	gp->waitsince = 0;
	gp->preempt = false;
	gp->stackguard0 = gp->stackguard;
	m->p->schedtick++;
	m->curg = gp;
	gp->m = m;

	// Check whether the profiler needs to be turned on or off.
	hz = runtime·sched.profilehz;
	if(m->profilehz != hz)
		runtime·resetcpuprofiler(hz);

	runtime·gogo(&gp->sched);
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
	if(runtime·sched.gcwaiting) {
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
		gp = globrunqget(m->p, 0);
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
		if(runtime·sched.gcwaiting)
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
	if(runtime·sched.gcwaiting) {
		runtime·unlock(&runtime·sched);
		goto top;
	}
	if(runtime·sched.runqsize) {
		gp = globrunqget(m->p, 0);
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

static void
resetspinning(void)
{
	int32 nmspinning;

	if(m->spinning) {
		m->spinning = false;
		nmspinning = runtime·xadd(&runtime·sched.nmspinning, -1);
		if(nmspinning < 0)
			runtime·throw("findrunnable: negative nmspinning");
	} else
		nmspinning = runtime·atomicload(&runtime·sched.nmspinning);

	// M wakeup policy is deliberately somewhat conservative (see nmspinning handling),
	// so see if we need to wakeup another P here.
	if (nmspinning == 0 && runtime·atomicload(&runtime·sched.npidle) > 0)
		wakep();
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
	uint32 tick;

	if(m->locks)
		runtime·throw("schedule: holding locks");

top:
	if(runtime·sched.gcwaiting) {
		gcstopm();
		goto top;
	}

	gp = nil;
	// Check the global runnable queue once in a while to ensure fairness.
	// Otherwise two goroutines can completely occupy the local runqueue
	// by constantly respawning each other.
	tick = m->p->schedtick;
	// This is a fancy way to say tick%61==0,
	// it uses 2 MUL instructions instead of a single DIV and so is faster on modern processors.
	if(tick - (((uint64)tick*0x4325c53fu)>>36)*61 == 0 && runtime·sched.runqsize > 0) {
		runtime·lock(&runtime·sched);
		gp = globrunqget(m->p, 1);
		runtime·unlock(&runtime·sched);
		if(gp)
			resetspinning();
	}
	if(gp == nil) {
		gp = runqget(m->p);
		if(gp && m->spinning)
			runtime·throw("schedule: spinning with local work");
	}
	if(gp == nil) {
		gp = findrunnable();  // blocks until work is available
		resetspinning();
	}

	if(gp->lockedm) {
		// Hands off own p to the locked m,
		// then blocks waiting for a new p.
		startlockedm(gp);
		goto top;
	}

	execute(gp);
}

// Puts the current goroutine into a waiting state and calls unlockf.
// If unlockf returns false, the goroutine is resumed.
void
runtime·park(bool(*unlockf)(G*, void*), void *lock, int8 *reason)
{
	m->waitlock = lock;
	m->waitunlockf = unlockf;
	g->waitreason = reason;
	runtime·mcall(park0);
}

static bool
parkunlock(G *gp, void *lock)
{
	USED(gp);
	runtime·unlock(lock);
	return true;
}

// Puts the current goroutine into a waiting state and unlocks the lock.
// The goroutine can be made runnable again by calling runtime·ready(gp).
void
runtime·parkunlock(Lock *lock, int8 *reason)
{
	runtime·park(parkunlock, lock, reason);
}

// runtime·park continuation on g0.
static void
park0(G *gp)
{
	bool ok;

	gp->status = Gwaiting;
	gp->m = nil;
	m->curg = nil;
	if(m->waitunlockf) {
		ok = m->waitunlockf(gp, m->waitlock);
		m->waitunlockf = nil;
		m->waitlock = nil;
		if(!ok) {
			gp->status = Grunnable;
			execute(gp);  // Schedule it back, never returns.
		}
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
	runtime·mcall(runtime·gosched0);
}

// runtime·gosched continuation on g0.
void
runtime·gosched0(G *gp)
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
// Need to mark it as nosplit, because it runs with sp > stackbase (as runtime·lessstack).
// Since it does not return it does not matter.  But if it is preempted
// at the split stack check, GC will complain about inconsistent sp.
#pragma textflag NOSPLIT
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
		runtime·printf("invalid m->locked = %d\n", m->locked);
		runtime·throw("internal lockOSThread error");
	}	
	m->locked = 0;
	runtime·unwindstack(gp, nil);
	gfput(m->p, gp);
	schedule();
}

#pragma textflag NOSPLIT
static void
save(void *pc, uintptr sp)
{
	g->sched.pc = (uintptr)pc;
	g->sched.sp = sp;
	g->sched.lr = 0;
	g->sched.ret = 0;
	g->sched.ctxt = 0;
	g->sched.g = g;
}

// The goroutine g is about to enter a system call.
// Record that it's not using the cpu anymore.
// This is called only from the go syscall library and cgocall,
// not from the low-level system calls used by the runtime.
//
// Entersyscall cannot split the stack: the runtime·gosave must
// make g->sched refer to the caller's stack segment, because
// entersyscall is going to return immediately after.
#pragma textflag NOSPLIT
void
·entersyscall(int32 dummy)
{
	// Disable preemption because during this function g is in Gsyscall status,
	// but can have inconsistent g->sched, do not let GC observe it.
	m->locks++;

	// Leave SP around for GC and traceback.
	save(runtime·getcallerpc(&dummy), runtime·getcallersp(&dummy));
	g->syscallsp = g->sched.sp;
	g->syscallpc = g->sched.pc;
	g->syscallstack = g->stackbase;
	g->syscallguard = g->stackguard;
	g->status = Gsyscall;
	if(g->syscallsp < g->syscallguard-StackGuard || g->syscallstack < g->syscallsp) {
		// runtime·printf("entersyscall inconsistent %p [%p,%p]\n",
		//	g->syscallsp, g->syscallguard-StackGuard, g->syscallstack);
		runtime·throw("entersyscall");
	}

	if(runtime·atomicload(&runtime·sched.sysmonwait)) {  // TODO: fast atomic
		runtime·lock(&runtime·sched);
		if(runtime·atomicload(&runtime·sched.sysmonwait)) {
			runtime·atomicstore(&runtime·sched.sysmonwait, 0);
			runtime·notewakeup(&runtime·sched.sysmonnote);
		}
		runtime·unlock(&runtime·sched);
		save(runtime·getcallerpc(&dummy), runtime·getcallersp(&dummy));
	}

	m->mcache = nil;
	m->p->m = nil;
	runtime·atomicstore(&m->p->status, Psyscall);
	if(runtime·sched.gcwaiting) {
		runtime·lock(&runtime·sched);
		if (runtime·sched.stopwait > 0 && runtime·cas(&m->p->status, Psyscall, Pgcstop)) {
			if(--runtime·sched.stopwait == 0)
				runtime·notewakeup(&runtime·sched.stopnote);
		}
		runtime·unlock(&runtime·sched);
		save(runtime·getcallerpc(&dummy), runtime·getcallersp(&dummy));
	}

	// Goroutines must not split stacks in Gsyscall status (it would corrupt g->sched).
	// We set stackguard to StackPreempt so that first split stack check calls morestack.
	// Morestack detects this case and throws.
	g->stackguard0 = StackPreempt;
	m->locks--;
}

// The same as runtime·entersyscall(), but with a hint that the syscall is blocking.
#pragma textflag NOSPLIT
void
·entersyscallblock(int32 dummy)
{
	P *p;

	m->locks++;  // see comment in entersyscall

	// Leave SP around for GC and traceback.
	save(runtime·getcallerpc(&dummy), runtime·getcallersp(&dummy));
	g->syscallsp = g->sched.sp;
	g->syscallpc = g->sched.pc;
	g->syscallstack = g->stackbase;
	g->syscallguard = g->stackguard;
	g->status = Gsyscall;
	if(g->syscallsp < g->syscallguard-StackGuard || g->syscallstack < g->syscallsp) {
		// runtime·printf("entersyscall inconsistent %p [%p,%p]\n",
		//	g->syscallsp, g->syscallguard-StackGuard, g->syscallstack);
		runtime·throw("entersyscallblock");
	}

	p = releasep();
	handoffp(p);
	if(g->isbackground)  // do not consider blocked scavenger for deadlock detection
		incidlelocked(1);

	// Resave for traceback during blocked call.
	save(runtime·getcallerpc(&dummy), runtime·getcallersp(&dummy));

	g->stackguard0 = StackPreempt;  // see comment in entersyscall
	m->locks--;
}

// The goroutine g exited its system call.
// Arrange for it to run on a cpu again.
// This is called only from the go syscall library, not
// from the low-level system calls used by the runtime.
#pragma textflag NOSPLIT
void
runtime·exitsyscall(void)
{
	m->locks++;  // see comment in entersyscall

	if(g->isbackground)  // do not consider blocked scavenger for deadlock detection
		incidlelocked(-1);

	g->waitsince = 0;
	if(exitsyscallfast()) {
		// There's a cpu for us, so we can run.
		m->p->syscalltick++;
		g->status = Grunning;
		// Garbage collector isn't running (since we are),
		// so okay to clear gcstack and gcsp.
		g->syscallstack = (uintptr)nil;
		g->syscallsp = (uintptr)nil;
		m->locks--;
		if(g->preempt) {
			// restore the preemption request in case we've cleared it in newstack
			g->stackguard0 = StackPreempt;
		} else {
			// otherwise restore the real stackguard, we've spoiled it in entersyscall/entersyscallblock
			g->stackguard0 = g->stackguard;
		}
		return;
	}

	m->locks--;

	// Call the scheduler.
	runtime·mcall(exitsyscall0);

	// Scheduler returned, so we're allowed to run now.
	// Delete the gcstack information that we left for
	// the garbage collector during the system call.
	// Must wait until now because until gosched returns
	// we don't know for sure that the garbage collector
	// is not running.
	g->syscallstack = (uintptr)nil;
	g->syscallsp = (uintptr)nil;
	m->p->syscalltick++;
}

#pragma textflag NOSPLIT
static bool
exitsyscallfast(void)
{
	P *p;

	// Freezetheworld sets stopwait but does not retake P's.
	if(runtime·sched.stopwait) {
		m->p = nil;
		return false;
	}

	// Try to re-acquire the last P.
	if(m->p && m->p->status == Psyscall && runtime·cas(&m->p->status, Psyscall, Prunning)) {
		// There's a cpu for us, so we can run.
		m->mcache = m->p->mcache;
		m->p->m = m;
		return true;
	}
	// Try to get any other idle P.
	m->p = nil;
	if(runtime·sched.pidle) {
		runtime·lock(&runtime·sched);
		p = pidleget();
		if(p && runtime·atomicload(&runtime·sched.sysmonwait)) {
			runtime·atomicstore(&runtime·sched.sysmonwait, 0);
			runtime·notewakeup(&runtime·sched.sysmonnote);
		}
		runtime·unlock(&runtime·sched);
		if(p) {
			acquirep(p);
			return true;
		}
	}
	return false;
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
	else if(runtime·atomicload(&runtime·sched.sysmonwait)) {
		runtime·atomicstore(&runtime·sched.sysmonwait, 0);
		runtime·notewakeup(&runtime·sched.sysmonnote);
	}
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

// Called from syscall package before fork.
void
syscall·runtime_BeforeFork(void)
{
	// Fork can hang if preempted with signals frequently enough (see issue 5517).
	// Ensure that we stay on the same M where we disable profiling.
	m->locks++;
	if(m->profilehz != 0)
		runtime·resetcpuprofiler(0);
}

// Called from syscall package after fork in parent.
void
syscall·runtime_AfterFork(void)
{
	int32 hz;

	hz = runtime·sched.profilehz;
	if(hz != 0)
		runtime·resetcpuprofiler(hz);
	m->locks--;
}

// Hook used by runtime·malg to call runtime·stackalloc on the
// scheduler stack.  This exists because runtime·stackalloc insists
// on being called on the scheduler stack, to avoid trying to grow
// the stack while allocating a new stack segment.
static void
mstackalloc(G *gp)
{
	gp->param = runtime·stackalloc((uintptr)gp->param);
	runtime·gogo(&gp->sched);
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
		newg->stacksize = StackSystem + stacksize;
		newg->stack0 = (uintptr)stk;
		newg->stackguard = (uintptr)stk + StackGuard;
		newg->stackguard0 = newg->stackguard;
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
#pragma textflag NOSPLIT
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
	P *p;
	int32 siz;

//runtime·printf("newproc1 %p %p narg=%d nret=%d\n", fn->fn, argp, narg, nret);
	m->locks++;  // disable preemption because it can be holding p in a local var
	siz = narg + nret;
	siz = (siz+7) & ~7;

	// We could instead create a secondary stack frame
	// and make it look like goexit was on the original but
	// the call to the actual goroutine function was split.
	// Not worth it: this is almost always an error.
	if(siz > StackMin - 1024)
		runtime·throw("runtime.newproc: function arguments too large for new goroutine");

	p = m->p;
	if((newg = gfget(p)) != nil) {
		if(newg->stackguard - StackGuard != newg->stack0)
			runtime·throw("invalid stack in newg");
	} else {
		newg = runtime·malg(StackMin);
		allgadd(newg);
	}

	sp = (byte*)newg->stackbase;
	sp -= siz;
	runtime·memmove(sp, argp, narg);
	if(thechar == '5') {
		// caller's LR
		sp -= sizeof(void*);
		*(void**)sp = nil;
	}

	runtime·memclr((byte*)&newg->sched, sizeof newg->sched);
	newg->sched.sp = (uintptr)sp;
	newg->sched.pc = (uintptr)runtime·goexit;
	newg->sched.g = newg;
	runtime·gostartcallfn(&newg->sched, fn);
	newg->gopc = (uintptr)callerpc;
	newg->status = Grunnable;
	if(p->goidcache == p->goidcacheend) {
		p->goidcache = runtime·xadd64(&runtime·sched.goidgen, GoidCacheBatch);
		p->goidcacheend = p->goidcache + GoidCacheBatch;
	}
	newg->goid = p->goidcache++;
	newg->panicwrap = 0;
	if(raceenabled)
		newg->racectx = runtime·racegostart((void*)callerpc);
	runqput(p, newg);

	if(runtime·atomicload(&runtime·sched.npidle) != 0 && runtime·atomicload(&runtime·sched.nmspinning) == 0 && fn->fn != runtime·main)  // TODO: fast atomic
		wakep();
	m->locks--;
	if(m->locks == 0 && g->preempt)  // restore the preemption request in case we've cleared it in newstack
		g->stackguard0 = StackPreempt;
	return newg;
}

static void
allgadd(G *gp)
{
	G **new;
	uintptr cap;

	runtime·lock(&allglock);
	if(runtime·allglen >= allgcap) {
		cap = 4096/sizeof(new[0]);
		if(cap < 2*allgcap)
			cap = 2*allgcap;
		new = runtime·malloc(cap*sizeof(new[0]));
		if(new == nil)
			runtime·throw("runtime: cannot allocate memory");
		if(runtime·allg != nil) {
			runtime·memmove(new, runtime·allg, runtime·allglen*sizeof(new[0]));
			runtime·free(runtime·allg);
		}
		runtime·allg = new;
		allgcap = cap;
	}
	runtime·allg[runtime·allglen++] = gp;
	runtime·unlock(&allglock);
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

	runtime·semacquire(&runtime·worldsema, false);
	m->gcing = 1;
	runtime·stoptheworld();
	newprocs = n;
	m->gcing = 0;
	runtime·semrelease(&runtime·worldsema);
	runtime·starttheworld();

	return ret;
}

// lockOSThread is called by runtime.LockOSThread and runtime.lockOSThread below
// after they modify m->locked. Do not allow preemption during this call,
// or else the m might be different in this function than in the caller.
#pragma textflag NOSPLIT
static void
lockOSThread(void)
{
	m->lockedg = g;
	g->lockedm = m;
}

void
runtime·LockOSThread(void)
{
	m->locked |= LockExternal;
	lockOSThread();
}

void
runtime·lockOSThread(void)
{
	m->locked += LockInternal;
	lockOSThread();
}


// unlockOSThread is called by runtime.UnlockOSThread and runtime.unlockOSThread below
// after they update m->locked. Do not allow preemption during this call,
// or else the m might be in different in this function than in the caller.
#pragma textflag NOSPLIT
static void
unlockOSThread(void)
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
	unlockOSThread();
}

void
runtime·unlockOSThread(void)
{
	if(m->locked < LockInternal)
		runtime·throw("runtime: internal error: misuse of lockOSThread/unlockOSThread");
	m->locked -= LockInternal;
	unlockOSThread();
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
	uintptr i;

	n = 0;
	runtime·lock(&allglock);
	// TODO(dvyukov): runtime.NumGoroutine() is O(N).
	// We do not want to increment/decrement centralized counter in newproc/goexit,
	// just to make runtime.NumGoroutine() faster.
	// Compromise solution is to introduce per-P counters of active goroutines.
	for(i = 0; i < runtime·allglen; i++) {
		gp = runtime·allg[i];
		s = gp->status;
		if(s == Grunnable || s == Grunning || s == Gsyscall || s == Gwaiting)
			n++;
	}
	runtime·unlock(&allglock);
	return n;
}

int32
runtime·mcount(void)
{
	return runtime·sched.mcount;
}

void
runtime·badmcall(void (*fn)(G*))  // called from assembly
{
	USED(fn); // TODO: print fn?
	runtime·throw("runtime: mcall called on m->g0 stack");
}

void
runtime·badmcall2(void (*fn)(G*))  // called from assembly
{
	USED(fn);
	runtime·throw("runtime: mcall function returned");
}

void
runtime·badreflectcall(void) // called from assembly
{
	runtime·panicstring("runtime: arg size to reflect.call more than 1GB");
}

static struct {
	Lock;
	void (*fn)(uintptr*, int32);
	int32 hz;
	uintptr pcbuf[100];
} prof;

static void
System(void)
{
}

// Called if we receive a SIGPROF signal.
void
runtime·sigprof(uint8 *pc, uint8 *sp, uint8 *lr, G *gp, M *mp)
{
	int32 n;
	bool traceback;
	// Do not use global m in this function, use mp instead.
	// On windows one m is sending reports about all the g's, so m means a wrong thing.
	byte m;

	m = 0;
	USED(m);

	if(prof.fn == nil || prof.hz == 0)
		return;

	// Profiling runs concurrently with GC, so it must not allocate.
	mp->mallocing++;

	// Define that a "user g" is a user-created goroutine, and a "system g"
	// is one that is m->g0 or m->gsignal. We've only made sure that we
	// can unwind user g's, so exclude the system g's.
	//
	// It is not quite as easy as testing gp == m->curg (the current user g)
	// because we might be interrupted for profiling halfway through a
	// goroutine switch. The switch involves updating three (or four) values:
	// g, PC, SP, and (on arm) LR. The PC must be the last to be updated,
	// because once it gets updated the new g is running.
	//
	// When switching from a user g to a system g, LR is not considered live,
	// so the update only affects g, SP, and PC. Since PC must be last, there
	// the possible partial transitions in ordinary execution are (1) g alone is updated,
	// (2) both g and SP are updated, and (3) SP alone is updated.
	// If g is updated, we'll see a system g and not look closer.
	// If SP alone is updated, we can detect the partial transition by checking
	// whether the SP is within g's stack bounds. (We could also require that SP
	// be changed only after g, but the stack bounds check is needed by other
	// cases, so there is no need to impose an additional requirement.)
	//
	// There is one exceptional transition to a system g, not in ordinary execution.
	// When a signal arrives, the operating system starts the signal handler running
	// with an updated PC and SP. The g is updated last, at the beginning of the
	// handler. There are two reasons this is okay. First, until g is updated the
	// g and SP do not match, so the stack bounds check detects the partial transition.
	// Second, signal handlers currently run with signals disabled, so a profiling
	// signal cannot arrive during the handler.
	//
	// When switching from a system g to a user g, there are three possibilities.
	//
	// First, it may be that the g switch has no PC update, because the SP
	// either corresponds to a user g throughout (as in runtime.asmcgocall)
	// or because it has been arranged to look like a user g frame
	// (as in runtime.cgocallback_gofunc). In this case, since the entire
	// transition is a g+SP update, a partial transition updating just one of 
	// those will be detected by the stack bounds check.
	//
	// Second, when returning from a signal handler, the PC and SP updates
	// are performed by the operating system in an atomic update, so the g
	// update must be done before them. The stack bounds check detects
	// the partial transition here, and (again) signal handlers run with signals
	// disabled, so a profiling signal cannot arrive then anyway.
	//
	// Third, the common case: it may be that the switch updates g, SP, and PC
	// separately, as in runtime.gogo.
	//
	// Because runtime.gogo is the only instance, we check whether the PC lies
	// within that function, and if so, not ask for a traceback. This approach
	// requires knowing the size of the runtime.gogo function, which we
	// record in arch_*.h and check in runtime_test.go.
	//
	// There is another apparently viable approach, recorded here in case
	// the "PC within runtime.gogo" check turns out not to be usable.
	// It would be possible to delay the update of either g or SP until immediately
	// before the PC update instruction. Then, because of the stack bounds check,
	// the only problematic interrupt point is just before that PC update instruction,
	// and the sigprof handler can detect that instruction and simulate stepping past
	// it in order to reach a consistent state. On ARM, the update of g must be made
	// in two places (in R10 and also in a TLS slot), so the delayed update would
	// need to be the SP update. The sigprof handler must read the instruction at
	// the current PC and if it was the known instruction (for example, JMP BX or 
	// MOV R2, PC), use that other register in place of the PC value.
	// The biggest drawback to this solution is that it requires that we can tell
	// whether it's safe to read from the memory pointed at by PC.
	// In a correct program, we can test PC == nil and otherwise read,
	// but if a profiling signal happens at the instant that a program executes
	// a bad jump (before the program manages to handle the resulting fault)
	// the profiling handler could fault trying to read nonexistent memory.
	//
	// To recap, there are no constraints on the assembly being used for the
	// transition. We simply require that g and SP match and that the PC is not
	// in runtime.gogo.
	traceback = true;
	if(gp == nil || gp != mp->curg ||
	   (uintptr)sp < gp->stackguard - StackGuard || gp->stackbase < (uintptr)sp ||
	   ((uint8*)runtime·gogo <= pc && pc < (uint8*)runtime·gogo + RuntimeGogoBytes))
		traceback = false;

	// Race detector calls asmcgocall w/o entersyscall/exitsyscall,
	// we can not currently unwind through asmcgocall.
	if(mp != nil && mp->racecall)
		traceback = false;

	runtime·lock(&prof);
	if(prof.fn == nil) {
		runtime·unlock(&prof);
		mp->mallocing--;
		return;
	}
	n = 0;
	if(traceback)
		n = runtime·gentraceback((uintptr)pc, (uintptr)sp, (uintptr)lr, gp, 0, prof.pcbuf, nelem(prof.pcbuf), nil, nil, false);
	if(!traceback || n <= 0) {
		n = 2;
		prof.pcbuf[0] = (uintptr)pc;
		prof.pcbuf[1] = (uintptr)System + 1;
	}
	prof.fn(prof.pcbuf, n);
	runtime·unlock(&prof);
	mp->mallocing--;
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

	// Disable preemption, otherwise we can be rescheduled to another thread
	// that has profiling enabled.
	m->locks++;

	// Stop profiler on this thread so that it is safe to lock prof.
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

	m->locks--;
}

// Change number of processors.  The world is stopped, sched is locked.
static void
procresize(int32 new)
{
	int32 i, old;
	bool empty;
	G *gp;
	P *p;

	old = runtime·gomaxprocs;
	if(old < 0 || old > MaxGomaxprocs || new <= 0 || new >MaxGomaxprocs)
		runtime·throw("procresize: invalid arg");
	// initialize new P's
	for(i = 0; i < new; i++) {
		p = runtime·allp[i];
		if(p == nil) {
			p = (P*)runtime·mallocgc(sizeof(*p), 0, FlagNoInvokeGC);
			p->id = i;
			p->status = Pgcstop;
			runtime·atomicstorep(&runtime·allp[i], p);
		}
		if(p->mcache == nil) {
			if(old==0 && i==0)
				p->mcache = m->mcache;  // bootstrap
			else
				p->mcache = runtime·allocmcache();
		}
	}

	// redistribute runnable G's evenly
	// collect all runnable goroutines in global queue preserving FIFO order
	// FIFO order is required to ensure fairness even during frequent GCs
	// see http://golang.org/issue/7126
	empty = false;
	while(!empty) {
		empty = true;
		for(i = 0; i < old; i++) {
			p = runtime·allp[i];
			if(p->runqhead == p->runqtail)
				continue;
			empty = false;
			// pop from tail of local queue
			p->runqtail--;
			gp = p->runq[p->runqtail%nelem(p->runq)];
			// push onto head of global queue
			gp->schedlink = runtime·sched.runqhead;
			runtime·sched.runqhead = gp;
			if(runtime·sched.runqtail == nil)
				runtime·sched.runqtail = gp;
			runtime·sched.runqsize++;
		}
	}
	// fill local queues with at most nelem(p->runq)/2 goroutines
	// start at 1 because current M already executes some G and will acquire allp[0] below,
	// so if we have a spare G we want to put it into allp[1].
	for(i = 1; i < new * nelem(p->runq)/2 && runtime·sched.runqsize > 0; i++) {
		gp = runtime·sched.runqhead;
		runtime·sched.runqhead = gp->schedlink;
		if(runtime·sched.runqhead == nil)
			runtime·sched.runqtail = nil;
		runtime·sched.runqsize--;
		runqput(runtime·allp[i%new], gp);
	}

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
incidlelocked(int32 v)
{
	runtime·lock(&runtime·sched);
	runtime·sched.nmidlelocked += v;
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
	uintptr i;

	// -1 for sysmon
	run = runtime·sched.mcount - runtime·sched.nmidle - runtime·sched.nmidlelocked - 1;
	if(run > 0)
		return;
	if(run < 0) {
		runtime·printf("checkdead: nmidle=%d nmidlelocked=%d mcount=%d\n",
			runtime·sched.nmidle, runtime·sched.nmidlelocked, runtime·sched.mcount);
		runtime·throw("checkdead: inconsistent counts");
	}
	grunning = 0;
	runtime·lock(&allglock);
	for(i = 0; i < runtime·allglen; i++) {
		gp = runtime·allg[i];
		if(gp->isbackground)
			continue;
		s = gp->status;
		if(s == Gwaiting)
			grunning++;
		else if(s == Grunnable || s == Grunning || s == Gsyscall) {
			runtime·unlock(&allglock);
			runtime·printf("checkdead: find g %D in status %d\n", gp->goid, s);
			runtime·throw("checkdead: runnable g");
		}
	}
	runtime·unlock(&allglock);
	if(grunning == 0)  // possible if main goroutine calls runtime·Goexit()
		runtime·exit(0);
	m->throwing = -1;  // do not dump full stacks
	runtime·throw("all goroutines are asleep - deadlock!");
}

static void
sysmon(void)
{
	uint32 idle, delay;
	int64 now, lastpoll, lasttrace;
	G *gp;

	lasttrace = 0;
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
		if(runtime·debug.schedtrace <= 0 &&
			(runtime·sched.gcwaiting || runtime·atomicload(&runtime·sched.npidle) == runtime·gomaxprocs)) {  // TODO: fast atomic
			runtime·lock(&runtime·sched);
			if(runtime·atomicload(&runtime·sched.gcwaiting) || runtime·atomicload(&runtime·sched.npidle) == runtime·gomaxprocs) {
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
		if(lastpoll != 0 && lastpoll + 10*1000*1000 < now) {
			runtime·cas64(&runtime·sched.lastpoll, lastpoll, now);
			gp = runtime·netpoll(false);  // non-blocking
			if(gp) {
				// Need to decrement number of idle locked M's
				// (pretending that one more is running) before injectglist.
				// Otherwise it can lead to the following situation:
				// injectglist grabs all P's but before it starts M's to run the P's,
				// another M returns from syscall, finishes running its G,
				// observes that there is no work to do and no other running M's
				// and reports deadlock.
				incidlelocked(-1);
				injectglist(gp);
				incidlelocked(1);
			}
		}
		// retake P's blocked in syscalls
		// and preempt long running G's
		if(retake(now))
			idle = 0;
		else
			idle++;

		if(runtime·debug.schedtrace > 0 && lasttrace + runtime·debug.schedtrace*1000000ll <= now) {
			lasttrace = now;
			runtime·schedtrace(runtime·debug.scheddetail);
		}
	}
}

typedef struct Pdesc Pdesc;
struct Pdesc
{
	uint32	schedtick;
	int64	schedwhen;
	uint32	syscalltick;
	int64	syscallwhen;
};
static Pdesc pdesc[MaxGomaxprocs];

static uint32
retake(int64 now)
{
	uint32 i, s, n;
	int64 t;
	P *p;
	Pdesc *pd;

	n = 0;
	for(i = 0; i < runtime·gomaxprocs; i++) {
		p = runtime·allp[i];
		if(p==nil)
			continue;
		pd = &pdesc[i];
		s = p->status;
		if(s == Psyscall) {
			// Retake P from syscall if it's there for more than 1 sysmon tick (at least 20us).
			t = p->syscalltick;
			if(pd->syscalltick != t) {
				pd->syscalltick = t;
				pd->syscallwhen = now;
				continue;
			}
			// On the one hand we don't want to retake Ps if there is no other work to do,
			// but on the other hand we want to retake them eventually
			// because they can prevent the sysmon thread from deep sleep.
			if(p->runqhead == p->runqtail &&
				runtime·atomicload(&runtime·sched.nmspinning) + runtime·atomicload(&runtime·sched.npidle) > 0 &&
				pd->syscallwhen + 10*1000*1000 > now)
				continue;
			// Need to decrement number of idle locked M's
			// (pretending that one more is running) before the CAS.
			// Otherwise the M from which we retake can exit the syscall,
			// increment nmidle and report deadlock.
			incidlelocked(-1);
			if(runtime·cas(&p->status, s, Pidle)) {
				n++;
				handoffp(p);
			}
			incidlelocked(1);
		} else if(s == Prunning) {
			// Preempt G if it's running for more than 10ms.
			t = p->schedtick;
			if(pd->schedtick != t) {
				pd->schedtick = t;
				pd->schedwhen = now;
				continue;
			}
			if(pd->schedwhen + 10*1000*1000 > now)
				continue;
			preemptone(p);
		}
	}
	return n;
}

// Tell all goroutines that they have been preempted and they should stop.
// This function is purely best-effort.  It can fail to inform a goroutine if a
// processor just started running it.
// No locks need to be held.
// Returns true if preemption request was issued to at least one goroutine.
static bool
preemptall(void)
{
	P *p;
	int32 i;
	bool res;

	res = false;
	for(i = 0; i < runtime·gomaxprocs; i++) {
		p = runtime·allp[i];
		if(p == nil || p->status != Prunning)
			continue;
		res |= preemptone(p);
	}
	return res;
}

// Tell the goroutine running on processor P to stop.
// This function is purely best-effort.  It can incorrectly fail to inform the
// goroutine.  It can send inform the wrong goroutine.  Even if it informs the
// correct goroutine, that goroutine might ignore the request if it is
// simultaneously executing runtime·newstack.
// No lock needs to be held.
// Returns true if preemption request was issued.
static bool
preemptone(P *p)
{
	M *mp;
	G *gp;

	mp = p->m;
	if(mp == nil || mp == m)
		return false;
	gp = mp->curg;
	if(gp == nil || gp == mp->g0)
		return false;
	gp->preempt = true;
	gp->stackguard0 = StackPreempt;
	return true;
}

void
runtime·schedtrace(bool detailed)
{
	static int64 starttime;
	int64 now;
	int64 id1, id2, id3;
	int32 i, t, h;
	uintptr gi;
	int8 *fmt;
	M *mp, *lockedm;
	G *gp, *lockedg;
	P *p;

	now = runtime·nanotime();
	if(starttime == 0)
		starttime = now;

	runtime·lock(&runtime·sched);
	runtime·printf("SCHED %Dms: gomaxprocs=%d idleprocs=%d threads=%d idlethreads=%d runqueue=%d",
		(now-starttime)/1000000, runtime·gomaxprocs, runtime·sched.npidle, runtime·sched.mcount,
		runtime·sched.nmidle, runtime·sched.runqsize);
	if(detailed) {
		runtime·printf(" gcwaiting=%d nmidlelocked=%d nmspinning=%d stopwait=%d sysmonwait=%d\n",
			runtime·sched.gcwaiting, runtime·sched.nmidlelocked, runtime·sched.nmspinning,
			runtime·sched.stopwait, runtime·sched.sysmonwait);
	}
	// We must be careful while reading data from P's, M's and G's.
	// Even if we hold schedlock, most data can be changed concurrently.
	// E.g. (p->m ? p->m->id : -1) can crash if p->m changes from non-nil to nil.
	for(i = 0; i < runtime·gomaxprocs; i++) {
		p = runtime·allp[i];
		if(p == nil)
			continue;
		mp = p->m;
		h = runtime·atomicload(&p->runqhead);
		t = runtime·atomicload(&p->runqtail);
		if(detailed)
			runtime·printf("  P%d: status=%d schedtick=%d syscalltick=%d m=%d runqsize=%d gfreecnt=%d\n",
				i, p->status, p->schedtick, p->syscalltick, mp ? mp->id : -1, t-h, p->gfreecnt);
		else {
			// In non-detailed mode format lengths of per-P run queues as:
			// [len1 len2 len3 len4]
			fmt = " %d";
			if(runtime·gomaxprocs == 1)
				fmt = " [%d]\n";
			else if(i == 0)
				fmt = " [%d";
			else if(i == runtime·gomaxprocs-1)
				fmt = " %d]\n";
			runtime·printf(fmt, t-h);
		}
	}
	if(!detailed) {
		runtime·unlock(&runtime·sched);
		return;
	}
	for(mp = runtime·allm; mp; mp = mp->alllink) {
		p = mp->p;
		gp = mp->curg;
		lockedg = mp->lockedg;
		id1 = -1;
		if(p)
			id1 = p->id;
		id2 = -1;
		if(gp)
			id2 = gp->goid;
		id3 = -1;
		if(lockedg)
			id3 = lockedg->goid;
		runtime·printf("  M%d: p=%D curg=%D mallocing=%d throwing=%d gcing=%d"
			" locks=%d dying=%d helpgc=%d spinning=%d blocked=%d lockedg=%D\n",
			mp->id, id1, id2,
			mp->mallocing, mp->throwing, mp->gcing, mp->locks, mp->dying, mp->helpgc,
			mp->spinning, m->blocked, id3);
	}
	runtime·lock(&allglock);
	for(gi = 0; gi < runtime·allglen; gi++) {
		gp = runtime·allg[gi];
		mp = gp->m;
		lockedm = gp->lockedm;
		runtime·printf("  G%D: status=%d(%s) m=%d lockedm=%d\n",
			gp->goid, gp->status, gp->waitreason, mp ? mp->id : -1,
			lockedm ? lockedm->id : -1);
	}
	runtime·unlock(&allglock);
	runtime·unlock(&runtime·sched);
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

// Put a batch of runnable goroutines on the global runnable queue.
// Sched must be locked.
static void
globrunqputbatch(G *ghead, G *gtail, int32 n)
{
	gtail->schedlink = nil;
	if(runtime·sched.runqtail)
		runtime·sched.runqtail->schedlink = ghead;
	else
		runtime·sched.runqhead = ghead;
	runtime·sched.runqtail = gtail;
	runtime·sched.runqsize += n;
}

// Try get a batch of G's from the global runnable queue.
// Sched must be locked.
static G*
globrunqget(P *p, int32 max)
{
	G *gp, *gp1;
	int32 n;

	if(runtime·sched.runqsize == 0)
		return nil;
	n = runtime·sched.runqsize/runtime·gomaxprocs+1;
	if(n > runtime·sched.runqsize)
		n = runtime·sched.runqsize;
	if(max > 0 && n > max)
		n = max;
	if(n > nelem(p->runq)/2)
		n = nelem(p->runq)/2;
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

// Try to put g on local runnable queue.
// If it's full, put onto global queue.
// Executed only by the owner P.
static void
runqput(P *p, G *gp)
{
	uint32 h, t;

retry:
	h = runtime·atomicload(&p->runqhead);  // load-acquire, synchronize with consumers
	t = p->runqtail;
	if(t - h < nelem(p->runq)) {
		p->runq[t%nelem(p->runq)] = gp;
		runtime·atomicstore(&p->runqtail, t+1);  // store-release, makes the item available for consumption
		return;
	}
	if(runqputslow(p, gp, h, t))
		return;
	// the queue is not full, now the put above must suceed
	goto retry;
}

// Put g and a batch of work from local runnable queue on global queue.
// Executed only by the owner P.
static bool
runqputslow(P *p, G *gp, uint32 h, uint32 t)
{
	G *batch[nelem(p->runq)/2+1];
	uint32 n, i;

	// First, grab a batch from local queue.
	n = t-h;
	n = n/2;
	if(n != nelem(p->runq)/2)
		runtime·throw("runqputslow: queue is not full");
	for(i=0; i<n; i++)
		batch[i] = p->runq[(h+i)%nelem(p->runq)];
	if(!runtime·cas(&p->runqhead, h, h+n))  // cas-release, commits consume
		return false;
	batch[n] = gp;
	// Link the goroutines.
	for(i=0; i<n; i++)
		batch[i]->schedlink = batch[i+1];
	// Now put the batch on global queue.
	runtime·lock(&runtime·sched);
	globrunqputbatch(batch[0], batch[n], n+1);
	runtime·unlock(&runtime·sched);
	return true;
}

// Get g from local runnable queue.
// Executed only by the owner P.
static G*
runqget(P *p)
{
	G *gp;
	uint32 t, h;

	for(;;) {
		h = runtime·atomicload(&p->runqhead);  // load-acquire, synchronize with other consumers
		t = p->runqtail;
		if(t == h)
			return nil;
		gp = p->runq[h%nelem(p->runq)];
		if(runtime·cas(&p->runqhead, h, h+1))  // cas-release, commits consume
			return gp;
	}
}

// Grabs a batch of goroutines from local runnable queue.
// batch array must be of size nelem(p->runq)/2. Returns number of grabbed goroutines.
// Can be executed by any P.
static uint32
runqgrab(P *p, G **batch)
{
	uint32 t, h, n, i;

	for(;;) {
		h = runtime·atomicload(&p->runqhead);  // load-acquire, synchronize with other consumers
		t = runtime·atomicload(&p->runqtail);  // load-acquire, synchronize with the producer
		n = t-h;
		n = n - n/2;
		if(n == 0)
			break;
		if(n > nelem(p->runq)/2)  // read inconsistent h and t
			continue;
		for(i=0; i<n; i++)
			batch[i] = p->runq[(h+i)%nelem(p->runq)];
		if(runtime·cas(&p->runqhead, h, h+n))  // cas-release, commits consume
			break;
	}
	return n;
}

// Steal half of elements from local runnable queue of p2
// and put onto local runnable queue of p.
// Returns one of the stolen elements (or nil if failed).
static G*
runqsteal(P *p, P *p2)
{
	G *gp;
	G *batch[nelem(p->runq)/2];
	uint32 t, h, n, i;

	n = runqgrab(p2, batch);
	if(n == 0)
		return nil;
	n--;
	gp = batch[n];
	if(n == 0)
		return gp;
	h = runtime·atomicload(&p->runqhead);  // load-acquire, synchronize with consumers
	t = p->runqtail;
	if(t - h + n >= nelem(p->runq))
		runtime·throw("runqsteal: runq overflow");
	for(i=0; i<n; i++, t++)
		p->runq[t%nelem(p->runq)] = batch[i];
	runtime·atomicstore(&p->runqtail, t);  // store-release, makes the item available for consumption
	return gp;
}

void
runtime·testSchedLocalQueue(void)
{
	P p;
	G gs[nelem(p.runq)];
	int32 i, j;

	runtime·memclr((byte*)&p, sizeof(p));

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
	G gs[nelem(p1.runq)], *gp;
	int32 i, j, s;

	runtime·memclr((byte*)&p1, sizeof(p1));
	runtime·memclr((byte*)&p2, sizeof(p2));

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

extern void runtime·morestack(void);

// Does f mark the top of a goroutine stack?
bool
runtime·topofstack(Func *f)
{
	return f->entry == (uintptr)runtime·goexit ||
		f->entry == (uintptr)runtime·mstart ||
		f->entry == (uintptr)runtime·mcall ||
		f->entry == (uintptr)runtime·morestack ||
		f->entry == (uintptr)runtime·lessstack ||
		f->entry == (uintptr)_rt0_go;
}

void
runtime∕debug·setMaxThreads(intgo in, intgo out)
{
	runtime·lock(&runtime·sched);
	out = runtime·sched.maxmcount;
	runtime·sched.maxmcount = in;
	checkmcount();
	runtime·unlock(&runtime·sched);
	FLUSH(&out);
}

static int8 experiment[] = GOEXPERIMENT; // defined in zaexperiment.h

static bool
haveexperiment(int8 *name)
{
	int32 i, j;
	
	for(i=0; i<sizeof(experiment); i++) {
		if((i == 0 || experiment[i-1] == ',') && experiment[i] == name[0]) {
			for(j=0; name[j]; j++)
				if(experiment[i+j] != name[j])
					goto nomatch;
			if(experiment[i+j] != '\0' && experiment[i+j] != ',')
				goto nomatch;
			return 1;
		}
	nomatch:;
	}
	return 0;
}

// func runtime_procPin() int
void
sync·runtime_procPin(intgo p)
{
	M *mp;

	mp = m;
	// Disable preemption.
	mp->locks++;
	p = mp->p->id;
	FLUSH(&p);
}

// func runtime_procUnpin()
void
sync·runtime_procUnpin(void)
{
	m->locks--;
}
