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
#include "mgc0.h"
#include "textflag.h"

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

enum
{
	// Number of goroutine ids to grab from runtime·sched.goidgen to local per-P cache at once.
	// 16 seems to provide enough amortization, but other than that it's mostly arbitrary number.
	GoidCacheBatch = 16,
};

SchedT	runtime·sched;
int32	runtime·gomaxprocs;
uint32	runtime·needextram;
bool	runtime·iscgo;
M	runtime·m0;
G	runtime·g0;	// idle goroutine for m0
G*	runtime·lastg;
M*	runtime·allm;
M*	runtime·extram;
P*	runtime·allp[MaxGomaxprocs+1];
int8*	runtime·goos;
int32	runtime·ncpu;
int32	runtime·newprocs;

Mutex runtime·allglock;	// the following vars are protected by this lock or by stoptheworld
G**	runtime·allg;
Slice	runtime·allgs;
uintptr runtime·allglen;
ForceGCState	runtime·forcegc;

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
void runtime·park_m(G*);
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
void runtime·allgadd(G*);
static void dropg(void);

extern String runtime·buildVersion;

// For cgo-using programs with external linking,
// export "main" (defined in assembly) so that libc can handle basic
// C runtime startup and call the Go program as if it were
// the C main function.
#pragma cgo_export_static main

// Filled in by dynamic linker when Cgo is available.
void (*_cgo_init)(void);
void (*_cgo_malloc)(void);
void (*_cgo_free)(void);

// Copy for Go code.
void* runtime·cgoMalloc;
void* runtime·cgoFree;

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

	// raceinit must be the first call to race detector.
	// In particular, it must be done before mallocinit below calls racemapshadow.
	if(raceenabled)
		g->racectx = runtime·raceinit();

	runtime·sched.maxmcount = 10000;

	runtime·tracebackinit();
	runtime·symtabinit();
	runtime·stackinit();
	runtime·mallocinit();
	mcommoninit(g->m);
	
	runtime·goargs();
	runtime·goenvs();
	runtime·parsedebugvars();
	runtime·gcinit();

	runtime·sched.lastpoll = runtime·nanotime();
	procs = 1;
	p = runtime·getenv("GOMAXPROCS");
	if(p != nil && (n = runtime·atoi(p)) > 0) {
		if(n > MaxGomaxprocs)
			n = MaxGomaxprocs;
		procs = n;
	}
	procresize(procs);

	if(runtime·buildVersion.str == nil) {
		// Condition should never trigger.  This code just serves
		// to ensure runtime·buildVersion is kept in the resulting binary.
		runtime·buildVersion.str = (uint8*)"unknown";
		runtime·buildVersion.len = 7;
	}

	runtime·cgoMalloc = _cgo_malloc;
	runtime·cgoFree = _cgo_free;
}

void
runtime·newsysmon(void)
{
	newm(sysmon, nil);
}

static void
dumpgstatus(G* gp)
{
	runtime·printf("runtime: gp: gp=%p, goid=%D, gp->atomicstatus=%x\n", gp, gp->goid, runtime·readgstatus(gp));
	runtime·printf("runtime:  g:  g=%p, goid=%D,  g->atomicstatus=%x\n", g, g->goid, runtime·readgstatus(g));
}

static void
checkmcount(void)
{
	// sched lock is held
	if(runtime·sched.mcount > runtime·sched.maxmcount){
		runtime·printf("runtime: program exceeds %d-thread limit\n", runtime·sched.maxmcount);
		runtime·throw("thread exhaustion");
	}
}

static void
mcommoninit(M *mp)
{
	// g0 stack won't make sense for user (and is not necessary unwindable).
	if(g != g->m->g0)
		runtime·callers(1, mp->createstack, nelem(mp->createstack));

	mp->fastrand = 0x49f6428aUL + mp->id + runtime·cputicks();

	runtime·lock(&runtime·sched.lock);
	mp->id = runtime·sched.mcount++;
	checkmcount();
	runtime·mpreinit(mp);
	if(mp->gsignal)
		mp->gsignal->stackguard1 = mp->gsignal->stack.lo + StackGuard;

	// Add to runtime·allm so garbage collector doesn't free g->m
	// when it is just in a register or thread-local storage.
	mp->alllink = runtime·allm;
	// runtime·NumCgoCall() iterates over allm w/o schedlock,
	// so we need to publish it safely.
	runtime·atomicstorep(&runtime·allm, mp);
	runtime·unlock(&runtime·sched.lock);
}

// Mark gp ready to run.
void
runtime·ready(G *gp)
{
	uint32 status;

	status = runtime·readgstatus(gp);
	// Mark runnable.
	g->m->locks++;  // disable preemption because it can be holding p in a local var
	if((status&~Gscan) != Gwaiting){
		dumpgstatus(gp);
		runtime·throw("bad g->status in ready");
	}
	// status is Gwaiting or Gscanwaiting, make Grunnable and put on runq
	runtime·casgstatus(gp, Gwaiting, Grunnable);
	runqput(g->m->p, gp);
	if(runtime·atomicload(&runtime·sched.npidle) != 0 && runtime·atomicload(&runtime·sched.nmspinning) == 0)  // TODO: fast atomic
		wakep();
	g->m->locks--;
	if(g->m->locks == 0 && g->preempt)  // restore the preemption request in case we've cleared it in newstack
		g->stackguard0 = StackPreempt;
}

void
runtime·ready_m(void)
{
	G *gp;

	gp = g->m->ptrarg[0];
	g->m->ptrarg[0] = nil;
	runtime·ready(gp);
}

int32
runtime·gcprocs(void)
{
	int32 n;

	// Figure out how many CPUs to use during GC.
	// Limited by gomaxprocs, number of actual CPUs, and MaxGcproc.
	runtime·lock(&runtime·sched.lock);
	n = runtime·gomaxprocs;
	if(n > runtime·ncpu)
		n = runtime·ncpu;
	if(n > MaxGcproc)
		n = MaxGcproc;
	if(n > runtime·sched.nmidle+1) // one M is currently running
		n = runtime·sched.nmidle+1;
	runtime·unlock(&runtime·sched.lock);
	return n;
}

static bool
needaddgcproc(void)
{
	int32 n;

	runtime·lock(&runtime·sched.lock);
	n = runtime·gomaxprocs;
	if(n > runtime·ncpu)
		n = runtime·ncpu;
	if(n > MaxGcproc)
		n = MaxGcproc;
	n -= runtime·sched.nmidle+1; // one M is currently running
	runtime·unlock(&runtime·sched.lock);
	return n > 0;
}

void
runtime·helpgc(int32 nproc)
{
	M *mp;
	int32 n, pos;

	runtime·lock(&runtime·sched.lock);
	pos = 0;
	for(n = 1; n < nproc; n++) {  // one M is currently running
		if(runtime·allp[pos]->mcache == g->m->mcache)
			pos++;
		mp = mget();
		if(mp == nil)
			runtime·throw("runtime·gcprocs inconsistency");
		mp->helpgc = n;
		mp->mcache = runtime·allp[pos]->mcache;
		pos++;
		runtime·notewakeup(&mp->park);
	}
	runtime·unlock(&runtime·sched.lock);
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

static bool
isscanstatus(uint32 status)
{
	if(status == Gscan)
		runtime·throw("isscanstatus: Bad status Gscan");
	return (status&Gscan) == Gscan;
}

// All reads and writes of g's status go through readgstatus, casgstatus
// castogscanstatus, casfromgscanstatus.
#pragma textflag NOSPLIT
uint32
runtime·readgstatus(G *gp)
{
	return runtime·atomicload(&gp->atomicstatus);
}

// The Gscanstatuses are acting like locks and this releases them.
// If it proves to be a performance hit we should be able to make these
// simple atomic stores but for now we are going to throw if
// we see an inconsistent state.
void
runtime·casfromgscanstatus(G *gp, uint32 oldval, uint32 newval)
{
	bool success = false;

	// Check that transition is valid.
	switch(oldval) {
	case Gscanrunnable:
	case Gscanwaiting:
	case Gscanrunning:
	case Gscansyscall:
		if(newval == (oldval&~Gscan))
			success = runtime·cas(&gp->atomicstatus, oldval, newval);
		break;
	case Gscanenqueue:
		if(newval == Gwaiting)
			success = runtime·cas(&gp->atomicstatus, oldval, newval);
		break;
	}	
	if(!success){
		runtime·printf("runtime: casfromgscanstatus failed gp=%p, oldval=%d, newval=%d\n",  
			gp, oldval, newval);
		dumpgstatus(gp);
		runtime·throw("casfromgscanstatus: gp->status is not in scan state");
	}
}

// This will return false if the gp is not in the expected status and the cas fails. 
// This acts like a lock acquire while the casfromgstatus acts like a lock release.
bool
runtime·castogscanstatus(G *gp, uint32 oldval, uint32 newval)
{
	switch(oldval) {
	case Grunnable:
	case Gwaiting:
	case Gsyscall:
		if(newval == (oldval|Gscan))
			return runtime·cas(&gp->atomicstatus, oldval, newval);
		break;
	case Grunning:
		if(newval == Gscanrunning || newval == Gscanenqueue)
			return runtime·cas(&gp->atomicstatus, oldval, newval);
		break;   
	}

	runtime·printf("runtime: castogscanstatus oldval=%d newval=%d\n", oldval, newval);
	runtime·throw("castogscanstatus");
	return false; // not reached
}

static void badcasgstatus(void);
static void helpcasgstatus(void);
static void badgstatusrunnable(void);

// If asked to move to or from a Gscanstatus this will throw. Use the castogscanstatus
// and casfromgscanstatus instead.
// casgstatus will loop if the g->atomicstatus is in a Gscan status until the routine that 
// put it in the Gscan state is finished.
#pragma textflag NOSPLIT
void
runtime·casgstatus(G *gp, uint32 oldval, uint32 newval)
{
	void (*fn)(void);

	if((oldval&Gscan) || (newval&Gscan) || oldval == newval) {
		g->m->scalararg[0] = oldval;
		g->m->scalararg[1] = newval;
		fn = badcasgstatus;
		runtime·onM(&fn);
	}

	// loop if gp->atomicstatus is in a scan state giving
	// GC time to finish and change the state to oldval.
	while(!runtime·cas(&gp->atomicstatus, oldval, newval)) {
		if(oldval == Gwaiting && gp->atomicstatus == Grunnable) {
			fn = badgstatusrunnable;
			runtime·onM(&fn);
		}
		// Help GC if needed. 
		if(gp->preemptscan && !gp->gcworkdone && (oldval == Grunning || oldval == Gsyscall)) {
			gp->preemptscan = false;
			g->m->ptrarg[0] = gp;
			fn = helpcasgstatus;
			runtime·onM(&fn);
		}
	}	
}

static void
badgstatusrunnable(void)
{
	runtime·throw("casgstatus: waiting for Gwaiting but is Grunnable");
}

// casgstatus(gp, oldstatus, Gcopystack), assuming oldstatus is Gwaiting or Grunnable.
// Returns old status. Cannot call casgstatus directly, because we are racing with an
// async wakeup that might come in from netpoll. If we see Gwaiting from the readgstatus,
// it might have become Grunnable by the time we get to the cas. If we called casgstatus,
// it would loop waiting for the status to go back to Gwaiting, which it never will.
#pragma textflag NOSPLIT
uint32
runtime·casgcopystack(G *gp)
{
	uint32 oldstatus;

	for(;;) {
		oldstatus = runtime·readgstatus(gp) & ~Gscan;
		if(oldstatus != Gwaiting && oldstatus != Grunnable)
			runtime·throw("copystack: bad status, not Gwaiting or Grunnable");
		if(runtime·cas(&gp->atomicstatus, oldstatus, Gcopystack))
			break;
	}
	return oldstatus;
}

static void
badcasgstatus(void)
{
	uint32 oldval, newval;
	
	oldval = g->m->scalararg[0];
	newval = g->m->scalararg[1];
	g->m->scalararg[0] = 0;
	g->m->scalararg[1] = 0;

	runtime·printf("casgstatus: oldval=%d, newval=%d\n", oldval, newval);
	runtime·throw("casgstatus: bad incoming values");
}

static void
helpcasgstatus(void)
{
	G *gp;
	
	gp = g->m->ptrarg[0];
	g->m->ptrarg[0] = 0;
	runtime·gcphasework(gp);
}

// stopg ensures that gp is stopped at a GC safe point where its stack can be scanned
// or in the context of a moving collector the pointers can be flipped from pointing 
// to old object to pointing to new objects. 
// If stopg returns true, the caller knows gp is at a GC safe point and will remain there until
// the caller calls restartg.
// If stopg returns false, the caller is not responsible for calling restartg. This can happen
// if another thread, either the gp itself or another GC thread is taking the responsibility 
// to do the GC work related to this thread.
bool
runtime·stopg(G *gp)
{
	uint32 s;

	for(;;) {
		if(gp->gcworkdone)
			return false;

		s = runtime·readgstatus(gp);
		switch(s) {
		default:
			dumpgstatus(gp);
			runtime·throw("stopg: gp->atomicstatus is not valid");

		case Gdead:
			return false;

		case Gcopystack:
			// Loop until a new stack is in place.
			break;

		case Grunnable:
		case Gsyscall:
		case Gwaiting:
			// Claim goroutine by setting scan bit.
			if(!runtime·castogscanstatus(gp, s, s|Gscan))
				break;
			// In scan state, do work.
			runtime·gcphasework(gp);
			return true;

		case Gscanrunnable:
		case Gscanwaiting:
		case Gscansyscall:
			// Goroutine already claimed by another GC helper.
			return false;

		case Grunning:
			// Claim goroutine, so we aren't racing with a status
			// transition away from Grunning.
			if(!runtime·castogscanstatus(gp, Grunning, Gscanrunning))
				break;

			// Mark gp for preemption.
			if(!gp->gcworkdone) {
				gp->preemptscan = true;
				gp->preempt = true;
				gp->stackguard0 = StackPreempt;
			}

			// Unclaim.
			runtime·casfromgscanstatus(gp, Gscanrunning, Grunning);
			return false;
		}
	}
	// Should not be here....
}

// The GC requests that this routine be moved from a scanmumble state to a mumble state.
void 
runtime·restartg (G *gp)
{
	uint32 s;

	s = runtime·readgstatus(gp);
	switch(s) {
	default:
		dumpgstatus(gp); 
		runtime·throw("restartg: unexpected status");

	case Gdead:
		break;

	case Gscanrunnable:
	case Gscanwaiting:
	case Gscansyscall:
		runtime·casfromgscanstatus(gp, s, s&~Gscan);
		break;

	case Gscanenqueue:
		// Scan is now completed.
		// Goroutine now needs to be made runnable.
		// We put it on the global run queue; ready blocks on the global scheduler lock.
		runtime·casfromgscanstatus(gp, Gscanenqueue, Gwaiting);
		if(gp != g->m->curg)
			runtime·throw("processing Gscanenqueue on wrong m");
		dropg();
		runtime·ready(gp);
		break;
	}
}

static void
stopscanstart(G* gp)
{
	if(g == gp)
		runtime·throw("GC not moved to G0");
	if(runtime·stopg(gp)) {
		if(!isscanstatus(runtime·readgstatus(gp))) {
			dumpgstatus(gp);
			runtime·throw("GC not in scan state");
		}
		runtime·restartg(gp);
	}
}

// Runs on g0 and does the actual work after putting the g back on the run queue.
static void
mquiesce(G *gpmaster)
{
	G* gp;
	uint32 i;
	uint32 status;
	uint32 activeglen;

	activeglen = runtime·allglen;
	// enqueue the calling goroutine.
	runtime·restartg(gpmaster);
	for(i = 0; i < activeglen; i++) {
		gp = runtime·allg[i];
		if(runtime·readgstatus(gp) == Gdead) 
			gp->gcworkdone = true; // noop scan.
		else 
			gp->gcworkdone = false; 
		stopscanstart(gp); 
	}

	// Check that the G's gcwork (such as scanning) has been done. If not do it now. 
	// You can end up doing work here if the page trap on a Grunning Goroutine has
	// not been sprung or in some race situations. For example a runnable goes dead
	// and is started up again with a gp->gcworkdone set to false.
	for(i = 0; i < activeglen; i++) {
		gp = runtime·allg[i];
		while (!gp->gcworkdone) {
			status = runtime·readgstatus(gp);
			if(status == Gdead) {
				gp->gcworkdone = true; // scan is a noop
				break;
				//do nothing, scan not needed. 
			}
			if(status == Grunning && gp->stackguard0 == (uintptr)StackPreempt && runtime·notetsleep(&runtime·sched.stopnote, 100*1000)) // nanosecond arg 
				runtime·noteclear(&runtime·sched.stopnote);
			else 
				stopscanstart(gp);
		}
	}

	for(i = 0; i < activeglen; i++) {
		gp = runtime·allg[i];
		status = runtime·readgstatus(gp);
		if(isscanstatus(status)) {
			runtime·printf("mstopandscang:bottom: post scan bad status gp=%p has status %x\n", gp, status);
			dumpgstatus(gp);
		}
		if(!gp->gcworkdone && status != Gdead) {
			runtime·printf("mstopandscang:bottom: post scan gp=%p->gcworkdone still false\n", gp);
			dumpgstatus(gp);
		}
	}

	schedule(); // Never returns.
}

// quiesce moves all the goroutines to a GC safepoint which for now is a at preemption point.
// If the global runtime·gcphase is GCmark quiesce will ensure that all of the goroutine's stacks
// have been scanned before it returns.
void
runtime·quiesce(G* mastergp)
{
	void (*fn)(G*);

	runtime·castogscanstatus(mastergp, Grunning, Gscanenqueue);
	// Now move this to the g0 (aka m) stack.
	// g0 will potentially scan this thread and put mastergp on the runqueue 
	fn = mquiesce;
	runtime·mcall(&fn);
}

// This is used by the GC as well as the routines that do stack dumps. In the case
// of GC all the routines can be reliably stopped. This is not always the case
// when the system is in panic or being exited.
void
runtime·stoptheworld(void)
{
	int32 i;
	uint32 s;
	P *p;
	bool wait;

	// If we hold a lock, then we won't be able to stop another M
	// that is blocked trying to acquire the lock.
	if(g->m->locks > 0)
		runtime·throw("stoptheworld: holding locks");

	runtime·lock(&runtime·sched.lock);
	runtime·sched.stopwait = runtime·gomaxprocs;
	runtime·atomicstore((uint32*)&runtime·sched.gcwaiting, 1);
	preemptall();
	// stop current P
	g->m->p->status = Pgcstop; // Pgcstop is only diagnostic.
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
	runtime·unlock(&runtime·sched.lock);

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
	g->m->helpgc = -1;
}

void
runtime·starttheworld(void)
{
	P *p, *p1;
	M *mp;
	G *gp;
	bool add;

	g->m->locks++;  // disable preemption because it can be holding p in a local var
	gp = runtime·netpoll(false);  // non-blocking
	injectglist(gp);
	add = needaddgcproc();
	runtime·lock(&runtime·sched.lock);
	if(runtime·newprocs) {
		procresize(runtime·newprocs);
		runtime·newprocs = 0;
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
	runtime·unlock(&runtime·sched.lock);

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
	g->m->locks--;
	if(g->m->locks == 0 && g->preempt)  // restore the preemption request in case we've cleared it in newstack
		g->stackguard0 = StackPreempt;
}

static void mstart(void);

// Called to start an M.
#pragma textflag NOSPLIT
void
runtime·mstart(void)
{
	uintptr x, size;
	
	if(g->stack.lo == 0) {
		// Initialize stack bounds from system stack.
		// Cgo may have left stack size in stack.hi.
		size = g->stack.hi;
		if(size == 0)
			size = 8192;
		g->stack.hi = (uintptr)&x;
		g->stack.lo = g->stack.hi - size + 1024;
	}
	
	// Initialize stack guards so that we can start calling
	// both Go and C functions with stack growth prologues.
	g->stackguard0 = g->stack.lo + StackGuard;
	g->stackguard1 = g->stackguard0;
	mstart();
}

static void
mstart(void)
{
	if(g != g->m->g0)
		runtime·throw("bad runtime·mstart");

	// Record top of stack for use by mcall.
	// Once we call schedule we're never coming back,
	// so other calls can reuse this stack space.
	runtime·gosave(&g->m->g0->sched);
	g->m->g0->sched.pc = (uintptr)-1;  // make sure it is never used
	runtime·asminit();
	runtime·minit();

	// Install signal handlers; after minit so that minit can
	// prepare the thread to be able to handle the signals.
	if(g->m == &runtime·m0)
		runtime·initsig();
	
	if(g->m->mstartfn)
		g->m->mstartfn();

	if(g->m->helpgc) {
		g->m->helpgc = 0;
		stopm();
	} else if(g->m != &runtime·m0) {
		acquirep(g->m->nextp);
		g->m->nextp = nil;
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
	G *g;
	uintptr *tls;
	void (*fn)(void);
};

M *runtime·newM(void); // in proc.go

// Allocate a new m unassociated with any thread.
// Can use p for allocation context if needed.
M*
runtime·allocm(P *p)
{
	M *mp;

	g->m->locks++;  // disable GC because it can be called from sysmon
	if(g->m->p == nil)
		acquirep(p);  // temporarily borrow p for mallocs in this function
	mp = runtime·newM();
	mcommoninit(mp);

	// In case of cgo or Solaris, pthread_create will make us a stack.
	// Windows and Plan 9 will layout sched stack on OS stack.
	if(runtime·iscgo || Solaris || Windows || Plan9)
		mp->g0 = runtime·malg(-1);
	else
		mp->g0 = runtime·malg(8192);
	mp->g0->m = mp;

	if(p == g->m->p)
		releasep();
	g->m->locks--;
	if(g->m->locks == 0 && g->preempt)  // restore the preemption request in case we've cleared it in newstack
		g->stackguard0 = StackPreempt;

	return mp;
}

G *runtime·newG(void); // in proc.go

static G*
allocg(void)
{
	return runtime·newG();
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

	// Install g (= m->g0) and set the stack bounds
	// to match the current stack. We don't actually know
	// how big the stack is, like we don't know how big any
	// scheduling stack is, but we assume there's at least 32 kB,
	// which is more than enough for us.
	runtime·setg(mp->g0);
	g->stack.hi = (uintptr)(&x + 1024);
	g->stack.lo = (uintptr)(&x - 32*1024);
	g->stackguard0 = g->stack.lo + StackGuard;

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
	gp->sched.pc = (uintptr)runtime·goexit + PCQuantum;
	gp->sched.sp = gp->stack.hi;
	gp->sched.sp -= 4*sizeof(uintreg); // extra space in case of reads slightly beyond frame
	gp->sched.lr = 0;
	gp->sched.g = gp;
	gp->syscallpc = gp->sched.pc;
	gp->syscallsp = gp->sched.sp;
	// malg returns status as Gidle, change to Gsyscall before adding to allg
	// where GC will see it.
	runtime·casgstatus(gp, Gidle, Gsyscall);
	gp->m = mp;
	mp->curg = gp;
	mp->locked = LockInternal;
	mp->lockedg = gp;
	gp->lockedm = mp;
	gp->goid = runtime·xadd64(&runtime·sched.goidgen, 1);
	if(raceenabled)
		gp->racectx = runtime·racegostart(runtime·newextram);
	// put on allg for garbage collector
	runtime·allgadd(gp);

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

	// Clear m and g, and return m to the extra list.
	// After the call to setmg we can only call nosplit functions.
	mp = g->m;
	runtime·setg(nil);

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
		ts.g = mp->g0;
		ts.tls = mp->tls;
		ts.fn = runtime·mstart;
		runtime·asmcgocall(_cgo_thread_start, &ts);
		return;
	}
	runtime·newosproc(mp, (byte*)mp->g0->stack.hi);
}

// Stops execution of the current m until new work is available.
// Returns with acquired P.
static void
stopm(void)
{
	if(g->m->locks)
		runtime·throw("stopm holding locks");
	if(g->m->p)
		runtime·throw("stopm holding p");
	if(g->m->spinning) {
		g->m->spinning = false;
		runtime·xadd(&runtime·sched.nmspinning, -1);
	}

retry:
	runtime·lock(&runtime·sched.lock);
	mput(g->m);
	runtime·unlock(&runtime·sched.lock);
	runtime·notesleep(&g->m->park);
	runtime·noteclear(&g->m->park);
	if(g->m->helpgc) {
		runtime·gchelper();
		g->m->helpgc = 0;
		g->m->mcache = nil;
		goto retry;
	}
	acquirep(g->m->nextp);
	g->m->nextp = nil;
}

static void
mspinning(void)
{
	g->m->spinning = true;
}

// Schedules some M to run the p (creates an M if necessary).
// If p==nil, tries to get an idle P, if no idle P's does nothing.
static void
startm(P *p, bool spinning)
{
	M *mp;
	void (*fn)(void);

	runtime·lock(&runtime·sched.lock);
	if(p == nil) {
		p = pidleget();
		if(p == nil) {
			runtime·unlock(&runtime·sched.lock);
			if(spinning)
				runtime·xadd(&runtime·sched.nmspinning, -1);
			return;
		}
	}
	mp = mget();
	runtime·unlock(&runtime·sched.lock);
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
		runtime·cas(&runtime·sched.nmspinning, 0, 1)){
		startm(p, true);
		return;
	}
	runtime·lock(&runtime·sched.lock);
	if(runtime·sched.gcwaiting) {
		p->status = Pgcstop;
		if(--runtime·sched.stopwait == 0)
			runtime·notewakeup(&runtime·sched.stopnote);
		runtime·unlock(&runtime·sched.lock);
		return;
	}
	if(runtime·sched.runqsize) {
		runtime·unlock(&runtime·sched.lock);
		startm(p, false);
		return;
	}
	// If this is the last running P and nobody is polling network,
	// need to wakeup another M to poll network.
	if(runtime·sched.npidle == runtime·gomaxprocs-1 && runtime·atomicload64(&runtime·sched.lastpoll) != 0) {
		runtime·unlock(&runtime·sched.lock);
		startm(p, false);
		return;
	}
	pidleput(p);
	runtime·unlock(&runtime·sched.lock);
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
	uint32 status;

	if(g->m->lockedg == nil || g->m->lockedg->lockedm != g->m)
		runtime·throw("stoplockedm: inconsistent locking");
	if(g->m->p) {
		// Schedule another M to run this p.
		p = releasep();
		handoffp(p);
	}
	incidlelocked(1);
	// Wait until another thread schedules lockedg again.
	runtime·notesleep(&g->m->park);
	runtime·noteclear(&g->m->park);
	status = runtime·readgstatus(g->m->lockedg);
	if((status&~Gscan) != Grunnable){
		runtime·printf("runtime:stoplockedm: g is not Grunnable or Gscanrunnable");
		dumpgstatus(g);
		runtime·throw("stoplockedm: not runnable");
	}
	acquirep(g->m->nextp);
	g->m->nextp = nil;
}

// Schedules the locked m to run the locked gp.
static void
startlockedm(G *gp)
{
	M *mp;
	P *p;

	mp = gp->lockedm;
	if(mp == g->m)
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
	if(g->m->spinning) {
		g->m->spinning = false;
		runtime·xadd(&runtime·sched.nmspinning, -1);
	}
	p = releasep();
	runtime·lock(&runtime·sched.lock);
	p->status = Pgcstop;
	if(--runtime·sched.stopwait == 0)
		runtime·notewakeup(&runtime·sched.stopnote);
	runtime·unlock(&runtime·sched.lock);
	stopm();
}

// Schedules gp to run on the current M.
// Never returns.
static void
execute(G *gp)
{
	int32 hz;
	
	runtime·casgstatus(gp, Grunnable, Grunning);
	gp->waitsince = 0;
	gp->preempt = false;
	gp->stackguard0 = gp->stack.lo + StackGuard;
	g->m->p->schedtick++;
	g->m->curg = gp;
	gp->m = g->m;

	// Check whether the profiler needs to be turned on or off.
	hz = runtime·sched.profilehz;
	if(g->m->profilehz != hz)
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
	if(runtime·fingwait && runtime·fingwake && (gp = runtime·wakefing()) != nil)
		runtime·ready(gp);
	// local runq
	gp = runqget(g->m->p);
	if(gp)
		return gp;
	// global runq
	if(runtime·sched.runqsize) {
		runtime·lock(&runtime·sched.lock);
		gp = globrunqget(g->m->p, 0);
		runtime·unlock(&runtime·sched.lock);
		if(gp)
			return gp;
	}
	// poll network
	gp = runtime·netpoll(false);  // non-blocking
	if(gp) {
		injectglist(gp->schedlink);
		runtime·casgstatus(gp, Gwaiting, Grunnable);
		return gp;
	}
	// If number of spinning M's >= number of busy P's, block.
	// This is necessary to prevent excessive CPU consumption
	// when GOMAXPROCS>>1 but the program parallelism is low.
	if(!g->m->spinning && 2 * runtime·atomicload(&runtime·sched.nmspinning) >= runtime·gomaxprocs - runtime·atomicload(&runtime·sched.npidle))  // TODO: fast atomic
		goto stop;
	if(!g->m->spinning) {
		g->m->spinning = true;
		runtime·xadd(&runtime·sched.nmspinning, 1);
	}
	// random steal from other P's
	for(i = 0; i < 2*runtime·gomaxprocs; i++) {
		if(runtime·sched.gcwaiting)
			goto top;
		p = runtime·allp[runtime·fastrand1()%runtime·gomaxprocs];
		if(p == g->m->p)
			gp = runqget(p);
		else
			gp = runqsteal(g->m->p, p);
		if(gp)
			return gp;
	}
stop:
	// return P and block
	runtime·lock(&runtime·sched.lock);
	if(runtime·sched.gcwaiting) {
		runtime·unlock(&runtime·sched.lock);
		goto top;
	}
	if(runtime·sched.runqsize) {
		gp = globrunqget(g->m->p, 0);
		runtime·unlock(&runtime·sched.lock);
		return gp;
	}
	p = releasep();
	pidleput(p);
	runtime·unlock(&runtime·sched.lock);
	if(g->m->spinning) {
		g->m->spinning = false;
		runtime·xadd(&runtime·sched.nmspinning, -1);
	}
	// check all runqueues once again
	for(i = 0; i < runtime·gomaxprocs; i++) {
		p = runtime·allp[i];
		if(p && p->runqhead != p->runqtail) {
			runtime·lock(&runtime·sched.lock);
			p = pidleget();
			runtime·unlock(&runtime·sched.lock);
			if(p) {
				acquirep(p);
				goto top;
			}
			break;
		}
	}
	// poll network
	if(runtime·xchg64(&runtime·sched.lastpoll, 0) != 0) {
		if(g->m->p)
			runtime·throw("findrunnable: netpoll with p");
		if(g->m->spinning)
			runtime·throw("findrunnable: netpoll with spinning");
		gp = runtime·netpoll(true);  // block until new work is available
		runtime·atomicstore64(&runtime·sched.lastpoll, runtime·nanotime());
		if(gp) {
			runtime·lock(&runtime·sched.lock);
			p = pidleget();
			runtime·unlock(&runtime·sched.lock);
			if(p) {
				acquirep(p);
				injectglist(gp->schedlink);
				runtime·casgstatus(gp, Gwaiting, Grunnable);
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

	if(g->m->spinning) {
		g->m->spinning = false;
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
	runtime·lock(&runtime·sched.lock);
	for(n = 0; glist; n++) {
		gp = glist;
		glist = gp->schedlink;
		runtime·casgstatus(gp, Gwaiting, Grunnable); 
		globrunqput(gp);
	}
	runtime·unlock(&runtime·sched.lock);

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

	if(g->m->locks)
		runtime·throw("schedule: holding locks");

	if(g->m->lockedg) {
		stoplockedm();
		execute(g->m->lockedg);  // Never returns.
	}

top:
	if(runtime·sched.gcwaiting) {
		gcstopm();
		goto top;
	}

	gp = nil;
	// Check the global runnable queue once in a while to ensure fairness.
	// Otherwise two goroutines can completely occupy the local runqueue
	// by constantly respawning each other.
	tick = g->m->p->schedtick;
	// This is a fancy way to say tick%61==0,
	// it uses 2 MUL instructions instead of a single DIV and so is faster on modern processors.
	if(tick - (((uint64)tick*0x4325c53fu)>>36)*61 == 0 && runtime·sched.runqsize > 0) {
		runtime·lock(&runtime·sched.lock);
		gp = globrunqget(g->m->p, 1);
		runtime·unlock(&runtime·sched.lock);
		if(gp)
			resetspinning();
	}
	if(gp == nil) {
		gp = runqget(g->m->p);
		if(gp && g->m->spinning)
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

// dropg removes the association between m and the current goroutine m->curg (gp for short).
// Typically a caller sets gp's status away from Grunning and then
// immediately calls dropg to finish the job. The caller is also responsible
// for arranging that gp will be restarted using runtime·ready at an
// appropriate time. After calling dropg and arranging for gp to be
// readied later, the caller can do other work but eventually should
// call schedule to restart the scheduling of goroutines on this m.
static void
dropg(void)
{
	if(g->m->lockedg == nil) {
		g->m->curg->m = nil;
		g->m->curg = nil;
	}
}

// Puts the current goroutine into a waiting state and calls unlockf.
// If unlockf returns false, the goroutine is resumed.
void
runtime·park(bool(*unlockf)(G*, void*), void *lock, String reason)
{
	void (*fn)(G*);

	g->m->waitlock = lock;
	g->m->waitunlockf = unlockf;
	g->waitreason = reason;
	fn = runtime·park_m;
	runtime·mcall(&fn);
}

bool
runtime·parkunlock_c(G *gp, void *lock)
{
	USED(gp);
	runtime·unlock(lock);
	return true;
}

// Puts the current goroutine into a waiting state and unlocks the lock.
// The goroutine can be made runnable again by calling runtime·ready(gp).
void
runtime·parkunlock(Mutex *lock, String reason)
{
	runtime·park(runtime·parkunlock_c, lock, reason);
}

// runtime·park continuation on g0.
void
runtime·park_m(G *gp)
{
	bool ok;

	runtime·casgstatus(gp, Grunning, Gwaiting);
	dropg();

	if(g->m->waitunlockf) {
		ok = g->m->waitunlockf(gp, g->m->waitlock);
		g->m->waitunlockf = nil;
		g->m->waitlock = nil;
		if(!ok) {
			runtime·casgstatus(gp, Gwaiting, Grunnable); 
			execute(gp);  // Schedule it back, never returns.
		}
	}

	schedule();
}

// Gosched continuation on g0.
void
runtime·gosched_m(G *gp)
{
	uint32 status;

	status = runtime·readgstatus(gp);
	if((status&~Gscan) != Grunning){
		dumpgstatus(gp);
		runtime·throw("bad g status");
	}
	runtime·casgstatus(gp, Grunning, Grunnable);
	dropg();
	runtime·lock(&runtime·sched.lock);
	globrunqput(gp);
	runtime·unlock(&runtime·sched.lock);

	schedule();
}

// Finishes execution of the current goroutine.
// Must be NOSPLIT because it is called from Go.
#pragma textflag NOSPLIT
void
runtime·goexit1(void)
{
	void (*fn)(G*);

	if(raceenabled)
		runtime·racegoend();
	fn = goexit0;
	runtime·mcall(&fn);
}

// runtime·goexit continuation on g0.
static void
goexit0(G *gp)
{
	runtime·casgstatus(gp, Grunning, Gdead);
	gp->m = nil;
	gp->lockedm = nil;
	g->m->lockedg = nil;
	gp->paniconfault = 0;
	gp->defer = nil; // should be true already but just in case.
	gp->panic = nil; // non-nil for Goexit during panic. points at stack-allocated data.
	gp->writebuf.array = nil;
	gp->writebuf.len = 0;
	gp->writebuf.cap = 0;
	gp->waitreason.str = nil;
	gp->waitreason.len = 0;
	gp->param = nil;

	dropg();

	if(g->m->locked & ~LockExternal) {
		runtime·printf("invalid m->locked = %d\n", g->m->locked);
		runtime·throw("internal lockOSThread error");
	}	
	g->m->locked = 0;
	gfput(g->m->p, gp);
	schedule();
}

#pragma textflag NOSPLIT
static void
save(uintptr pc, uintptr sp)
{
	g->sched.pc = pc;
	g->sched.sp = sp;
	g->sched.lr = 0;
	g->sched.ret = 0;
	g->sched.ctxt = 0;
	g->sched.g = g;
}

static void entersyscall_bad(void);
static void entersyscall_sysmon(void);
static void entersyscall_gcwait(void);

// The goroutine g is about to enter a system call.
// Record that it's not using the cpu anymore.
// This is called only from the go syscall library and cgocall,
// not from the low-level system calls used by the runtime.
//
// Entersyscall cannot split the stack: the runtime·gosave must
// make g->sched refer to the caller's stack segment, because
// entersyscall is going to return immediately after.
//
// Nothing entersyscall calls can split the stack either.
// We cannot safely move the stack during an active call to syscall,
// because we do not know which of the uintptr arguments are
// really pointers (back into the stack).
// In practice, this means that we make the fast path run through
// entersyscall doing no-split things, and the slow path has to use onM
// to run bigger things on the m stack.
//
// reentersyscall is the entry point used by cgo callbacks, where explicitly
// saved SP and PC are restored. This is needed when exitsyscall will be called
// from a function further up in the call stack than the parent, as g->syscallsp
// must always point to a valid stack frame. entersyscall below is the normal
// entry point for syscalls, which obtains the SP and PC from the caller.
#pragma textflag NOSPLIT
void
runtime·reentersyscall(uintptr pc, uintptr sp)
{
	void (*fn)(void);

	// Disable preemption because during this function g is in Gsyscall status,
	// but can have inconsistent g->sched, do not let GC observe it.
	g->m->locks++;
	
	// Entersyscall must not call any function that might split/grow the stack.
	// (See details in comment above.)
	// Catch calls that might, by replacing the stack guard with something that
	// will trip any stack check and leaving a flag to tell newstack to die.
	g->stackguard0 = StackPreempt;
	g->throwsplit = 1;

	// Leave SP around for GC and traceback.
	save(pc, sp);
	g->syscallsp = sp;
	g->syscallpc = pc;
	runtime·casgstatus(g, Grunning, Gsyscall);
	if(g->syscallsp < g->stack.lo || g->stack.hi < g->syscallsp) {
		fn = entersyscall_bad;
		runtime·onM(&fn);
	}

	if(runtime·atomicload(&runtime·sched.sysmonwait)) {  // TODO: fast atomic
		fn = entersyscall_sysmon;
		runtime·onM(&fn);
		save(pc, sp);
	}

	g->m->mcache = nil;
	g->m->p->m = nil;
	runtime·atomicstore(&g->m->p->status, Psyscall);
	if(runtime·sched.gcwaiting) {
		fn = entersyscall_gcwait;
		runtime·onM(&fn);
		save(pc, sp);
	}

	// Goroutines must not split stacks in Gsyscall status (it would corrupt g->sched).
	// We set stackguard to StackPreempt so that first split stack check calls morestack.
	// Morestack detects this case and throws.
	g->stackguard0 = StackPreempt;
	g->m->locks--;
}

// Standard syscall entry used by the go syscall library and normal cgo calls.
#pragma textflag NOSPLIT
void
·entersyscall(int32 dummy)
{
	runtime·reentersyscall((uintptr)runtime·getcallerpc(&dummy), runtime·getcallersp(&dummy));
}

static void
entersyscall_bad(void)
{
	G *gp;
	
	gp = g->m->curg;
	runtime·printf("entersyscall inconsistent %p [%p,%p]\n",
		gp->syscallsp, gp->stack.lo, gp->stack.hi);
	runtime·throw("entersyscall");
}

static void
entersyscall_sysmon(void)
{
	runtime·lock(&runtime·sched.lock);
	if(runtime·atomicload(&runtime·sched.sysmonwait)) {
		runtime·atomicstore(&runtime·sched.sysmonwait, 0);
		runtime·notewakeup(&runtime·sched.sysmonnote);
	}
	runtime·unlock(&runtime·sched.lock);
}

static void
entersyscall_gcwait(void)
{
	runtime·lock(&runtime·sched.lock);
	if (runtime·sched.stopwait > 0 && runtime·cas(&g->m->p->status, Psyscall, Pgcstop)) {
		if(--runtime·sched.stopwait == 0)
			runtime·notewakeup(&runtime·sched.stopnote);
	}
	runtime·unlock(&runtime·sched.lock);
}

static void entersyscallblock_handoff(void);

// The same as runtime·entersyscall(), but with a hint that the syscall is blocking.
#pragma textflag NOSPLIT
void
·entersyscallblock(int32 dummy)
{
	void (*fn)(void);

	g->m->locks++;  // see comment in entersyscall
	g->throwsplit = 1;
	g->stackguard0 = StackPreempt;  // see comment in entersyscall

	// Leave SP around for GC and traceback.
	save((uintptr)runtime·getcallerpc(&dummy), runtime·getcallersp(&dummy));
	g->syscallsp = g->sched.sp;
	g->syscallpc = g->sched.pc;
	runtime·casgstatus(g, Grunning, Gsyscall);
	if(g->syscallsp < g->stack.lo || g->stack.hi < g->syscallsp) {
		fn = entersyscall_bad;
		runtime·onM(&fn);
	}
	
	fn = entersyscallblock_handoff;
	runtime·onM(&fn);

	// Resave for traceback during blocked call.
	save((uintptr)runtime·getcallerpc(&dummy), runtime·getcallersp(&dummy));

	g->m->locks--;
}

static void
entersyscallblock_handoff(void)
{
	handoffp(releasep());
}

// The goroutine g exited its system call.
// Arrange for it to run on a cpu again.
// This is called only from the go syscall library, not
// from the low-level system calls used by the runtime.
#pragma textflag NOSPLIT
void
·exitsyscall(int32 dummy)
{
	void (*fn)(G*);

	g->m->locks++;  // see comment in entersyscall

	if(runtime·getcallersp(&dummy) > g->syscallsp)
		runtime·throw("exitsyscall: syscall frame is no longer valid");

	g->waitsince = 0;
	if(exitsyscallfast()) {
		// There's a cpu for us, so we can run.
		g->m->p->syscalltick++;
		// We need to cas the status and scan before resuming...
		runtime·casgstatus(g, Gsyscall, Grunning);

		// Garbage collector isn't running (since we are),
		// so okay to clear syscallsp.
		g->syscallsp = (uintptr)nil;
		g->m->locks--;
		if(g->preempt) {
			// restore the preemption request in case we've cleared it in newstack
			g->stackguard0 = StackPreempt;
		} else {
			// otherwise restore the real stackguard, we've spoiled it in entersyscall/entersyscallblock
			g->stackguard0 = g->stack.lo + StackGuard;
		}
		g->throwsplit = 0;
		return;
	}

	g->m->locks--;

	// Call the scheduler.
	fn = exitsyscall0;
	runtime·mcall(&fn);

	// Scheduler returned, so we're allowed to run now.
	// Delete the syscallsp information that we left for
	// the garbage collector during the system call.
	// Must wait until now because until gosched returns
	// we don't know for sure that the garbage collector
	// is not running.
	g->syscallsp = (uintptr)nil;
	g->m->p->syscalltick++;
	g->throwsplit = 0;
}

static void exitsyscallfast_pidle(void);

#pragma textflag NOSPLIT
static bool
exitsyscallfast(void)
{
	void (*fn)(void);

	// Freezetheworld sets stopwait but does not retake P's.
	if(runtime·sched.stopwait) {
		g->m->p = nil;
		return false;
	}

	// Try to re-acquire the last P.
	if(g->m->p && g->m->p->status == Psyscall && runtime·cas(&g->m->p->status, Psyscall, Prunning)) {
		// There's a cpu for us, so we can run.
		g->m->mcache = g->m->p->mcache;
		g->m->p->m = g->m;
		return true;
	}
	// Try to get any other idle P.
	g->m->p = nil;
	if(runtime·sched.pidle) {
		fn = exitsyscallfast_pidle;
		runtime·onM(&fn);
		if(g->m->scalararg[0]) {
			g->m->scalararg[0] = 0;
			return true;
		}
	}
	return false;
}

static void
exitsyscallfast_pidle(void)
{
	P *p;

	runtime·lock(&runtime·sched.lock);
	p = pidleget();
	if(p && runtime·atomicload(&runtime·sched.sysmonwait)) {
		runtime·atomicstore(&runtime·sched.sysmonwait, 0);
		runtime·notewakeup(&runtime·sched.sysmonnote);
	}
	runtime·unlock(&runtime·sched.lock);
	if(p) {
		acquirep(p);
		g->m->scalararg[0] = 1;
	} else
		g->m->scalararg[0] = 0;
}

// runtime·exitsyscall slow path on g0.
// Failed to acquire P, enqueue gp as runnable.
static void
exitsyscall0(G *gp)
{
	P *p;

	runtime·casgstatus(gp, Gsyscall, Grunnable);
	dropg();
	runtime·lock(&runtime·sched.lock);
	p = pidleget();
	if(p == nil)
		globrunqput(gp);
	else if(runtime·atomicload(&runtime·sched.sysmonwait)) {
		runtime·atomicstore(&runtime·sched.sysmonwait, 0);
		runtime·notewakeup(&runtime·sched.sysmonnote);
	}
	runtime·unlock(&runtime·sched.lock);
	if(p) {
		acquirep(p);
		execute(gp);  // Never returns.
	}
	if(g->m->lockedg) {
		// Wait until another thread schedules gp and so m again.
		stoplockedm();
		execute(gp);  // Never returns.
	}
	stopm();
	schedule();  // Never returns.
}

static void
beforefork(void)
{
	G *gp;
	
	gp = g->m->curg;
	// Fork can hang if preempted with signals frequently enough (see issue 5517).
	// Ensure that we stay on the same M where we disable profiling.
	gp->m->locks++;
	if(gp->m->profilehz != 0)
		runtime·resetcpuprofiler(0);

	// This function is called before fork in syscall package.
	// Code between fork and exec must not allocate memory nor even try to grow stack.
	// Here we spoil g->stackguard to reliably detect any attempts to grow stack.
	// runtime_AfterFork will undo this in parent process, but not in child.
	gp->stackguard0 = StackFork;
}

// Called from syscall package before fork.
#pragma textflag NOSPLIT
void
syscall·runtime_BeforeFork(void)
{
	void (*fn)(void);
	
	fn = beforefork;
	runtime·onM(&fn);
}

static void
afterfork(void)
{
	int32 hz;
	G *gp;
	
	gp = g->m->curg;
	// See the comment in runtime_BeforeFork.
	gp->stackguard0 = gp->stack.lo + StackGuard;

	hz = runtime·sched.profilehz;
	if(hz != 0)
		runtime·resetcpuprofiler(hz);
	gp->m->locks--;
}

// Called from syscall package after fork in parent.
#pragma textflag NOSPLIT
void
syscall·runtime_AfterFork(void)
{
	void (*fn)(void);
	
	fn = afterfork;
	runtime·onM(&fn);
}

// Hook used by runtime·malg to call runtime·stackalloc on the
// scheduler stack.  This exists because runtime·stackalloc insists
// on being called on the scheduler stack, to avoid trying to grow
// the stack while allocating a new stack segment.
static void
mstackalloc(G *gp)
{
	G *newg;
	uintptr size;

	newg = g->m->ptrarg[0];
	size = g->m->scalararg[0];

	newg->stack = runtime·stackalloc(size);

	runtime·gogo(&gp->sched);
}

// Allocate a new g, with a stack big enough for stacksize bytes.
G*
runtime·malg(int32 stacksize)
{
	G *newg;
	void (*fn)(G*);

	newg = allocg();
	if(stacksize >= 0) {
		stacksize = runtime·round2(StackSystem + stacksize);
		if(g == g->m->g0) {
			// running on scheduler stack already.
			newg->stack = runtime·stackalloc(stacksize);
		} else {
			// have to call stackalloc on scheduler stack.
			g->m->scalararg[0] = stacksize;
			g->m->ptrarg[0] = newg;
			fn = mstackalloc;
			runtime·mcall(&fn);
			g->m->ptrarg[0] = nil;
		}
		newg->stackguard0 = newg->stack.lo + StackGuard;
		newg->stackguard1 = ~(uintptr)0;
	}
	return newg;
}

static void
newproc_m(void)
{
	byte *argp;
	void *callerpc;
	FuncVal *fn;
	int32 siz;

	siz = g->m->scalararg[0];
	callerpc = (void*)g->m->scalararg[1];	
	argp = g->m->ptrarg[0];
	fn = (FuncVal*)g->m->ptrarg[1];

	runtime·newproc1(fn, argp, siz, 0, callerpc);
	g->m->ptrarg[0] = nil;
	g->m->ptrarg[1] = nil;
}

// Create a new g running fn with siz bytes of arguments.
// Put it on the queue of g's waiting to run.
// The compiler turns a go statement into a call to this.
// Cannot split the stack because it assumes that the arguments
// are available sequentially after &fn; they would not be
// copied if a stack split occurred.
#pragma textflag NOSPLIT
void
runtime·newproc(int32 siz, FuncVal* fn, ...)
{
	byte *argp;
	void (*mfn)(void);

	if(thechar == '5')
		argp = (byte*)(&fn+2);  // skip caller's saved LR
	else
		argp = (byte*)(&fn+1);

	g->m->locks++;
	g->m->scalararg[0] = siz;
	g->m->scalararg[1] = (uintptr)runtime·getcallerpc(&siz);
	g->m->ptrarg[0] = argp;
	g->m->ptrarg[1] = fn;
	mfn = newproc_m;
	runtime·onM(&mfn);
	g->m->locks--;
}

void runtime·main(void);

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

	if(fn == nil) {
		g->m->throwing = -1;  // do not dump full stacks
		runtime·throw("go of nil func value");
	}
	g->m->locks++;  // disable preemption because it can be holding p in a local var
	siz = narg + nret;
	siz = (siz+7) & ~7;

	// We could allocate a larger initial stack if necessary.
	// Not worth it: this is almost always an error.
	// 4*sizeof(uintreg): extra space added below
	// sizeof(uintreg): caller's LR (arm) or return address (x86, in gostartcall).
	if(siz >= StackMin - 4*sizeof(uintreg) - sizeof(uintreg))
		runtime·throw("runtime.newproc: function arguments too large for new goroutine");

	p = g->m->p;
	if((newg = gfget(p)) == nil) {
		newg = runtime·malg(StackMin);
		runtime·casgstatus(newg, Gidle, Gdead);
		runtime·allgadd(newg); // publishes with a g->status of Gdead so GC scanner doesn't look at uninitialized stack.
	}
	if(newg->stack.hi == 0)
		runtime·throw("newproc1: newg missing stack");

	if(runtime·readgstatus(newg) != Gdead) 
		runtime·throw("newproc1: new g is not Gdead");

	sp = (byte*)newg->stack.hi;
	sp -= 4*sizeof(uintreg); // extra space in case of reads slightly beyond frame
	sp -= siz;
	runtime·memmove(sp, argp, narg);
	if(thechar == '5') {
		// caller's LR
		sp -= sizeof(void*);
		*(void**)sp = nil;
	}

	runtime·memclr((byte*)&newg->sched, sizeof newg->sched);
	newg->sched.sp = (uintptr)sp;
	newg->sched.pc = (uintptr)runtime·goexit + PCQuantum; // +PCQuantum so that previous instruction is in same function
	newg->sched.g = newg;
	runtime·gostartcallfn(&newg->sched, fn);
	newg->gopc = (uintptr)callerpc;
	runtime·casgstatus(newg, Gdead, Grunnable);

	if(p->goidcache == p->goidcacheend) {
		// Sched.goidgen is the last allocated id,
		// this batch must be [sched.goidgen+1, sched.goidgen+GoidCacheBatch].
		// At startup sched.goidgen=0, so main goroutine receives goid=1.
		p->goidcache = runtime·xadd64(&runtime·sched.goidgen, GoidCacheBatch);
		p->goidcache -= GoidCacheBatch - 1;
		p->goidcacheend = p->goidcache + GoidCacheBatch;
	}
	newg->goid = p->goidcache++;
	if(raceenabled)
		newg->racectx = runtime·racegostart((void*)callerpc);
	runqput(p, newg);

	if(runtime·atomicload(&runtime·sched.npidle) != 0 && runtime·atomicload(&runtime·sched.nmspinning) == 0 && fn->fn != runtime·main)  // TODO: fast atomic
		wakep();
	g->m->locks--;
	if(g->m->locks == 0 && g->preempt)  // restore the preemption request in case we've cleared it in newstack
		g->stackguard0 = StackPreempt;
	return newg;
}

// Put on gfree list.
// If local list is too long, transfer a batch to the global list.
static void
gfput(P *p, G *gp)
{
	uintptr stksize;

	if(runtime·readgstatus(gp) != Gdead) 
		runtime·throw("gfput: bad status (not Gdead)");

	stksize = gp->stack.hi - gp->stack.lo;
	
	if(stksize != FixedStack) {
		// non-standard stack size - free it.
		runtime·stackfree(gp->stack);
		gp->stack.lo = 0;
		gp->stack.hi = 0;
		gp->stackguard0 = 0;
	}
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
			runtime·sched.ngfree++;
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
	void (*fn)(G*);

retry:
	gp = p->gfree;
	if(gp == nil && runtime·sched.gfree) {
		runtime·lock(&runtime·sched.gflock);
		while(p->gfreecnt < 32 && runtime·sched.gfree != nil) {
			p->gfreecnt++;
			gp = runtime·sched.gfree;
			runtime·sched.gfree = gp->schedlink;
			runtime·sched.ngfree--;
			gp->schedlink = p->gfree;
			p->gfree = gp;
		}
		runtime·unlock(&runtime·sched.gflock);
		goto retry;
	}
	if(gp) {
		p->gfree = gp->schedlink;
		p->gfreecnt--;

		if(gp->stack.lo == 0) {
			// Stack was deallocated in gfput.  Allocate a new one.
			if(g == g->m->g0) {
				gp->stack = runtime·stackalloc(FixedStack);
			} else {
				g->m->scalararg[0] = FixedStack;
				g->m->ptrarg[0] = gp;
				fn = mstackalloc;
				runtime·mcall(&fn);
				g->m->ptrarg[0] = nil;
			}
			gp->stackguard0 = gp->stack.lo + StackGuard;
		} else {
			if(raceenabled)
				runtime·racemalloc((void*)gp->stack.lo, gp->stack.hi - gp->stack.lo);
		}
	}
	return gp;
}

// Purge all cached G's from gfree list to the global list.
static void
gfpurge(P *p)
{
	G *gp;

	runtime·lock(&runtime·sched.gflock);
	while(p->gfreecnt != 0) {
		p->gfreecnt--;
		gp = p->gfree;
		p->gfree = gp->schedlink;
		gp->schedlink = runtime·sched.gfree;
		runtime·sched.gfree = gp;
		runtime·sched.ngfree++;
	}
	runtime·unlock(&runtime·sched.gflock);
}

#pragma textflag NOSPLIT
void
runtime·Breakpoint(void)
{
	runtime·breakpoint();
}

// lockOSThread is called by runtime.LockOSThread and runtime.lockOSThread below
// after they modify m->locked. Do not allow preemption during this call,
// or else the m might be different in this function than in the caller.
#pragma textflag NOSPLIT
static void
lockOSThread(void)
{
	g->m->lockedg = g;
	g->lockedm = g->m;
}

#pragma textflag NOSPLIT
void
runtime·LockOSThread(void)
{
	g->m->locked |= LockExternal;
	lockOSThread();
}

#pragma textflag NOSPLIT
void
runtime·lockOSThread(void)
{
	g->m->locked += LockInternal;
	lockOSThread();
}


// unlockOSThread is called by runtime.UnlockOSThread and runtime.unlockOSThread below
// after they update m->locked. Do not allow preemption during this call,
// or else the m might be in different in this function than in the caller.
#pragma textflag NOSPLIT
static void
unlockOSThread(void)
{
	if(g->m->locked != 0)
		return;
	g->m->lockedg = nil;
	g->lockedm = nil;
}

#pragma textflag NOSPLIT
void
runtime·UnlockOSThread(void)
{
	g->m->locked &= ~LockExternal;
	unlockOSThread();
}

static void badunlockOSThread(void);

#pragma textflag NOSPLIT
void
runtime·unlockOSThread(void)
{
	void (*fn)(void);

	if(g->m->locked < LockInternal) {
		fn = badunlockOSThread;
		runtime·onM(&fn);
	}
	g->m->locked -= LockInternal;
	unlockOSThread();
}

static void
badunlockOSThread(void)
{
	runtime·throw("runtime: internal error: misuse of lockOSThread/unlockOSThread");
}

#pragma textflag NOSPLIT
int32
runtime·gcount(void)
{
	P *p, **pp;
	int32 n;

	n = runtime·allglen - runtime·sched.ngfree;
	for(pp=runtime·allp; p=*pp; pp++)
		n -= p->gfreecnt;
	// All these variables can be changed concurrently, so the result can be inconsistent.
	// But at least the current goroutine is running.
	if(n < 1)
		n = 1;
	return n;
}

int32
runtime·mcount(void)
{
	return runtime·sched.mcount;
}

static struct ProfState {
	uint32 lock;
	int32 hz;
} prof;

static void System(void) { System(); }
static void ExternalCode(void) { ExternalCode(); }
static void GC(void) { GC(); }

extern void runtime·cpuproftick(uintptr*, int32);
extern byte runtime·etext[];

// Called if we receive a SIGPROF signal.
void
runtime·sigprof(uint8 *pc, uint8 *sp, uint8 *lr, G *gp, M *mp)
{
	int32 n;
	bool traceback;
	// Do not use global m in this function, use mp instead.
	// On windows one m is sending reports about all the g's, so m means a wrong thing.
	byte m;
	uintptr stk[100];

	m = 0;
	USED(m);

	if(prof.hz == 0)
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
	   (uintptr)sp < gp->stack.lo || gp->stack.hi < (uintptr)sp ||
	   ((uint8*)runtime·gogo <= pc && pc < (uint8*)runtime·gogo + RuntimeGogoBytes))
		traceback = false;

	n = 0;
	if(traceback)
		n = runtime·gentraceback((uintptr)pc, (uintptr)sp, (uintptr)lr, gp, 0, stk, nelem(stk), nil, nil, TraceTrap);
	if(!traceback || n <= 0) {
		// Normal traceback is impossible or has failed.
		// See if it falls into several common cases.
		n = 0;
		if(mp->ncgo > 0 && mp->curg != nil &&
			mp->curg->syscallpc != 0 && mp->curg->syscallsp != 0) {
			// Cgo, we can't unwind and symbolize arbitrary C code,
			// so instead collect Go stack that leads to the cgo call.
			// This is especially important on windows, since all syscalls are cgo calls.
			n = runtime·gentraceback(mp->curg->syscallpc, mp->curg->syscallsp, 0, mp->curg, 0, stk, nelem(stk), nil, nil, 0);
		}
#ifdef GOOS_windows
		if(n == 0 && mp->libcallg != nil && mp->libcallpc != 0 && mp->libcallsp != 0) {
			// Libcall, i.e. runtime syscall on windows.
			// Collect Go stack that leads to the call.
			n = runtime·gentraceback(mp->libcallpc, mp->libcallsp, 0, mp->libcallg, 0, stk, nelem(stk), nil, nil, 0);
		}
#endif
		if(n == 0) {
			// If all of the above has failed, account it against abstract "System" or "GC".
			n = 2;
			// "ExternalCode" is better than "etext".
			if((uintptr)pc > (uintptr)runtime·etext)
				pc = (byte*)ExternalCode + PCQuantum;
			stk[0] = (uintptr)pc;
			if(mp->gcing || mp->helpgc)
				stk[1] = (uintptr)GC + PCQuantum;
			else
				stk[1] = (uintptr)System + PCQuantum;
		}
	}

	if(prof.hz != 0) {
		// Simple cas-lock to coordinate with setcpuprofilerate.
		while(!runtime·cas(&prof.lock, 0, 1))
			runtime·osyield();
		if(prof.hz != 0)
			runtime·cpuproftick(stk, n);
		runtime·atomicstore(&prof.lock, 0);
	}
	mp->mallocing--;
}

// Arrange to call fn with a traceback hz times a second.
void
runtime·setcpuprofilerate_m(void)
{
	int32 hz;
	
	hz = g->m->scalararg[0];
	g->m->scalararg[0] = 0;

	// Force sane arguments.
	if(hz < 0)
		hz = 0;

	// Disable preemption, otherwise we can be rescheduled to another thread
	// that has profiling enabled.
	g->m->locks++;

	// Stop profiler on this thread so that it is safe to lock prof.
	// if a profiling signal came in while we had prof locked,
	// it would deadlock.
	runtime·resetcpuprofiler(0);

	while(!runtime·cas(&prof.lock, 0, 1))
		runtime·osyield();
	prof.hz = hz;
	runtime·atomicstore(&prof.lock, 0);

	runtime·lock(&runtime·sched.lock);
	runtime·sched.profilehz = hz;
	runtime·unlock(&runtime·sched.lock);

	if(hz != 0)
		runtime·resetcpuprofiler(hz);

	g->m->locks--;
}

P *runtime·newP(void);

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
			p = runtime·newP();
			p->id = i;
			p->status = Pgcstop;
			runtime·atomicstorep(&runtime·allp[i], p);
		}
		if(p->mcache == nil) {
			if(old==0 && i==0)
				p->mcache = g->m->mcache;  // bootstrap
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

	if(g->m->p)
		g->m->p->m = nil;
	g->m->p = nil;
	g->m->mcache = nil;
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
	if(g->m->p || g->m->mcache)
		runtime·throw("acquirep: already in go");
	if(p->m || p->status != Pidle) {
		runtime·printf("acquirep: p->m=%p(%d) p->status=%d\n", p->m, p->m ? p->m->id : 0, p->status);
		runtime·throw("acquirep: invalid p state");
	}
	g->m->mcache = p->mcache;
	g->m->p = p;
	p->m = g->m;
	p->status = Prunning;
}

// Disassociate p and the current m.
static P*
releasep(void)
{
	P *p;

	if(g->m->p == nil || g->m->mcache == nil)
		runtime·throw("releasep: invalid arg");
	p = g->m->p;
	if(p->m != g->m || p->mcache != g->m->mcache || p->status != Prunning) {
		runtime·printf("releasep: m=%p m->p=%p p->m=%p m->mcache=%p p->mcache=%p p->status=%d\n",
			g->m, g->m->p, p->m, g->m->mcache, p->mcache, p->status);
		runtime·throw("releasep: invalid p state");
	}
	g->m->p = nil;
	g->m->mcache = nil;
	p->m = nil;
	p->status = Pidle;
	return p;
}

static void
incidlelocked(int32 v)
{
	runtime·lock(&runtime·sched.lock);
	runtime·sched.nmidlelocked += v;
	if(v > 0)
		checkdead();
	runtime·unlock(&runtime·sched.lock);
}

// Check for deadlock situation.
// The check is based on number of running M's, if 0 -> deadlock.
static void
checkdead(void)
{
	G *gp;
	P *p;
	M *mp;
	int32 run, grunning, s;
	uintptr i;

	// -1 for sysmon
	run = runtime·sched.mcount - runtime·sched.nmidle - runtime·sched.nmidlelocked - 1;
	if(run > 0)
		return;
	// If we are dying because of a signal caught on an already idle thread,
	// freezetheworld will cause all running threads to block.
	// And runtime will essentially enter into deadlock state,
	// except that there is a thread that will call runtime·exit soon.
	if(runtime·panicking > 0)
		return;
	if(run < 0) {
		runtime·printf("runtime: checkdead: nmidle=%d nmidlelocked=%d mcount=%d\n",
			runtime·sched.nmidle, runtime·sched.nmidlelocked, runtime·sched.mcount);
		runtime·throw("checkdead: inconsistent counts");
	}
	grunning = 0;
	runtime·lock(&runtime·allglock);
	for(i = 0; i < runtime·allglen; i++) {
		gp = runtime·allg[i];
		if(gp->issystem)
			continue;
		s = runtime·readgstatus(gp);
		switch(s&~Gscan) {
		case Gwaiting:
			grunning++;
			break;
		case Grunnable:
		case Grunning:
		case Gsyscall:
			runtime·unlock(&runtime·allglock);
			runtime·printf("runtime: checkdead: find g %D in status %d\n", gp->goid, s);
			runtime·throw("checkdead: runnable g");
			break;
		}
	}
	runtime·unlock(&runtime·allglock);
	if(grunning == 0)  // possible if main goroutine calls runtime·Goexit()
		runtime·throw("no goroutines (main called runtime.Goexit) - deadlock!");

	// Maybe jump time forward for playground.
	if((gp = runtime·timejump()) != nil) {
		runtime·casgstatus(gp, Gwaiting, Grunnable);
		globrunqput(gp);
 		p = pidleget();
 		if(p == nil)
 			runtime·throw("checkdead: no p for timer");
 		mp = mget();
 		if(mp == nil)
 			newm(nil, p);
 		else {
 			mp->nextp = p;
 			runtime·notewakeup(&mp->park);
 		}
 		return;
 	}

	g->m->throwing = -1;  // do not dump full stacks
	runtime·throw("all goroutines are asleep - deadlock!");
}

static void
sysmon(void)
{
	uint32 idle, delay, nscavenge;
	int64 now, unixnow, lastpoll, lasttrace, lastgc;
	int64 forcegcperiod, scavengelimit, lastscavenge, maxsleep;
	G *gp;

	// If we go two minutes without a garbage collection, force one to run.
	forcegcperiod = 2*60*1e9;
	// If a heap span goes unused for 5 minutes after a garbage collection,
	// we hand it back to the operating system.
	scavengelimit = 5*60*1e9;
	if(runtime·debug.scavenge > 0) {
		// Scavenge-a-lot for testing.
		forcegcperiod = 10*1e6;
		scavengelimit = 20*1e6;
	}
	lastscavenge = runtime·nanotime();
	nscavenge = 0;
	// Make wake-up period small enough for the sampling to be correct.
	maxsleep = forcegcperiod/2;
	if(scavengelimit < forcegcperiod)
		maxsleep = scavengelimit/2;

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
			runtime·lock(&runtime·sched.lock);
			if(runtime·atomicload(&runtime·sched.gcwaiting) || runtime·atomicload(&runtime·sched.npidle) == runtime·gomaxprocs) {
				runtime·atomicstore(&runtime·sched.sysmonwait, 1);
				runtime·unlock(&runtime·sched.lock);
				runtime·notetsleep(&runtime·sched.sysmonnote, maxsleep);
				runtime·lock(&runtime·sched.lock);
				runtime·atomicstore(&runtime·sched.sysmonwait, 0);
				runtime·noteclear(&runtime·sched.sysmonnote);
				idle = 0;
				delay = 20;
			}
			runtime·unlock(&runtime·sched.lock);
		}
		// poll network if not polled for more than 10ms
		lastpoll = runtime·atomicload64(&runtime·sched.lastpoll);
		now = runtime·nanotime();
		unixnow = runtime·unixnanotime();
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

		// check if we need to force a GC
		lastgc = runtime·atomicload64(&mstats.last_gc);
		if(lastgc != 0 && unixnow - lastgc > forcegcperiod && runtime·atomicload(&runtime·forcegc.idle)) {
			runtime·lock(&runtime·forcegc.lock);
			runtime·forcegc.idle = 0;
			runtime·forcegc.g->schedlink = nil;
			injectglist(runtime·forcegc.g);
			runtime·unlock(&runtime·forcegc.lock);
		}

		// scavenge heap once in a while
		if(lastscavenge + scavengelimit/2 < now) {
			runtime·MHeap_Scavenge(nscavenge, now, scavengelimit);
			lastscavenge = now;
			nscavenge++;
		}

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
#pragma dataflag NOPTR
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
// The actual preemption will happen at some point in the future
// and will be indicated by the gp->status no longer being
// Grunning
static bool
preemptone(P *p)
{
	M *mp;
	G *gp;

	mp = p->m;
	if(mp == nil || mp == g->m)
		return false;
	gp = mp->curg;
	if(gp == nil || gp == mp->g0)
		return false;
	gp->preempt = true;
	// Every call in a go routine checks for stack overflow by
	// comparing the current stack pointer to gp->stackguard0.
	// Setting gp->stackguard0 to StackPreempt folds
	// preemption into the normal stack overflow check.
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

	runtime·lock(&runtime·sched.lock);
	runtime·printf("SCHED %Dms: gomaxprocs=%d idleprocs=%d threads=%d spinningthreads=%d idlethreads=%d runqueue=%d",
		(now-starttime)/1000000, runtime·gomaxprocs, runtime·sched.npidle, runtime·sched.mcount,
		runtime·sched.nmspinning, runtime·sched.nmidle, runtime·sched.runqsize);
	if(detailed) {
		runtime·printf(" gcwaiting=%d nmidlelocked=%d stopwait=%d sysmonwait=%d\n",
			runtime·sched.gcwaiting, runtime·sched.nmidlelocked,
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
		runtime·unlock(&runtime·sched.lock);
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
			mp->spinning, g->m->blocked, id3);
	}
	runtime·lock(&runtime·allglock);
	for(gi = 0; gi < runtime·allglen; gi++) {
		gp = runtime·allg[gi];
		mp = gp->m;
		lockedm = gp->lockedm;
		runtime·printf("  G%D: status=%d(%S) m=%d lockedm=%d\n",
			gp->goid, runtime·readgstatus(gp), gp->waitreason, mp ? mp->id : -1,
			lockedm ? lockedm->id : -1);
	}
	runtime·unlock(&runtime·allglock);
	runtime·unlock(&runtime·sched.lock);
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
	runtime·lock(&runtime·sched.lock);
	globrunqputbatch(batch[0], batch[n], n+1);
	runtime·unlock(&runtime·sched.lock);
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
	P *p;
	G *gs;
	int32 i, j;

	p = (P*)runtime·mallocgc(sizeof(*p), nil, FlagNoScan);
	gs = (G*)runtime·mallocgc(nelem(p->runq)*sizeof(*gs), nil, FlagNoScan);

	for(i = 0; i < nelem(p->runq); i++) {
		if(runqget(p) != nil)
			runtime·throw("runq is not empty initially");
		for(j = 0; j < i; j++)
			runqput(p, &gs[i]);
		for(j = 0; j < i; j++) {
			if(runqget(p) != &gs[i]) {
				runtime·printf("bad element at iter %d/%d\n", i, j);
				runtime·throw("bad element");
			}
		}
		if(runqget(p) != nil)
			runtime·throw("runq is not empty afterwards");
	}
}

void
runtime·testSchedLocalQueueSteal(void)
{
	P *p1, *p2;
	G *gs, *gp;
	int32 i, j, s;

	p1 = (P*)runtime·mallocgc(sizeof(*p1), nil, FlagNoScan);
	p2 = (P*)runtime·mallocgc(sizeof(*p2), nil, FlagNoScan);
	gs = (G*)runtime·mallocgc(nelem(p1->runq)*sizeof(*gs), nil, FlagNoScan);

	for(i = 0; i < nelem(p1->runq); i++) {
		for(j = 0; j < i; j++) {
			gs[j].sig = 0;
			runqput(p1, &gs[j]);
		}
		gp = runqsteal(p2, p1);
		s = 0;
		if(gp) {
			s++;
			gp->sig++;
		}
		while(gp = runqget(p2)) {
			s++;
			gp->sig++;
		}
		while(gp = runqget(p1))
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

void
runtime·setmaxthreads_m(void)
{
	int32 in;
	int32 out;

	in = g->m->scalararg[0];

	runtime·lock(&runtime·sched.lock);
	out = runtime·sched.maxmcount;
	runtime·sched.maxmcount = in;
	checkmcount();
	runtime·unlock(&runtime·sched.lock);

	g->m->scalararg[0] = out;
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

#pragma textflag NOSPLIT
void
sync·runtime_procPin(intptr p)
{
	M *mp;

	mp = g->m;
	// Disable preemption.
	mp->locks++;
	p = mp->p->id;
	FLUSH(&p);
}

#pragma textflag NOSPLIT
void
sync·runtime_procUnpin()
{
	g->m->locks--;
}
