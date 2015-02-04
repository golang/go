// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

var (
	m0 m
	g0 g
)

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

const (
	// Number of goroutine ids to grab from sched.goidgen to local per-P cache at once.
	// 16 seems to provide enough amortization, but other than that it's mostly arbitrary number.
	_GoidCacheBatch = 16
)

// The bootstrap sequence is:
//
//	call osinit
//	call schedinit
//	make & queue new G
//	call runtime·mstart
//
// The new G calls runtime·main.
func schedinit() {
	// raceinit must be the first call to race detector.
	// In particular, it must be done before mallocinit below calls racemapshadow.
	_g_ := getg()
	if raceenabled {
		_g_.racectx = raceinit()
	}

	sched.maxmcount = 10000

	// Cache the framepointer experiment.  This affects stack unwinding.
	framepointer_enabled = haveexperiment("framepointer")

	tracebackinit()
	symtabinit()
	stackinit()
	mallocinit()
	mcommoninit(_g_.m)

	goargs()
	goenvs()
	parsedebugvars()
	wbshadowinit()
	gcinit()

	sched.lastpoll = uint64(nanotime())
	procs := 1
	if n := atoi(gogetenv("GOMAXPROCS")); n > 0 {
		if n > _MaxGomaxprocs {
			n = _MaxGomaxprocs
		}
		procs = n
	}
	if procresize(int32(procs)) != nil {
		throw("unknown runnable goroutine during bootstrap")
	}

	if buildVersion == "" {
		// Condition should never trigger.  This code just serves
		// to ensure runtime·buildVersion is kept in the resulting binary.
		buildVersion = "unknown"
	}
}

func newsysmon() {
	_newm(sysmon, nil)
}

func dumpgstatus(gp *g) {
	_g_ := getg()
	print("runtime: gp: gp=", gp, ", goid=", gp.goid, ", gp->atomicstatus=", readgstatus(gp), "\n")
	print("runtime:  g:  g=", _g_, ", goid=", _g_.goid, ",  g->atomicstatus=", readgstatus(_g_), "\n")
}

func checkmcount() {
	// sched lock is held
	if sched.mcount > sched.maxmcount {
		print("runtime: program exceeds ", sched.maxmcount, "-thread limit\n")
		throw("thread exhaustion")
	}
}

func mcommoninit(mp *m) {
	_g_ := getg()

	// g0 stack won't make sense for user (and is not necessary unwindable).
	if _g_ != _g_.m.g0 {
		callers(1, &mp.createstack[0], len(mp.createstack))
	}

	mp.fastrand = 0x49f6428a + uint32(mp.id) + uint32(cputicks())
	if mp.fastrand == 0 {
		mp.fastrand = 0x49f6428a
	}

	lock(&sched.lock)
	mp.id = sched.mcount
	sched.mcount++
	checkmcount()
	mpreinit(mp)
	if mp.gsignal != nil {
		mp.gsignal.stackguard1 = mp.gsignal.stack.lo + _StackGuard
	}

	// Add to allm so garbage collector doesn't free g->m
	// when it is just in a register or thread-local storage.
	mp.alllink = allm

	// NumCgoCall() iterates over allm w/o schedlock,
	// so we need to publish it safely.
	atomicstorep(unsafe.Pointer(&allm), unsafe.Pointer(mp))
	unlock(&sched.lock)
}

// Mark gp ready to run.
func ready(gp *g) {
	if trace.enabled {
		traceGoUnpark(gp)
	}

	status := readgstatus(gp)

	// Mark runnable.
	_g_ := getg()
	_g_.m.locks++ // disable preemption because it can be holding p in a local var
	if status&^_Gscan != _Gwaiting {
		dumpgstatus(gp)
		throw("bad g->status in ready")
	}

	// status is Gwaiting or Gscanwaiting, make Grunnable and put on runq
	casgstatus(gp, _Gwaiting, _Grunnable)
	runqput(_g_.m.p, gp)
	if atomicload(&sched.npidle) != 0 && atomicload(&sched.nmspinning) == 0 { // TODO: fast atomic
		wakep()
	}
	_g_.m.locks--
	if _g_.m.locks == 0 && _g_.preempt { // restore the preemption request in case we've cleared it in newstack
		_g_.stackguard0 = stackPreempt
	}
}

func gcprocs() int32 {
	// Figure out how many CPUs to use during GC.
	// Limited by gomaxprocs, number of actual CPUs, and MaxGcproc.
	lock(&sched.lock)
	n := gomaxprocs
	if n > ncpu {
		n = ncpu
	}
	if n > _MaxGcproc {
		n = _MaxGcproc
	}
	if n > sched.nmidle+1 { // one M is currently running
		n = sched.nmidle + 1
	}
	unlock(&sched.lock)
	return n
}

func needaddgcproc() bool {
	lock(&sched.lock)
	n := gomaxprocs
	if n > ncpu {
		n = ncpu
	}
	if n > _MaxGcproc {
		n = _MaxGcproc
	}
	n -= sched.nmidle + 1 // one M is currently running
	unlock(&sched.lock)
	return n > 0
}

func helpgc(nproc int32) {
	_g_ := getg()
	lock(&sched.lock)
	pos := 0
	for n := int32(1); n < nproc; n++ { // one M is currently running
		if allp[pos].mcache == _g_.m.mcache {
			pos++
		}
		mp := mget()
		if mp == nil {
			throw("gcprocs inconsistency")
		}
		mp.helpgc = n
		mp.p = allp[pos]
		mp.mcache = allp[pos].mcache
		pos++
		notewakeup(&mp.park)
	}
	unlock(&sched.lock)
}

// Similar to stoptheworld but best-effort and can be called several times.
// There is no reverse operation, used during crashing.
// This function must not lock any mutexes.
func freezetheworld() {
	if gomaxprocs == 1 {
		return
	}
	// stopwait and preemption requests can be lost
	// due to races with concurrently executing threads,
	// so try several times
	for i := 0; i < 5; i++ {
		// this should tell the scheduler to not start any new goroutines
		sched.stopwait = 0x7fffffff
		atomicstore(&sched.gcwaiting, 1)
		// this should stop running goroutines
		if !preemptall() {
			break // no running goroutines
		}
		usleep(1000)
	}
	// to be sure
	usleep(1000)
	preemptall()
	usleep(1000)
}

func isscanstatus(status uint32) bool {
	if status == _Gscan {
		throw("isscanstatus: Bad status Gscan")
	}
	return status&_Gscan == _Gscan
}

// All reads and writes of g's status go through readgstatus, casgstatus
// castogscanstatus, casfrom_Gscanstatus.
//go:nosplit
func readgstatus(gp *g) uint32 {
	return atomicload(&gp.atomicstatus)
}

// The Gscanstatuses are acting like locks and this releases them.
// If it proves to be a performance hit we should be able to make these
// simple atomic stores but for now we are going to throw if
// we see an inconsistent state.
func casfrom_Gscanstatus(gp *g, oldval, newval uint32) {
	success := false

	// Check that transition is valid.
	switch oldval {
	default:
		print("runtime: casfrom_Gscanstatus bad oldval gp=", gp, ", oldval=", hex(oldval), ", newval=", hex(newval), "\n")
		dumpgstatus(gp)
		throw("casfrom_Gscanstatus:top gp->status is not in scan state")
	case _Gscanrunnable,
		_Gscanwaiting,
		_Gscanrunning,
		_Gscansyscall:
		if newval == oldval&^_Gscan {
			success = cas(&gp.atomicstatus, oldval, newval)
		}
	case _Gscanenqueue:
		if newval == _Gwaiting {
			success = cas(&gp.atomicstatus, oldval, newval)
		}
	}
	if !success {
		print("runtime: casfrom_Gscanstatus failed gp=", gp, ", oldval=", hex(oldval), ", newval=", hex(newval), "\n")
		dumpgstatus(gp)
		throw("casfrom_Gscanstatus: gp->status is not in scan state")
	}
	if newval == _Grunning {
		gp.gcscanvalid = false
	}
}

// This will return false if the gp is not in the expected status and the cas fails.
// This acts like a lock acquire while the casfromgstatus acts like a lock release.
func castogscanstatus(gp *g, oldval, newval uint32) bool {
	switch oldval {
	case _Grunnable,
		_Gwaiting,
		_Gsyscall:
		if newval == oldval|_Gscan {
			return cas(&gp.atomicstatus, oldval, newval)
		}
	case _Grunning:
		if gp.gcscanvalid {
			print("runtime: castogscanstatus _Grunning and gp.gcscanvalid is true, newval=", hex(newval), "\n")
			throw("castogscanstatus")
		}
		if newval == _Gscanrunning || newval == _Gscanenqueue {
			return cas(&gp.atomicstatus, oldval, newval)
		}
	}
	print("runtime: castogscanstatus oldval=", hex(oldval), " newval=", hex(newval), "\n")
	throw("castogscanstatus")
	panic("not reached")
}

// If asked to move to or from a Gscanstatus this will throw. Use the castogscanstatus
// and casfrom_Gscanstatus instead.
// casgstatus will loop if the g->atomicstatus is in a Gscan status until the routine that
// put it in the Gscan state is finished.
//go:nosplit
func casgstatus(gp *g, oldval, newval uint32) {
	if (oldval&_Gscan != 0) || (newval&_Gscan != 0) || oldval == newval {
		systemstack(func() {
			print("runtime: casgstatus: oldval=", hex(oldval), " newval=", hex(newval), "\n")
			throw("casgstatus: bad incoming values")
		})
	}

	// loop if gp->atomicstatus is in a scan state giving
	// GC time to finish and change the state to oldval.
	for !cas(&gp.atomicstatus, oldval, newval) {
		if oldval == _Gwaiting && gp.atomicstatus == _Grunnable {
			systemstack(func() {
				throw("casgstatus: waiting for Gwaiting but is Grunnable")
			})
		}
		// Help GC if needed.
		// if gp.preemptscan && !gp.gcworkdone && (oldval == _Grunning || oldval == _Gsyscall) {
		// 	gp.preemptscan = false
		// 	systemstack(func() {
		// 		gcphasework(gp)
		// 	})
		// }
	}
	if newval == _Grunning {
		gp.gcscanvalid = false
	}
}

// casgstatus(gp, oldstatus, Gcopystack), assuming oldstatus is Gwaiting or Grunnable.
// Returns old status. Cannot call casgstatus directly, because we are racing with an
// async wakeup that might come in from netpoll. If we see Gwaiting from the readgstatus,
// it might have become Grunnable by the time we get to the cas. If we called casgstatus,
// it would loop waiting for the status to go back to Gwaiting, which it never will.
//go:nosplit
func casgcopystack(gp *g) uint32 {
	for {
		oldstatus := readgstatus(gp) &^ _Gscan
		if oldstatus != _Gwaiting && oldstatus != _Grunnable {
			throw("copystack: bad status, not Gwaiting or Grunnable")
		}
		if cas(&gp.atomicstatus, oldstatus, _Gcopystack) {
			return oldstatus
		}
	}
}

// stopg ensures that gp is stopped at a GC safe point where its stack can be scanned
// or in the context of a moving collector the pointers can be flipped from pointing
// to old object to pointing to new objects.
// If stopg returns true, the caller knows gp is at a GC safe point and will remain there until
// the caller calls restartg.
// If stopg returns false, the caller is not responsible for calling restartg. This can happen
// if another thread, either the gp itself or another GC thread is taking the responsibility
// to do the GC work related to this thread.
func stopg(gp *g) bool {
	for {
		if gp.gcworkdone {
			return false
		}

		switch s := readgstatus(gp); s {
		default:
			dumpgstatus(gp)
			throw("stopg: gp->atomicstatus is not valid")

		case _Gdead:
			return false

		case _Gcopystack:
			// Loop until a new stack is in place.

		case _Grunnable,
			_Gsyscall,
			_Gwaiting:
			// Claim goroutine by setting scan bit.
			if !castogscanstatus(gp, s, s|_Gscan) {
				break
			}
			// In scan state, do work.
			gcphasework(gp)
			return true

		case _Gscanrunnable,
			_Gscanwaiting,
			_Gscansyscall:
			// Goroutine already claimed by another GC helper.
			return false

		case _Grunning:
			// Claim goroutine, so we aren't racing with a status
			// transition away from Grunning.
			if !castogscanstatus(gp, _Grunning, _Gscanrunning) {
				break
			}

			// Mark gp for preemption.
			if !gp.gcworkdone {
				gp.preemptscan = true
				gp.preempt = true
				gp.stackguard0 = stackPreempt
			}

			// Unclaim.
			casfrom_Gscanstatus(gp, _Gscanrunning, _Grunning)
			return false
		}
	}
}

// The GC requests that this routine be moved from a scanmumble state to a mumble state.
func restartg(gp *g) {
	s := readgstatus(gp)
	switch s {
	default:
		dumpgstatus(gp)
		throw("restartg: unexpected status")

	case _Gdead:
		// ok

	case _Gscanrunnable,
		_Gscanwaiting,
		_Gscansyscall:
		casfrom_Gscanstatus(gp, s, s&^_Gscan)

	// Scan is now completed.
	// Goroutine now needs to be made runnable.
	// We put it on the global run queue; ready blocks on the global scheduler lock.
	case _Gscanenqueue:
		casfrom_Gscanstatus(gp, _Gscanenqueue, _Gwaiting)
		if gp != getg().m.curg {
			throw("processing Gscanenqueue on wrong m")
		}
		dropg()
		ready(gp)
	}
}

func stopscanstart(gp *g) {
	_g_ := getg()
	if _g_ == gp {
		throw("GC not moved to G0")
	}
	if stopg(gp) {
		if !isscanstatus(readgstatus(gp)) {
			dumpgstatus(gp)
			throw("GC not in scan state")
		}
		restartg(gp)
	}
}

// Runs on g0 and does the actual work after putting the g back on the run queue.
func mquiesce(gpmaster *g) {
	// enqueue the calling goroutine.
	restartg(gpmaster)

	activeglen := len(allgs)
	for i := 0; i < activeglen; i++ {
		gp := allgs[i]
		if readgstatus(gp) == _Gdead {
			gp.gcworkdone = true // noop scan.
		} else {
			gp.gcworkdone = false
		}
		stopscanstart(gp)
	}

	// Check that the G's gcwork (such as scanning) has been done. If not do it now.
	// You can end up doing work here if the page trap on a Grunning Goroutine has
	// not been sprung or in some race situations. For example a runnable goes dead
	// and is started up again with a gp->gcworkdone set to false.
	for i := 0; i < activeglen; i++ {
		gp := allgs[i]
		for !gp.gcworkdone {
			status := readgstatus(gp)
			if status == _Gdead {
				//do nothing, scan not needed.
				gp.gcworkdone = true // scan is a noop
				break
			}
			if status == _Grunning && gp.stackguard0 == uintptr(stackPreempt) && notetsleep(&sched.stopnote, 100*1000) { // nanosecond arg
				noteclear(&sched.stopnote)
			} else {
				stopscanstart(gp)
			}
		}
	}

	for i := 0; i < activeglen; i++ {
		gp := allgs[i]
		status := readgstatus(gp)
		if isscanstatus(status) {
			print("mstopandscang:bottom: post scan bad status gp=", gp, " has status ", hex(status), "\n")
			dumpgstatus(gp)
		}
		if !gp.gcworkdone && status != _Gdead {
			print("mstopandscang:bottom: post scan gp=", gp, "->gcworkdone still false\n")
			dumpgstatus(gp)
		}
	}

	schedule() // Never returns.
}

// quiesce moves all the goroutines to a GC safepoint which for now is a at preemption point.
// If the global gcphase is GCmark quiesce will ensure that all of the goroutine's stacks
// have been scanned before it returns.
func quiesce(mastergp *g) {
	castogscanstatus(mastergp, _Grunning, _Gscanenqueue)
	// Now move this to the g0 (aka m) stack.
	// g0 will potentially scan this thread and put mastergp on the runqueue
	mcall(mquiesce)
}

// This is used by the GC as well as the routines that do stack dumps. In the case
// of GC all the routines can be reliably stopped. This is not always the case
// when the system is in panic or being exited.
func stoptheworld() {
	_g_ := getg()

	// If we hold a lock, then we won't be able to stop another M
	// that is blocked trying to acquire the lock.
	if _g_.m.locks > 0 {
		throw("stoptheworld: holding locks")
	}

	lock(&sched.lock)
	sched.stopwait = gomaxprocs
	atomicstore(&sched.gcwaiting, 1)
	preemptall()
	// stop current P
	_g_.m.p.status = _Pgcstop // Pgcstop is only diagnostic.
	sched.stopwait--
	// try to retake all P's in Psyscall status
	for i := 0; i < int(gomaxprocs); i++ {
		p := allp[i]
		s := p.status
		if s == _Psyscall && cas(&p.status, s, _Pgcstop) {
			if trace.enabled {
				traceGoSysBlock(p)
				traceProcStop(p)
			}
			p.syscalltick++
			sched.stopwait--
		}
	}
	// stop idle P's
	for {
		p := pidleget()
		if p == nil {
			break
		}
		p.status = _Pgcstop
		sched.stopwait--
	}
	wait := sched.stopwait > 0
	unlock(&sched.lock)

	// wait for remaining P's to stop voluntarily
	if wait {
		for {
			// wait for 100us, then try to re-preempt in case of any races
			if notetsleep(&sched.stopnote, 100*1000) {
				noteclear(&sched.stopnote)
				break
			}
			preemptall()
		}
	}
	if sched.stopwait != 0 {
		throw("stoptheworld: not stopped")
	}
	for i := 0; i < int(gomaxprocs); i++ {
		p := allp[i]
		if p.status != _Pgcstop {
			throw("stoptheworld: not stopped")
		}
	}
}

func mhelpgc() {
	_g_ := getg()
	_g_.m.helpgc = -1
}

func starttheworld() {
	_g_ := getg()

	_g_.m.locks++        // disable preemption because it can be holding p in a local var
	gp := netpoll(false) // non-blocking
	injectglist(gp)
	add := needaddgcproc()
	lock(&sched.lock)

	procs := gomaxprocs
	if newprocs != 0 {
		procs = newprocs
		newprocs = 0
	}
	p1 := procresize(procs)
	sched.gcwaiting = 0
	if sched.sysmonwait != 0 {
		sched.sysmonwait = 0
		notewakeup(&sched.sysmonnote)
	}
	unlock(&sched.lock)

	for p1 != nil {
		p := p1
		p1 = p1.link
		if p.m != nil {
			mp := p.m
			p.m = nil
			if mp.nextp != nil {
				throw("starttheworld: inconsistent mp->nextp")
			}
			mp.nextp = p
			notewakeup(&mp.park)
		} else {
			// Start M to run P.  Do not start another M below.
			_newm(nil, p)
			add = false
		}
	}

	// Wakeup an additional proc in case we have excessive runnable goroutines
	// in local queues or in the global queue. If we don't, the proc will park itself.
	// If we have lots of excessive work, resetspinning will unpark additional procs as necessary.
	if atomicload(&sched.npidle) != 0 && atomicload(&sched.nmspinning) == 0 {
		wakep()
	}

	if add {
		// If GC could have used another helper proc, start one now,
		// in the hope that it will be available next time.
		// It would have been even better to start it before the collection,
		// but doing so requires allocating memory, so it's tricky to
		// coordinate.  This lazy approach works out in practice:
		// we don't mind if the first couple gc rounds don't have quite
		// the maximum number of procs.
		_newm(mhelpgc, nil)
	}
	_g_.m.locks--
	if _g_.m.locks == 0 && _g_.preempt { // restore the preemption request in case we've cleared it in newstack
		_g_.stackguard0 = stackPreempt
	}
}

// Called to start an M.
//go:nosplit
func mstart() {
	_g_ := getg()

	if _g_.stack.lo == 0 {
		// Initialize stack bounds from system stack.
		// Cgo may have left stack size in stack.hi.
		size := _g_.stack.hi
		if size == 0 {
			size = 8192
		}
		_g_.stack.hi = uintptr(noescape(unsafe.Pointer(&size)))
		_g_.stack.lo = _g_.stack.hi - size + 1024
	}
	// Initialize stack guards so that we can start calling
	// both Go and C functions with stack growth prologues.
	_g_.stackguard0 = _g_.stack.lo + _StackGuard
	_g_.stackguard1 = _g_.stackguard0
	mstart1()
}

func mstart1() {
	_g_ := getg()

	if _g_ != _g_.m.g0 {
		throw("bad runtime·mstart")
	}

	// Record top of stack for use by mcall.
	// Once we call schedule we're never coming back,
	// so other calls can reuse this stack space.
	gosave(&_g_.m.g0.sched)
	_g_.m.g0.sched.pc = ^uintptr(0) // make sure it is never used
	asminit()
	minit()

	// Install signal handlers; after minit so that minit can
	// prepare the thread to be able to handle the signals.
	if _g_.m == &m0 {
		initsig()
	}

	if _g_.m.mstartfn != nil {
		fn := *(*func())(unsafe.Pointer(&_g_.m.mstartfn))
		fn()
	}

	if _g_.m.helpgc != 0 {
		_g_.m.helpgc = 0
		stopm()
	} else if _g_.m != &m0 {
		acquirep(_g_.m.nextp)
		_g_.m.nextp = nil
	}
	schedule()

	// TODO(brainman): This point is never reached, because scheduler
	// does not release os threads at the moment. But once this path
	// is enabled, we must remove our seh here.
}

// When running with cgo, we call _cgo_thread_start
// to start threads for us so that we can play nicely with
// foreign code.
var cgoThreadStart unsafe.Pointer

type cgothreadstart struct {
	g   *g
	tls *uint64
	fn  unsafe.Pointer
}

// Allocate a new m unassociated with any thread.
// Can use p for allocation context if needed.
func allocm(_p_ *p) *m {
	_g_ := getg()
	_g_.m.locks++ // disable GC because it can be called from sysmon
	if _g_.m.p == nil {
		acquirep(_p_) // temporarily borrow p for mallocs in this function
	}
	mp := new(m)
	mcommoninit(mp)

	// In case of cgo or Solaris, pthread_create will make us a stack.
	// Windows and Plan 9 will layout sched stack on OS stack.
	if iscgo || GOOS == "solaris" || GOOS == "windows" || GOOS == "plan9" {
		mp.g0 = malg(-1)
	} else {
		mp.g0 = malg(8192)
	}
	mp.g0.m = mp

	if _p_ == _g_.m.p {
		releasep()
	}
	_g_.m.locks--
	if _g_.m.locks == 0 && _g_.preempt { // restore the preemption request in case we've cleared it in newstack
		_g_.stackguard0 = stackPreempt
	}

	return mp
}

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
//go:nosplit
func needm(x byte) {
	if needextram != 0 {
		// Can happen if C/C++ code calls Go from a global ctor.
		// Can not throw, because scheduler is not initialized yet.
		write(2, unsafe.Pointer(&earlycgocallback[0]), int32(len(earlycgocallback)))
		exit(1)
	}

	// Lock extra list, take head, unlock popped list.
	// nilokay=false is safe here because of the invariant above,
	// that the extra list always contains or will soon contain
	// at least one m.
	mp := lockextra(false)

	// Set needextram when we've just emptied the list,
	// so that the eventual call into cgocallbackg will
	// allocate a new m for the extra list. We delay the
	// allocation until then so that it can be done
	// after exitsyscall makes sure it is okay to be
	// running at all (that is, there's no garbage collection
	// running right now).
	mp.needextram = mp.schedlink == nil
	unlockextra(mp.schedlink)

	// Install g (= m->g0) and set the stack bounds
	// to match the current stack. We don't actually know
	// how big the stack is, like we don't know how big any
	// scheduling stack is, but we assume there's at least 32 kB,
	// which is more than enough for us.
	setg(mp.g0)
	_g_ := getg()
	_g_.stack.hi = uintptr(noescape(unsafe.Pointer(&x))) + 1024
	_g_.stack.lo = uintptr(noescape(unsafe.Pointer(&x))) - 32*1024
	_g_.stackguard0 = _g_.stack.lo + _StackGuard

	// Initialize this thread to use the m.
	asminit()
	minit()
}

var earlycgocallback = []byte("fatal error: cgo callback before cgo call\n")

// newextram allocates an m and puts it on the extra list.
// It is called with a working local m, so that it can do things
// like call schedlock and allocate.
func newextram() {
	// Create extra goroutine locked to extra m.
	// The goroutine is the context in which the cgo callback will run.
	// The sched.pc will never be returned to, but setting it to
	// goexit makes clear to the traceback routines where
	// the goroutine stack ends.
	mp := allocm(nil)
	gp := malg(4096)
	gp.sched.pc = funcPC(goexit) + _PCQuantum
	gp.sched.sp = gp.stack.hi
	gp.sched.sp -= 4 * regSize // extra space in case of reads slightly beyond frame
	gp.sched.lr = 0
	gp.sched.g = guintptr(unsafe.Pointer(gp))
	gp.syscallpc = gp.sched.pc
	gp.syscallsp = gp.sched.sp
	// malg returns status as Gidle, change to Gsyscall before adding to allg
	// where GC will see it.
	casgstatus(gp, _Gidle, _Gsyscall)
	gp.m = mp
	mp.curg = gp
	mp.locked = _LockInternal
	mp.lockedg = gp
	gp.lockedm = mp
	gp.goid = int64(xadd64(&sched.goidgen, 1))
	if raceenabled {
		gp.racectx = racegostart(funcPC(newextram))
	}
	// put on allg for garbage collector
	allgadd(gp)

	// Add m to the extra list.
	mnext := lockextra(true)
	mp.schedlink = mnext
	unlockextra(mp)
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
func dropm() {
	// Undo whatever initialization minit did during needm.
	unminit()

	// Clear m and g, and return m to the extra list.
	// After the call to setg we can only call nosplit functions
	// with no pointer manipulation.
	mp := getg().m
	mnext := lockextra(true)
	mp.schedlink = mnext

	setg(nil)
	unlockextra(mp)
}

var extram uintptr

// lockextra locks the extra list and returns the list head.
// The caller must unlock the list by storing a new list head
// to extram. If nilokay is true, then lockextra will
// return a nil list head if that's what it finds. If nilokay is false,
// lockextra will keep waiting until the list head is no longer nil.
//go:nosplit
func lockextra(nilokay bool) *m {
	const locked = 1

	for {
		old := atomicloaduintptr(&extram)
		if old == locked {
			yield := osyield
			yield()
			continue
		}
		if old == 0 && !nilokay {
			usleep(1)
			continue
		}
		if casuintptr(&extram, old, locked) {
			return (*m)(unsafe.Pointer(old))
		}
		yield := osyield
		yield()
		continue
	}
}

//go:nosplit
func unlockextra(mp *m) {
	atomicstoreuintptr(&extram, uintptr(unsafe.Pointer(mp)))
}

// Create a new m.  It will start off with a call to fn, or else the scheduler.
func _newm(fn func(), _p_ *p) {
	mp := allocm(_p_)
	mp.nextp = _p_
	mp.mstartfn = *(*unsafe.Pointer)(unsafe.Pointer(&fn))

	if iscgo {
		var ts cgothreadstart
		if _cgo_thread_start == nil {
			throw("_cgo_thread_start missing")
		}
		ts.g = mp.g0
		ts.tls = (*uint64)(unsafe.Pointer(&mp.tls[0]))
		ts.fn = unsafe.Pointer(funcPC(mstart))
		asmcgocall(_cgo_thread_start, unsafe.Pointer(&ts))
		return
	}
	newosproc(mp, unsafe.Pointer(mp.g0.stack.hi))
}

// Stops execution of the current m until new work is available.
// Returns with acquired P.
func stopm() {
	_g_ := getg()

	if _g_.m.locks != 0 {
		throw("stopm holding locks")
	}
	if _g_.m.p != nil {
		throw("stopm holding p")
	}
	if _g_.m.spinning {
		_g_.m.spinning = false
		xadd(&sched.nmspinning, -1)
	}

retry:
	lock(&sched.lock)
	mput(_g_.m)
	unlock(&sched.lock)
	notesleep(&_g_.m.park)
	noteclear(&_g_.m.park)
	if _g_.m.helpgc != 0 {
		gchelper()
		_g_.m.helpgc = 0
		_g_.m.mcache = nil
		_g_.m.p = nil
		goto retry
	}
	acquirep(_g_.m.nextp)
	_g_.m.nextp = nil
}

func mspinning() {
	getg().m.spinning = true
}

// Schedules some M to run the p (creates an M if necessary).
// If p==nil, tries to get an idle P, if no idle P's does nothing.
func startm(_p_ *p, spinning bool) {
	lock(&sched.lock)
	if _p_ == nil {
		_p_ = pidleget()
		if _p_ == nil {
			unlock(&sched.lock)
			if spinning {
				xadd(&sched.nmspinning, -1)
			}
			return
		}
	}
	mp := mget()
	unlock(&sched.lock)
	if mp == nil {
		var fn func()
		if spinning {
			fn = mspinning
		}
		_newm(fn, _p_)
		return
	}
	if mp.spinning {
		throw("startm: m is spinning")
	}
	if mp.nextp != nil {
		throw("startm: m has p")
	}
	mp.spinning = spinning
	mp.nextp = _p_
	notewakeup(&mp.park)
}

// Hands off P from syscall or locked M.
func handoffp(_p_ *p) {
	// if it has local work, start it straight away
	if _p_.runqhead != _p_.runqtail || sched.runqsize != 0 {
		startm(_p_, false)
		return
	}
	// no local work, check that there are no spinning/idle M's,
	// otherwise our help is not required
	if atomicload(&sched.nmspinning)+atomicload(&sched.npidle) == 0 && cas(&sched.nmspinning, 0, 1) { // TODO: fast atomic
		startm(_p_, true)
		return
	}
	lock(&sched.lock)
	if sched.gcwaiting != 0 {
		_p_.status = _Pgcstop
		sched.stopwait--
		if sched.stopwait == 0 {
			notewakeup(&sched.stopnote)
		}
		unlock(&sched.lock)
		return
	}
	if sched.runqsize != 0 {
		unlock(&sched.lock)
		startm(_p_, false)
		return
	}
	// If this is the last running P and nobody is polling network,
	// need to wakeup another M to poll network.
	if sched.npidle == uint32(gomaxprocs-1) && atomicload64(&sched.lastpoll) != 0 {
		unlock(&sched.lock)
		startm(_p_, false)
		return
	}
	pidleput(_p_)
	unlock(&sched.lock)
}

// Tries to add one more P to execute G's.
// Called when a G is made runnable (newproc, ready).
func wakep() {
	// be conservative about spinning threads
	if !cas(&sched.nmspinning, 0, 1) {
		return
	}
	startm(nil, true)
}

// Stops execution of the current m that is locked to a g until the g is runnable again.
// Returns with acquired P.
func stoplockedm() {
	_g_ := getg()

	if _g_.m.lockedg == nil || _g_.m.lockedg.lockedm != _g_.m {
		throw("stoplockedm: inconsistent locking")
	}
	if _g_.m.p != nil {
		// Schedule another M to run this p.
		_p_ := releasep()
		handoffp(_p_)
	}
	incidlelocked(1)
	// Wait until another thread schedules lockedg again.
	notesleep(&_g_.m.park)
	noteclear(&_g_.m.park)
	status := readgstatus(_g_.m.lockedg)
	if status&^_Gscan != _Grunnable {
		print("runtime:stoplockedm: g is not Grunnable or Gscanrunnable\n")
		dumpgstatus(_g_)
		throw("stoplockedm: not runnable")
	}
	acquirep(_g_.m.nextp)
	_g_.m.nextp = nil
}

// Schedules the locked m to run the locked gp.
func startlockedm(gp *g) {
	_g_ := getg()

	mp := gp.lockedm
	if mp == _g_.m {
		throw("startlockedm: locked to me")
	}
	if mp.nextp != nil {
		throw("startlockedm: m has p")
	}
	// directly handoff current P to the locked m
	incidlelocked(-1)
	_p_ := releasep()
	mp.nextp = _p_
	notewakeup(&mp.park)
	stopm()
}

// Stops the current m for stoptheworld.
// Returns when the world is restarted.
func gcstopm() {
	_g_ := getg()

	if sched.gcwaiting == 0 {
		throw("gcstopm: not waiting for gc")
	}
	if _g_.m.spinning {
		_g_.m.spinning = false
		xadd(&sched.nmspinning, -1)
	}
	_p_ := releasep()
	lock(&sched.lock)
	_p_.status = _Pgcstop
	sched.stopwait--
	if sched.stopwait == 0 {
		notewakeup(&sched.stopnote)
	}
	unlock(&sched.lock)
	stopm()
}

// Schedules gp to run on the current M.
// Never returns.
func execute(gp *g) {
	_g_ := getg()

	casgstatus(gp, _Grunnable, _Grunning)
	gp.waitsince = 0
	gp.preempt = false
	gp.stackguard0 = gp.stack.lo + _StackGuard
	_g_.m.p.schedtick++
	_g_.m.curg = gp
	gp.m = _g_.m

	// Check whether the profiler needs to be turned on or off.
	hz := sched.profilehz
	if _g_.m.profilehz != hz {
		resetcpuprofiler(hz)
	}

	if trace.enabled {
		traceGoStart()
	}

	gogo(&gp.sched)
}

// Finds a runnable goroutine to execute.
// Tries to steal from other P's, get g from global queue, poll network.
func findrunnable() *g {
	_g_ := getg()

top:
	if sched.gcwaiting != 0 {
		gcstopm()
		goto top
	}
	if fingwait && fingwake {
		if gp := wakefing(); gp != nil {
			ready(gp)
		}
	}

	// local runq
	if gp := runqget(_g_.m.p); gp != nil {
		return gp
	}

	// global runq
	if sched.runqsize != 0 {
		lock(&sched.lock)
		gp := globrunqget(_g_.m.p, 0)
		unlock(&sched.lock)
		if gp != nil {
			return gp
		}
	}

	// poll network - returns list of goroutines
	if gp := netpoll(false); gp != nil { // non-blocking
		injectglist(gp.schedlink)
		casgstatus(gp, _Gwaiting, _Grunnable)
		if trace.enabled {
			traceGoUnpark(gp)
		}
		return gp
	}

	// If number of spinning M's >= number of busy P's, block.
	// This is necessary to prevent excessive CPU consumption
	// when GOMAXPROCS>>1 but the program parallelism is low.
	if !_g_.m.spinning && 2*atomicload(&sched.nmspinning) >= uint32(gomaxprocs)-atomicload(&sched.npidle) { // TODO: fast atomic
		goto stop
	}
	if !_g_.m.spinning {
		_g_.m.spinning = true
		xadd(&sched.nmspinning, 1)
	}
	// random steal from other P's
	for i := 0; i < int(2*gomaxprocs); i++ {
		if sched.gcwaiting != 0 {
			goto top
		}
		_p_ := allp[fastrand1()%uint32(gomaxprocs)]
		var gp *g
		if _p_ == _g_.m.p {
			gp = runqget(_p_)
		} else {
			gp = runqsteal(_g_.m.p, _p_)
		}
		if gp != nil {
			return gp
		}
	}
stop:

	// return P and block
	lock(&sched.lock)
	if sched.gcwaiting != 0 {
		unlock(&sched.lock)
		goto top
	}
	if sched.runqsize != 0 {
		gp := globrunqget(_g_.m.p, 0)
		unlock(&sched.lock)
		return gp
	}
	_p_ := releasep()
	pidleput(_p_)
	unlock(&sched.lock)
	if _g_.m.spinning {
		_g_.m.spinning = false
		xadd(&sched.nmspinning, -1)
	}

	// check all runqueues once again
	for i := 0; i < int(gomaxprocs); i++ {
		_p_ := allp[i]
		if _p_ != nil && _p_.runqhead != _p_.runqtail {
			lock(&sched.lock)
			_p_ = pidleget()
			unlock(&sched.lock)
			if _p_ != nil {
				acquirep(_p_)
				goto top
			}
			break
		}
	}

	// poll network
	if netpollinited() && xchg64(&sched.lastpoll, 0) != 0 {
		if _g_.m.p != nil {
			throw("findrunnable: netpoll with p")
		}
		if _g_.m.spinning {
			throw("findrunnable: netpoll with spinning")
		}
		gp := netpoll(true) // block until new work is available
		atomicstore64(&sched.lastpoll, uint64(nanotime()))
		if gp != nil {
			lock(&sched.lock)
			_p_ = pidleget()
			unlock(&sched.lock)
			if _p_ != nil {
				acquirep(_p_)
				injectglist(gp.schedlink)
				casgstatus(gp, _Gwaiting, _Grunnable)
				if trace.enabled {
					traceGoUnpark(gp)
				}
				return gp
			}
			injectglist(gp)
		}
	}
	stopm()
	goto top
}

func resetspinning() {
	_g_ := getg()

	var nmspinning uint32
	if _g_.m.spinning {
		_g_.m.spinning = false
		nmspinning = xadd(&sched.nmspinning, -1)
		if nmspinning < 0 {
			throw("findrunnable: negative nmspinning")
		}
	} else {
		nmspinning = atomicload(&sched.nmspinning)
	}

	// M wakeup policy is deliberately somewhat conservative (see nmspinning handling),
	// so see if we need to wakeup another P here.
	if nmspinning == 0 && atomicload(&sched.npidle) > 0 {
		wakep()
	}
}

// Injects the list of runnable G's into the scheduler.
// Can run concurrently with GC.
func injectglist(glist *g) {
	if glist == nil {
		return
	}
	if trace.enabled {
		for gp := glist; gp != nil; gp = gp.schedlink {
			traceGoUnpark(gp)
		}
	}
	lock(&sched.lock)
	var n int
	for n = 0; glist != nil; n++ {
		gp := glist
		glist = gp.schedlink
		casgstatus(gp, _Gwaiting, _Grunnable)
		globrunqput(gp)
	}
	unlock(&sched.lock)
	for ; n != 0 && sched.npidle != 0; n-- {
		startm(nil, false)
	}
}

// One round of scheduler: find a runnable goroutine and execute it.
// Never returns.
func schedule() {
	_g_ := getg()

	if _g_.m.locks != 0 {
		throw("schedule: holding locks")
	}

	if _g_.m.lockedg != nil {
		stoplockedm()
		execute(_g_.m.lockedg) // Never returns.
	}

top:
	if sched.gcwaiting != 0 {
		gcstopm()
		goto top
	}

	var gp *g
	if trace.enabled || trace.shutdown {
		gp = traceReader()
		if gp != nil {
			casgstatus(gp, _Gwaiting, _Grunnable)
			traceGoUnpark(gp)
			resetspinning()
		}
	}
	if gp == nil {
		// Check the global runnable queue once in a while to ensure fairness.
		// Otherwise two goroutines can completely occupy the local runqueue
		// by constantly respawning each other.
		if _g_.m.p.schedtick%61 == 0 && sched.runqsize > 0 {
			lock(&sched.lock)
			gp = globrunqget(_g_.m.p, 1)
			unlock(&sched.lock)
			if gp != nil {
				resetspinning()
			}
		}
	}
	if gp == nil {
		gp = runqget(_g_.m.p)
		if gp != nil && _g_.m.spinning {
			throw("schedule: spinning with local work")
		}
	}
	if gp == nil {
		gp = findrunnable() // blocks until work is available
		resetspinning()
	}

	if gp.lockedm != nil {
		// Hands off own p to the locked m,
		// then blocks waiting for a new p.
		startlockedm(gp)
		goto top
	}

	execute(gp)
}

// dropg removes the association between m and the current goroutine m->curg (gp for short).
// Typically a caller sets gp's status away from Grunning and then
// immediately calls dropg to finish the job. The caller is also responsible
// for arranging that gp will be restarted using ready at an
// appropriate time. After calling dropg and arranging for gp to be
// readied later, the caller can do other work but eventually should
// call schedule to restart the scheduling of goroutines on this m.
func dropg() {
	_g_ := getg()

	if _g_.m.lockedg == nil {
		_g_.m.curg.m = nil
		_g_.m.curg = nil
	}
}

// Puts the current goroutine into a waiting state and calls unlockf.
// If unlockf returns false, the goroutine is resumed.
func park(unlockf func(*g, unsafe.Pointer) bool, lock unsafe.Pointer, reason string, traceev byte) {
	_g_ := getg()

	_g_.m.waitlock = lock
	_g_.m.waitunlockf = *(*unsafe.Pointer)(unsafe.Pointer(&unlockf))
	_g_.m.waittraceev = traceev
	_g_.waitreason = reason
	mcall(park_m)
}

func parkunlock_c(gp *g, lock unsafe.Pointer) bool {
	unlock((*mutex)(lock))
	return true
}

// Puts the current goroutine into a waiting state and unlocks the lock.
// The goroutine can be made runnable again by calling ready(gp).
func parkunlock(lock *mutex, reason string, traceev byte) {
	park(parkunlock_c, unsafe.Pointer(lock), reason, traceev)
}

// park continuation on g0.
func park_m(gp *g) {
	_g_ := getg()

	if trace.enabled {
		traceGoPark(_g_.m.waittraceev, gp)
	}

	casgstatus(gp, _Grunning, _Gwaiting)
	dropg()

	if _g_.m.waitunlockf != nil {
		fn := *(*func(*g, unsafe.Pointer) bool)(unsafe.Pointer(&_g_.m.waitunlockf))
		ok := fn(gp, _g_.m.waitlock)
		_g_.m.waitunlockf = nil
		_g_.m.waitlock = nil
		if !ok {
			if trace.enabled {
				traceGoUnpark(gp)
			}
			casgstatus(gp, _Gwaiting, _Grunnable)
			execute(gp) // Schedule it back, never returns.
		}
	}
	schedule()
}

func goschedImpl(gp *g) {
	status := readgstatus(gp)
	if status&^_Gscan != _Grunning {
		dumpgstatus(gp)
		throw("bad g status")
	}
	casgstatus(gp, _Grunning, _Grunnable)
	dropg()
	lock(&sched.lock)
	globrunqput(gp)
	unlock(&sched.lock)

	schedule()
}

// Gosched continuation on g0.
func gosched_m(gp *g) {
	if trace.enabled {
		traceGoSched()
	}
	goschedImpl(gp)
}

func gopreempt_m(gp *g) {
	if trace.enabled {
		traceGoPreempt()
	}
	goschedImpl(gp)
}

// Finishes execution of the current goroutine.
// Must be NOSPLIT because it is called from Go. (TODO - probably not anymore)
//go:nosplit
func goexit1() {
	if raceenabled {
		racegoend()
	}
	if trace.enabled {
		traceGoEnd()
	}
	mcall(goexit0)
}

// goexit continuation on g0.
func goexit0(gp *g) {
	_g_ := getg()

	casgstatus(gp, _Grunning, _Gdead)
	gp.m = nil
	gp.lockedm = nil
	_g_.m.lockedg = nil
	gp.paniconfault = false
	gp._defer = nil // should be true already but just in case.
	gp._panic = nil // non-nil for Goexit during panic. points at stack-allocated data.
	gp.writebuf = nil
	gp.waitreason = ""
	gp.param = nil

	dropg()

	if _g_.m.locked&^_LockExternal != 0 {
		print("invalid m->locked = ", _g_.m.locked, "\n")
		throw("internal lockOSThread error")
	}
	_g_.m.locked = 0
	gfput(_g_.m.p, gp)
	schedule()
}

//go:nosplit
//go:nowritebarrier
func save(pc, sp uintptr) {
	_g_ := getg()

	_g_.sched.pc = pc
	_g_.sched.sp = sp
	_g_.sched.lr = 0
	_g_.sched.ret = 0
	_g_.sched.ctxt = nil
	_g_.sched.g = guintptr(unsafe.Pointer(_g_))
}

// The goroutine g is about to enter a system call.
// Record that it's not using the cpu anymore.
// This is called only from the go syscall library and cgocall,
// not from the low-level system calls used by the
//
// Entersyscall cannot split the stack: the gosave must
// make g->sched refer to the caller's stack segment, because
// entersyscall is going to return immediately after.
//
// Nothing entersyscall calls can split the stack either.
// We cannot safely move the stack during an active call to syscall,
// because we do not know which of the uintptr arguments are
// really pointers (back into the stack).
// In practice, this means that we make the fast path run through
// entersyscall doing no-split things, and the slow path has to use systemstack
// to run bigger things on the system stack.
//
// reentersyscall is the entry point used by cgo callbacks, where explicitly
// saved SP and PC are restored. This is needed when exitsyscall will be called
// from a function further up in the call stack than the parent, as g->syscallsp
// must always point to a valid stack frame. entersyscall below is the normal
// entry point for syscalls, which obtains the SP and PC from the caller.
//
// Syscall tracing:
// At the start of a syscall we emit traceGoSysCall to capture the stack trace.
// If the syscall does not block, that is it, we do not emit any other events.
// If the syscall blocks (that is, P is retaken), retaker emits traceGoSysBlock;
// when syscall returns we emit traceGoSysExit and when the goroutine starts running
// (potentially instantly, if exitsyscallfast returns true) we emit traceGoStart.
// To ensure that traceGoSysExit is emitted strictly after traceGoSysBlock,
// we remember current value of syscalltick in m (_g_.m.syscalltick = _g_.m.p.syscalltick),
// whoever emits traceGoSysBlock increments p.syscalltick afterwards;
// and we wait for the increment before emitting traceGoSysExit.
// Note that the increment is done even if tracing is not enabled,
// because tracing can be enabled in the middle of syscall. We don't want the wait to hang.
//
//go:nosplit
func reentersyscall(pc, sp uintptr) {
	_g_ := getg()

	// Disable preemption because during this function g is in Gsyscall status,
	// but can have inconsistent g->sched, do not let GC observe it.
	_g_.m.locks++

	if trace.enabled {
		systemstack(traceGoSysCall)
	}

	// Entersyscall must not call any function that might split/grow the stack.
	// (See details in comment above.)
	// Catch calls that might, by replacing the stack guard with something that
	// will trip any stack check and leaving a flag to tell newstack to die.
	_g_.stackguard0 = stackPreempt
	_g_.throwsplit = true

	// Leave SP around for GC and traceback.
	save(pc, sp)
	_g_.syscallsp = sp
	_g_.syscallpc = pc
	casgstatus(_g_, _Grunning, _Gsyscall)
	if _g_.syscallsp < _g_.stack.lo || _g_.stack.hi < _g_.syscallsp {
		systemstack(func() {
			print("entersyscall inconsistent ", hex(_g_.syscallsp), " [", hex(_g_.stack.lo), ",", hex(_g_.stack.hi), "]\n")
			throw("entersyscall")
		})
	}

	if atomicload(&sched.sysmonwait) != 0 { // TODO: fast atomic
		systemstack(entersyscall_sysmon)
		save(pc, sp)
	}

	_g_.m.syscalltick = _g_.m.p.syscalltick
	_g_.m.mcache = nil
	_g_.m.p.m = nil
	atomicstore(&_g_.m.p.status, _Psyscall)
	if sched.gcwaiting != 0 {
		systemstack(entersyscall_gcwait)
		save(pc, sp)
	}

	// Goroutines must not split stacks in Gsyscall status (it would corrupt g->sched).
	// We set _StackGuard to StackPreempt so that first split stack check calls morestack.
	// Morestack detects this case and throws.
	_g_.stackguard0 = stackPreempt
	_g_.m.locks--
}

// Standard syscall entry used by the go syscall library and normal cgo calls.
//go:nosplit
func entersyscall(dummy int32) {
	reentersyscall(getcallerpc(unsafe.Pointer(&dummy)), getcallersp(unsafe.Pointer(&dummy)))
}

func entersyscall_sysmon() {
	lock(&sched.lock)
	if atomicload(&sched.sysmonwait) != 0 {
		atomicstore(&sched.sysmonwait, 0)
		notewakeup(&sched.sysmonnote)
	}
	unlock(&sched.lock)
}

func entersyscall_gcwait() {
	_g_ := getg()
	_p_ := _g_.m.p

	lock(&sched.lock)
	if sched.stopwait > 0 && cas(&_p_.status, _Psyscall, _Pgcstop) {
		if trace.enabled {
			traceGoSysBlock(_p_)
			traceProcStop(_p_)
		}
		_p_.syscalltick++
		if sched.stopwait--; sched.stopwait == 0 {
			notewakeup(&sched.stopnote)
		}
	}
	unlock(&sched.lock)
}

// The same as entersyscall(), but with a hint that the syscall is blocking.
//go:nosplit
func entersyscallblock(dummy int32) {
	_g_ := getg()

	_g_.m.locks++ // see comment in entersyscall
	_g_.throwsplit = true
	_g_.stackguard0 = stackPreempt // see comment in entersyscall
	_g_.m.syscalltick = _g_.m.p.syscalltick
	_g_.m.p.syscalltick++

	// Leave SP around for GC and traceback.
	pc := getcallerpc(unsafe.Pointer(&dummy))
	sp := getcallersp(unsafe.Pointer(&dummy))
	save(pc, sp)
	_g_.syscallsp = _g_.sched.sp
	_g_.syscallpc = _g_.sched.pc
	if _g_.syscallsp < _g_.stack.lo || _g_.stack.hi < _g_.syscallsp {
		sp1 := sp
		sp2 := _g_.sched.sp
		sp3 := _g_.syscallsp
		systemstack(func() {
			print("entersyscallblock inconsistent ", hex(sp1), " ", hex(sp2), " ", hex(sp3), " [", hex(_g_.stack.lo), ",", hex(_g_.stack.hi), "]\n")
			throw("entersyscallblock")
		})
	}
	casgstatus(_g_, _Grunning, _Gsyscall)
	if _g_.syscallsp < _g_.stack.lo || _g_.stack.hi < _g_.syscallsp {
		systemstack(func() {
			print("entersyscallblock inconsistent ", hex(sp), " ", hex(_g_.sched.sp), " ", hex(_g_.syscallsp), " [", hex(_g_.stack.lo), ",", hex(_g_.stack.hi), "]\n")
			throw("entersyscallblock")
		})
	}

	systemstack(entersyscallblock_handoff)

	// Resave for traceback during blocked call.
	save(getcallerpc(unsafe.Pointer(&dummy)), getcallersp(unsafe.Pointer(&dummy)))

	_g_.m.locks--
}

func entersyscallblock_handoff() {
	if trace.enabled {
		traceGoSysCall()
		traceGoSysBlock(getg().m.p)
	}
	handoffp(releasep())
}

// The goroutine g exited its system call.
// Arrange for it to run on a cpu again.
// This is called only from the go syscall library, not
// from the low-level system calls used by the
//go:nosplit
func exitsyscall(dummy int32) {
	_g_ := getg()

	_g_.m.locks++ // see comment in entersyscall
	if getcallersp(unsafe.Pointer(&dummy)) > _g_.syscallsp {
		throw("exitsyscall: syscall frame is no longer valid")
	}

	_g_.waitsince = 0
	oldp := _g_.m.p
	if exitsyscallfast() {
		if _g_.m.mcache == nil {
			throw("lost mcache")
		}
		if trace.enabled {
			if oldp != _g_.m.p || _g_.m.syscalltick != _g_.m.p.syscalltick {
				systemstack(traceGoStart)
			}
		}
		// There's a cpu for us, so we can run.
		_g_.m.p.syscalltick++
		// We need to cas the status and scan before resuming...
		casgstatus(_g_, _Gsyscall, _Grunning)

		// Garbage collector isn't running (since we are),
		// so okay to clear syscallsp.
		_g_.syscallsp = 0
		_g_.m.locks--
		if _g_.preempt {
			// restore the preemption request in case we've cleared it in newstack
			_g_.stackguard0 = stackPreempt
		} else {
			// otherwise restore the real _StackGuard, we've spoiled it in entersyscall/entersyscallblock
			_g_.stackguard0 = _g_.stack.lo + _StackGuard
		}
		_g_.throwsplit = false
		return
	}

	if trace.enabled {
		// Wait till traceGoSysBlock event is emited.
		// This ensures consistency of the trace (the goroutine is started after it is blocked).
		for oldp != nil && oldp.syscalltick == _g_.m.syscalltick {
			osyield()
		}
		systemstack(traceGoSysExit)
	}

	_g_.m.locks--

	// Call the scheduler.
	mcall(exitsyscall0)

	if _g_.m.mcache == nil {
		throw("lost mcache")
	}

	// Scheduler returned, so we're allowed to run now.
	// Delete the syscallsp information that we left for
	// the garbage collector during the system call.
	// Must wait until now because until gosched returns
	// we don't know for sure that the garbage collector
	// is not running.
	_g_.syscallsp = 0
	_g_.m.p.syscalltick++
	_g_.throwsplit = false
}

//go:nosplit
func exitsyscallfast() bool {
	_g_ := getg()

	// Freezetheworld sets stopwait but does not retake P's.
	if sched.stopwait != 0 {
		_g_.m.mcache = nil
		_g_.m.p = nil
		return false
	}

	// Try to re-acquire the last P.
	if _g_.m.p != nil && _g_.m.p.status == _Psyscall && cas(&_g_.m.p.status, _Psyscall, _Prunning) {
		// There's a cpu for us, so we can run.
		_g_.m.mcache = _g_.m.p.mcache
		_g_.m.p.m = _g_.m
		if _g_.m.syscalltick != _g_.m.p.syscalltick {
			if trace.enabled {
				// The p was retaken and then enter into syscall again (since _g_.m.syscalltick has changed).
				// traceGoSysBlock for this syscall was already emitted,
				// but here we effectively retake the p from the new syscall running on the same p.
				systemstack(func() {
					// Denote blocking of the new syscall.
					traceGoSysBlock(_g_.m.p)
					// Denote completion of the current syscall.
					traceGoSysExit()
				})
			}
			_g_.m.p.syscalltick++
		}
		return true
	}

	// Try to get any other idle P.
	oldp := _g_.m.p
	_g_.m.mcache = nil
	_g_.m.p = nil
	if sched.pidle != nil {
		var ok bool
		systemstack(func() {
			ok = exitsyscallfast_pidle()
			if ok && trace.enabled {
				if oldp != nil {
					// Wait till traceGoSysBlock event is emited.
					// This ensures consistency of the trace (the goroutine is started after it is blocked).
					for oldp.syscalltick == _g_.m.syscalltick {
						osyield()
					}
				}
				traceGoSysExit()
			}
		})
		if ok {
			return true
		}
	}
	return false
}

func exitsyscallfast_pidle() bool {
	lock(&sched.lock)
	_p_ := pidleget()
	if _p_ != nil && atomicload(&sched.sysmonwait) != 0 {
		atomicstore(&sched.sysmonwait, 0)
		notewakeup(&sched.sysmonnote)
	}
	unlock(&sched.lock)
	if _p_ != nil {
		acquirep(_p_)
		return true
	}
	return false
}

// exitsyscall slow path on g0.
// Failed to acquire P, enqueue gp as runnable.
func exitsyscall0(gp *g) {
	_g_ := getg()

	casgstatus(gp, _Gsyscall, _Grunnable)
	dropg()
	lock(&sched.lock)
	_p_ := pidleget()
	if _p_ == nil {
		globrunqput(gp)
	} else if atomicload(&sched.sysmonwait) != 0 {
		atomicstore(&sched.sysmonwait, 0)
		notewakeup(&sched.sysmonnote)
	}
	unlock(&sched.lock)
	if _p_ != nil {
		acquirep(_p_)
		execute(gp) // Never returns.
	}
	if _g_.m.lockedg != nil {
		// Wait until another thread schedules gp and so m again.
		stoplockedm()
		execute(gp) // Never returns.
	}
	stopm()
	schedule() // Never returns.
}

func beforefork() {
	gp := getg().m.curg

	// Fork can hang if preempted with signals frequently enough (see issue 5517).
	// Ensure that we stay on the same M where we disable profiling.
	gp.m.locks++
	if gp.m.profilehz != 0 {
		resetcpuprofiler(0)
	}

	// This function is called before fork in syscall package.
	// Code between fork and exec must not allocate memory nor even try to grow stack.
	// Here we spoil g->_StackGuard to reliably detect any attempts to grow stack.
	// runtime_AfterFork will undo this in parent process, but not in child.
	gp.stackguard0 = stackFork
}

// Called from syscall package before fork.
//go:linkname syscall_runtime_BeforeFork syscall.runtime_BeforeFork
//go:nosplit
func syscall_runtime_BeforeFork() {
	systemstack(beforefork)
}

func afterfork() {
	gp := getg().m.curg

	// See the comment in beforefork.
	gp.stackguard0 = gp.stack.lo + _StackGuard

	hz := sched.profilehz
	if hz != 0 {
		resetcpuprofiler(hz)
	}
	gp.m.locks--
}

// Called from syscall package after fork in parent.
//go:linkname syscall_runtime_AfterFork syscall.runtime_AfterFork
//go:nosplit
func syscall_runtime_AfterFork() {
	systemstack(afterfork)
}

// Allocate a new g, with a stack big enough for stacksize bytes.
func malg(stacksize int32) *g {
	newg := new(g)
	if stacksize >= 0 {
		stacksize = round2(_StackSystem + stacksize)
		systemstack(func() {
			newg.stack = stackalloc(uint32(stacksize))
		})
		newg.stackguard0 = newg.stack.lo + _StackGuard
		newg.stackguard1 = ^uintptr(0)
	}
	return newg
}

// Create a new g running fn with siz bytes of arguments.
// Put it on the queue of g's waiting to run.
// The compiler turns a go statement into a call to this.
// Cannot split the stack because it assumes that the arguments
// are available sequentially after &fn; they would not be
// copied if a stack split occurred.
//go:nosplit
func newproc(siz int32, fn *funcval) {
	argp := add(unsafe.Pointer(&fn), ptrSize)
	pc := getcallerpc(unsafe.Pointer(&siz))
	systemstack(func() {
		newproc1(fn, (*uint8)(argp), siz, 0, pc)
	})
}

// Create a new g running fn with narg bytes of arguments starting
// at argp and returning nret bytes of results.  callerpc is the
// address of the go statement that created this.  The new g is put
// on the queue of g's waiting to run.
func newproc1(fn *funcval, argp *uint8, narg int32, nret int32, callerpc uintptr) *g {
	_g_ := getg()

	if fn == nil {
		_g_.m.throwing = -1 // do not dump full stacks
		throw("go of nil func value")
	}
	_g_.m.locks++ // disable preemption because it can be holding p in a local var
	siz := narg + nret
	siz = (siz + 7) &^ 7

	// We could allocate a larger initial stack if necessary.
	// Not worth it: this is almost always an error.
	// 4*sizeof(uintreg): extra space added below
	// sizeof(uintreg): caller's LR (arm) or return address (x86, in gostartcall).
	if siz >= _StackMin-4*regSize-regSize {
		throw("newproc: function arguments too large for new goroutine")
	}

	_p_ := _g_.m.p
	newg := gfget(_p_)
	if newg == nil {
		newg = malg(_StackMin)
		casgstatus(newg, _Gidle, _Gdead)
		allgadd(newg) // publishes with a g->status of Gdead so GC scanner doesn't look at uninitialized stack.
	}
	if newg.stack.hi == 0 {
		throw("newproc1: newg missing stack")
	}

	if readgstatus(newg) != _Gdead {
		throw("newproc1: new g is not Gdead")
	}

	sp := newg.stack.hi
	sp -= 4 * regSize // extra space in case of reads slightly beyond frame
	sp -= uintptr(siz)
	memmove(unsafe.Pointer(sp), unsafe.Pointer(argp), uintptr(narg))
	if hasLinkRegister {
		// caller's LR
		sp -= ptrSize
		*(*unsafe.Pointer)(unsafe.Pointer(sp)) = nil
	}

	memclr(unsafe.Pointer(&newg.sched), unsafe.Sizeof(newg.sched))
	newg.sched.sp = sp
	newg.sched.pc = funcPC(goexit) + _PCQuantum // +PCQuantum so that previous instruction is in same function
	newg.sched.g = guintptr(unsafe.Pointer(newg))
	gostartcallfn(&newg.sched, fn)
	newg.gopc = callerpc
	newg.startpc = fn.fn
	casgstatus(newg, _Gdead, _Grunnable)

	if _p_.goidcache == _p_.goidcacheend {
		// Sched.goidgen is the last allocated id,
		// this batch must be [sched.goidgen+1, sched.goidgen+GoidCacheBatch].
		// At startup sched.goidgen=0, so main goroutine receives goid=1.
		_p_.goidcache = xadd64(&sched.goidgen, _GoidCacheBatch)
		_p_.goidcache -= _GoidCacheBatch - 1
		_p_.goidcacheend = _p_.goidcache + _GoidCacheBatch
	}
	newg.goid = int64(_p_.goidcache)
	_p_.goidcache++
	if raceenabled {
		newg.racectx = racegostart(callerpc)
	}
	if trace.enabled {
		traceGoCreate(newg, newg.startpc)
	}
	runqput(_p_, newg)

	if atomicload(&sched.npidle) != 0 && atomicload(&sched.nmspinning) == 0 && unsafe.Pointer(fn.fn) != unsafe.Pointer(funcPC(main)) { // TODO: fast atomic
		wakep()
	}
	_g_.m.locks--
	if _g_.m.locks == 0 && _g_.preempt { // restore the preemption request in case we've cleared it in newstack
		_g_.stackguard0 = stackPreempt
	}
	return newg
}

// Put on gfree list.
// If local list is too long, transfer a batch to the global list.
func gfput(_p_ *p, gp *g) {
	if readgstatus(gp) != _Gdead {
		throw("gfput: bad status (not Gdead)")
	}

	stksize := gp.stack.hi - gp.stack.lo

	if stksize != _FixedStack {
		// non-standard stack size - free it.
		stackfree(gp.stack)
		gp.stack.lo = 0
		gp.stack.hi = 0
		gp.stackguard0 = 0
	}

	gp.schedlink = _p_.gfree
	_p_.gfree = gp
	_p_.gfreecnt++
	if _p_.gfreecnt >= 64 {
		lock(&sched.gflock)
		for _p_.gfreecnt >= 32 {
			_p_.gfreecnt--
			gp = _p_.gfree
			_p_.gfree = gp.schedlink
			gp.schedlink = sched.gfree
			sched.gfree = gp
			sched.ngfree++
		}
		unlock(&sched.gflock)
	}
}

// Get from gfree list.
// If local list is empty, grab a batch from global list.
func gfget(_p_ *p) *g {
retry:
	gp := _p_.gfree
	if gp == nil && sched.gfree != nil {
		lock(&sched.gflock)
		for _p_.gfreecnt < 32 && sched.gfree != nil {
			_p_.gfreecnt++
			gp = sched.gfree
			sched.gfree = gp.schedlink
			sched.ngfree--
			gp.schedlink = _p_.gfree
			_p_.gfree = gp
		}
		unlock(&sched.gflock)
		goto retry
	}
	if gp != nil {
		_p_.gfree = gp.schedlink
		_p_.gfreecnt--
		if gp.stack.lo == 0 {
			// Stack was deallocated in gfput.  Allocate a new one.
			systemstack(func() {
				gp.stack = stackalloc(_FixedStack)
			})
			gp.stackguard0 = gp.stack.lo + _StackGuard
		} else {
			if raceenabled {
				racemalloc(unsafe.Pointer(gp.stack.lo), gp.stack.hi-gp.stack.lo)
			}
		}
	}
	return gp
}

// Purge all cached G's from gfree list to the global list.
func gfpurge(_p_ *p) {
	lock(&sched.gflock)
	for _p_.gfreecnt != 0 {
		_p_.gfreecnt--
		gp := _p_.gfree
		_p_.gfree = gp.schedlink
		gp.schedlink = sched.gfree
		sched.gfree = gp
		sched.ngfree++
	}
	unlock(&sched.gflock)
}

// Breakpoint executes a breakpoint trap.
func Breakpoint() {
	breakpoint()
}

// dolockOSThread is called by LockOSThread and lockOSThread below
// after they modify m.locked. Do not allow preemption during this call,
// or else the m might be different in this function than in the caller.
//go:nosplit
func dolockOSThread() {
	_g_ := getg()
	_g_.m.lockedg = _g_
	_g_.lockedm = _g_.m
}

//go:nosplit

// LockOSThread wires the calling goroutine to its current operating system thread.
// Until the calling goroutine exits or calls UnlockOSThread, it will always
// execute in that thread, and no other goroutine can.
func LockOSThread() {
	getg().m.locked |= _LockExternal
	dolockOSThread()
}

//go:nosplit
func lockOSThread() {
	getg().m.locked += _LockInternal
	dolockOSThread()
}

// dounlockOSThread is called by UnlockOSThread and unlockOSThread below
// after they update m->locked. Do not allow preemption during this call,
// or else the m might be in different in this function than in the caller.
//go:nosplit
func dounlockOSThread() {
	_g_ := getg()
	if _g_.m.locked != 0 {
		return
	}
	_g_.m.lockedg = nil
	_g_.lockedm = nil
}

//go:nosplit

// UnlockOSThread unwires the calling goroutine from its fixed operating system thread.
// If the calling goroutine has not called LockOSThread, UnlockOSThread is a no-op.
func UnlockOSThread() {
	getg().m.locked &^= _LockExternal
	dounlockOSThread()
}

//go:nosplit
func unlockOSThread() {
	_g_ := getg()
	if _g_.m.locked < _LockInternal {
		systemstack(badunlockosthread)
	}
	_g_.m.locked -= _LockInternal
	dounlockOSThread()
}

func badunlockosthread() {
	throw("runtime: internal error: misuse of lockOSThread/unlockOSThread")
}

func gcount() int32 {
	n := int32(allglen) - sched.ngfree
	for i := 0; ; i++ {
		_p_ := allp[i]
		if _p_ == nil {
			break
		}
		n -= _p_.gfreecnt
	}

	// All these variables can be changed concurrently, so the result can be inconsistent.
	// But at least the current goroutine is running.
	if n < 1 {
		n = 1
	}
	return n
}

func mcount() int32 {
	return sched.mcount
}

var prof struct {
	lock uint32
	hz   int32
}

func _System()       { _System() }
func _ExternalCode() { _ExternalCode() }
func _GC()           { _GC() }

var etext struct{}

// Called if we receive a SIGPROF signal.
func sigprof(pc *uint8, sp *uint8, lr *uint8, gp *g, mp *m) {
	var n int32
	var traceback bool
	var stk [100]uintptr

	if prof.hz == 0 {
		return
	}

	// Profiling runs concurrently with GC, so it must not allocate.
	mp.mallocing++

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
	// either corresponds to a user g throughout (as in asmcgocall)
	// or because it has been arranged to look like a user g frame
	// (as in cgocallback_gofunc). In this case, since the entire
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
	// separately, as in gogo.
	//
	// Because gogo is the only instance, we check whether the PC lies
	// within that function, and if so, not ask for a traceback. This approach
	// requires knowing the size of the gogo function, which we
	// record in arch_*.h and check in runtime_test.go.
	//
	// There is another apparently viable approach, recorded here in case
	// the "PC within gogo" check turns out not to be usable.
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
	// in gogo.
	traceback = true
	usp := uintptr(unsafe.Pointer(sp))
	gogo := funcPC(gogo)
	if gp == nil || gp != mp.curg ||
		usp < gp.stack.lo || gp.stack.hi < usp ||
		(gogo <= uintptr(unsafe.Pointer(pc)) && uintptr(unsafe.Pointer(pc)) < gogo+_RuntimeGogoBytes) {
		traceback = false
	}

	n = 0
	if traceback {
		n = int32(gentraceback(uintptr(unsafe.Pointer(pc)), uintptr(unsafe.Pointer(sp)), uintptr(unsafe.Pointer(lr)), gp, 0, &stk[0], len(stk), nil, nil, _TraceTrap))
	}
	if !traceback || n <= 0 {
		// Normal traceback is impossible or has failed.
		// See if it falls into several common cases.
		n = 0
		if mp.ncgo > 0 && mp.curg != nil && mp.curg.syscallpc != 0 && mp.curg.syscallsp != 0 {
			// Cgo, we can't unwind and symbolize arbitrary C code,
			// so instead collect Go stack that leads to the cgo call.
			// This is especially important on windows, since all syscalls are cgo calls.
			n = int32(gentraceback(mp.curg.syscallpc, mp.curg.syscallsp, 0, mp.curg, 0, &stk[0], len(stk), nil, nil, 0))
		}
		if GOOS == "windows" && n == 0 && mp.libcallg != nil && mp.libcallpc != 0 && mp.libcallsp != 0 {
			// Libcall, i.e. runtime syscall on windows.
			// Collect Go stack that leads to the call.
			n = int32(gentraceback(mp.libcallpc, mp.libcallsp, 0, mp.libcallg, 0, &stk[0], len(stk), nil, nil, 0))
		}
		if n == 0 {
			// If all of the above has failed, account it against abstract "System" or "GC".
			n = 2
			// "ExternalCode" is better than "etext".
			if uintptr(unsafe.Pointer(pc)) > uintptr(unsafe.Pointer(&etext)) {
				pc = (*uint8)(unsafe.Pointer(uintptr(funcPC(_ExternalCode) + _PCQuantum)))
			}
			stk[0] = uintptr(unsafe.Pointer(pc))
			if mp.preemptoff != "" || mp.helpgc != 0 {
				stk[1] = funcPC(_GC) + _PCQuantum
			} else {
				stk[1] = funcPC(_System) + _PCQuantum
			}
		}
	}

	if prof.hz != 0 {
		// Simple cas-lock to coordinate with setcpuprofilerate.
		for !cas(&prof.lock, 0, 1) {
			osyield()
		}
		if prof.hz != 0 {
			cpuproftick(&stk[0], n)
		}
		atomicstore(&prof.lock, 0)
	}
	mp.mallocing--
}

// Arrange to call fn with a traceback hz times a second.
func setcpuprofilerate_m(hz int32) {
	// Force sane arguments.
	if hz < 0 {
		hz = 0
	}

	// Disable preemption, otherwise we can be rescheduled to another thread
	// that has profiling enabled.
	_g_ := getg()
	_g_.m.locks++

	// Stop profiler on this thread so that it is safe to lock prof.
	// if a profiling signal came in while we had prof locked,
	// it would deadlock.
	resetcpuprofiler(0)

	for !cas(&prof.lock, 0, 1) {
		osyield()
	}
	prof.hz = hz
	atomicstore(&prof.lock, 0)

	lock(&sched.lock)
	sched.profilehz = hz
	unlock(&sched.lock)

	if hz != 0 {
		resetcpuprofiler(hz)
	}

	_g_.m.locks--
}

// Change number of processors.  The world is stopped, sched is locked.
// gcworkbufs are not being modified by either the GC or
// the write barrier code.
// Returns list of Ps with local work, they need to be scheduled by the caller.
func procresize(nprocs int32) *p {
	old := gomaxprocs
	if old < 0 || old > _MaxGomaxprocs || nprocs <= 0 || nprocs > _MaxGomaxprocs {
		throw("procresize: invalid arg")
	}
	if trace.enabled {
		traceGomaxprocs(nprocs)
	}

	// initialize new P's
	for i := int32(0); i < nprocs; i++ {
		pp := allp[i]
		if pp == nil {
			pp = new(p)
			pp.id = i
			pp.status = _Pgcstop
			atomicstorep(unsafe.Pointer(&allp[i]), unsafe.Pointer(pp))
		}
		if pp.mcache == nil {
			if old == 0 && i == 0 {
				if getg().m.mcache == nil {
					throw("missing mcache?")
				}
				pp.mcache = getg().m.mcache // bootstrap
			} else {
				pp.mcache = allocmcache()
			}
		}
	}

	// free unused P's
	for i := nprocs; i < old; i++ {
		p := allp[i]
		if trace.enabled {
			if p == getg().m.p {
				// moving to p[0], pretend that we were descheduled
				// and then scheduled again to keep the trace sane.
				traceGoSched()
				traceProcStop(p)
			}
		}
		// move all runable goroutines to the global queue
		for p.runqhead != p.runqtail {
			// pop from tail of local queue
			p.runqtail--
			gp := p.runq[p.runqtail%uint32(len(p.runq))]
			// push onto head of global queue
			gp.schedlink = sched.runqhead
			sched.runqhead = gp
			if sched.runqtail == nil {
				sched.runqtail = gp
			}
			sched.runqsize++
		}
		freemcache(p.mcache)
		p.mcache = nil
		gfpurge(p)
		traceProcFree(p)
		p.status = _Pdead
		// can't free P itself because it can be referenced by an M in syscall
	}

	_g_ := getg()
	if _g_.m.p != nil && _g_.m.p.id < nprocs {
		// continue to use the current P
		_g_.m.p.status = _Prunning
	} else {
		// release the current P and acquire allp[0]
		if _g_.m.p != nil {
			_g_.m.p.m = nil
		}
		_g_.m.p = nil
		_g_.m.mcache = nil
		p := allp[0]
		p.m = nil
		p.status = _Pidle
		acquirep(p)
		if trace.enabled {
			traceGoStart()
		}
	}
	var runnablePs *p
	for i := nprocs - 1; i >= 0; i-- {
		p := allp[i]
		if _g_.m.p == p {
			continue
		}
		p.status = _Pidle
		if p.runqhead == p.runqtail {
			pidleput(p)
		} else {
			p.m = mget()
			p.link = runnablePs
			runnablePs = p
		}
	}
	var int32p *int32 = &gomaxprocs // make compiler check that gomaxprocs is an int32
	atomicstore((*uint32)(unsafe.Pointer(int32p)), uint32(nprocs))
	return runnablePs
}

// Associate p and the current m.
func acquirep(_p_ *p) {
	_g_ := getg()

	if _g_.m.p != nil || _g_.m.mcache != nil {
		throw("acquirep: already in go")
	}
	if _p_.m != nil || _p_.status != _Pidle {
		id := int32(0)
		if _p_.m != nil {
			id = _p_.m.id
		}
		print("acquirep: p->m=", _p_.m, "(", id, ") p->status=", _p_.status, "\n")
		throw("acquirep: invalid p state")
	}
	_g_.m.mcache = _p_.mcache
	_g_.m.p = _p_
	_p_.m = _g_.m
	_p_.status = _Prunning

	if trace.enabled {
		traceProcStart()
	}
}

// Disassociate p and the current m.
func releasep() *p {
	_g_ := getg()

	if _g_.m.p == nil || _g_.m.mcache == nil {
		throw("releasep: invalid arg")
	}
	_p_ := _g_.m.p
	if _p_.m != _g_.m || _p_.mcache != _g_.m.mcache || _p_.status != _Prunning {
		print("releasep: m=", _g_.m, " m->p=", _g_.m.p, " p->m=", _p_.m, " m->mcache=", _g_.m.mcache, " p->mcache=", _p_.mcache, " p->status=", _p_.status, "\n")
		throw("releasep: invalid p state")
	}
	if trace.enabled {
		traceProcStop(_g_.m.p)
	}
	_g_.m.p = nil
	_g_.m.mcache = nil
	_p_.m = nil
	_p_.status = _Pidle
	return _p_
}

func incidlelocked(v int32) {
	lock(&sched.lock)
	sched.nmidlelocked += v
	if v > 0 {
		checkdead()
	}
	unlock(&sched.lock)
}

// Check for deadlock situation.
// The check is based on number of running M's, if 0 -> deadlock.
func checkdead() {
	// If we are dying because of a signal caught on an already idle thread,
	// freezetheworld will cause all running threads to block.
	// And runtime will essentially enter into deadlock state,
	// except that there is a thread that will call exit soon.
	if panicking > 0 {
		return
	}

	// -1 for sysmon
	run := sched.mcount - sched.nmidle - sched.nmidlelocked - 1
	if run > 0 {
		return
	}
	if run < 0 {
		print("runtime: checkdead: nmidle=", sched.nmidle, " nmidlelocked=", sched.nmidlelocked, " mcount=", sched.mcount, "\n")
		throw("checkdead: inconsistent counts")
	}

	grunning := 0
	lock(&allglock)
	for i := 0; i < len(allgs); i++ {
		gp := allgs[i]
		if gp.issystem {
			continue
		}
		s := readgstatus(gp)
		switch s &^ _Gscan {
		case _Gwaiting:
			grunning++
		case _Grunnable,
			_Grunning,
			_Gsyscall:
			unlock(&allglock)
			print("runtime: checkdead: find g ", gp.goid, " in status ", s, "\n")
			throw("checkdead: runnable g")
		}
	}
	unlock(&allglock)
	if grunning == 0 { // possible if main goroutine calls runtime·Goexit()
		throw("no goroutines (main called runtime.Goexit) - deadlock!")
	}

	// Maybe jump time forward for playground.
	gp := timejump()
	if gp != nil {
		casgstatus(gp, _Gwaiting, _Grunnable)
		globrunqput(gp)
		_p_ := pidleget()
		if _p_ == nil {
			throw("checkdead: no p for timer")
		}
		mp := mget()
		if mp == nil {
			_newm(nil, _p_)
		} else {
			mp.nextp = _p_
			notewakeup(&mp.park)
		}
		return
	}

	getg().m.throwing = -1 // do not dump full stacks
	throw("all goroutines are asleep - deadlock!")
}

func sysmon() {
	// If we go two minutes without a garbage collection, force one to run.
	forcegcperiod := int64(2 * 60 * 1e9)

	// If a heap span goes unused for 5 minutes after a garbage collection,
	// we hand it back to the operating system.
	scavengelimit := int64(5 * 60 * 1e9)

	if debug.scavenge > 0 {
		// Scavenge-a-lot for testing.
		forcegcperiod = 10 * 1e6
		scavengelimit = 20 * 1e6
	}

	lastscavenge := nanotime()
	nscavenge := 0

	// Make wake-up period small enough for the sampling to be correct.
	maxsleep := forcegcperiod / 2
	if scavengelimit < forcegcperiod {
		maxsleep = scavengelimit / 2
	}

	lasttrace := int64(0)
	idle := 0 // how many cycles in succession we had not wokeup somebody
	delay := uint32(0)
	for {
		if idle == 0 { // start with 20us sleep...
			delay = 20
		} else if idle > 50 { // start doubling the sleep after 1ms...
			delay *= 2
		}
		if delay > 10*1000 { // up to 10ms
			delay = 10 * 1000
		}
		usleep(delay)
		if debug.schedtrace <= 0 && (sched.gcwaiting != 0 || atomicload(&sched.npidle) == uint32(gomaxprocs)) { // TODO: fast atomic
			lock(&sched.lock)
			if atomicload(&sched.gcwaiting) != 0 || atomicload(&sched.npidle) == uint32(gomaxprocs) {
				atomicstore(&sched.sysmonwait, 1)
				unlock(&sched.lock)
				notetsleep(&sched.sysmonnote, maxsleep)
				lock(&sched.lock)
				atomicstore(&sched.sysmonwait, 0)
				noteclear(&sched.sysmonnote)
				idle = 0
				delay = 20
			}
			unlock(&sched.lock)
		}
		// poll network if not polled for more than 10ms
		lastpoll := int64(atomicload64(&sched.lastpoll))
		now := nanotime()
		unixnow := unixnanotime()
		if lastpoll != 0 && lastpoll+10*1000*1000 < now {
			cas64(&sched.lastpoll, uint64(lastpoll), uint64(now))
			gp := netpoll(false) // non-blocking - returns list of goroutines
			if gp != nil {
				// Need to decrement number of idle locked M's
				// (pretending that one more is running) before injectglist.
				// Otherwise it can lead to the following situation:
				// injectglist grabs all P's but before it starts M's to run the P's,
				// another M returns from syscall, finishes running its G,
				// observes that there is no work to do and no other running M's
				// and reports deadlock.
				incidlelocked(-1)
				injectglist(gp)
				incidlelocked(1)
			}
		}
		// retake P's blocked in syscalls
		// and preempt long running G's
		if retake(now) != 0 {
			idle = 0
		} else {
			idle++
		}
		// check if we need to force a GC
		lastgc := int64(atomicload64(&memstats.last_gc))
		if lastgc != 0 && unixnow-lastgc > forcegcperiod && atomicload(&forcegc.idle) != 0 {
			lock(&forcegc.lock)
			forcegc.idle = 0
			forcegc.g.schedlink = nil
			injectglist(forcegc.g)
			unlock(&forcegc.lock)
		}
		// scavenge heap once in a while
		if lastscavenge+scavengelimit/2 < now {
			mHeap_Scavenge(int32(nscavenge), uint64(now), uint64(scavengelimit))
			lastscavenge = now
			nscavenge++
		}
		if debug.schedtrace > 0 && lasttrace+int64(debug.schedtrace*1000000) <= now {
			lasttrace = now
			schedtrace(debug.scheddetail > 0)
		}
	}
}

var pdesc [_MaxGomaxprocs]struct {
	schedtick   uint32
	schedwhen   int64
	syscalltick uint32
	syscallwhen int64
}

func retake(now int64) uint32 {
	n := 0
	for i := int32(0); i < gomaxprocs; i++ {
		_p_ := allp[i]
		if _p_ == nil {
			continue
		}
		pd := &pdesc[i]
		s := _p_.status
		if s == _Psyscall {
			// Retake P from syscall if it's there for more than 1 sysmon tick (at least 20us).
			t := int64(_p_.syscalltick)
			if int64(pd.syscalltick) != t {
				pd.syscalltick = uint32(t)
				pd.syscallwhen = now
				continue
			}
			// On the one hand we don't want to retake Ps if there is no other work to do,
			// but on the other hand we want to retake them eventually
			// because they can prevent the sysmon thread from deep sleep.
			if _p_.runqhead == _p_.runqtail && atomicload(&sched.nmspinning)+atomicload(&sched.npidle) > 0 && pd.syscallwhen+10*1000*1000 > now {
				continue
			}
			// Need to decrement number of idle locked M's
			// (pretending that one more is running) before the CAS.
			// Otherwise the M from which we retake can exit the syscall,
			// increment nmidle and report deadlock.
			incidlelocked(-1)
			if cas(&_p_.status, s, _Pidle) {
				if trace.enabled {
					traceGoSysBlock(_p_)
					traceProcStop(_p_)
				}
				n++
				_p_.syscalltick++
				handoffp(_p_)
			}
			incidlelocked(1)
		} else if s == _Prunning {
			// Preempt G if it's running for more than 10ms.
			t := int64(_p_.schedtick)
			if int64(pd.schedtick) != t {
				pd.schedtick = uint32(t)
				pd.schedwhen = now
				continue
			}
			if pd.schedwhen+10*1000*1000 > now {
				continue
			}
			preemptone(_p_)
		}
	}
	return uint32(n)
}

// Tell all goroutines that they have been preempted and they should stop.
// This function is purely best-effort.  It can fail to inform a goroutine if a
// processor just started running it.
// No locks need to be held.
// Returns true if preemption request was issued to at least one goroutine.
func preemptall() bool {
	res := false
	for i := int32(0); i < gomaxprocs; i++ {
		_p_ := allp[i]
		if _p_ == nil || _p_.status != _Prunning {
			continue
		}
		if preemptone(_p_) {
			res = true
		}
	}
	return res
}

// Tell the goroutine running on processor P to stop.
// This function is purely best-effort.  It can incorrectly fail to inform the
// goroutine.  It can send inform the wrong goroutine.  Even if it informs the
// correct goroutine, that goroutine might ignore the request if it is
// simultaneously executing newstack.
// No lock needs to be held.
// Returns true if preemption request was issued.
// The actual preemption will happen at some point in the future
// and will be indicated by the gp->status no longer being
// Grunning
func preemptone(_p_ *p) bool {
	mp := _p_.m
	if mp == nil || mp == getg().m {
		return false
	}
	gp := mp.curg
	if gp == nil || gp == mp.g0 {
		return false
	}

	gp.preempt = true

	// Every call in a go routine checks for stack overflow by
	// comparing the current stack pointer to gp->stackguard0.
	// Setting gp->stackguard0 to StackPreempt folds
	// preemption into the normal stack overflow check.
	gp.stackguard0 = stackPreempt
	return true
}

var starttime int64

func schedtrace(detailed bool) {
	now := nanotime()
	if starttime == 0 {
		starttime = now
	}

	lock(&sched.lock)
	print("SCHED ", (now-starttime)/1e6, "ms: gomaxprocs=", gomaxprocs, " idleprocs=", sched.npidle, " threads=", sched.mcount, " spinningthreads=", sched.nmspinning, " idlethreads=", sched.nmidle, " runqueue=", sched.runqsize)
	if detailed {
		print(" gcwaiting=", sched.gcwaiting, " nmidlelocked=", sched.nmidlelocked, " stopwait=", sched.stopwait, " sysmonwait=", sched.sysmonwait, "\n")
	}
	// We must be careful while reading data from P's, M's and G's.
	// Even if we hold schedlock, most data can be changed concurrently.
	// E.g. (p->m ? p->m->id : -1) can crash if p->m changes from non-nil to nil.
	for i := int32(0); i < gomaxprocs; i++ {
		_p_ := allp[i]
		if _p_ == nil {
			continue
		}
		mp := _p_.m
		h := atomicload(&_p_.runqhead)
		t := atomicload(&_p_.runqtail)
		if detailed {
			id := int32(-1)
			if mp != nil {
				id = mp.id
			}
			print("  P", i, ": status=", _p_.status, " schedtick=", _p_.schedtick, " syscalltick=", _p_.syscalltick, " m=", id, " runqsize=", t-h, " gfreecnt=", _p_.gfreecnt, "\n")
		} else {
			// In non-detailed mode format lengths of per-P run queues as:
			// [len1 len2 len3 len4]
			print(" ")
			if i == 0 {
				print("[")
			}
			print(t - h)
			if i == gomaxprocs-1 {
				print("]\n")
			}
		}
	}

	if !detailed {
		unlock(&sched.lock)
		return
	}

	for mp := allm; mp != nil; mp = mp.alllink {
		_p_ := mp.p
		gp := mp.curg
		lockedg := mp.lockedg
		id1 := int32(-1)
		if _p_ != nil {
			id1 = _p_.id
		}
		id2 := int64(-1)
		if gp != nil {
			id2 = gp.goid
		}
		id3 := int64(-1)
		if lockedg != nil {
			id3 = lockedg.goid
		}
		print("  M", mp.id, ": p=", id1, " curg=", id2, " mallocing=", mp.mallocing, " throwing=", mp.throwing, " preemptoff=", mp.preemptoff, ""+" locks=", mp.locks, " dying=", mp.dying, " helpgc=", mp.helpgc, " spinning=", mp.spinning, " blocked=", getg().m.blocked, " lockedg=", id3, "\n")
	}

	lock(&allglock)
	for gi := 0; gi < len(allgs); gi++ {
		gp := allgs[gi]
		mp := gp.m
		lockedm := gp.lockedm
		id1 := int32(-1)
		if mp != nil {
			id1 = mp.id
		}
		id2 := int32(-1)
		if lockedm != nil {
			id2 = lockedm.id
		}
		print("  G", gp.goid, ": status=", readgstatus(gp), "(", gp.waitreason, ") m=", id1, " lockedm=", id2, "\n")
	}
	unlock(&allglock)
	unlock(&sched.lock)
}

// Put mp on midle list.
// Sched must be locked.
func mput(mp *m) {
	mp.schedlink = sched.midle
	sched.midle = mp
	sched.nmidle++
	checkdead()
}

// Try to get an m from midle list.
// Sched must be locked.
func mget() *m {
	mp := sched.midle
	if mp != nil {
		sched.midle = mp.schedlink
		sched.nmidle--
	}
	return mp
}

// Put gp on the global runnable queue.
// Sched must be locked.
func globrunqput(gp *g) {
	gp.schedlink = nil
	if sched.runqtail != nil {
		sched.runqtail.schedlink = gp
	} else {
		sched.runqhead = gp
	}
	sched.runqtail = gp
	sched.runqsize++
}

// Put a batch of runnable goroutines on the global runnable queue.
// Sched must be locked.
func globrunqputbatch(ghead *g, gtail *g, n int32) {
	gtail.schedlink = nil
	if sched.runqtail != nil {
		sched.runqtail.schedlink = ghead
	} else {
		sched.runqhead = ghead
	}
	sched.runqtail = gtail
	sched.runqsize += n
}

// Try get a batch of G's from the global runnable queue.
// Sched must be locked.
func globrunqget(_p_ *p, max int32) *g {
	if sched.runqsize == 0 {
		return nil
	}

	n := sched.runqsize/gomaxprocs + 1
	if n > sched.runqsize {
		n = sched.runqsize
	}
	if max > 0 && n > max {
		n = max
	}
	if n > int32(len(_p_.runq))/2 {
		n = int32(len(_p_.runq)) / 2
	}

	sched.runqsize -= n
	if sched.runqsize == 0 {
		sched.runqtail = nil
	}

	gp := sched.runqhead
	sched.runqhead = gp.schedlink
	n--
	for ; n > 0; n-- {
		gp1 := sched.runqhead
		sched.runqhead = gp1.schedlink
		runqput(_p_, gp1)
	}
	return gp
}

// Put p to on _Pidle list.
// Sched must be locked.
func pidleput(_p_ *p) {
	_p_.link = sched.pidle
	sched.pidle = _p_
	xadd(&sched.npidle, 1) // TODO: fast atomic
}

// Try get a p from _Pidle list.
// Sched must be locked.
func pidleget() *p {
	_p_ := sched.pidle
	if _p_ != nil {
		sched.pidle = _p_.link
		xadd(&sched.npidle, -1) // TODO: fast atomic
	}
	return _p_
}

// Try to put g on local runnable queue.
// If it's full, put onto global queue.
// Executed only by the owner P.
func runqput(_p_ *p, gp *g) {
retry:
	h := atomicload(&_p_.runqhead) // load-acquire, synchronize with consumers
	t := _p_.runqtail
	if t-h < uint32(len(_p_.runq)) {
		_p_.runq[t%uint32(len(_p_.runq))] = gp
		atomicstore(&_p_.runqtail, t+1) // store-release, makes the item available for consumption
		return
	}
	if runqputslow(_p_, gp, h, t) {
		return
	}
	// the queue is not full, now the put above must suceed
	goto retry
}

// Put g and a batch of work from local runnable queue on global queue.
// Executed only by the owner P.
func runqputslow(_p_ *p, gp *g, h, t uint32) bool {
	var batch [len(_p_.runq)/2 + 1]*g

	// First, grab a batch from local queue.
	n := t - h
	n = n / 2
	if n != uint32(len(_p_.runq)/2) {
		throw("runqputslow: queue is not full")
	}
	for i := uint32(0); i < n; i++ {
		batch[i] = _p_.runq[(h+i)%uint32(len(_p_.runq))]
	}
	if !cas(&_p_.runqhead, h, h+n) { // cas-release, commits consume
		return false
	}
	batch[n] = gp

	// Link the goroutines.
	for i := uint32(0); i < n; i++ {
		batch[i].schedlink = batch[i+1]
	}

	// Now put the batch on global queue.
	lock(&sched.lock)
	globrunqputbatch(batch[0], batch[n], int32(n+1))
	unlock(&sched.lock)
	return true
}

// Get g from local runnable queue.
// Executed only by the owner P.
func runqget(_p_ *p) *g {
	for {
		h := atomicload(&_p_.runqhead) // load-acquire, synchronize with other consumers
		t := _p_.runqtail
		if t == h {
			return nil
		}
		gp := _p_.runq[h%uint32(len(_p_.runq))]
		if cas(&_p_.runqhead, h, h+1) { // cas-release, commits consume
			return gp
		}
	}
}

// Grabs a batch of goroutines from local runnable queue.
// batch array must be of size len(p->runq)/2. Returns number of grabbed goroutines.
// Can be executed by any P.
func runqgrab(_p_ *p, batch []*g) uint32 {
	for {
		h := atomicload(&_p_.runqhead) // load-acquire, synchronize with other consumers
		t := atomicload(&_p_.runqtail) // load-acquire, synchronize with the producer
		n := t - h
		n = n - n/2
		if n == 0 {
			return 0
		}
		if n > uint32(len(_p_.runq)/2) { // read inconsistent h and t
			continue
		}
		for i := uint32(0); i < n; i++ {
			batch[i] = _p_.runq[(h+i)%uint32(len(_p_.runq))]
		}
		if cas(&_p_.runqhead, h, h+n) { // cas-release, commits consume
			return n
		}
	}
}

// Steal half of elements from local runnable queue of p2
// and put onto local runnable queue of p.
// Returns one of the stolen elements (or nil if failed).
func runqsteal(_p_, p2 *p) *g {
	var batch [len(_p_.runq) / 2]*g

	n := runqgrab(p2, batch[:])
	if n == 0 {
		return nil
	}
	n--
	gp := batch[n]
	if n == 0 {
		return gp
	}
	h := atomicload(&_p_.runqhead) // load-acquire, synchronize with consumers
	t := _p_.runqtail
	if t-h+n >= uint32(len(_p_.runq)) {
		throw("runqsteal: runq overflow")
	}
	for i := uint32(0); i < n; i++ {
		_p_.runq[(t+i)%uint32(len(_p_.runq))] = batch[i]
	}
	atomicstore(&_p_.runqtail, t+n) // store-release, makes the item available for consumption
	return gp
}

func testSchedLocalQueue() {
	_p_ := new(p)
	gs := make([]g, len(_p_.runq))
	for i := 0; i < len(_p_.runq); i++ {
		if runqget(_p_) != nil {
			throw("runq is not empty initially")
		}
		for j := 0; j < i; j++ {
			runqput(_p_, &gs[i])
		}
		for j := 0; j < i; j++ {
			if runqget(_p_) != &gs[i] {
				print("bad element at iter ", i, "/", j, "\n")
				throw("bad element")
			}
		}
		if runqget(_p_) != nil {
			throw("runq is not empty afterwards")
		}
	}
}

func testSchedLocalQueueSteal() {
	p1 := new(p)
	p2 := new(p)
	gs := make([]g, len(p1.runq))
	for i := 0; i < len(p1.runq); i++ {
		for j := 0; j < i; j++ {
			gs[j].sig = 0
			runqput(p1, &gs[j])
		}
		gp := runqsteal(p2, p1)
		s := 0
		if gp != nil {
			s++
			gp.sig++
		}
		for {
			gp = runqget(p2)
			if gp == nil {
				break
			}
			s++
			gp.sig++
		}
		for {
			gp = runqget(p1)
			if gp == nil {
				break
			}
			gp.sig++
		}
		for j := 0; j < i; j++ {
			if gs[j].sig != 1 {
				print("bad element ", j, "(", gs[j].sig, ") at iter ", i, "\n")
				throw("bad element")
			}
		}
		if s != i/2 && s != i/2+1 {
			print("bad steal ", s, ", want ", i/2, " or ", i/2+1, ", iter ", i, "\n")
			throw("bad steal")
		}
	}
}

func setMaxThreads(in int) (out int) {
	lock(&sched.lock)
	out = int(sched.maxmcount)
	sched.maxmcount = int32(in)
	checkmcount()
	unlock(&sched.lock)
	return
}

func haveexperiment(name string) bool {
	x := goexperiment
	for x != "" {
		xname := ""
		i := index(x, ",")
		if i < 0 {
			xname, x = x, ""
		} else {
			xname, x = x[:i], x[i+1:]
		}
		if xname == name {
			return true
		}
	}
	return false
}

//go:nosplit
func procPin() int {
	_g_ := getg()
	mp := _g_.m

	mp.locks++
	return int(mp.p.id)
}

//go:nosplit
func procUnpin() {
	_g_ := getg()
	_g_.m.locks--
}

//go:linkname sync_runtime_procPin sync.runtime_procPin
//go:nosplit
func sync_runtime_procPin() int {
	return procPin()
}

//go:linkname sync_runtime_procUnpin sync.runtime_procUnpin
//go:nosplit
func sync_runtime_procUnpin() {
	procUnpin()
}

//go:linkname sync_atomic_runtime_procPin sync/atomic.runtime_procPin
//go:nosplit
func sync_atomic_runtime_procPin() int {
	return procPin()
}

//go:linkname sync_atomic_runtime_procUnpin sync/atomic.runtime_procUnpin
//go:nosplit
func sync_atomic_runtime_procUnpin() {
	procUnpin()
}
