// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/cpu"
	"internal/goarch"
	"internal/goexperiment"
	"internal/goos"
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// set using cmd/go/internal/modload.ModInfoProg
var modinfo string

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
// Design doc at https://golang.org/s/go11sched.

// Worker thread parking/unparking.
// We need to balance between keeping enough running worker threads to utilize
// available hardware parallelism and parking excessive running worker threads
// to conserve CPU resources and power. This is not simple for two reasons:
// (1) scheduler state is intentionally distributed (in particular, per-P work
// queues), so it is not possible to compute global predicates on fast paths;
// (2) for optimal thread management we would need to know the future (don't park
// a worker thread when a new goroutine will be readied in near future).
//
// Three rejected approaches that would work badly:
// 1. Centralize all scheduler state (would inhibit scalability).
// 2. Direct goroutine handoff. That is, when we ready a new goroutine and there
//    is a spare P, unpark a thread and handoff it the thread and the goroutine.
//    This would lead to thread state thrashing, as the thread that readied the
//    goroutine can be out of work the very next moment, we will need to park it.
//    Also, it would destroy locality of computation as we want to preserve
//    dependent goroutines on the same thread; and introduce additional latency.
// 3. Unpark an additional thread whenever we ready a goroutine and there is an
//    idle P, but don't do handoff. This would lead to excessive thread parking/
//    unparking as the additional threads will instantly park without discovering
//    any work to do.
//
// The current approach:
//
// This approach applies to three primary sources of potential work: readying a
// goroutine, new/modified-earlier timers, and idle-priority GC. See below for
// additional details.
//
// We unpark an additional thread when we submit work if (this is wakep()):
// 1. There is an idle P, and
// 2. There are no "spinning" worker threads.
//
// A worker thread is considered spinning if it is out of local work and did
// not find work in the global run queue or netpoller; the spinning state is
// denoted in m.spinning and in sched.nmspinning. Threads unparked this way are
// also considered spinning; we don't do goroutine handoff so such threads are
// out of work initially. Spinning threads spin on looking for work in per-P
// run queues and timer heaps or from the GC before parking. If a spinning
// thread finds work it takes itself out of the spinning state and proceeds to
// execution. If it does not find work it takes itself out of the spinning
// state and then parks.
//
// If there is at least one spinning thread (sched.nmspinning>1), we don't
// unpark new threads when submitting work. To compensate for that, if the last
// spinning thread finds work and stops spinning, it must unpark a new spinning
// thread. This approach smooths out unjustified spikes of thread unparking,
// but at the same time guarantees eventual maximal CPU parallelism
// utilization.
//
// The main implementation complication is that we need to be very careful
// during spinning->non-spinning thread transition. This transition can race
// with submission of new work, and either one part or another needs to unpark
// another worker thread. If they both fail to do that, we can end up with
// semi-persistent CPU underutilization.
//
// The general pattern for submission is:
// 1. Submit work to the local or global run queue, timer heap, or GC state.
// 2. #StoreLoad-style memory barrier.
// 3. Check sched.nmspinning.
//
// The general pattern for spinning->non-spinning transition is:
// 1. Decrement nmspinning.
// 2. #StoreLoad-style memory barrier.
// 3. Check all per-P work queues and GC for new work.
//
// Note that all this complexity does not apply to global run queue as we are
// not sloppy about thread unparking when submitting to global queue. Also see
// comments for nmspinning manipulation.
//
// How these different sources of work behave varies, though it doesn't affect
// the synchronization approach:
// * Ready goroutine: this is an obvious source of work; the goroutine is
//   immediately ready and must run on some thread eventually.
// * New/modified-earlier timer: The current timer implementation (see time.go)
//   uses netpoll in a thread with no work available to wait for the soonest
//   timer. If there is no thread waiting, we want a new spinning thread to go
//   wait.
// * Idle-priority GC: The GC wakes a stopped idle thread to contribute to
//   background GC work (note: currently disabled per golang.org/issue/19112).
//   Also see golang.org/issue/44313, as this should be extended to all GC
//   workers.

var (
	m0           m
	g0           g
	mcache0      *mcache
	raceprocctx0 uintptr
	raceFiniLock mutex
)

// This slice records the initializing tasks that need to be
// done to start up the runtime. It is built by the linker.
var runtime_inittasks []*initTask

// main_init_done is a signal used by cgocallbackg that initialization
// has been completed. It is made before _cgo_notify_runtime_init_done,
// so all cgo calls can rely on it existing. When main_init is complete,
// it is closed, meaning cgocallbackg can reliably receive from it.
var main_init_done chan bool

//go:linkname main_main main.main
func main_main()

// mainStarted indicates that the main M has started.
var mainStarted bool

// runtimeInitTime is the nanotime() at which the runtime started.
var runtimeInitTime int64

// Value to use for signal mask for newly created M's.
var initSigmask sigset

// The main goroutine.
func main() {
	mp := getg().m

	// Racectx of m0->g0 is used only as the parent of the main goroutine.
	// It must not be used for anything else.
	mp.g0.racectx = 0

	// Max stack size is 1 GB on 64-bit, 250 MB on 32-bit.
	// Using decimal instead of binary GB and MB because
	// they look nicer in the stack overflow failure message.
	if goarch.PtrSize == 8 {
		maxstacksize = 1000000000
	} else {
		maxstacksize = 250000000
	}

	// An upper limit for max stack size. Used to avoid random crashes
	// after calling SetMaxStack and trying to allocate a stack that is too big,
	// since stackalloc works with 32-bit sizes.
	maxstackceiling = 2 * maxstacksize

	// Allow newproc to start new Ms.
	mainStarted = true

	if haveSysmon {
		systemstack(func() {
			newm(sysmon, nil, -1)
		})
	}

	// Lock the main goroutine onto this, the main OS thread,
	// during initialization. Most programs won't care, but a few
	// do require certain calls to be made by the main thread.
	// Those can arrange for main.main to run in the main thread
	// by calling runtime.LockOSThread during initialization
	// to preserve the lock.
	lockOSThread()

	if mp != &m0 {
		throw("runtime.main not on m0")
	}

	// Record when the world started.
	// Must be before doInit for tracing init.
	runtimeInitTime = nanotime()
	if runtimeInitTime == 0 {
		throw("nanotime returning zero")
	}

	if debug.inittrace != 0 {
		inittrace.id = getg().goid
		inittrace.active = true
	}

	doInit(runtime_inittasks) // Must be before defer.

	// Defer unlock so that runtime.Goexit during init does the unlock too.
	needUnlock := true
	defer func() {
		if needUnlock {
			unlockOSThread()
		}
	}()

	gcenable()

	main_init_done = make(chan bool)
	if iscgo {
		if _cgo_pthread_key_created == nil {
			throw("_cgo_pthread_key_created missing")
		}

		if _cgo_thread_start == nil {
			throw("_cgo_thread_start missing")
		}
		if GOOS != "windows" {
			if _cgo_setenv == nil {
				throw("_cgo_setenv missing")
			}
			if _cgo_unsetenv == nil {
				throw("_cgo_unsetenv missing")
			}
		}
		if _cgo_notify_runtime_init_done == nil {
			throw("_cgo_notify_runtime_init_done missing")
		}

		// Set the x_crosscall2_ptr C function pointer variable point to crosscall2.
		if set_crosscall2 == nil {
			throw("set_crosscall2 missing")
		}
		set_crosscall2()

		// Start the template thread in case we enter Go from
		// a C-created thread and need to create a new thread.
		startTemplateThread()
		cgocall(_cgo_notify_runtime_init_done, nil)
	}

	// Run the initializing tasks. Depending on build mode this
	// list can arrive a few different ways, but it will always
	// contain the init tasks computed by the linker for all the
	// packages in the program (excluding those added at runtime
	// by package plugin). Run through the modules in dependency
	// order (the order they are initialized by the dynamic
	// loader, i.e. they are added to the moduledata linked list).
	for m := &firstmoduledata; m != nil; m = m.next {
		doInit(m.inittasks)
	}

	// Disable init tracing after main init done to avoid overhead
	// of collecting statistics in malloc and newproc
	inittrace.active = false

	close(main_init_done)

	needUnlock = false
	unlockOSThread()

	if isarchive || islibrary {
		// A program compiled with -buildmode=c-archive or c-shared
		// has a main, but it is not executed.
		return
	}
	fn := main_main // make an indirect call, as the linker doesn't know the address of the main package when laying down the runtime
	fn()
	if raceenabled {
		runExitHooks(0) // run hooks now, since racefini does not return
		racefini()
	}

	// Make racy client program work: if panicking on
	// another goroutine at the same time as main returns,
	// let the other goroutine finish printing the panic trace.
	// Once it does, it will exit. See issues 3934 and 20018.
	if runningPanicDefers.Load() != 0 {
		// Running deferred functions should not take long.
		for c := 0; c < 1000; c++ {
			if runningPanicDefers.Load() == 0 {
				break
			}
			Gosched()
		}
	}
	if panicking.Load() != 0 {
		gopark(nil, nil, waitReasonPanicWait, traceBlockForever, 1)
	}
	runExitHooks(0)

	exit(0)
	for {
		var x *int32
		*x = 0
	}
}

// os_beforeExit is called from os.Exit(0).
//
//go:linkname os_beforeExit os.runtime_beforeExit
func os_beforeExit(exitCode int) {
	runExitHooks(exitCode)
	if exitCode == 0 && raceenabled {
		racefini()
	}
}

// start forcegc helper goroutine
func init() {
	go forcegchelper()
}

func forcegchelper() {
	forcegc.g = getg()
	lockInit(&forcegc.lock, lockRankForcegc)
	for {
		lock(&forcegc.lock)
		if forcegc.idle.Load() {
			throw("forcegc: phase error")
		}
		forcegc.idle.Store(true)
		goparkunlock(&forcegc.lock, waitReasonForceGCIdle, traceBlockSystemGoroutine, 1)
		// this goroutine is explicitly resumed by sysmon
		if debug.gctrace > 0 {
			println("GC forced")
		}
		// Time-triggered, fully concurrent.
		gcStart(gcTrigger{kind: gcTriggerTime, now: nanotime()})
	}
}

// Gosched yields the processor, allowing other goroutines to run. It does not
// suspend the current goroutine, so execution resumes automatically.
//
//go:nosplit
func Gosched() {
	checkTimeouts()
	mcall(gosched_m)
}

// goschedguarded yields the processor like gosched, but also checks
// for forbidden states and opts out of the yield in those cases.
//
//go:nosplit
func goschedguarded() {
	mcall(goschedguarded_m)
}

// goschedIfBusy yields the processor like gosched, but only does so if
// there are no idle Ps or if we're on the only P and there's nothing in
// the run queue. In both cases, there is freely available idle time.
//
//go:nosplit
func goschedIfBusy() {
	gp := getg()
	// Call gosched if gp.preempt is set; we may be in a tight loop that
	// doesn't otherwise yield.
	if !gp.preempt && sched.npidle.Load() > 0 {
		return
	}
	mcall(gosched_m)
}

// Puts the current goroutine into a waiting state and calls unlockf on the
// system stack.
//
// If unlockf returns false, the goroutine is resumed.
//
// unlockf must not access this G's stack, as it may be moved between
// the call to gopark and the call to unlockf.
//
// Note that because unlockf is called after putting the G into a waiting
// state, the G may have already been readied by the time unlockf is called
// unless there is external synchronization preventing the G from being
// readied. If unlockf returns false, it must guarantee that the G cannot be
// externally readied.
//
// Reason explains why the goroutine has been parked. It is displayed in stack
// traces and heap dumps. Reasons should be unique and descriptive. Do not
// re-use reasons, add new ones.
func gopark(unlockf func(*g, unsafe.Pointer) bool, lock unsafe.Pointer, reason waitReason, traceReason traceBlockReason, traceskip int) {
	if reason != waitReasonSleep {
		checkTimeouts() // timeouts may expire while two goroutines keep the scheduler busy
	}
	mp := acquirem()
	gp := mp.curg
	status := readgstatus(gp)
	if status != _Grunning && status != _Gscanrunning {
		throw("gopark: bad g status")
	}
	mp.waitlock = lock
	mp.waitunlockf = unlockf
	gp.waitreason = reason
	mp.waitTraceBlockReason = traceReason
	mp.waitTraceSkip = traceskip
	releasem(mp)
	// can't do anything that might move the G between Ms here.
	mcall(park_m)
}

// Puts the current goroutine into a waiting state and unlocks the lock.
// The goroutine can be made runnable again by calling goready(gp).
func goparkunlock(lock *mutex, reason waitReason, traceReason traceBlockReason, traceskip int) {
	gopark(parkunlock_c, unsafe.Pointer(lock), reason, traceReason, traceskip)
}

func goready(gp *g, traceskip int) {
	systemstack(func() {
		ready(gp, traceskip, true)
	})
}

//go:nosplit
func acquireSudog() *sudog {
	// Delicate dance: the semaphore implementation calls
	// acquireSudog, acquireSudog calls new(sudog),
	// new calls malloc, malloc can call the garbage collector,
	// and the garbage collector calls the semaphore implementation
	// in stopTheWorld.
	// Break the cycle by doing acquirem/releasem around new(sudog).
	// The acquirem/releasem increments m.locks during new(sudog),
	// which keeps the garbage collector from being invoked.
	mp := acquirem()
	pp := mp.p.ptr()
	if len(pp.sudogcache) == 0 {
		lock(&sched.sudoglock)
		// First, try to grab a batch from central cache.
		for len(pp.sudogcache) < cap(pp.sudogcache)/2 && sched.sudogcache != nil {
			s := sched.sudogcache
			sched.sudogcache = s.next
			s.next = nil
			pp.sudogcache = append(pp.sudogcache, s)
		}
		unlock(&sched.sudoglock)
		// If the central cache is empty, allocate a new one.
		if len(pp.sudogcache) == 0 {
			pp.sudogcache = append(pp.sudogcache, new(sudog))
		}
	}
	n := len(pp.sudogcache)
	s := pp.sudogcache[n-1]
	pp.sudogcache[n-1] = nil
	pp.sudogcache = pp.sudogcache[:n-1]
	if s.elem != nil {
		throw("acquireSudog: found s.elem != nil in cache")
	}
	releasem(mp)
	return s
}

//go:nosplit
func releaseSudog(s *sudog) {
	if s.elem != nil {
		throw("runtime: sudog with non-nil elem")
	}
	if s.isSelect {
		throw("runtime: sudog with non-false isSelect")
	}
	if s.next != nil {
		throw("runtime: sudog with non-nil next")
	}
	if s.prev != nil {
		throw("runtime: sudog with non-nil prev")
	}
	if s.waitlink != nil {
		throw("runtime: sudog with non-nil waitlink")
	}
	if s.c != nil {
		throw("runtime: sudog with non-nil c")
	}
	gp := getg()
	if gp.param != nil {
		throw("runtime: releaseSudog with non-nil gp.param")
	}
	mp := acquirem() // avoid rescheduling to another P
	pp := mp.p.ptr()
	if len(pp.sudogcache) == cap(pp.sudogcache) {
		// Transfer half of local cache to the central cache.
		var first, last *sudog
		for len(pp.sudogcache) > cap(pp.sudogcache)/2 {
			n := len(pp.sudogcache)
			p := pp.sudogcache[n-1]
			pp.sudogcache[n-1] = nil
			pp.sudogcache = pp.sudogcache[:n-1]
			if first == nil {
				first = p
			} else {
				last.next = p
			}
			last = p
		}
		lock(&sched.sudoglock)
		last.next = sched.sudogcache
		sched.sudogcache = first
		unlock(&sched.sudoglock)
	}
	pp.sudogcache = append(pp.sudogcache, s)
	releasem(mp)
}

// called from assembly.
func badmcall(fn func(*g)) {
	throw("runtime: mcall called on m->g0 stack")
}

func badmcall2(fn func(*g)) {
	throw("runtime: mcall function returned")
}

func badreflectcall() {
	panic(plainError("arg size to reflect.call more than 1GB"))
}

//go:nosplit
//go:nowritebarrierrec
func badmorestackg0() {
	if !crashStackImplemented {
		writeErrStr("fatal: morestack on g0\n")
		return
	}

	g := getg()
	switchToCrashStack(func() {
		print("runtime: morestack on g0, stack [", hex(g.stack.lo), " ", hex(g.stack.hi), "], sp=", hex(g.sched.sp), ", called from\n")
		g.m.traceback = 2 // include pc and sp in stack trace
		traceback1(g.sched.pc, g.sched.sp, g.sched.lr, g, 0)
		print("\n")

		throw("morestack on g0")
	})
}

//go:nosplit
//go:nowritebarrierrec
func badmorestackgsignal() {
	writeErrStr("fatal: morestack on gsignal\n")
}

//go:nosplit
func badctxt() {
	throw("ctxt != 0")
}

// gcrash is a fake g that can be used when crashing due to bad
// stack conditions.
var gcrash g

var crashingG atomic.Pointer[g]

// Switch to crashstack and call fn, with special handling of
// concurrent and recursive cases.
//
// Nosplit as it is called in a bad stack condition (we know
// morestack would fail).
//
//go:nosplit
//go:nowritebarrierrec
func switchToCrashStack(fn func()) {
	me := getg()
	if crashingG.CompareAndSwapNoWB(nil, me) {
		switchToCrashStack0(fn) // should never return
		abort()
	}
	if crashingG.Load() == me {
		// recursive crashing. too bad.
		writeErrStr("fatal: recursive switchToCrashStack\n")
		abort()
	}
	// Another g is crashing. Give it some time, hopefully it will finish traceback.
	usleep_no_g(100)
	writeErrStr("fatal: concurrent switchToCrashStack\n")
	abort()
}

// Disable crash stack on Windows for now. Apparently, throwing an exception
// on a non-system-allocated crash stack causes EXCEPTION_STACK_OVERFLOW and
// hangs the process (see issue 63938).
const crashStackImplemented = (GOARCH == "amd64" || GOARCH == "arm" || GOARCH == "arm64" || GOARCH == "loong64" || GOARCH == "mips64" || GOARCH == "mips64le" || GOARCH == "ppc64" || GOARCH == "ppc64le" || GOARCH == "riscv64" || GOARCH == "s390x" || GOARCH == "wasm") && GOOS != "windows"

//go:noescape
func switchToCrashStack0(fn func()) // in assembly

func lockedOSThread() bool {
	gp := getg()
	return gp.lockedm != 0 && gp.m.lockedg != 0
}

var (
	// allgs contains all Gs ever created (including dead Gs), and thus
	// never shrinks.
	//
	// Access via the slice is protected by allglock or stop-the-world.
	// Readers that cannot take the lock may (carefully!) use the atomic
	// variables below.
	allglock mutex
	allgs    []*g

	// allglen and allgptr are atomic variables that contain len(allgs) and
	// &allgs[0] respectively. Proper ordering depends on totally-ordered
	// loads and stores. Writes are protected by allglock.
	//
	// allgptr is updated before allglen. Readers should read allglen
	// before allgptr to ensure that allglen is always <= len(allgptr). New
	// Gs appended during the race can be missed. For a consistent view of
	// all Gs, allglock must be held.
	//
	// allgptr copies should always be stored as a concrete type or
	// unsafe.Pointer, not uintptr, to ensure that GC can still reach it
	// even if it points to a stale array.
	allglen uintptr
	allgptr **g
)

func allgadd(gp *g) {
	if readgstatus(gp) == _Gidle {
		throw("allgadd: bad status Gidle")
	}

	lock(&allglock)
	allgs = append(allgs, gp)
	if &allgs[0] != allgptr {
		atomicstorep(unsafe.Pointer(&allgptr), unsafe.Pointer(&allgs[0]))
	}
	atomic.Storeuintptr(&allglen, uintptr(len(allgs)))
	unlock(&allglock)
}

// allGsSnapshot returns a snapshot of the slice of all Gs.
//
// The world must be stopped or allglock must be held.
func allGsSnapshot() []*g {
	assertWorldStoppedOrLockHeld(&allglock)

	// Because the world is stopped or allglock is held, allgadd
	// cannot happen concurrently with this. allgs grows
	// monotonically and existing entries never change, so we can
	// simply return a copy of the slice header. For added safety,
	// we trim everything past len because that can still change.
	return allgs[:len(allgs):len(allgs)]
}

// atomicAllG returns &allgs[0] and len(allgs) for use with atomicAllGIndex.
func atomicAllG() (**g, uintptr) {
	length := atomic.Loaduintptr(&allglen)
	ptr := (**g)(atomic.Loadp(unsafe.Pointer(&allgptr)))
	return ptr, length
}

// atomicAllGIndex returns ptr[i] with the allgptr returned from atomicAllG.
func atomicAllGIndex(ptr **g, i uintptr) *g {
	return *(**g)(add(unsafe.Pointer(ptr), i*goarch.PtrSize))
}

// forEachG calls fn on every G from allgs.
//
// forEachG takes a lock to exclude concurrent addition of new Gs.
func forEachG(fn func(gp *g)) {
	lock(&allglock)
	for _, gp := range allgs {
		fn(gp)
	}
	unlock(&allglock)
}

// forEachGRace calls fn on every G from allgs.
//
// forEachGRace avoids locking, but does not exclude addition of new Gs during
// execution, which may be missed.
func forEachGRace(fn func(gp *g)) {
	ptr, length := atomicAllG()
	for i := uintptr(0); i < length; i++ {
		gp := atomicAllGIndex(ptr, i)
		fn(gp)
	}
	return
}

const (
	// Number of goroutine ids to grab from sched.goidgen to local per-P cache at once.
	// 16 seems to provide enough amortization, but other than that it's mostly arbitrary number.
	_GoidCacheBatch = 16
)

// cpuinit sets up CPU feature flags and calls internal/cpu.Initialize. env should be the complete
// value of the GODEBUG environment variable.
func cpuinit(env string) {
	switch GOOS {
	case "aix", "darwin", "ios", "dragonfly", "freebsd", "netbsd", "openbsd", "illumos", "solaris", "linux":
		cpu.DebugOptions = true
	}
	cpu.Initialize(env)

	// Support cpu feature variables are used in code generated by the compiler
	// to guard execution of instructions that can not be assumed to be always supported.
	switch GOARCH {
	case "386", "amd64":
		x86HasPOPCNT = cpu.X86.HasPOPCNT
		x86HasSSE41 = cpu.X86.HasSSE41
		x86HasFMA = cpu.X86.HasFMA

	case "arm":
		armHasVFPv4 = cpu.ARM.HasVFPv4

	case "arm64":
		arm64HasATOMICS = cpu.ARM64.HasATOMICS
	}
}

// getGodebugEarly extracts the environment variable GODEBUG from the environment on
// Unix-like operating systems and returns it. This function exists to extract GODEBUG
// early before much of the runtime is initialized.
func getGodebugEarly() string {
	const prefix = "GODEBUG="
	var env string
	switch GOOS {
	case "aix", "darwin", "ios", "dragonfly", "freebsd", "netbsd", "openbsd", "illumos", "solaris", "linux":
		// Similar to goenv_unix but extracts the environment value for
		// GODEBUG directly.
		// TODO(moehrmann): remove when general goenvs() can be called before cpuinit()
		n := int32(0)
		for argv_index(argv, argc+1+n) != nil {
			n++
		}

		for i := int32(0); i < n; i++ {
			p := argv_index(argv, argc+1+i)
			s := unsafe.String(p, findnull(p))

			if hasPrefix(s, prefix) {
				env = gostring(p)[len(prefix):]
				break
			}
		}
	}
	return env
}

// The bootstrap sequence is:
//
//	call osinit
//	call schedinit
//	make & queue new G
//	call runtime·mstart
//
// The new G calls runtime·main.
func schedinit() {
	lockInit(&sched.lock, lockRankSched)
	lockInit(&sched.sysmonlock, lockRankSysmon)
	lockInit(&sched.deferlock, lockRankDefer)
	lockInit(&sched.sudoglock, lockRankSudog)
	lockInit(&deadlock, lockRankDeadlock)
	lockInit(&paniclk, lockRankPanic)
	lockInit(&allglock, lockRankAllg)
	lockInit(&allpLock, lockRankAllp)
	lockInit(&reflectOffs.lock, lockRankReflectOffs)
	lockInit(&finlock, lockRankFin)
	lockInit(&cpuprof.lock, lockRankCpuprof)
	allocmLock.init(lockRankAllocmR, lockRankAllocmRInternal, lockRankAllocmW)
	execLock.init(lockRankExecR, lockRankExecRInternal, lockRankExecW)
	traceLockInit()
	// Enforce that this lock is always a leaf lock.
	// All of this lock's critical sections should be
	// extremely short.
	lockInit(&memstats.heapStats.noPLock, lockRankLeafRank)

	// raceinit must be the first call to race detector.
	// In particular, it must be done before mallocinit below calls racemapshadow.
	gp := getg()
	if raceenabled {
		gp.racectx, raceprocctx0 = raceinit()
	}

	sched.maxmcount = 10000
	crashFD.Store(^uintptr(0))

	// The world starts stopped.
	worldStopped()

	ticks.init() // run as early as possible
	moduledataverify()
	stackinit()
	mallocinit()
	godebug := getGodebugEarly()
	initPageTrace(godebug) // must run after mallocinit but before anything allocates
	cpuinit(godebug)       // must run before alginit
	randinit()             // must run before alginit, mcommoninit
	alginit()              // maps, hash, rand must not be used before this call
	mcommoninit(gp.m, -1)
	modulesinit()   // provides activeModules
	typelinksinit() // uses maps, activeModules
	itabsinit()     // uses activeModules
	stkobjinit()    // must run before GC starts

	sigsave(&gp.m.sigmask)
	initSigmask = gp.m.sigmask

	goargs()
	goenvs()
	secure()
	checkfds()
	parsedebugvars()
	gcinit()

	// Allocate stack space that can be used when crashing due to bad stack
	// conditions, e.g. morestack on g0.
	gcrash.stack = stackalloc(16384)
	gcrash.stackguard0 = gcrash.stack.lo + 1000
	gcrash.stackguard1 = gcrash.stack.lo + 1000

	// if disableMemoryProfiling is set, update MemProfileRate to 0 to turn off memprofile.
	// Note: parsedebugvars may update MemProfileRate, but when disableMemoryProfiling is
	// set to true by the linker, it means that nothing is consuming the profile, it is
	// safe to set MemProfileRate to 0.
	if disableMemoryProfiling {
		MemProfileRate = 0
	}

	lock(&sched.lock)
	sched.lastpoll.Store(nanotime())
	procs := ncpu
	if n, ok := atoi32(gogetenv("GOMAXPROCS")); ok && n > 0 {
		procs = n
	}
	if procresize(procs) != nil {
		throw("unknown runnable goroutine during bootstrap")
	}
	unlock(&sched.lock)

	// World is effectively started now, as P's can run.
	worldStarted()

	if buildVersion == "" {
		// Condition should never trigger. This code just serves
		// to ensure runtime·buildVersion is kept in the resulting binary.
		buildVersion = "unknown"
	}
	if len(modinfo) == 1 {
		// Condition should never trigger. This code just serves
		// to ensure runtime·modinfo is kept in the resulting binary.
		modinfo = ""
	}
}

func dumpgstatus(gp *g) {
	thisg := getg()
	print("runtime:   gp: gp=", gp, ", goid=", gp.goid, ", gp->atomicstatus=", readgstatus(gp), "\n")
	print("runtime: getg:  g=", thisg, ", goid=", thisg.goid, ",  g->atomicstatus=", readgstatus(thisg), "\n")
}

// sched.lock must be held.
func checkmcount() {
	assertLockHeld(&sched.lock)

	// Exclude extra M's, which are used for cgocallback from threads
	// created in C.
	//
	// The purpose of the SetMaxThreads limit is to avoid accidental fork
	// bomb from something like millions of goroutines blocking on system
	// calls, causing the runtime to create millions of threads. By
	// definition, this isn't a problem for threads created in C, so we
	// exclude them from the limit. See https://go.dev/issue/60004.
	count := mcount() - int32(extraMInUse.Load()) - int32(extraMLength.Load())
	if count > sched.maxmcount {
		print("runtime: program exceeds ", sched.maxmcount, "-thread limit\n")
		throw("thread exhaustion")
	}
}

// mReserveID returns the next ID to use for a new m. This new m is immediately
// considered 'running' by checkdead.
//
// sched.lock must be held.
func mReserveID() int64 {
	assertLockHeld(&sched.lock)

	if sched.mnext+1 < sched.mnext {
		throw("runtime: thread ID overflow")
	}
	id := sched.mnext
	sched.mnext++
	checkmcount()
	return id
}

// Pre-allocated ID may be passed as 'id', or omitted by passing -1.
func mcommoninit(mp *m, id int64) {
	gp := getg()

	// g0 stack won't make sense for user (and is not necessary unwindable).
	if gp != gp.m.g0 {
		callers(1, mp.createstack[:])
	}

	lock(&sched.lock)

	if id >= 0 {
		mp.id = id
	} else {
		mp.id = mReserveID()
	}

	mrandinit(mp)

	mpreinit(mp)
	if mp.gsignal != nil {
		mp.gsignal.stackguard1 = mp.gsignal.stack.lo + stackGuard
	}

	// Add to allm so garbage collector doesn't free g->m
	// when it is just in a register or thread-local storage.
	mp.alllink = allm

	// NumCgoCall() and others iterate over allm w/o schedlock,
	// so we need to publish it safely.
	atomicstorep(unsafe.Pointer(&allm), unsafe.Pointer(mp))
	unlock(&sched.lock)

	// Allocate memory to hold a cgo traceback if the cgo call crashes.
	if iscgo || GOOS == "solaris" || GOOS == "illumos" || GOOS == "windows" {
		mp.cgoCallers = new(cgoCallers)
	}
}

func (mp *m) becomeSpinning() {
	mp.spinning = true
	sched.nmspinning.Add(1)
	sched.needspinning.Store(0)
}

func (mp *m) hasCgoOnStack() bool {
	return mp.ncgo > 0 || mp.isextra
}

const (
	// osHasLowResTimer indicates that the platform's internal timer system has a low resolution,
	// typically on the order of 1 ms or more.
	osHasLowResTimer = GOOS == "windows" || GOOS == "openbsd" || GOOS == "netbsd"

	// osHasLowResClockInt is osHasLowResClock but in integer form, so it can be used to create
	// constants conditionally.
	osHasLowResClockInt = goos.IsWindows

	// osHasLowResClock indicates that timestamps produced by nanotime on the platform have a
	// low resolution, typically on the order of 1 ms or more.
	osHasLowResClock = osHasLowResClockInt > 0
)

// Mark gp ready to run.
func ready(gp *g, traceskip int, next bool) {
	status := readgstatus(gp)

	// Mark runnable.
	mp := acquirem() // disable preemption because it can be holding p in a local var
	if status&^_Gscan != _Gwaiting {
		dumpgstatus(gp)
		throw("bad g->status in ready")
	}

	// status is Gwaiting or Gscanwaiting, make Grunnable and put on runq
	trace := traceAcquire()
	casgstatus(gp, _Gwaiting, _Grunnable)
	if trace.ok() {
		trace.GoUnpark(gp, traceskip)
		traceRelease(trace)
	}
	runqput(mp.p.ptr(), gp, next)
	wakep()
	releasem(mp)
}

// freezeStopWait is a large value that freezetheworld sets
// sched.stopwait to in order to request that all Gs permanently stop.
const freezeStopWait = 0x7fffffff

// freezing is set to non-zero if the runtime is trying to freeze the
// world.
var freezing atomic.Bool

// Similar to stopTheWorld but best-effort and can be called several times.
// There is no reverse operation, used during crashing.
// This function must not lock any mutexes.
func freezetheworld() {
	freezing.Store(true)
	if debug.dontfreezetheworld > 0 {
		// Don't prempt Ps to stop goroutines. That will perturb
		// scheduler state, making debugging more difficult. Instead,
		// allow goroutines to continue execution.
		//
		// fatalpanic will tracebackothers to trace all goroutines. It
		// is unsafe to trace a running goroutine, so tracebackothers
		// will skip running goroutines. That is OK and expected, we
		// expect users of dontfreezetheworld to use core files anyway.
		//
		// However, allowing the scheduler to continue running free
		// introduces a race: a goroutine may be stopped when
		// tracebackothers checks its status, and then start running
		// later when we are in the middle of traceback, potentially
		// causing a crash.
		//
		// To mitigate this, when an M naturally enters the scheduler,
		// schedule checks if freezing is set and if so stops
		// execution. This guarantees that while Gs can transition from
		// running to stopped, they can never transition from stopped
		// to running.
		//
		// The sleep here allows racing Ms that missed freezing and are
		// about to run a G to complete the transition to running
		// before we start traceback.
		usleep(1000)
		return
	}

	// stopwait and preemption requests can be lost
	// due to races with concurrently executing threads,
	// so try several times
	for i := 0; i < 5; i++ {
		// this should tell the scheduler to not start any new goroutines
		sched.stopwait = freezeStopWait
		sched.gcwaiting.Store(true)
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

// All reads and writes of g's status go through readgstatus, casgstatus
// castogscanstatus, casfrom_Gscanstatus.
//
//go:nosplit
func readgstatus(gp *g) uint32 {
	return gp.atomicstatus.Load()
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
		_Gscansyscall,
		_Gscanpreempted:
		if newval == oldval&^_Gscan {
			success = gp.atomicstatus.CompareAndSwap(oldval, newval)
		}
	}
	if !success {
		print("runtime: casfrom_Gscanstatus failed gp=", gp, ", oldval=", hex(oldval), ", newval=", hex(newval), "\n")
		dumpgstatus(gp)
		throw("casfrom_Gscanstatus: gp->status is not in scan state")
	}
	releaseLockRank(lockRankGscan)
}

// This will return false if the gp is not in the expected status and the cas fails.
// This acts like a lock acquire while the casfromgstatus acts like a lock release.
func castogscanstatus(gp *g, oldval, newval uint32) bool {
	switch oldval {
	case _Grunnable,
		_Grunning,
		_Gwaiting,
		_Gsyscall:
		if newval == oldval|_Gscan {
			r := gp.atomicstatus.CompareAndSwap(oldval, newval)
			if r {
				acquireLockRank(lockRankGscan)
			}
			return r

		}
	}
	print("runtime: castogscanstatus oldval=", hex(oldval), " newval=", hex(newval), "\n")
	throw("castogscanstatus")
	panic("not reached")
}

// casgstatusAlwaysTrack is a debug flag that causes casgstatus to always track
// various latencies on every transition instead of sampling them.
var casgstatusAlwaysTrack = false

// If asked to move to or from a Gscanstatus this will throw. Use the castogscanstatus
// and casfrom_Gscanstatus instead.
// casgstatus will loop if the g->atomicstatus is in a Gscan status until the routine that
// put it in the Gscan state is finished.
//
//go:nosplit
func casgstatus(gp *g, oldval, newval uint32) {
	if (oldval&_Gscan != 0) || (newval&_Gscan != 0) || oldval == newval {
		systemstack(func() {
			print("runtime: casgstatus: oldval=", hex(oldval), " newval=", hex(newval), "\n")
			throw("casgstatus: bad incoming values")
		})
	}

	acquireLockRank(lockRankGscan)
	releaseLockRank(lockRankGscan)

	// See https://golang.org/cl/21503 for justification of the yield delay.
	const yieldDelay = 5 * 1000
	var nextYield int64

	// loop if gp->atomicstatus is in a scan state giving
	// GC time to finish and change the state to oldval.
	for i := 0; !gp.atomicstatus.CompareAndSwap(oldval, newval); i++ {
		if oldval == _Gwaiting && gp.atomicstatus.Load() == _Grunnable {
			throw("casgstatus: waiting for Gwaiting but is Grunnable")
		}
		if i == 0 {
			nextYield = nanotime() + yieldDelay
		}
		if nanotime() < nextYield {
			for x := 0; x < 10 && gp.atomicstatus.Load() != oldval; x++ {
				procyield(1)
			}
		} else {
			osyield()
			nextYield = nanotime() + yieldDelay/2
		}
	}

	if oldval == _Grunning {
		// Track every gTrackingPeriod time a goroutine transitions out of running.
		if casgstatusAlwaysTrack || gp.trackingSeq%gTrackingPeriod == 0 {
			gp.tracking = true
		}
		gp.trackingSeq++
	}
	if !gp.tracking {
		return
	}

	// Handle various kinds of tracking.
	//
	// Currently:
	// - Time spent in runnable.
	// - Time spent blocked on a sync.Mutex or sync.RWMutex.
	switch oldval {
	case _Grunnable:
		// We transitioned out of runnable, so measure how much
		// time we spent in this state and add it to
		// runnableTime.
		now := nanotime()
		gp.runnableTime += now - gp.trackingStamp
		gp.trackingStamp = 0
	case _Gwaiting:
		if !gp.waitreason.isMutexWait() {
			// Not blocking on a lock.
			break
		}
		// Blocking on a lock, measure it. Note that because we're
		// sampling, we have to multiply by our sampling period to get
		// a more representative estimate of the absolute value.
		// gTrackingPeriod also represents an accurate sampling period
		// because we can only enter this state from _Grunning.
		now := nanotime()
		sched.totalMutexWaitTime.Add((now - gp.trackingStamp) * gTrackingPeriod)
		gp.trackingStamp = 0
	}
	switch newval {
	case _Gwaiting:
		if !gp.waitreason.isMutexWait() {
			// Not blocking on a lock.
			break
		}
		// Blocking on a lock. Write down the timestamp.
		now := nanotime()
		gp.trackingStamp = now
	case _Grunnable:
		// We just transitioned into runnable, so record what
		// time that happened.
		now := nanotime()
		gp.trackingStamp = now
	case _Grunning:
		// We're transitioning into running, so turn off
		// tracking and record how much time we spent in
		// runnable.
		gp.tracking = false
		sched.timeToRun.record(gp.runnableTime)
		gp.runnableTime = 0
	}
}

// casGToWaiting transitions gp from old to _Gwaiting, and sets the wait reason.
//
// Use this over casgstatus when possible to ensure that a waitreason is set.
func casGToWaiting(gp *g, old uint32, reason waitReason) {
	// Set the wait reason before calling casgstatus, because casgstatus will use it.
	gp.waitreason = reason
	casgstatus(gp, old, _Gwaiting)
}

// casgstatus(gp, oldstatus, Gcopystack), assuming oldstatus is Gwaiting or Grunnable.
// Returns old status. Cannot call casgstatus directly, because we are racing with an
// async wakeup that might come in from netpoll. If we see Gwaiting from the readgstatus,
// it might have become Grunnable by the time we get to the cas. If we called casgstatus,
// it would loop waiting for the status to go back to Gwaiting, which it never will.
//
//go:nosplit
func casgcopystack(gp *g) uint32 {
	for {
		oldstatus := readgstatus(gp) &^ _Gscan
		if oldstatus != _Gwaiting && oldstatus != _Grunnable {
			throw("copystack: bad status, not Gwaiting or Grunnable")
		}
		if gp.atomicstatus.CompareAndSwap(oldstatus, _Gcopystack) {
			return oldstatus
		}
	}
}

// casGToPreemptScan transitions gp from _Grunning to _Gscan|_Gpreempted.
//
// TODO(austin): This is the only status operation that both changes
// the status and locks the _Gscan bit. Rethink this.
func casGToPreemptScan(gp *g, old, new uint32) {
	if old != _Grunning || new != _Gscan|_Gpreempted {
		throw("bad g transition")
	}
	acquireLockRank(lockRankGscan)
	for !gp.atomicstatus.CompareAndSwap(_Grunning, _Gscan|_Gpreempted) {
	}
}

// casGFromPreempted attempts to transition gp from _Gpreempted to
// _Gwaiting. If successful, the caller is responsible for
// re-scheduling gp.
func casGFromPreempted(gp *g, old, new uint32) bool {
	if old != _Gpreempted || new != _Gwaiting {
		throw("bad g transition")
	}
	gp.waitreason = waitReasonPreempted
	return gp.atomicstatus.CompareAndSwap(_Gpreempted, _Gwaiting)
}

// stwReason is an enumeration of reasons the world is stopping.
type stwReason uint8

// Reasons to stop-the-world.
//
// Avoid reusing reasons and add new ones instead.
const (
	stwUnknown                     stwReason = iota // "unknown"
	stwGCMarkTerm                                   // "GC mark termination"
	stwGCSweepTerm                                  // "GC sweep termination"
	stwWriteHeapDump                                // "write heap dump"
	stwGoroutineProfile                             // "goroutine profile"
	stwGoroutineProfileCleanup                      // "goroutine profile cleanup"
	stwAllGoroutinesStack                           // "all goroutines stack trace"
	stwReadMemStats                                 // "read mem stats"
	stwAllThreadsSyscall                            // "AllThreadsSyscall"
	stwGOMAXPROCS                                   // "GOMAXPROCS"
	stwStartTrace                                   // "start trace"
	stwStopTrace                                    // "stop trace"
	stwForTestCountPagesInUse                       // "CountPagesInUse (test)"
	stwForTestReadMetricsSlow                       // "ReadMetricsSlow (test)"
	stwForTestReadMemStatsSlow                      // "ReadMemStatsSlow (test)"
	stwForTestPageCachePagesLeaked                  // "PageCachePagesLeaked (test)"
	stwForTestResetDebugLog                         // "ResetDebugLog (test)"
)

func (r stwReason) String() string {
	return stwReasonStrings[r]
}

func (r stwReason) isGC() bool {
	return r == stwGCMarkTerm || r == stwGCSweepTerm
}

// If you add to this list, also add it to src/internal/trace/parser.go.
// If you change the values of any of the stw* constants, bump the trace
// version number and make a copy of this.
var stwReasonStrings = [...]string{
	stwUnknown:                     "unknown",
	stwGCMarkTerm:                  "GC mark termination",
	stwGCSweepTerm:                 "GC sweep termination",
	stwWriteHeapDump:               "write heap dump",
	stwGoroutineProfile:            "goroutine profile",
	stwGoroutineProfileCleanup:     "goroutine profile cleanup",
	stwAllGoroutinesStack:          "all goroutines stack trace",
	stwReadMemStats:                "read mem stats",
	stwAllThreadsSyscall:           "AllThreadsSyscall",
	stwGOMAXPROCS:                  "GOMAXPROCS",
	stwStartTrace:                  "start trace",
	stwStopTrace:                   "stop trace",
	stwForTestCountPagesInUse:      "CountPagesInUse (test)",
	stwForTestReadMetricsSlow:      "ReadMetricsSlow (test)",
	stwForTestReadMemStatsSlow:     "ReadMemStatsSlow (test)",
	stwForTestPageCachePagesLeaked: "PageCachePagesLeaked (test)",
	stwForTestResetDebugLog:        "ResetDebugLog (test)",
}

// worldStop provides context from the stop-the-world required by the
// start-the-world.
type worldStop struct {
	reason stwReason
	start  int64
}

// Temporary variable for stopTheWorld, when it can't write to the stack.
//
// Protected by worldsema.
var stopTheWorldContext worldStop

// stopTheWorld stops all P's from executing goroutines, interrupting
// all goroutines at GC safe points and records reason as the reason
// for the stop. On return, only the current goroutine's P is running.
// stopTheWorld must not be called from a system stack and the caller
// must not hold worldsema. The caller must call startTheWorld when
// other P's should resume execution.
//
// stopTheWorld is safe for multiple goroutines to call at the
// same time. Each will execute its own stop, and the stops will
// be serialized.
//
// This is also used by routines that do stack dumps. If the system is
// in panic or being exited, this may not reliably stop all
// goroutines.
//
// Returns the STW context. When starting the world, this context must be
// passed to startTheWorld.
func stopTheWorld(reason stwReason) worldStop {
	semacquire(&worldsema)
	gp := getg()
	gp.m.preemptoff = reason.String()
	systemstack(func() {
		// Mark the goroutine which called stopTheWorld preemptible so its
		// stack may be scanned.
		// This lets a mark worker scan us while we try to stop the world
		// since otherwise we could get in a mutual preemption deadlock.
		// We must not modify anything on the G stack because a stack shrink
		// may occur. A stack shrink is otherwise OK though because in order
		// to return from this function (and to leave the system stack) we
		// must have preempted all goroutines, including any attempting
		// to scan our stack, in which case, any stack shrinking will
		// have already completed by the time we exit.
		//
		// N.B. The execution tracer is not aware of this status
		// transition and handles it specially based on the
		// wait reason.
		casGToWaiting(gp, _Grunning, waitReasonStoppingTheWorld)
		stopTheWorldContext = stopTheWorldWithSema(reason) // avoid write to stack
		casgstatus(gp, _Gwaiting, _Grunning)
	})
	return stopTheWorldContext
}

// startTheWorld undoes the effects of stopTheWorld.
//
// w must be the worldStop returned by stopTheWorld.
func startTheWorld(w worldStop) {
	systemstack(func() { startTheWorldWithSema(0, w) })

	// worldsema must be held over startTheWorldWithSema to ensure
	// gomaxprocs cannot change while worldsema is held.
	//
	// Release worldsema with direct handoff to the next waiter, but
	// acquirem so that semrelease1 doesn't try to yield our time.
	//
	// Otherwise if e.g. ReadMemStats is being called in a loop,
	// it might stomp on other attempts to stop the world, such as
	// for starting or ending GC. The operation this blocks is
	// so heavy-weight that we should just try to be as fair as
	// possible here.
	//
	// We don't want to just allow us to get preempted between now
	// and releasing the semaphore because then we keep everyone
	// (including, for example, GCs) waiting longer.
	mp := acquirem()
	mp.preemptoff = ""
	semrelease1(&worldsema, true, 0)
	releasem(mp)
}

// stopTheWorldGC has the same effect as stopTheWorld, but blocks
// until the GC is not running. It also blocks a GC from starting
// until startTheWorldGC is called.
func stopTheWorldGC(reason stwReason) worldStop {
	semacquire(&gcsema)
	return stopTheWorld(reason)
}

// startTheWorldGC undoes the effects of stopTheWorldGC.
//
// w must be the worldStop returned by stopTheWorld.
func startTheWorldGC(w worldStop) {
	startTheWorld(w)
	semrelease(&gcsema)
}

// Holding worldsema grants an M the right to try to stop the world.
var worldsema uint32 = 1

// Holding gcsema grants the M the right to block a GC, and blocks
// until the current GC is done. In particular, it prevents gomaxprocs
// from changing concurrently.
//
// TODO(mknyszek): Once gomaxprocs and the execution tracer can handle
// being changed/enabled during a GC, remove this.
var gcsema uint32 = 1

// stopTheWorldWithSema is the core implementation of stopTheWorld.
// The caller is responsible for acquiring worldsema and disabling
// preemption first and then should stopTheWorldWithSema on the system
// stack:
//
//	semacquire(&worldsema, 0)
//	m.preemptoff = "reason"
//	var stw worldStop
//	systemstack(func() {
//		stw = stopTheWorldWithSema(reason)
//	})
//
// When finished, the caller must either call startTheWorld or undo
// these three operations separately:
//
//	m.preemptoff = ""
//	systemstack(func() {
//		now = startTheWorldWithSema(stw)
//	})
//	semrelease(&worldsema)
//
// It is allowed to acquire worldsema once and then execute multiple
// startTheWorldWithSema/stopTheWorldWithSema pairs.
// Other P's are able to execute between successive calls to
// startTheWorldWithSema and stopTheWorldWithSema.
// Holding worldsema causes any other goroutines invoking
// stopTheWorld to block.
//
// Returns the STW context. When starting the world, this context must be
// passed to startTheWorldWithSema.
func stopTheWorldWithSema(reason stwReason) worldStop {
	trace := traceAcquire()
	if trace.ok() {
		trace.STWStart(reason)
		traceRelease(trace)
	}
	gp := getg()

	// If we hold a lock, then we won't be able to stop another M
	// that is blocked trying to acquire the lock.
	if gp.m.locks > 0 {
		throw("stopTheWorld: holding locks")
	}

	lock(&sched.lock)
	start := nanotime() // exclude time waiting for sched.lock from start and total time metrics.
	sched.stopwait = gomaxprocs
	sched.gcwaiting.Store(true)
	preemptall()
	// stop current P
	gp.m.p.ptr().status = _Pgcstop // Pgcstop is only diagnostic.
	sched.stopwait--
	// try to retake all P's in Psyscall status
	trace = traceAcquire()
	for _, pp := range allp {
		s := pp.status
		if s == _Psyscall && atomic.Cas(&pp.status, s, _Pgcstop) {
			if trace.ok() {
				trace.GoSysBlock(pp)
				trace.ProcSteal(pp, false)
			}
			pp.syscalltick++
			sched.stopwait--
		}
	}
	if trace.ok() {
		traceRelease(trace)
	}

	// stop idle P's
	now := nanotime()
	for {
		pp, _ := pidleget(now)
		if pp == nil {
			break
		}
		pp.status = _Pgcstop
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

	startTime := nanotime() - start
	if reason.isGC() {
		sched.stwStoppingTimeGC.record(startTime)
	} else {
		sched.stwStoppingTimeOther.record(startTime)
	}

	// sanity checks
	bad := ""
	if sched.stopwait != 0 {
		bad = "stopTheWorld: not stopped (stopwait != 0)"
	} else {
		for _, pp := range allp {
			if pp.status != _Pgcstop {
				bad = "stopTheWorld: not stopped (status != _Pgcstop)"
			}
		}
	}
	if freezing.Load() {
		// Some other thread is panicking. This can cause the
		// sanity checks above to fail if the panic happens in
		// the signal handler on a stopped thread. Either way,
		// we should halt this thread.
		lock(&deadlock)
		lock(&deadlock)
	}
	if bad != "" {
		throw(bad)
	}

	worldStopped()

	return worldStop{reason: reason, start: start}
}

// reason is the same STW reason passed to stopTheWorld. start is the start
// time returned by stopTheWorld.
//
// now is the current time; prefer to pass 0 to capture a fresh timestamp.
//
// stattTheWorldWithSema returns now.
func startTheWorldWithSema(now int64, w worldStop) int64 {
	assertWorldStopped()

	mp := acquirem() // disable preemption because it can be holding p in a local var
	if netpollinited() {
		list, delta := netpoll(0) // non-blocking
		injectglist(&list)
		netpollAdjustWaiters(delta)
	}
	lock(&sched.lock)

	procs := gomaxprocs
	if newprocs != 0 {
		procs = newprocs
		newprocs = 0
	}
	p1 := procresize(procs)
	sched.gcwaiting.Store(false)
	if sched.sysmonwait.Load() {
		sched.sysmonwait.Store(false)
		notewakeup(&sched.sysmonnote)
	}
	unlock(&sched.lock)

	worldStarted()

	for p1 != nil {
		p := p1
		p1 = p1.link.ptr()
		if p.m != 0 {
			mp := p.m.ptr()
			p.m = 0
			if mp.nextp != 0 {
				throw("startTheWorld: inconsistent mp->nextp")
			}
			mp.nextp.set(p)
			notewakeup(&mp.park)
		} else {
			// Start M to run P.  Do not start another M below.
			newm(nil, p, -1)
		}
	}

	// Capture start-the-world time before doing clean-up tasks.
	if now == 0 {
		now = nanotime()
	}
	totalTime := now - w.start
	if w.reason.isGC() {
		sched.stwTotalTimeGC.record(totalTime)
	} else {
		sched.stwTotalTimeOther.record(totalTime)
	}
	trace := traceAcquire()
	if trace.ok() {
		trace.STWDone()
		traceRelease(trace)
	}

	// Wakeup an additional proc in case we have excessive runnable goroutines
	// in local queues or in the global queue. If we don't, the proc will park itself.
	// If we have lots of excessive work, resetspinning will unpark additional procs as necessary.
	wakep()

	releasem(mp)

	return now
}

// usesLibcall indicates whether this runtime performs system calls
// via libcall.
func usesLibcall() bool {
	switch GOOS {
	case "aix", "darwin", "illumos", "ios", "solaris", "windows":
		return true
	case "openbsd":
		return GOARCH != "mips64"
	}
	return false
}

// mStackIsSystemAllocated indicates whether this runtime starts on a
// system-allocated stack.
func mStackIsSystemAllocated() bool {
	switch GOOS {
	case "aix", "darwin", "plan9", "illumos", "ios", "solaris", "windows":
		return true
	case "openbsd":
		return GOARCH != "mips64"
	}
	return false
}

// mstart is the entry-point for new Ms.
// It is written in assembly, uses ABI0, is marked TOPFRAME, and calls mstart0.
func mstart()

// mstart0 is the Go entry-point for new Ms.
// This must not split the stack because we may not even have stack
// bounds set up yet.
//
// May run during STW (because it doesn't have a P yet), so write
// barriers are not allowed.
//
//go:nosplit
//go:nowritebarrierrec
func mstart0() {
	gp := getg()

	osStack := gp.stack.lo == 0
	if osStack {
		// Initialize stack bounds from system stack.
		// Cgo may have left stack size in stack.hi.
		// minit may update the stack bounds.
		//
		// Note: these bounds may not be very accurate.
		// We set hi to &size, but there are things above
		// it. The 1024 is supposed to compensate this,
		// but is somewhat arbitrary.
		size := gp.stack.hi
		if size == 0 {
			size = 16384 * sys.StackGuardMultiplier
		}
		gp.stack.hi = uintptr(noescape(unsafe.Pointer(&size)))
		gp.stack.lo = gp.stack.hi - size + 1024
	}
	// Initialize stack guard so that we can start calling regular
	// Go code.
	gp.stackguard0 = gp.stack.lo + stackGuard
	// This is the g0, so we can also call go:systemstack
	// functions, which check stackguard1.
	gp.stackguard1 = gp.stackguard0
	mstart1()

	// Exit this thread.
	if mStackIsSystemAllocated() {
		// Windows, Solaris, illumos, Darwin, AIX and Plan 9 always system-allocate
		// the stack, but put it in gp.stack before mstart,
		// so the logic above hasn't set osStack yet.
		osStack = true
	}
	mexit(osStack)
}

// The go:noinline is to guarantee the getcallerpc/getcallersp below are safe,
// so that we can set up g0.sched to return to the call of mstart1 above.
//
//go:noinline
func mstart1() {
	gp := getg()

	if gp != gp.m.g0 {
		throw("bad runtime·mstart")
	}

	// Set up m.g0.sched as a label returning to just
	// after the mstart1 call in mstart0 above, for use by goexit0 and mcall.
	// We're never coming back to mstart1 after we call schedule,
	// so other calls can reuse the current frame.
	// And goexit0 does a gogo that needs to return from mstart1
	// and let mstart0 exit the thread.
	gp.sched.g = guintptr(unsafe.Pointer(gp))
	gp.sched.pc = getcallerpc()
	gp.sched.sp = getcallersp()

	asminit()
	minit()

	// Install signal handlers; after minit so that minit can
	// prepare the thread to be able to handle the signals.
	if gp.m == &m0 {
		mstartm0()
	}

	if fn := gp.m.mstartfn; fn != nil {
		fn()
	}

	if gp.m != &m0 {
		acquirep(gp.m.nextp.ptr())
		gp.m.nextp = 0
	}
	schedule()
}

// mstartm0 implements part of mstart1 that only runs on the m0.
//
// Write barriers are allowed here because we know the GC can't be
// running yet, so they'll be no-ops.
//
//go:yeswritebarrierrec
func mstartm0() {
	// Create an extra M for callbacks on threads not created by Go.
	// An extra M is also needed on Windows for callbacks created by
	// syscall.NewCallback. See issue #6751 for details.
	if (iscgo || GOOS == "windows") && !cgoHasExtraM {
		cgoHasExtraM = true
		newextram()
	}
	initsig(false)
}

// mPark causes a thread to park itself, returning once woken.
//
//go:nosplit
func mPark() {
	gp := getg()
	notesleep(&gp.m.park)
	noteclear(&gp.m.park)
}

// mexit tears down and exits the current thread.
//
// Don't call this directly to exit the thread, since it must run at
// the top of the thread stack. Instead, use gogo(&gp.m.g0.sched) to
// unwind the stack to the point that exits the thread.
//
// It is entered with m.p != nil, so write barriers are allowed. It
// will release the P before exiting.
//
//go:yeswritebarrierrec
func mexit(osStack bool) {
	mp := getg().m

	if mp == &m0 {
		// This is the main thread. Just wedge it.
		//
		// On Linux, exiting the main thread puts the process
		// into a non-waitable zombie state. On Plan 9,
		// exiting the main thread unblocks wait even though
		// other threads are still running. On Solaris we can
		// neither exitThread nor return from mstart. Other
		// bad things probably happen on other platforms.
		//
		// We could try to clean up this M more before wedging
		// it, but that complicates signal handling.
		handoffp(releasep())
		lock(&sched.lock)
		sched.nmfreed++
		checkdead()
		unlock(&sched.lock)
		mPark()
		throw("locked m0 woke up")
	}

	sigblock(true)
	unminit()

	// Free the gsignal stack.
	if mp.gsignal != nil {
		stackfree(mp.gsignal.stack)
		// On some platforms, when calling into VDSO (e.g. nanotime)
		// we store our g on the gsignal stack, if there is one.
		// Now the stack is freed, unlink it from the m, so we
		// won't write to it when calling VDSO code.
		mp.gsignal = nil
	}

	// Remove m from allm.
	lock(&sched.lock)
	for pprev := &allm; *pprev != nil; pprev = &(*pprev).alllink {
		if *pprev == mp {
			*pprev = mp.alllink
			goto found
		}
	}
	throw("m not found in allm")
found:
	// Events must not be traced after this point.

	// Delay reaping m until it's done with the stack.
	//
	// Put mp on the free list, though it will not be reaped while freeWait
	// is freeMWait. mp is no longer reachable via allm, so even if it is
	// on an OS stack, we must keep a reference to mp alive so that the GC
	// doesn't free mp while we are still using it.
	//
	// Note that the free list must not be linked through alllink because
	// some functions walk allm without locking, so may be using alllink.
	//
	// N.B. It's important that the M appears on the free list simultaneously
	// with it being removed so that the tracer can find it.
	mp.freeWait.Store(freeMWait)
	mp.freelink = sched.freem
	sched.freem = mp
	unlock(&sched.lock)

	atomic.Xadd64(&ncgocall, int64(mp.ncgocall))
	sched.totalRuntimeLockWaitTime.Add(mp.mLockProfile.waitTime.Load())

	// Release the P.
	handoffp(releasep())
	// After this point we must not have write barriers.

	// Invoke the deadlock detector. This must happen after
	// handoffp because it may have started a new M to take our
	// P's work.
	lock(&sched.lock)
	sched.nmfreed++
	checkdead()
	unlock(&sched.lock)

	if GOOS == "darwin" || GOOS == "ios" {
		// Make sure pendingPreemptSignals is correct when an M exits.
		// For #41702.
		if mp.signalPending.Load() != 0 {
			pendingPreemptSignals.Add(-1)
		}
	}

	// Destroy all allocated resources. After this is called, we may no
	// longer take any locks.
	mdestroy(mp)

	if osStack {
		// No more uses of mp, so it is safe to drop the reference.
		mp.freeWait.Store(freeMRef)

		// Return from mstart and let the system thread
		// library free the g0 stack and terminate the thread.
		return
	}

	// mstart is the thread's entry point, so there's nothing to
	// return to. Exit the thread directly. exitThread will clear
	// m.freeWait when it's done with the stack and the m can be
	// reaped.
	exitThread(&mp.freeWait)
}

// forEachP calls fn(p) for every P p when p reaches a GC safe point.
// If a P is currently executing code, this will bring the P to a GC
// safe point and execute fn on that P. If the P is not executing code
// (it is idle or in a syscall), this will call fn(p) directly while
// preventing the P from exiting its state. This does not ensure that
// fn will run on every CPU executing Go code, but it acts as a global
// memory barrier. GC uses this as a "ragged barrier."
//
// The caller must hold worldsema. fn must not refer to any
// part of the current goroutine's stack, since the GC may move it.
func forEachP(reason waitReason, fn func(*p)) {
	systemstack(func() {
		gp := getg().m.curg
		// Mark the user stack as preemptible so that it may be scanned.
		// Otherwise, our attempt to force all P's to a safepoint could
		// result in a deadlock as we attempt to preempt a worker that's
		// trying to preempt us (e.g. for a stack scan).
		//
		// N.B. The execution tracer is not aware of this status
		// transition and handles it specially based on the
		// wait reason.
		casGToWaiting(gp, _Grunning, reason)
		forEachPInternal(fn)
		casgstatus(gp, _Gwaiting, _Grunning)
	})
}

// forEachPInternal calls fn(p) for every P p when p reaches a GC safe point.
// It is the internal implementation of forEachP.
//
// The caller must hold worldsema and either must ensure that a GC is not
// running (otherwise this may deadlock with the GC trying to preempt this P)
// or it must leave its goroutine in a preemptible state before it switches
// to the systemstack. Due to these restrictions, prefer forEachP when possible.
//
//go:systemstack
func forEachPInternal(fn func(*p)) {
	mp := acquirem()
	pp := getg().m.p.ptr()

	lock(&sched.lock)
	if sched.safePointWait != 0 {
		throw("forEachP: sched.safePointWait != 0")
	}
	sched.safePointWait = gomaxprocs - 1
	sched.safePointFn = fn

	// Ask all Ps to run the safe point function.
	for _, p2 := range allp {
		if p2 != pp {
			atomic.Store(&p2.runSafePointFn, 1)
		}
	}
	preemptall()

	// Any P entering _Pidle or _Psyscall from now on will observe
	// p.runSafePointFn == 1 and will call runSafePointFn when
	// changing its status to _Pidle/_Psyscall.

	// Run safe point function for all idle Ps. sched.pidle will
	// not change because we hold sched.lock.
	for p := sched.pidle.ptr(); p != nil; p = p.link.ptr() {
		if atomic.Cas(&p.runSafePointFn, 1, 0) {
			fn(p)
			sched.safePointWait--
		}
	}

	wait := sched.safePointWait > 0
	unlock(&sched.lock)

	// Run fn for the current P.
	fn(pp)

	// Force Ps currently in _Psyscall into _Pidle and hand them
	// off to induce safe point function execution.
	for _, p2 := range allp {
		s := p2.status

		// We need to be fine-grained about tracing here, since handoffp
		// might call into the tracer, and the tracer is non-reentrant.
		trace := traceAcquire()
		if s == _Psyscall && p2.runSafePointFn == 1 && atomic.Cas(&p2.status, s, _Pidle) {
			if trace.ok() {
				// It's important that we traceRelease before we call handoffp, which may also traceAcquire.
				trace.GoSysBlock(p2)
				trace.ProcSteal(p2, false)
				traceRelease(trace)
			}
			p2.syscalltick++
			handoffp(p2)
		} else if trace.ok() {
			traceRelease(trace)
		}
	}

	// Wait for remaining Ps to run fn.
	if wait {
		for {
			// Wait for 100us, then try to re-preempt in
			// case of any races.
			//
			// Requires system stack.
			if notetsleep(&sched.safePointNote, 100*1000) {
				noteclear(&sched.safePointNote)
				break
			}
			preemptall()
		}
	}
	if sched.safePointWait != 0 {
		throw("forEachP: not done")
	}
	for _, p2 := range allp {
		if p2.runSafePointFn != 0 {
			throw("forEachP: P did not run fn")
		}
	}

	lock(&sched.lock)
	sched.safePointFn = nil
	unlock(&sched.lock)
	releasem(mp)
}

// runSafePointFn runs the safe point function, if any, for this P.
// This should be called like
//
//	if getg().m.p.runSafePointFn != 0 {
//	    runSafePointFn()
//	}
//
// runSafePointFn must be checked on any transition in to _Pidle or
// _Psyscall to avoid a race where forEachP sees that the P is running
// just before the P goes into _Pidle/_Psyscall and neither forEachP
// nor the P run the safe-point function.
func runSafePointFn() {
	p := getg().m.p.ptr()
	// Resolve the race between forEachP running the safe-point
	// function on this P's behalf and this P running the
	// safe-point function directly.
	if !atomic.Cas(&p.runSafePointFn, 1, 0) {
		return
	}
	sched.safePointFn(p)
	lock(&sched.lock)
	sched.safePointWait--
	if sched.safePointWait == 0 {
		notewakeup(&sched.safePointNote)
	}
	unlock(&sched.lock)
}

// When running with cgo, we call _cgo_thread_start
// to start threads for us so that we can play nicely with
// foreign code.
var cgoThreadStart unsafe.Pointer

type cgothreadstart struct {
	g   guintptr
	tls *uint64
	fn  unsafe.Pointer
}

// Allocate a new m unassociated with any thread.
// Can use p for allocation context if needed.
// fn is recorded as the new m's m.mstartfn.
// id is optional pre-allocated m ID. Omit by passing -1.
//
// This function is allowed to have write barriers even if the caller
// isn't because it borrows pp.
//
//go:yeswritebarrierrec
func allocm(pp *p, fn func(), id int64) *m {
	allocmLock.rlock()

	// The caller owns pp, but we may borrow (i.e., acquirep) it. We must
	// disable preemption to ensure it is not stolen, which would make the
	// caller lose ownership.
	acquirem()

	gp := getg()
	if gp.m.p == 0 {
		acquirep(pp) // temporarily borrow p for mallocs in this function
	}

	// Release the free M list. We need to do this somewhere and
	// this may free up a stack we can use.
	if sched.freem != nil {
		lock(&sched.lock)
		var newList *m
		for freem := sched.freem; freem != nil; {
			// Wait for freeWait to indicate that freem's stack is unused.
			wait := freem.freeWait.Load()
			if wait == freeMWait {
				next := freem.freelink
				freem.freelink = newList
				newList = freem
				freem = next
				continue
			}
			// Drop any remaining trace resources.
			// Ms can continue to emit events all the way until wait != freeMWait,
			// so it's only safe to call traceThreadDestroy at this point.
			if traceEnabled() || traceShuttingDown() {
				traceThreadDestroy(freem)
			}
			// Free the stack if needed. For freeMRef, there is
			// nothing to do except drop freem from the sched.freem
			// list.
			if wait == freeMStack {
				// stackfree must be on the system stack, but allocm is
				// reachable off the system stack transitively from
				// startm.
				systemstack(func() {
					stackfree(freem.g0.stack)
				})
			}
			freem = freem.freelink
		}
		sched.freem = newList
		unlock(&sched.lock)
	}

	mp := new(m)
	mp.mstartfn = fn
	mcommoninit(mp, id)

	// In case of cgo or Solaris or illumos or Darwin, pthread_create will make us a stack.
	// Windows and Plan 9 will layout sched stack on OS stack.
	if iscgo || mStackIsSystemAllocated() {
		mp.g0 = malg(-1)
	} else {
		mp.g0 = malg(16384 * sys.StackGuardMultiplier)
	}
	mp.g0.m = mp

	if pp == gp.m.p.ptr() {
		releasep()
	}

	releasem(gp.m)
	allocmLock.runlock()
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
// a lock by doing an exchange (via Casuintptr) to steal the stack
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
// It calls dropm to put the m back on the list,
// 1. when the callback is done with the m in non-pthread platforms,
// 2. or when the C thread exiting on pthread platforms.
//
// The signal argument indicates whether we're called from a signal
// handler.
//
//go:nosplit
func needm(signal bool) {
	if (iscgo || GOOS == "windows") && !cgoHasExtraM {
		// Can happen if C/C++ code calls Go from a global ctor.
		// Can also happen on Windows if a global ctor uses a
		// callback created by syscall.NewCallback. See issue #6751
		// for details.
		//
		// Can not throw, because scheduler is not initialized yet.
		writeErrStr("fatal error: cgo callback before cgo call\n")
		exit(1)
	}

	// Save and block signals before getting an M.
	// The signal handler may call needm itself,
	// and we must avoid a deadlock. Also, once g is installed,
	// any incoming signals will try to execute,
	// but we won't have the sigaltstack settings and other data
	// set up appropriately until the end of minit, which will
	// unblock the signals. This is the same dance as when
	// starting a new m to run Go code via newosproc.
	var sigmask sigset
	sigsave(&sigmask)
	sigblock(false)

	// getExtraM is safe here because of the invariant above,
	// that the extra list always contains or will soon contain
	// at least one m.
	mp, last := getExtraM()

	// Set needextram when we've just emptied the list,
	// so that the eventual call into cgocallbackg will
	// allocate a new m for the extra list. We delay the
	// allocation until then so that it can be done
	// after exitsyscall makes sure it is okay to be
	// running at all (that is, there's no garbage collection
	// running right now).
	mp.needextram = last

	// Store the original signal mask for use by minit.
	mp.sigmask = sigmask

	// Install TLS on some platforms (previously setg
	// would do this if necessary).
	osSetupTLS(mp)

	// Install g (= m->g0) and set the stack bounds
	// to match the current stack.
	setg(mp.g0)
	sp := getcallersp()
	callbackUpdateSystemStack(mp, sp, signal)

	// Should mark we are already in Go now.
	// Otherwise, we may call needm again when we get a signal, before cgocallbackg1,
	// which means the extram list may be empty, that will cause a deadlock.
	mp.isExtraInC = false

	// Initialize this thread to use the m.
	asminit()
	minit()

	// Emit a trace event for this dead -> syscall transition,
	// but only in the new tracer and only if we're not in a signal handler.
	//
	// N.B. the tracer can run on a bare M just fine, we just have
	// to make sure to do this before setg(nil) and unminit.
	var trace traceLocker
	if goexperiment.ExecTracer2 && !signal {
		trace = traceAcquire()
	}

	// mp.curg is now a real goroutine.
	casgstatus(mp.curg, _Gdead, _Gsyscall)
	sched.ngsys.Add(-1)

	if goexperiment.ExecTracer2 && !signal {
		if trace.ok() {
			trace.GoCreateSyscall(mp.curg)
			traceRelease(trace)
		}
	}
	mp.isExtraInSig = signal
}

// Acquire an extra m and bind it to the C thread when a pthread key has been created.
//
//go:nosplit
func needAndBindM() {
	needm(false)

	if _cgo_pthread_key_created != nil && *(*uintptr)(_cgo_pthread_key_created) != 0 {
		cgoBindM()
	}
}

// newextram allocates m's and puts them on the extra list.
// It is called with a working local m, so that it can do things
// like call schedlock and allocate.
func newextram() {
	c := extraMWaiters.Swap(0)
	if c > 0 {
		for i := uint32(0); i < c; i++ {
			oneNewExtraM()
		}
	} else if extraMLength.Load() == 0 {
		// Make sure there is at least one extra M.
		oneNewExtraM()
	}
}

// oneNewExtraM allocates an m and puts it on the extra list.
func oneNewExtraM() {
	// Create extra goroutine locked to extra m.
	// The goroutine is the context in which the cgo callback will run.
	// The sched.pc will never be returned to, but setting it to
	// goexit makes clear to the traceback routines where
	// the goroutine stack ends.
	mp := allocm(nil, nil, -1)
	gp := malg(4096)
	gp.sched.pc = abi.FuncPCABI0(goexit) + sys.PCQuantum
	gp.sched.sp = gp.stack.hi
	gp.sched.sp -= 4 * goarch.PtrSize // extra space in case of reads slightly beyond frame
	gp.sched.lr = 0
	gp.sched.g = guintptr(unsafe.Pointer(gp))
	gp.syscallpc = gp.sched.pc
	gp.syscallsp = gp.sched.sp
	gp.stktopsp = gp.sched.sp
	// malg returns status as _Gidle. Change to _Gdead before
	// adding to allg where GC can see it. We use _Gdead to hide
	// this from tracebacks and stack scans since it isn't a
	// "real" goroutine until needm grabs it.
	casgstatus(gp, _Gidle, _Gdead)
	gp.m = mp
	mp.curg = gp
	mp.isextra = true
	// mark we are in C by default.
	mp.isExtraInC = true
	mp.lockedInt++
	mp.lockedg.set(gp)
	gp.lockedm.set(mp)
	gp.goid = sched.goidgen.Add(1)
	if raceenabled {
		gp.racectx = racegostart(abi.FuncPCABIInternal(newextram) + sys.PCQuantum)
	}
	trace := traceAcquire()
	if trace.ok() {
		trace.OneNewExtraM(gp)
		traceRelease(trace)
	}
	// put on allg for garbage collector
	allgadd(gp)

	// gp is now on the allg list, but we don't want it to be
	// counted by gcount. It would be more "proper" to increment
	// sched.ngfree, but that requires locking. Incrementing ngsys
	// has the same effect.
	sched.ngsys.Add(1)

	// Add m to the extra list.
	addExtraM(mp)
}

// dropm puts the current m back onto the extra list.
//
// 1. On systems without pthreads, like Windows
// dropm is called when a cgo callback has called needm but is now
// done with the callback and returning back into the non-Go thread.
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
// 2. On systems with pthreads
// dropm is called while a non-Go thread is exiting.
// We allocate a pthread per-thread variable using pthread_key_create,
// to register a thread-exit-time destructor.
// And store the g into a thread-specific value associated with the pthread key,
// when first return back to C.
// So that the destructor would invoke dropm while the non-Go thread is exiting.
// This is much faster since it avoids expensive signal-related syscalls.
//
// This always runs without a P, so //go:nowritebarrierrec is required.
//
// This may run with a different stack than was recorded in g0 (there is no
// call to callbackUpdateSystemStack prior to dropm), so this must be
// //go:nosplit to avoid the stack bounds check.
//
//go:nowritebarrierrec
//go:nosplit
func dropm() {
	// Clear m and g, and return m to the extra list.
	// After the call to setg we can only call nosplit functions
	// with no pointer manipulation.
	mp := getg().m

	// Emit a trace event for this syscall -> dead transition,
	// but only in the new tracer.
	//
	// N.B. the tracer can run on a bare M just fine, we just have
	// to make sure to do this before setg(nil) and unminit.
	var trace traceLocker
	if goexperiment.ExecTracer2 && !mp.isExtraInSig {
		trace = traceAcquire()
	}

	// Return mp.curg to dead state.
	casgstatus(mp.curg, _Gsyscall, _Gdead)
	mp.curg.preemptStop = false
	sched.ngsys.Add(1)

	if goexperiment.ExecTracer2 && !mp.isExtraInSig {
		if trace.ok() {
			trace.GoDestroySyscall()
			traceRelease(trace)
		}
	}

	if goexperiment.ExecTracer2 {
		// Trash syscalltick so that it doesn't line up with mp.old.syscalltick anymore.
		//
		// In the new tracer, we model needm and dropm and a goroutine being created and
		// destroyed respectively. The m then might get reused with a different procid but
		// still with a reference to oldp, and still with the same syscalltick. The next
		// time a G is "created" in needm, it'll return and quietly reacquire its P from a
		// different m with a different procid, which will confuse the trace parser. By
		// trashing syscalltick, we ensure that it'll appear as if we lost the P to the
		// tracer parser and that we just reacquired it.
		//
		// Trash the value by decrementing because that gets us as far away from the value
		// the syscall exit code expects as possible. Setting to zero is risky because
		// syscalltick could already be zero (and in fact, is initialized to zero).
		mp.syscalltick--
	}

	// Reset trace state unconditionally. This goroutine is being 'destroyed'
	// from the perspective of the tracer.
	mp.curg.trace.reset()

	// Flush all the M's buffers. This is necessary because the M might
	// be used on a different thread with a different procid, so we have
	// to make sure we don't write into the same buffer.
	//
	// N.B. traceThreadDestroy is a no-op in the old tracer, so avoid the
	// unnecessary acquire/release of the lock.
	if goexperiment.ExecTracer2 && (traceEnabled() || traceShuttingDown()) {
		// Acquire sched.lock across thread destruction. One of the invariants of the tracer
		// is that a thread cannot disappear from the tracer's view (allm or freem) without
		// it noticing, so it requires that sched.lock be held over traceThreadDestroy.
		//
		// This isn't strictly necessary in this case, because this thread never leaves allm,
		// but the critical section is short and dropm is rare on pthread platforms, so just
		// take the lock and play it safe. traceThreadDestroy also asserts that the lock is held.
		lock(&sched.lock)
		traceThreadDestroy(mp)
		unlock(&sched.lock)
	}
	mp.isExtraInSig = false

	// Block signals before unminit.
	// Unminit unregisters the signal handling stack (but needs g on some systems).
	// Setg(nil) clears g, which is the signal handler's cue not to run Go handlers.
	// It's important not to try to handle a signal between those two steps.
	sigmask := mp.sigmask
	sigblock(false)
	unminit()

	setg(nil)

	// Clear g0 stack bounds to ensure that needm always refreshes the
	// bounds when reusing this M.
	g0 := mp.g0
	g0.stack.hi = 0
	g0.stack.lo = 0
	g0.stackguard0 = 0
	g0.stackguard1 = 0

	putExtraM(mp)

	msigrestore(sigmask)
}

// bindm store the g0 of the current m into a thread-specific value.
//
// We allocate a pthread per-thread variable using pthread_key_create,
// to register a thread-exit-time destructor.
// We are here setting the thread-specific value of the pthread key, to enable the destructor.
// So that the pthread_key_destructor would dropm while the C thread is exiting.
//
// And the saved g will be used in pthread_key_destructor,
// since the g stored in the TLS by Go might be cleared in some platforms,
// before the destructor invoked, so, we restore g by the stored g, before dropm.
//
// We store g0 instead of m, to make the assembly code simpler,
// since we need to restore g0 in runtime.cgocallback.
//
// On systems without pthreads, like Windows, bindm shouldn't be used.
//
// NOTE: this always runs without a P, so, nowritebarrierrec required.
//
//go:nosplit
//go:nowritebarrierrec
func cgoBindM() {
	if GOOS == "windows" || GOOS == "plan9" {
		fatal("bindm in unexpected GOOS")
	}
	g := getg()
	if g.m.g0 != g {
		fatal("the current g is not g0")
	}
	if _cgo_bindm != nil {
		asmcgocall(_cgo_bindm, unsafe.Pointer(g))
	}
}

// A helper function for EnsureDropM.
func getm() uintptr {
	return uintptr(unsafe.Pointer(getg().m))
}

var (
	// Locking linked list of extra M's, via mp.schedlink. Must be accessed
	// only via lockextra/unlockextra.
	//
	// Can't be atomic.Pointer[m] because we use an invalid pointer as a
	// "locked" sentinel value. M's on this list remain visible to the GC
	// because their mp.curg is on allgs.
	extraM atomic.Uintptr
	// Number of M's in the extraM list.
	extraMLength atomic.Uint32
	// Number of waiters in lockextra.
	extraMWaiters atomic.Uint32

	// Number of extra M's in use by threads.
	extraMInUse atomic.Uint32
)

// lockextra locks the extra list and returns the list head.
// The caller must unlock the list by storing a new list head
// to extram. If nilokay is true, then lockextra will
// return a nil list head if that's what it finds. If nilokay is false,
// lockextra will keep waiting until the list head is no longer nil.
//
//go:nosplit
func lockextra(nilokay bool) *m {
	const locked = 1

	incr := false
	for {
		old := extraM.Load()
		if old == locked {
			osyield_no_g()
			continue
		}
		if old == 0 && !nilokay {
			if !incr {
				// Add 1 to the number of threads
				// waiting for an M.
				// This is cleared by newextram.
				extraMWaiters.Add(1)
				incr = true
			}
			usleep_no_g(1)
			continue
		}
		if extraM.CompareAndSwap(old, locked) {
			return (*m)(unsafe.Pointer(old))
		}
		osyield_no_g()
		continue
	}
}

//go:nosplit
func unlockextra(mp *m, delta int32) {
	extraMLength.Add(delta)
	extraM.Store(uintptr(unsafe.Pointer(mp)))
}

// Return an M from the extra M list. Returns last == true if the list becomes
// empty because of this call.
//
// Spins waiting for an extra M, so caller must ensure that the list always
// contains or will soon contain at least one M.
//
//go:nosplit
func getExtraM() (mp *m, last bool) {
	mp = lockextra(false)
	extraMInUse.Add(1)
	unlockextra(mp.schedlink.ptr(), -1)
	return mp, mp.schedlink.ptr() == nil
}

// Returns an extra M back to the list. mp must be from getExtraM. Newly
// allocated M's should use addExtraM.
//
//go:nosplit
func putExtraM(mp *m) {
	extraMInUse.Add(-1)
	addExtraM(mp)
}

// Adds a newly allocated M to the extra M list.
//
//go:nosplit
func addExtraM(mp *m) {
	mnext := lockextra(true)
	mp.schedlink.set(mnext)
	unlockextra(mp, 1)
}

var (
	// allocmLock is locked for read when creating new Ms in allocm and their
	// addition to allm. Thus acquiring this lock for write blocks the
	// creation of new Ms.
	allocmLock rwmutex

	// execLock serializes exec and clone to avoid bugs or unspecified
	// behaviour around exec'ing while creating/destroying threads. See
	// issue #19546.
	execLock rwmutex
)

// These errors are reported (via writeErrStr) by some OS-specific
// versions of newosproc and newosproc0.
const (
	failthreadcreate  = "runtime: failed to create new OS thread\n"
	failallocatestack = "runtime: failed to allocate stack for the new OS thread\n"
)

// newmHandoff contains a list of m structures that need new OS threads.
// This is used by newm in situations where newm itself can't safely
// start an OS thread.
var newmHandoff struct {
	lock mutex

	// newm points to a list of M structures that need new OS
	// threads. The list is linked through m.schedlink.
	newm muintptr

	// waiting indicates that wake needs to be notified when an m
	// is put on the list.
	waiting bool
	wake    note

	// haveTemplateThread indicates that the templateThread has
	// been started. This is not protected by lock. Use cas to set
	// to 1.
	haveTemplateThread uint32
}

// Create a new m. It will start off with a call to fn, or else the scheduler.
// fn needs to be static and not a heap allocated closure.
// May run with m.p==nil, so write barriers are not allowed.
//
// id is optional pre-allocated m ID. Omit by passing -1.
//
//go:nowritebarrierrec
func newm(fn func(), pp *p, id int64) {
	// allocm adds a new M to allm, but they do not start until created by
	// the OS in newm1 or the template thread.
	//
	// doAllThreadsSyscall requires that every M in allm will eventually
	// start and be signal-able, even with a STW.
	//
	// Disable preemption here until we start the thread to ensure that
	// newm is not preempted between allocm and starting the new thread,
	// ensuring that anything added to allm is guaranteed to eventually
	// start.
	acquirem()

	mp := allocm(pp, fn, id)
	mp.nextp.set(pp)
	mp.sigmask = initSigmask
	if gp := getg(); gp != nil && gp.m != nil && (gp.m.lockedExt != 0 || gp.m.incgo) && GOOS != "plan9" {
		// We're on a locked M or a thread that may have been
		// started by C. The kernel state of this thread may
		// be strange (the user may have locked it for that
		// purpose). We don't want to clone that into another
		// thread. Instead, ask a known-good thread to create
		// the thread for us.
		//
		// This is disabled on Plan 9. See golang.org/issue/22227.
		//
		// TODO: This may be unnecessary on Windows, which
		// doesn't model thread creation off fork.
		lock(&newmHandoff.lock)
		if newmHandoff.haveTemplateThread == 0 {
			throw("on a locked thread with no template thread")
		}
		mp.schedlink = newmHandoff.newm
		newmHandoff.newm.set(mp)
		if newmHandoff.waiting {
			newmHandoff.waiting = false
			notewakeup(&newmHandoff.wake)
		}
		unlock(&newmHandoff.lock)
		// The M has not started yet, but the template thread does not
		// participate in STW, so it will always process queued Ms and
		// it is safe to releasem.
		releasem(getg().m)
		return
	}
	newm1(mp)
	releasem(getg().m)
}

func newm1(mp *m) {
	if iscgo {
		var ts cgothreadstart
		if _cgo_thread_start == nil {
			throw("_cgo_thread_start missing")
		}
		ts.g.set(mp.g0)
		ts.tls = (*uint64)(unsafe.Pointer(&mp.tls[0]))
		ts.fn = unsafe.Pointer(abi.FuncPCABI0(mstart))
		if msanenabled {
			msanwrite(unsafe.Pointer(&ts), unsafe.Sizeof(ts))
		}
		if asanenabled {
			asanwrite(unsafe.Pointer(&ts), unsafe.Sizeof(ts))
		}
		execLock.rlock() // Prevent process clone.
		asmcgocall(_cgo_thread_start, unsafe.Pointer(&ts))
		execLock.runlock()
		return
	}
	execLock.rlock() // Prevent process clone.
	newosproc(mp)
	execLock.runlock()
}

// startTemplateThread starts the template thread if it is not already
// running.
//
// The calling thread must itself be in a known-good state.
func startTemplateThread() {
	if GOARCH == "wasm" { // no threads on wasm yet
		return
	}

	// Disable preemption to guarantee that the template thread will be
	// created before a park once haveTemplateThread is set.
	mp := acquirem()
	if !atomic.Cas(&newmHandoff.haveTemplateThread, 0, 1) {
		releasem(mp)
		return
	}
	newm(templateThread, nil, -1)
	releasem(mp)
}

// templateThread is a thread in a known-good state that exists solely
// to start new threads in known-good states when the calling thread
// may not be in a good state.
//
// Many programs never need this, so templateThread is started lazily
// when we first enter a state that might lead to running on a thread
// in an unknown state.
//
// templateThread runs on an M without a P, so it must not have write
// barriers.
//
//go:nowritebarrierrec
func templateThread() {
	lock(&sched.lock)
	sched.nmsys++
	checkdead()
	unlock(&sched.lock)

	for {
		lock(&newmHandoff.lock)
		for newmHandoff.newm != 0 {
			newm := newmHandoff.newm.ptr()
			newmHandoff.newm = 0
			unlock(&newmHandoff.lock)
			for newm != nil {
				next := newm.schedlink.ptr()
				newm.schedlink = 0
				newm1(newm)
				newm = next
			}
			lock(&newmHandoff.lock)
		}
		newmHandoff.waiting = true
		noteclear(&newmHandoff.wake)
		unlock(&newmHandoff.lock)
		notesleep(&newmHandoff.wake)
	}
}

// Stops execution of the current m until new work is available.
// Returns with acquired P.
func stopm() {
	gp := getg()

	if gp.m.locks != 0 {
		throw("stopm holding locks")
	}
	if gp.m.p != 0 {
		throw("stopm holding p")
	}
	if gp.m.spinning {
		throw("stopm spinning")
	}

	lock(&sched.lock)
	mput(gp.m)
	unlock(&sched.lock)
	mPark()
	acquirep(gp.m.nextp.ptr())
	gp.m.nextp = 0
}

func mspinning() {
	// startm's caller incremented nmspinning. Set the new M's spinning.
	getg().m.spinning = true
}

// Schedules some M to run the p (creates an M if necessary).
// If p==nil, tries to get an idle P, if no idle P's does nothing.
// May run with m.p==nil, so write barriers are not allowed.
// If spinning is set, the caller has incremented nmspinning and must provide a
// P. startm will set m.spinning in the newly started M.
//
// Callers passing a non-nil P must call from a non-preemptible context. See
// comment on acquirem below.
//
// Argument lockheld indicates whether the caller already acquired the
// scheduler lock. Callers holding the lock when making the call must pass
// true. The lock might be temporarily dropped, but will be reacquired before
// returning.
//
// Must not have write barriers because this may be called without a P.
//
//go:nowritebarrierrec
func startm(pp *p, spinning, lockheld bool) {
	// Disable preemption.
	//
	// Every owned P must have an owner that will eventually stop it in the
	// event of a GC stop request. startm takes transient ownership of a P
	// (either from argument or pidleget below) and transfers ownership to
	// a started M, which will be responsible for performing the stop.
	//
	// Preemption must be disabled during this transient ownership,
	// otherwise the P this is running on may enter GC stop while still
	// holding the transient P, leaving that P in limbo and deadlocking the
	// STW.
	//
	// Callers passing a non-nil P must already be in non-preemptible
	// context, otherwise such preemption could occur on function entry to
	// startm. Callers passing a nil P may be preemptible, so we must
	// disable preemption before acquiring a P from pidleget below.
	mp := acquirem()
	if !lockheld {
		lock(&sched.lock)
	}
	if pp == nil {
		if spinning {
			// TODO(prattmic): All remaining calls to this function
			// with _p_ == nil could be cleaned up to find a P
			// before calling startm.
			throw("startm: P required for spinning=true")
		}
		pp, _ = pidleget(0)
		if pp == nil {
			if !lockheld {
				unlock(&sched.lock)
			}
			releasem(mp)
			return
		}
	}
	nmp := mget()
	if nmp == nil {
		// No M is available, we must drop sched.lock and call newm.
		// However, we already own a P to assign to the M.
		//
		// Once sched.lock is released, another G (e.g., in a syscall),
		// could find no idle P while checkdead finds a runnable G but
		// no running M's because this new M hasn't started yet, thus
		// throwing in an apparent deadlock.
		// This apparent deadlock is possible when startm is called
		// from sysmon, which doesn't count as a running M.
		//
		// Avoid this situation by pre-allocating the ID for the new M,
		// thus marking it as 'running' before we drop sched.lock. This
		// new M will eventually run the scheduler to execute any
		// queued G's.
		id := mReserveID()
		unlock(&sched.lock)

		var fn func()
		if spinning {
			// The caller incremented nmspinning, so set m.spinning in the new M.
			fn = mspinning
		}
		newm(fn, pp, id)

		if lockheld {
			lock(&sched.lock)
		}
		// Ownership transfer of pp committed by start in newm.
		// Preemption is now safe.
		releasem(mp)
		return
	}
	if !lockheld {
		unlock(&sched.lock)
	}
	if nmp.spinning {
		throw("startm: m is spinning")
	}
	if nmp.nextp != 0 {
		throw("startm: m has p")
	}
	if spinning && !runqempty(pp) {
		throw("startm: p has runnable gs")
	}
	// The caller incremented nmspinning, so set m.spinning in the new M.
	nmp.spinning = spinning
	nmp.nextp.set(pp)
	notewakeup(&nmp.park)
	// Ownership transfer of pp committed by wakeup. Preemption is now
	// safe.
	releasem(mp)
}

// Hands off P from syscall or locked M.
// Always runs without a P, so write barriers are not allowed.
//
//go:nowritebarrierrec
func handoffp(pp *p) {
	// handoffp must start an M in any situation where
	// findrunnable would return a G to run on pp.

	// if it has local work, start it straight away
	if !runqempty(pp) || sched.runqsize != 0 {
		startm(pp, false, false)
		return
	}
	// if there's trace work to do, start it straight away
	if (traceEnabled() || traceShuttingDown()) && traceReaderAvailable() != nil {
		startm(pp, false, false)
		return
	}
	// if it has GC work, start it straight away
	if gcBlackenEnabled != 0 && gcMarkWorkAvailable(pp) {
		startm(pp, false, false)
		return
	}
	// no local work, check that there are no spinning/idle M's,
	// otherwise our help is not required
	if sched.nmspinning.Load()+sched.npidle.Load() == 0 && sched.nmspinning.CompareAndSwap(0, 1) { // TODO: fast atomic
		sched.needspinning.Store(0)
		startm(pp, true, false)
		return
	}
	lock(&sched.lock)
	if sched.gcwaiting.Load() {
		pp.status = _Pgcstop
		sched.stopwait--
		if sched.stopwait == 0 {
			notewakeup(&sched.stopnote)
		}
		unlock(&sched.lock)
		return
	}
	if pp.runSafePointFn != 0 && atomic.Cas(&pp.runSafePointFn, 1, 0) {
		sched.safePointFn(pp)
		sched.safePointWait--
		if sched.safePointWait == 0 {
			notewakeup(&sched.safePointNote)
		}
	}
	if sched.runqsize != 0 {
		unlock(&sched.lock)
		startm(pp, false, false)
		return
	}
	// If this is the last running P and nobody is polling network,
	// need to wakeup another M to poll network.
	if sched.npidle.Load() == gomaxprocs-1 && sched.lastpoll.Load() != 0 {
		unlock(&sched.lock)
		startm(pp, false, false)
		return
	}

	// The scheduler lock cannot be held when calling wakeNetPoller below
	// because wakeNetPoller may call wakep which may call startm.
	when := nobarrierWakeTime(pp)
	pidleput(pp, 0)
	unlock(&sched.lock)

	if when != 0 {
		wakeNetPoller(when)
	}
}

// Tries to add one more P to execute G's.
// Called when a G is made runnable (newproc, ready).
// Must be called with a P.
func wakep() {
	// Be conservative about spinning threads, only start one if none exist
	// already.
	if sched.nmspinning.Load() != 0 || !sched.nmspinning.CompareAndSwap(0, 1) {
		return
	}

	// Disable preemption until ownership of pp transfers to the next M in
	// startm. Otherwise preemption here would leave pp stuck waiting to
	// enter _Pgcstop.
	//
	// See preemption comment on acquirem in startm for more details.
	mp := acquirem()

	var pp *p
	lock(&sched.lock)
	pp, _ = pidlegetSpinning(0)
	if pp == nil {
		if sched.nmspinning.Add(-1) < 0 {
			throw("wakep: negative nmspinning")
		}
		unlock(&sched.lock)
		releasem(mp)
		return
	}
	// Since we always have a P, the race in the "No M is available"
	// comment in startm doesn't apply during the small window between the
	// unlock here and lock in startm. A checkdead in between will always
	// see at least one running M (ours).
	unlock(&sched.lock)

	startm(pp, true, false)

	releasem(mp)
}

// Stops execution of the current m that is locked to a g until the g is runnable again.
// Returns with acquired P.
func stoplockedm() {
	gp := getg()

	if gp.m.lockedg == 0 || gp.m.lockedg.ptr().lockedm.ptr() != gp.m {
		throw("stoplockedm: inconsistent locking")
	}
	if gp.m.p != 0 {
		// Schedule another M to run this p.
		pp := releasep()
		handoffp(pp)
	}
	incidlelocked(1)
	// Wait until another thread schedules lockedg again.
	mPark()
	status := readgstatus(gp.m.lockedg.ptr())
	if status&^_Gscan != _Grunnable {
		print("runtime:stoplockedm: lockedg (atomicstatus=", status, ") is not Grunnable or Gscanrunnable\n")
		dumpgstatus(gp.m.lockedg.ptr())
		throw("stoplockedm: not runnable")
	}
	acquirep(gp.m.nextp.ptr())
	gp.m.nextp = 0
}

// Schedules the locked m to run the locked gp.
// May run during STW, so write barriers are not allowed.
//
//go:nowritebarrierrec
func startlockedm(gp *g) {
	mp := gp.lockedm.ptr()
	if mp == getg().m {
		throw("startlockedm: locked to me")
	}
	if mp.nextp != 0 {
		throw("startlockedm: m has p")
	}
	// directly handoff current P to the locked m
	incidlelocked(-1)
	pp := releasep()
	mp.nextp.set(pp)
	notewakeup(&mp.park)
	stopm()
}

// Stops the current m for stopTheWorld.
// Returns when the world is restarted.
func gcstopm() {
	gp := getg()

	if !sched.gcwaiting.Load() {
		throw("gcstopm: not waiting for gc")
	}
	if gp.m.spinning {
		gp.m.spinning = false
		// OK to just drop nmspinning here,
		// startTheWorld will unpark threads as necessary.
		if sched.nmspinning.Add(-1) < 0 {
			throw("gcstopm: negative nmspinning")
		}
	}
	pp := releasep()
	lock(&sched.lock)
	pp.status = _Pgcstop
	sched.stopwait--
	if sched.stopwait == 0 {
		notewakeup(&sched.stopnote)
	}
	unlock(&sched.lock)
	stopm()
}

// Schedules gp to run on the current M.
// If inheritTime is true, gp inherits the remaining time in the
// current time slice. Otherwise, it starts a new time slice.
// Never returns.
//
// Write barriers are allowed because this is called immediately after
// acquiring a P in several places.
//
//go:yeswritebarrierrec
func execute(gp *g, inheritTime bool) {
	mp := getg().m

	if goroutineProfile.active {
		// Make sure that gp has had its stack written out to the goroutine
		// profile, exactly as it was when the goroutine profiler first stopped
		// the world.
		tryRecordGoroutineProfile(gp, osyield)
	}

	// Assign gp.m before entering _Grunning so running Gs have an
	// M.
	mp.curg = gp
	gp.m = mp
	casgstatus(gp, _Grunnable, _Grunning)
	gp.waitsince = 0
	gp.preempt = false
	gp.stackguard0 = gp.stack.lo + stackGuard
	if !inheritTime {
		mp.p.ptr().schedtick++
	}

	// Check whether the profiler needs to be turned on or off.
	hz := sched.profilehz
	if mp.profilehz != hz {
		setThreadCPUProfiler(hz)
	}

	trace := traceAcquire()
	if trace.ok() {
		// GoSysExit has to happen when we have a P, but before GoStart.
		// So we emit it here.
		if !goexperiment.ExecTracer2 && gp.syscallsp != 0 {
			trace.GoSysExit(true)
		}
		trace.GoStart()
		traceRelease(trace)
	}

	gogo(&gp.sched)
}

// Finds a runnable goroutine to execute.
// Tries to steal from other P's, get g from local or global queue, poll network.
// tryWakeP indicates that the returned goroutine is not normal (GC worker, trace
// reader) so the caller should try to wake a P.
func findRunnable() (gp *g, inheritTime, tryWakeP bool) {
	mp := getg().m

	// The conditions here and in handoffp must agree: if
	// findrunnable would return a G to run, handoffp must start
	// an M.

top:
	pp := mp.p.ptr()
	if sched.gcwaiting.Load() {
		gcstopm()
		goto top
	}
	if pp.runSafePointFn != 0 {
		runSafePointFn()
	}

	// now and pollUntil are saved for work stealing later,
	// which may steal timers. It's important that between now
	// and then, nothing blocks, so these numbers remain mostly
	// relevant.
	now, pollUntil, _ := checkTimers(pp, 0)

	// Try to schedule the trace reader.
	if traceEnabled() || traceShuttingDown() {
		gp := traceReader()
		if gp != nil {
			trace := traceAcquire()
			casgstatus(gp, _Gwaiting, _Grunnable)
			if trace.ok() {
				trace.GoUnpark(gp, 0)
				traceRelease(trace)
			}
			return gp, false, true
		}
	}

	// Try to schedule a GC worker.
	if gcBlackenEnabled != 0 {
		gp, tnow := gcController.findRunnableGCWorker(pp, now)
		if gp != nil {
			return gp, false, true
		}
		now = tnow
	}

	// Check the global runnable queue once in a while to ensure fairness.
	// Otherwise two goroutines can completely occupy the local runqueue
	// by constantly respawning each other.
	if pp.schedtick%61 == 0 && sched.runqsize > 0 {
		lock(&sched.lock)
		gp := globrunqget(pp, 1)
		unlock(&sched.lock)
		if gp != nil {
			return gp, false, false
		}
	}

	// Wake up the finalizer G.
	if fingStatus.Load()&(fingWait|fingWake) == fingWait|fingWake {
		if gp := wakefing(); gp != nil {
			ready(gp, 0, true)
		}
	}
	if *cgo_yield != nil {
		asmcgocall(*cgo_yield, nil)
	}

	// local runq
	if gp, inheritTime := runqget(pp); gp != nil {
		return gp, inheritTime, false
	}

	// global runq
	if sched.runqsize != 0 {
		lock(&sched.lock)
		gp := globrunqget(pp, 0)
		unlock(&sched.lock)
		if gp != nil {
			return gp, false, false
		}
	}

	// Poll network.
	// This netpoll is only an optimization before we resort to stealing.
	// We can safely skip it if there are no waiters or a thread is blocked
	// in netpoll already. If there is any kind of logical race with that
	// blocked thread (e.g. it has already returned from netpoll, but does
	// not set lastpoll yet), this thread will do blocking netpoll below
	// anyway.
	if netpollinited() && netpollAnyWaiters() && sched.lastpoll.Load() != 0 {
		if list, delta := netpoll(0); !list.empty() { // non-blocking
			gp := list.pop()
			injectglist(&list)
			netpollAdjustWaiters(delta)
			trace := traceAcquire()
			casgstatus(gp, _Gwaiting, _Grunnable)
			if trace.ok() {
				trace.GoUnpark(gp, 0)
				traceRelease(trace)
			}
			return gp, false, false
		}
	}

	// Spinning Ms: steal work from other Ps.
	//
	// Limit the number of spinning Ms to half the number of busy Ps.
	// This is necessary to prevent excessive CPU consumption when
	// GOMAXPROCS>>1 but the program parallelism is low.
	if mp.spinning || 2*sched.nmspinning.Load() < gomaxprocs-sched.npidle.Load() {
		if !mp.spinning {
			mp.becomeSpinning()
		}

		gp, inheritTime, tnow, w, newWork := stealWork(now)
		if gp != nil {
			// Successfully stole.
			return gp, inheritTime, false
		}
		if newWork {
			// There may be new timer or GC work; restart to
			// discover.
			goto top
		}

		now = tnow
		if w != 0 && (pollUntil == 0 || w < pollUntil) {
			// Earlier timer to wait for.
			pollUntil = w
		}
	}

	// We have nothing to do.
	//
	// If we're in the GC mark phase, can safely scan and blacken objects,
	// and have work to do, run idle-time marking rather than give up the P.
	if gcBlackenEnabled != 0 && gcMarkWorkAvailable(pp) && gcController.addIdleMarkWorker() {
		node := (*gcBgMarkWorkerNode)(gcBgMarkWorkerPool.pop())
		if node != nil {
			pp.gcMarkWorkerMode = gcMarkWorkerIdleMode
			gp := node.gp.ptr()

			trace := traceAcquire()
			casgstatus(gp, _Gwaiting, _Grunnable)
			if trace.ok() {
				trace.GoUnpark(gp, 0)
				traceRelease(trace)
			}
			return gp, false, false
		}
		gcController.removeIdleMarkWorker()
	}

	// wasm only:
	// If a callback returned and no other goroutine is awake,
	// then wake event handler goroutine which pauses execution
	// until a callback was triggered.
	gp, otherReady := beforeIdle(now, pollUntil)
	if gp != nil {
		trace := traceAcquire()
		casgstatus(gp, _Gwaiting, _Grunnable)
		if trace.ok() {
			trace.GoUnpark(gp, 0)
			traceRelease(trace)
		}
		return gp, false, false
	}
	if otherReady {
		goto top
	}

	// Before we drop our P, make a snapshot of the allp slice,
	// which can change underfoot once we no longer block
	// safe-points. We don't need to snapshot the contents because
	// everything up to cap(allp) is immutable.
	allpSnapshot := allp
	// Also snapshot masks. Value changes are OK, but we can't allow
	// len to change out from under us.
	idlepMaskSnapshot := idlepMask
	timerpMaskSnapshot := timerpMask

	// return P and block
	lock(&sched.lock)
	if sched.gcwaiting.Load() || pp.runSafePointFn != 0 {
		unlock(&sched.lock)
		goto top
	}
	if sched.runqsize != 0 {
		gp := globrunqget(pp, 0)
		unlock(&sched.lock)
		return gp, false, false
	}
	if !mp.spinning && sched.needspinning.Load() == 1 {
		// See "Delicate dance" comment below.
		mp.becomeSpinning()
		unlock(&sched.lock)
		goto top
	}
	if releasep() != pp {
		throw("findrunnable: wrong p")
	}
	now = pidleput(pp, now)
	unlock(&sched.lock)

	// Delicate dance: thread transitions from spinning to non-spinning
	// state, potentially concurrently with submission of new work. We must
	// drop nmspinning first and then check all sources again (with
	// #StoreLoad memory barrier in between). If we do it the other way
	// around, another thread can submit work after we've checked all
	// sources but before we drop nmspinning; as a result nobody will
	// unpark a thread to run the work.
	//
	// This applies to the following sources of work:
	//
	// * Goroutines added to the global or a per-P run queue.
	// * New/modified-earlier timers on a per-P timer heap.
	// * Idle-priority GC work (barring golang.org/issue/19112).
	//
	// If we discover new work below, we need to restore m.spinning as a
	// signal for resetspinning to unpark a new worker thread (because
	// there can be more than one starving goroutine).
	//
	// However, if after discovering new work we also observe no idle Ps
	// (either here or in resetspinning), we have a problem. We may be
	// racing with a non-spinning M in the block above, having found no
	// work and preparing to release its P and park. Allowing that P to go
	// idle will result in loss of work conservation (idle P while there is
	// runnable work). This could result in complete deadlock in the
	// unlikely event that we discover new work (from netpoll) right as we
	// are racing with _all_ other Ps going idle.
	//
	// We use sched.needspinning to synchronize with non-spinning Ms going
	// idle. If needspinning is set when they are about to drop their P,
	// they abort the drop and instead become a new spinning M on our
	// behalf. If we are not racing and the system is truly fully loaded
	// then no spinning threads are required, and the next thread to
	// naturally become spinning will clear the flag.
	//
	// Also see "Worker thread parking/unparking" comment at the top of the
	// file.
	wasSpinning := mp.spinning
	if mp.spinning {
		mp.spinning = false
		if sched.nmspinning.Add(-1) < 0 {
			throw("findrunnable: negative nmspinning")
		}

		// Note the for correctness, only the last M transitioning from
		// spinning to non-spinning must perform these rechecks to
		// ensure no missed work. However, the runtime has some cases
		// of transient increments of nmspinning that are decremented
		// without going through this path, so we must be conservative
		// and perform the check on all spinning Ms.
		//
		// See https://go.dev/issue/43997.

		// Check global and P runqueues again.

		lock(&sched.lock)
		if sched.runqsize != 0 {
			pp, _ := pidlegetSpinning(0)
			if pp != nil {
				gp := globrunqget(pp, 0)
				if gp == nil {
					throw("global runq empty with non-zero runqsize")
				}
				unlock(&sched.lock)
				acquirep(pp)
				mp.becomeSpinning()
				return gp, false, false
			}
		}
		unlock(&sched.lock)

		pp := checkRunqsNoP(allpSnapshot, idlepMaskSnapshot)
		if pp != nil {
			acquirep(pp)
			mp.becomeSpinning()
			goto top
		}

		// Check for idle-priority GC work again.
		pp, gp := checkIdleGCNoP()
		if pp != nil {
			acquirep(pp)
			mp.becomeSpinning()

			// Run the idle worker.
			pp.gcMarkWorkerMode = gcMarkWorkerIdleMode
			trace := traceAcquire()
			casgstatus(gp, _Gwaiting, _Grunnable)
			if trace.ok() {
				trace.GoUnpark(gp, 0)
				traceRelease(trace)
			}
			return gp, false, false
		}

		// Finally, check for timer creation or expiry concurrently with
		// transitioning from spinning to non-spinning.
		//
		// Note that we cannot use checkTimers here because it calls
		// adjusttimers which may need to allocate memory, and that isn't
		// allowed when we don't have an active P.
		pollUntil = checkTimersNoP(allpSnapshot, timerpMaskSnapshot, pollUntil)
	}

	// Poll network until next timer.
	if netpollinited() && (netpollAnyWaiters() || pollUntil != 0) && sched.lastpoll.Swap(0) != 0 {
		sched.pollUntil.Store(pollUntil)
		if mp.p != 0 {
			throw("findrunnable: netpoll with p")
		}
		if mp.spinning {
			throw("findrunnable: netpoll with spinning")
		}
		delay := int64(-1)
		if pollUntil != 0 {
			if now == 0 {
				now = nanotime()
			}
			delay = pollUntil - now
			if delay < 0 {
				delay = 0
			}
		}
		if faketime != 0 {
			// When using fake time, just poll.
			delay = 0
		}
		list, delta := netpoll(delay) // block until new work is available
		// Refresh now again, after potentially blocking.
		now = nanotime()
		sched.pollUntil.Store(0)
		sched.lastpoll.Store(now)
		if faketime != 0 && list.empty() {
			// Using fake time and nothing is ready; stop M.
			// When all M's stop, checkdead will call timejump.
			stopm()
			goto top
		}
		lock(&sched.lock)
		pp, _ := pidleget(now)
		unlock(&sched.lock)
		if pp == nil {
			injectglist(&list)
			netpollAdjustWaiters(delta)
		} else {
			acquirep(pp)
			if !list.empty() {
				gp := list.pop()
				injectglist(&list)
				netpollAdjustWaiters(delta)
				trace := traceAcquire()
				casgstatus(gp, _Gwaiting, _Grunnable)
				if trace.ok() {
					trace.GoUnpark(gp, 0)
					traceRelease(trace)
				}
				return gp, false, false
			}
			if wasSpinning {
				mp.becomeSpinning()
			}
			goto top
		}
	} else if pollUntil != 0 && netpollinited() {
		pollerPollUntil := sched.pollUntil.Load()
		if pollerPollUntil == 0 || pollerPollUntil > pollUntil {
			netpollBreak()
		}
	}
	stopm()
	goto top
}

// pollWork reports whether there is non-background work this P could
// be doing. This is a fairly lightweight check to be used for
// background work loops, like idle GC. It checks a subset of the
// conditions checked by the actual scheduler.
func pollWork() bool {
	if sched.runqsize != 0 {
		return true
	}
	p := getg().m.p.ptr()
	if !runqempty(p) {
		return true
	}
	if netpollinited() && netpollAnyWaiters() && sched.lastpoll.Load() != 0 {
		if list, delta := netpoll(0); !list.empty() {
			injectglist(&list)
			netpollAdjustWaiters(delta)
			return true
		}
	}
	return false
}

// stealWork attempts to steal a runnable goroutine or timer from any P.
//
// If newWork is true, new work may have been readied.
//
// If now is not 0 it is the current time. stealWork returns the passed time or
// the current time if now was passed as 0.
func stealWork(now int64) (gp *g, inheritTime bool, rnow, pollUntil int64, newWork bool) {
	pp := getg().m.p.ptr()

	ranTimer := false

	const stealTries = 4
	for i := 0; i < stealTries; i++ {
		stealTimersOrRunNextG := i == stealTries-1

		for enum := stealOrder.start(cheaprand()); !enum.done(); enum.next() {
			if sched.gcwaiting.Load() {
				// GC work may be available.
				return nil, false, now, pollUntil, true
			}
			p2 := allp[enum.position()]
			if pp == p2 {
				continue
			}

			// Steal timers from p2. This call to checkTimers is the only place
			// where we might hold a lock on a different P's timers. We do this
			// once on the last pass before checking runnext because stealing
			// from the other P's runnext should be the last resort, so if there
			// are timers to steal do that first.
			//
			// We only check timers on one of the stealing iterations because
			// the time stored in now doesn't change in this loop and checking
			// the timers for each P more than once with the same value of now
			// is probably a waste of time.
			//
			// timerpMask tells us whether the P may have timers at all. If it
			// can't, no need to check at all.
			if stealTimersOrRunNextG && timerpMask.read(enum.position()) {
				tnow, w, ran := checkTimers(p2, now)
				now = tnow
				if w != 0 && (pollUntil == 0 || w < pollUntil) {
					pollUntil = w
				}
				if ran {
					// Running the timers may have
					// made an arbitrary number of G's
					// ready and added them to this P's
					// local run queue. That invalidates
					// the assumption of runqsteal
					// that it always has room to add
					// stolen G's. So check now if there
					// is a local G to run.
					if gp, inheritTime := runqget(pp); gp != nil {
						return gp, inheritTime, now, pollUntil, ranTimer
					}
					ranTimer = true
				}
			}

			// Don't bother to attempt to steal if p2 is idle.
			if !idlepMask.read(enum.position()) {
				if gp := runqsteal(pp, p2, stealTimersOrRunNextG); gp != nil {
					return gp, false, now, pollUntil, ranTimer
				}
			}
		}
	}

	// No goroutines found to steal. Regardless, running a timer may have
	// made some goroutine ready that we missed. Indicate the next timer to
	// wait for.
	return nil, false, now, pollUntil, ranTimer
}

// Check all Ps for a runnable G to steal.
//
// On entry we have no P. If a G is available to steal and a P is available,
// the P is returned which the caller should acquire and attempt to steal the
// work to.
func checkRunqsNoP(allpSnapshot []*p, idlepMaskSnapshot pMask) *p {
	for id, p2 := range allpSnapshot {
		if !idlepMaskSnapshot.read(uint32(id)) && !runqempty(p2) {
			lock(&sched.lock)
			pp, _ := pidlegetSpinning(0)
			if pp == nil {
				// Can't get a P, don't bother checking remaining Ps.
				unlock(&sched.lock)
				return nil
			}
			unlock(&sched.lock)
			return pp
		}
	}

	// No work available.
	return nil
}

// Check all Ps for a timer expiring sooner than pollUntil.
//
// Returns updated pollUntil value.
func checkTimersNoP(allpSnapshot []*p, timerpMaskSnapshot pMask, pollUntil int64) int64 {
	for id, p2 := range allpSnapshot {
		if timerpMaskSnapshot.read(uint32(id)) {
			w := nobarrierWakeTime(p2)
			if w != 0 && (pollUntil == 0 || w < pollUntil) {
				pollUntil = w
			}
		}
	}

	return pollUntil
}

// Check for idle-priority GC, without a P on entry.
//
// If some GC work, a P, and a worker G are all available, the P and G will be
// returned. The returned P has not been wired yet.
func checkIdleGCNoP() (*p, *g) {
	// N.B. Since we have no P, gcBlackenEnabled may change at any time; we
	// must check again after acquiring a P. As an optimization, we also check
	// if an idle mark worker is needed at all. This is OK here, because if we
	// observe that one isn't needed, at least one is currently running. Even if
	// it stops running, its own journey into the scheduler should schedule it
	// again, if need be (at which point, this check will pass, if relevant).
	if atomic.Load(&gcBlackenEnabled) == 0 || !gcController.needIdleMarkWorker() {
		return nil, nil
	}
	if !gcMarkWorkAvailable(nil) {
		return nil, nil
	}

	// Work is available; we can start an idle GC worker only if there is
	// an available P and available worker G.
	//
	// We can attempt to acquire these in either order, though both have
	// synchronization concerns (see below). Workers are almost always
	// available (see comment in findRunnableGCWorker for the one case
	// there may be none). Since we're slightly less likely to find a P,
	// check for that first.
	//
	// Synchronization: note that we must hold sched.lock until we are
	// committed to keeping it. Otherwise we cannot put the unnecessary P
	// back in sched.pidle without performing the full set of idle
	// transition checks.
	//
	// If we were to check gcBgMarkWorkerPool first, we must somehow handle
	// the assumption in gcControllerState.findRunnableGCWorker that an
	// empty gcBgMarkWorkerPool is only possible if gcMarkDone is running.
	lock(&sched.lock)
	pp, now := pidlegetSpinning(0)
	if pp == nil {
		unlock(&sched.lock)
		return nil, nil
	}

	// Now that we own a P, gcBlackenEnabled can't change (as it requires STW).
	if gcBlackenEnabled == 0 || !gcController.addIdleMarkWorker() {
		pidleput(pp, now)
		unlock(&sched.lock)
		return nil, nil
	}

	node := (*gcBgMarkWorkerNode)(gcBgMarkWorkerPool.pop())
	if node == nil {
		pidleput(pp, now)
		unlock(&sched.lock)
		gcController.removeIdleMarkWorker()
		return nil, nil
	}

	unlock(&sched.lock)

	return pp, node.gp.ptr()
}

// wakeNetPoller wakes up the thread sleeping in the network poller if it isn't
// going to wake up before the when argument; or it wakes an idle P to service
// timers and the network poller if there isn't one already.
func wakeNetPoller(when int64) {
	if sched.lastpoll.Load() == 0 {
		// In findrunnable we ensure that when polling the pollUntil
		// field is either zero or the time to which the current
		// poll is expected to run. This can have a spurious wakeup
		// but should never miss a wakeup.
		pollerPollUntil := sched.pollUntil.Load()
		if pollerPollUntil == 0 || pollerPollUntil > when {
			netpollBreak()
		}
	} else {
		// There are no threads in the network poller, try to get
		// one there so it can handle new timers.
		if GOOS != "plan9" { // Temporary workaround - see issue #42303.
			wakep()
		}
	}
}

func resetspinning() {
	gp := getg()
	if !gp.m.spinning {
		throw("resetspinning: not a spinning m")
	}
	gp.m.spinning = false
	nmspinning := sched.nmspinning.Add(-1)
	if nmspinning < 0 {
		throw("findrunnable: negative nmspinning")
	}
	// M wakeup policy is deliberately somewhat conservative, so check if we
	// need to wakeup another P here. See "Worker thread parking/unparking"
	// comment at the top of the file for details.
	wakep()
}

// injectglist adds each runnable G on the list to some run queue,
// and clears glist. If there is no current P, they are added to the
// global queue, and up to npidle M's are started to run them.
// Otherwise, for each idle P, this adds a G to the global queue
// and starts an M. Any remaining G's are added to the current P's
// local run queue.
// This may temporarily acquire sched.lock.
// Can run concurrently with GC.
func injectglist(glist *gList) {
	if glist.empty() {
		return
	}
	trace := traceAcquire()
	if trace.ok() {
		for gp := glist.head.ptr(); gp != nil; gp = gp.schedlink.ptr() {
			trace.GoUnpark(gp, 0)
		}
		traceRelease(trace)
	}

	// Mark all the goroutines as runnable before we put them
	// on the run queues.
	head := glist.head.ptr()
	var tail *g
	qsize := 0
	for gp := head; gp != nil; gp = gp.schedlink.ptr() {
		tail = gp
		qsize++
		casgstatus(gp, _Gwaiting, _Grunnable)
	}

	// Turn the gList into a gQueue.
	var q gQueue
	q.head.set(head)
	q.tail.set(tail)
	*glist = gList{}

	startIdle := func(n int) {
		for i := 0; i < n; i++ {
			mp := acquirem() // See comment in startm.
			lock(&sched.lock)

			pp, _ := pidlegetSpinning(0)
			if pp == nil {
				unlock(&sched.lock)
				releasem(mp)
				break
			}

			startm(pp, false, true)
			unlock(&sched.lock)
			releasem(mp)
		}
	}

	pp := getg().m.p.ptr()
	if pp == nil {
		lock(&sched.lock)
		globrunqputbatch(&q, int32(qsize))
		unlock(&sched.lock)
		startIdle(qsize)
		return
	}

	npidle := int(sched.npidle.Load())
	var globq gQueue
	var n int
	for n = 0; n < npidle && !q.empty(); n++ {
		g := q.pop()
		globq.pushBack(g)
	}
	if n > 0 {
		lock(&sched.lock)
		globrunqputbatch(&globq, int32(n))
		unlock(&sched.lock)
		startIdle(n)
		qsize -= n
	}

	if !q.empty() {
		runqputbatch(pp, &q, qsize)
	}
}

// One round of scheduler: find a runnable goroutine and execute it.
// Never returns.
func schedule() {
	mp := getg().m

	if mp.locks != 0 {
		throw("schedule: holding locks")
	}

	if mp.lockedg != 0 {
		stoplockedm()
		execute(mp.lockedg.ptr(), false) // Never returns.
	}

	// We should not schedule away from a g that is executing a cgo call,
	// since the cgo call is using the m's g0 stack.
	if mp.incgo {
		throw("schedule: in cgo")
	}

top:
	pp := mp.p.ptr()
	pp.preempt = false

	// Safety check: if we are spinning, the run queue should be empty.
	// Check this before calling checkTimers, as that might call
	// goready to put a ready goroutine on the local run queue.
	if mp.spinning && (pp.runnext != 0 || pp.runqhead != pp.runqtail) {
		throw("schedule: spinning with local work")
	}

	gp, inheritTime, tryWakeP := findRunnable() // blocks until work is available

	if debug.dontfreezetheworld > 0 && freezing.Load() {
		// See comment in freezetheworld. We don't want to perturb
		// scheduler state, so we didn't gcstopm in findRunnable, but
		// also don't want to allow new goroutines to run.
		//
		// Deadlock here rather than in the findRunnable loop so if
		// findRunnable is stuck in a loop we don't perturb that
		// either.
		lock(&deadlock)
		lock(&deadlock)
	}

	// This thread is going to run a goroutine and is not spinning anymore,
	// so if it was marked as spinning we need to reset it now and potentially
	// start a new spinning M.
	if mp.spinning {
		resetspinning()
	}

	if sched.disable.user && !schedEnabled(gp) {
		// Scheduling of this goroutine is disabled. Put it on
		// the list of pending runnable goroutines for when we
		// re-enable user scheduling and look again.
		lock(&sched.lock)
		if schedEnabled(gp) {
			// Something re-enabled scheduling while we
			// were acquiring the lock.
			unlock(&sched.lock)
		} else {
			sched.disable.runnable.pushBack(gp)
			sched.disable.n++
			unlock(&sched.lock)
			goto top
		}
	}

	// If about to schedule a not-normal goroutine (a GCworker or tracereader),
	// wake a P if there is one.
	if tryWakeP {
		wakep()
	}
	if gp.lockedm != 0 {
		// Hands off own p to the locked m,
		// then blocks waiting for a new p.
		startlockedm(gp)
		goto top
	}

	execute(gp, inheritTime)
}

// dropg removes the association between m and the current goroutine m->curg (gp for short).
// Typically a caller sets gp's status away from Grunning and then
// immediately calls dropg to finish the job. The caller is also responsible
// for arranging that gp will be restarted using ready at an
// appropriate time. After calling dropg and arranging for gp to be
// readied later, the caller can do other work but eventually should
// call schedule to restart the scheduling of goroutines on this m.
func dropg() {
	gp := getg()

	setMNoWB(&gp.m.curg.m, nil)
	setGNoWB(&gp.m.curg, nil)
}

// checkTimers runs any timers for the P that are ready.
// If now is not 0 it is the current time.
// It returns the passed time or the current time if now was passed as 0.
// and the time when the next timer should run or 0 if there is no next timer,
// and reports whether it ran any timers.
// If the time when the next timer should run is not 0,
// it is always larger than the returned time.
// We pass now in and out to avoid extra calls of nanotime.
//
//go:yeswritebarrierrec
func checkTimers(pp *p, now int64) (rnow, pollUntil int64, ran bool) {
	// If it's not yet time for the first timer, or the first adjusted
	// timer, then there is nothing to do.
	next := pp.timer0When.Load()
	nextAdj := pp.timerModifiedEarliest.Load()
	if next == 0 || (nextAdj != 0 && nextAdj < next) {
		next = nextAdj
	}

	if next == 0 {
		// No timers to run or adjust.
		return now, 0, false
	}

	if now == 0 {
		now = nanotime()
	}
	if now < next {
		// Next timer is not ready to run, but keep going
		// if we would clear deleted timers.
		// This corresponds to the condition below where
		// we decide whether to call clearDeletedTimers.
		if pp != getg().m.p.ptr() || int(pp.deletedTimers.Load()) <= int(pp.numTimers.Load()/4) {
			return now, next, false
		}
	}

	lock(&pp.timersLock)

	if len(pp.timers) > 0 {
		adjusttimers(pp, now)
		for len(pp.timers) > 0 {
			// Note that runtimer may temporarily unlock
			// pp.timersLock.
			if tw := runtimer(pp, now); tw != 0 {
				if tw > 0 {
					pollUntil = tw
				}
				break
			}
			ran = true
		}
	}

	// If this is the local P, and there are a lot of deleted timers,
	// clear them out. We only do this for the local P to reduce
	// lock contention on timersLock.
	if pp == getg().m.p.ptr() && int(pp.deletedTimers.Load()) > len(pp.timers)/4 {
		clearDeletedTimers(pp)
	}

	unlock(&pp.timersLock)

	return now, pollUntil, ran
}

func parkunlock_c(gp *g, lock unsafe.Pointer) bool {
	unlock((*mutex)(lock))
	return true
}

// park continuation on g0.
func park_m(gp *g) {
	mp := getg().m

	trace := traceAcquire()

	// N.B. Not using casGToWaiting here because the waitreason is
	// set by park_m's caller.
	casgstatus(gp, _Grunning, _Gwaiting)
	if trace.ok() {
		trace.GoPark(mp.waitTraceBlockReason, mp.waitTraceSkip)
		traceRelease(trace)
	}

	dropg()

	if fn := mp.waitunlockf; fn != nil {
		ok := fn(gp, mp.waitlock)
		mp.waitunlockf = nil
		mp.waitlock = nil
		if !ok {
			trace := traceAcquire()
			casgstatus(gp, _Gwaiting, _Grunnable)
			if trace.ok() {
				trace.GoUnpark(gp, 2)
				traceRelease(trace)
			}
			execute(gp, true) // Schedule it back, never returns.
		}
	}
	schedule()
}

func goschedImpl(gp *g, preempted bool) {
	trace := traceAcquire()
	status := readgstatus(gp)
	if status&^_Gscan != _Grunning {
		dumpgstatus(gp)
		throw("bad g status")
	}
	casgstatus(gp, _Grunning, _Grunnable)
	if trace.ok() {
		if preempted {
			trace.GoPreempt()
		} else {
			trace.GoSched()
		}
		traceRelease(trace)
	}

	dropg()
	lock(&sched.lock)
	globrunqput(gp)
	unlock(&sched.lock)

	if mainStarted {
		wakep()
	}

	schedule()
}

// Gosched continuation on g0.
func gosched_m(gp *g) {
	goschedImpl(gp, false)
}

// goschedguarded is a forbidden-states-avoided version of gosched_m.
func goschedguarded_m(gp *g) {
	if !canPreemptM(gp.m) {
		gogo(&gp.sched) // never return
	}
	goschedImpl(gp, false)
}

func gopreempt_m(gp *g) {
	goschedImpl(gp, true)
}

// preemptPark parks gp and puts it in _Gpreempted.
//
//go:systemstack
func preemptPark(gp *g) {
	status := readgstatus(gp)
	if status&^_Gscan != _Grunning {
		dumpgstatus(gp)
		throw("bad g status")
	}

	if gp.asyncSafePoint {
		// Double-check that async preemption does not
		// happen in SPWRITE assembly functions.
		// isAsyncSafePoint must exclude this case.
		f := findfunc(gp.sched.pc)
		if !f.valid() {
			throw("preempt at unknown pc")
		}
		if f.flag&abi.FuncFlagSPWrite != 0 {
			println("runtime: unexpected SPWRITE function", funcname(f), "in async preempt")
			throw("preempt SPWRITE")
		}
	}

	// Transition from _Grunning to _Gscan|_Gpreempted. We can't
	// be in _Grunning when we dropg because then we'd be running
	// without an M, but the moment we're in _Gpreempted,
	// something could claim this G before we've fully cleaned it
	// up. Hence, we set the scan bit to lock down further
	// transitions until we can dropg.
	casGToPreemptScan(gp, _Grunning, _Gscan|_Gpreempted)
	dropg()

	// Be careful about how we trace this next event. The ordering
	// is subtle.
	//
	// The moment we CAS into _Gpreempted, suspendG could CAS to
	// _Gwaiting, do its work, and ready the goroutine. All of
	// this could happen before we even get the chance to emit
	// an event. The end result is that the events could appear
	// out of order, and the tracer generally assumes the scheduler
	// takes care of the ordering between GoPark and GoUnpark.
	//
	// The answer here is simple: emit the event while we still hold
	// the _Gscan bit on the goroutine. We still need to traceAcquire
	// and traceRelease across the CAS because the tracer could be
	// what's calling suspendG in the first place, and we want the
	// CAS and event emission to appear atomic to the tracer.
	trace := traceAcquire()
	if trace.ok() {
		trace.GoPark(traceBlockPreempted, 0)
	}
	casfrom_Gscanstatus(gp, _Gscan|_Gpreempted, _Gpreempted)
	if trace.ok() {
		traceRelease(trace)
	}
	schedule()
}

// goyield is like Gosched, but it:
// - emits a GoPreempt trace event instead of a GoSched trace event
// - puts the current G on the runq of the current P instead of the globrunq
func goyield() {
	checkTimeouts()
	mcall(goyield_m)
}

func goyield_m(gp *g) {
	trace := traceAcquire()
	pp := gp.m.p.ptr()
	casgstatus(gp, _Grunning, _Grunnable)
	if trace.ok() {
		trace.GoPreempt()
		traceRelease(trace)
	}
	dropg()
	runqput(pp, gp, false)
	schedule()
}

// Finishes execution of the current goroutine.
func goexit1() {
	if raceenabled {
		racegoend()
	}
	trace := traceAcquire()
	if trace.ok() {
		trace.GoEnd()
		traceRelease(trace)
	}
	mcall(goexit0)
}

// goexit continuation on g0.
func goexit0(gp *g) {
	gdestroy(gp)
	schedule()
}

func gdestroy(gp *g) {
	mp := getg().m
	pp := mp.p.ptr()

	casgstatus(gp, _Grunning, _Gdead)
	gcController.addScannableStack(pp, -int64(gp.stack.hi-gp.stack.lo))
	if isSystemGoroutine(gp, false) {
		sched.ngsys.Add(-1)
	}
	gp.m = nil
	locked := gp.lockedm != 0
	gp.lockedm = 0
	mp.lockedg = 0
	gp.preemptStop = false
	gp.paniconfault = false
	gp._defer = nil // should be true already but just in case.
	gp._panic = nil // non-nil for Goexit during panic. points at stack-allocated data.
	gp.writebuf = nil
	gp.waitreason = waitReasonZero
	gp.param = nil
	gp.labels = nil
	gp.timer = nil

	if gcBlackenEnabled != 0 && gp.gcAssistBytes > 0 {
		// Flush assist credit to the global pool. This gives
		// better information to pacing if the application is
		// rapidly creating an exiting goroutines.
		assistWorkPerByte := gcController.assistWorkPerByte.Load()
		scanCredit := int64(assistWorkPerByte * float64(gp.gcAssistBytes))
		gcController.bgScanCredit.Add(scanCredit)
		gp.gcAssistBytes = 0
	}

	dropg()

	if GOARCH == "wasm" { // no threads yet on wasm
		gfput(pp, gp)
		return
	}

	if mp.lockedInt != 0 {
		print("invalid m->lockedInt = ", mp.lockedInt, "\n")
		throw("internal lockOSThread error")
	}
	gfput(pp, gp)
	if locked {
		// The goroutine may have locked this thread because
		// it put it in an unusual kernel state. Kill it
		// rather than returning it to the thread pool.

		// Return to mstart, which will release the P and exit
		// the thread.
		if GOOS != "plan9" { // See golang.org/issue/22227.
			gogo(&mp.g0.sched)
		} else {
			// Clear lockedExt on plan9 since we may end up re-using
			// this thread.
			mp.lockedExt = 0
		}
	}
}

// save updates getg().sched to refer to pc and sp so that a following
// gogo will restore pc and sp.
//
// save must not have write barriers because invoking a write barrier
// can clobber getg().sched.
//
//go:nosplit
//go:nowritebarrierrec
func save(pc, sp uintptr) {
	gp := getg()

	if gp == gp.m.g0 || gp == gp.m.gsignal {
		// m.g0.sched is special and must describe the context
		// for exiting the thread. mstart1 writes to it directly.
		// m.gsignal.sched should not be used at all.
		// This check makes sure save calls do not accidentally
		// run in contexts where they'd write to system g's.
		throw("save on system g not allowed")
	}

	gp.sched.pc = pc
	gp.sched.sp = sp
	gp.sched.lr = 0
	gp.sched.ret = 0
	// We need to ensure ctxt is zero, but can't have a write
	// barrier here. However, it should always already be zero.
	// Assert that.
	if gp.sched.ctxt != nil {
		badctxt()
	}
}

// The goroutine g is about to enter a system call.
// Record that it's not using the cpu anymore.
// This is called only from the go syscall library and cgocall,
// not from the low-level system calls used by the runtime.
//
// Entersyscall cannot split the stack: the save must
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
// Syscall tracing (old tracer):
// At the start of a syscall we emit traceGoSysCall to capture the stack trace.
// If the syscall does not block, that is it, we do not emit any other events.
// If the syscall blocks (that is, P is retaken), retaker emits traceGoSysBlock;
// when syscall returns we emit traceGoSysExit and when the goroutine starts running
// (potentially instantly, if exitsyscallfast returns true) we emit traceGoStart.
// To ensure that traceGoSysExit is emitted strictly after traceGoSysBlock,
// we remember current value of syscalltick in m (gp.m.syscalltick = gp.m.p.ptr().syscalltick),
// whoever emits traceGoSysBlock increments p.syscalltick afterwards;
// and we wait for the increment before emitting traceGoSysExit.
// Note that the increment is done even if tracing is not enabled,
// because tracing can be enabled in the middle of syscall. We don't want the wait to hang.
//
//go:nosplit
func reentersyscall(pc, sp uintptr) {
	trace := traceAcquire()
	gp := getg()

	// Disable preemption because during this function g is in Gsyscall status,
	// but can have inconsistent g->sched, do not let GC observe it.
	gp.m.locks++

	// Entersyscall must not call any function that might split/grow the stack.
	// (See details in comment above.)
	// Catch calls that might, by replacing the stack guard with something that
	// will trip any stack check and leaving a flag to tell newstack to die.
	gp.stackguard0 = stackPreempt
	gp.throwsplit = true

	// Leave SP around for GC and traceback.
	save(pc, sp)
	gp.syscallsp = sp
	gp.syscallpc = pc
	casgstatus(gp, _Grunning, _Gsyscall)
	if staticLockRanking {
		// When doing static lock ranking casgstatus can call
		// systemstack which clobbers g.sched.
		save(pc, sp)
	}
	if gp.syscallsp < gp.stack.lo || gp.stack.hi < gp.syscallsp {
		systemstack(func() {
			print("entersyscall inconsistent ", hex(gp.syscallsp), " [", hex(gp.stack.lo), ",", hex(gp.stack.hi), "]\n")
			throw("entersyscall")
		})
	}

	if trace.ok() {
		systemstack(func() {
			trace.GoSysCall()
			traceRelease(trace)
		})
		// systemstack itself clobbers g.sched.{pc,sp} and we might
		// need them later when the G is genuinely blocked in a
		// syscall
		save(pc, sp)
	}

	if sched.sysmonwait.Load() {
		systemstack(entersyscall_sysmon)
		save(pc, sp)
	}

	if gp.m.p.ptr().runSafePointFn != 0 {
		// runSafePointFn may stack split if run on this stack
		systemstack(runSafePointFn)
		save(pc, sp)
	}

	gp.m.syscalltick = gp.m.p.ptr().syscalltick
	pp := gp.m.p.ptr()
	pp.m = 0
	gp.m.oldp.set(pp)
	gp.m.p = 0
	atomic.Store(&pp.status, _Psyscall)
	if sched.gcwaiting.Load() {
		systemstack(entersyscall_gcwait)
		save(pc, sp)
	}

	gp.m.locks--
}

// Standard syscall entry used by the go syscall library and normal cgo calls.
//
// This is exported via linkname to assembly in the syscall package and x/sys.
//
//go:nosplit
//go:linkname entersyscall
func entersyscall() {
	reentersyscall(getcallerpc(), getcallersp())
}

func entersyscall_sysmon() {
	lock(&sched.lock)
	if sched.sysmonwait.Load() {
		sched.sysmonwait.Store(false)
		notewakeup(&sched.sysmonnote)
	}
	unlock(&sched.lock)
}

func entersyscall_gcwait() {
	gp := getg()
	pp := gp.m.oldp.ptr()

	lock(&sched.lock)
	trace := traceAcquire()
	if sched.stopwait > 0 && atomic.Cas(&pp.status, _Psyscall, _Pgcstop) {
		if trace.ok() {
			if goexperiment.ExecTracer2 {
				// This is a steal in the new tracer. While it's very likely
				// that we were the ones to put this P into _Psyscall, between
				// then and now it's totally possible it had been stolen and
				// then put back into _Psyscall for us to acquire here. In such
				// case ProcStop would be incorrect.
				//
				// TODO(mknyszek): Consider emitting a ProcStop instead when
				// gp.m.syscalltick == pp.syscalltick, since then we know we never
				// lost the P.
				trace.ProcSteal(pp, true)
			} else {
				trace.GoSysBlock(pp)
				trace.ProcStop(pp)
			}
			traceRelease(trace)
		}
		pp.syscalltick++
		if sched.stopwait--; sched.stopwait == 0 {
			notewakeup(&sched.stopnote)
		}
	} else if trace.ok() {
		traceRelease(trace)
	}
	unlock(&sched.lock)
}

// The same as entersyscall(), but with a hint that the syscall is blocking.
//
//go:nosplit
func entersyscallblock() {
	gp := getg()

	gp.m.locks++ // see comment in entersyscall
	gp.throwsplit = true
	gp.stackguard0 = stackPreempt // see comment in entersyscall
	gp.m.syscalltick = gp.m.p.ptr().syscalltick
	gp.m.p.ptr().syscalltick++

	// Leave SP around for GC and traceback.
	pc := getcallerpc()
	sp := getcallersp()
	save(pc, sp)
	gp.syscallsp = gp.sched.sp
	gp.syscallpc = gp.sched.pc
	if gp.syscallsp < gp.stack.lo || gp.stack.hi < gp.syscallsp {
		sp1 := sp
		sp2 := gp.sched.sp
		sp3 := gp.syscallsp
		systemstack(func() {
			print("entersyscallblock inconsistent ", hex(sp1), " ", hex(sp2), " ", hex(sp3), " [", hex(gp.stack.lo), ",", hex(gp.stack.hi), "]\n")
			throw("entersyscallblock")
		})
	}
	casgstatus(gp, _Grunning, _Gsyscall)
	if gp.syscallsp < gp.stack.lo || gp.stack.hi < gp.syscallsp {
		systemstack(func() {
			print("entersyscallblock inconsistent ", hex(sp), " ", hex(gp.sched.sp), " ", hex(gp.syscallsp), " [", hex(gp.stack.lo), ",", hex(gp.stack.hi), "]\n")
			throw("entersyscallblock")
		})
	}

	systemstack(entersyscallblock_handoff)

	// Resave for traceback during blocked call.
	save(getcallerpc(), getcallersp())

	gp.m.locks--
}

func entersyscallblock_handoff() {
	trace := traceAcquire()
	if trace.ok() {
		trace.GoSysCall()
		trace.GoSysBlock(getg().m.p.ptr())
		traceRelease(trace)
	}
	handoffp(releasep())
}

// The goroutine g exited its system call.
// Arrange for it to run on a cpu again.
// This is called only from the go syscall library, not
// from the low-level system calls used by the runtime.
//
// Write barriers are not allowed because our P may have been stolen.
//
// This is exported via linkname to assembly in the syscall package.
//
//go:nosplit
//go:nowritebarrierrec
//go:linkname exitsyscall
func exitsyscall() {
	gp := getg()

	gp.m.locks++ // see comment in entersyscall
	if getcallersp() > gp.syscallsp {
		throw("exitsyscall: syscall frame is no longer valid")
	}

	gp.waitsince = 0
	oldp := gp.m.oldp.ptr()
	gp.m.oldp = 0
	if exitsyscallfast(oldp) {
		// When exitsyscallfast returns success, we have a P so can now use
		// write barriers
		if goroutineProfile.active {
			// Make sure that gp has had its stack written out to the goroutine
			// profile, exactly as it was when the goroutine profiler first
			// stopped the world.
			systemstack(func() {
				tryRecordGoroutineProfileWB(gp)
			})
		}
		trace := traceAcquire()
		if trace.ok() {
			lostP := oldp != gp.m.p.ptr() || gp.m.syscalltick != gp.m.p.ptr().syscalltick
			systemstack(func() {
				if goexperiment.ExecTracer2 {
					// Write out syscall exit eagerly in the experiment.
					//
					// It's important that we write this *after* we know whether we
					// lost our P or not (determined by exitsyscallfast).
					trace.GoSysExit(lostP)
				}
				if lostP {
					// We lost the P at some point, even though we got it back here.
					// Trace that we're starting again, because there was a traceGoSysBlock
					// call somewhere in exitsyscallfast (indicating that this goroutine
					// had blocked) and we're about to start running again.
					trace.GoStart()
				}
			})
		}
		// There's a cpu for us, so we can run.
		gp.m.p.ptr().syscalltick++
		// We need to cas the status and scan before resuming...
		casgstatus(gp, _Gsyscall, _Grunning)
		if trace.ok() {
			traceRelease(trace)
		}

		// Garbage collector isn't running (since we are),
		// so okay to clear syscallsp.
		gp.syscallsp = 0
		gp.m.locks--
		if gp.preempt {
			// restore the preemption request in case we've cleared it in newstack
			gp.stackguard0 = stackPreempt
		} else {
			// otherwise restore the real stackGuard, we've spoiled it in entersyscall/entersyscallblock
			gp.stackguard0 = gp.stack.lo + stackGuard
		}
		gp.throwsplit = false

		if sched.disable.user && !schedEnabled(gp) {
			// Scheduling of this goroutine is disabled.
			Gosched()
		}

		return
	}

	if !goexperiment.ExecTracer2 {
		// In the old tracer, because we don't have a P we can't
		// actually record the true time we exited the syscall.
		// Record it.
		trace := traceAcquire()
		if trace.ok() {
			trace.RecordSyscallExitedTime(gp, oldp)
			traceRelease(trace)
		}
	}

	gp.m.locks--

	// Call the scheduler.
	mcall(exitsyscall0)

	// Scheduler returned, so we're allowed to run now.
	// Delete the syscallsp information that we left for
	// the garbage collector during the system call.
	// Must wait until now because until gosched returns
	// we don't know for sure that the garbage collector
	// is not running.
	gp.syscallsp = 0
	gp.m.p.ptr().syscalltick++
	gp.throwsplit = false
}

//go:nosplit
func exitsyscallfast(oldp *p) bool {
	gp := getg()

	// Freezetheworld sets stopwait but does not retake P's.
	if sched.stopwait == freezeStopWait {
		return false
	}

	// Try to re-acquire the last P.
	trace := traceAcquire()
	if oldp != nil && oldp.status == _Psyscall && atomic.Cas(&oldp.status, _Psyscall, _Pidle) {
		// There's a cpu for us, so we can run.
		wirep(oldp)
		exitsyscallfast_reacquired(trace)
		if trace.ok() {
			traceRelease(trace)
		}
		return true
	}
	if trace.ok() {
		traceRelease(trace)
	}

	// Try to get any other idle P.
	if sched.pidle != 0 {
		var ok bool
		systemstack(func() {
			ok = exitsyscallfast_pidle()
			if ok && !goexperiment.ExecTracer2 {
				trace := traceAcquire()
				if trace.ok() {
					if oldp != nil {
						// Wait till traceGoSysBlock event is emitted.
						// This ensures consistency of the trace (the goroutine is started after it is blocked).
						for oldp.syscalltick == gp.m.syscalltick {
							osyield()
						}
					}
					// In the experiment, we write this in exitsyscall.
					// Don't write it here unless the experiment is off.
					trace.GoSysExit(true)
					traceRelease(trace)
				}
			}
		})
		if ok {
			return true
		}
	}
	return false
}

// exitsyscallfast_reacquired is the exitsyscall path on which this G
// has successfully reacquired the P it was running on before the
// syscall.
//
//go:nosplit
func exitsyscallfast_reacquired(trace traceLocker) {
	gp := getg()
	if gp.m.syscalltick != gp.m.p.ptr().syscalltick {
		if trace.ok() {
			// The p was retaken and then enter into syscall again (since gp.m.syscalltick has changed).
			// traceGoSysBlock for this syscall was already emitted,
			// but here we effectively retake the p from the new syscall running on the same p.
			systemstack(func() {
				if goexperiment.ExecTracer2 {
					// In the experiment, we're stealing the P. It's treated
					// as if it temporarily stopped running. Then, start running.
					trace.ProcSteal(gp.m.p.ptr(), true)
					trace.ProcStart()
				} else {
					// Denote blocking of the new syscall.
					trace.GoSysBlock(gp.m.p.ptr())
					// Denote completion of the current syscall.
					trace.GoSysExit(true)
				}
			})
		}
		gp.m.p.ptr().syscalltick++
	}
}

func exitsyscallfast_pidle() bool {
	lock(&sched.lock)
	pp, _ := pidleget(0)
	if pp != nil && sched.sysmonwait.Load() {
		sched.sysmonwait.Store(false)
		notewakeup(&sched.sysmonnote)
	}
	unlock(&sched.lock)
	if pp != nil {
		acquirep(pp)
		return true
	}
	return false
}

// exitsyscall slow path on g0.
// Failed to acquire P, enqueue gp as runnable.
//
// Called via mcall, so gp is the calling g from this M.
//
//go:nowritebarrierrec
func exitsyscall0(gp *g) {
	var trace traceLocker
	if goexperiment.ExecTracer2 {
		traceExitingSyscall()
		trace = traceAcquire()
	}
	casgstatus(gp, _Gsyscall, _Grunnable)
	if goexperiment.ExecTracer2 {
		traceExitedSyscall()
		if trace.ok() {
			// Write out syscall exit eagerly in the experiment.
			//
			// It's important that we write this *after* we know whether we
			// lost our P or not (determined by exitsyscallfast).
			trace.GoSysExit(true)
			traceRelease(trace)
		}
	}
	dropg()
	lock(&sched.lock)
	var pp *p
	if schedEnabled(gp) {
		pp, _ = pidleget(0)
	}
	var locked bool
	if pp == nil {
		globrunqput(gp)

		// Below, we stoplockedm if gp is locked. globrunqput releases
		// ownership of gp, so we must check if gp is locked prior to
		// committing the release by unlocking sched.lock, otherwise we
		// could race with another M transitioning gp from unlocked to
		// locked.
		locked = gp.lockedm != 0
	} else if sched.sysmonwait.Load() {
		sched.sysmonwait.Store(false)
		notewakeup(&sched.sysmonnote)
	}
	unlock(&sched.lock)
	if pp != nil {
		acquirep(pp)
		execute(gp, false) // Never returns.
	}
	if locked {
		// Wait until another thread schedules gp and so m again.
		//
		// N.B. lockedm must be this M, as this g was running on this M
		// before entersyscall.
		stoplockedm()
		execute(gp, false) // Never returns.
	}
	stopm()
	schedule() // Never returns.
}

// Called from syscall package before fork.
//
//go:linkname syscall_runtime_BeforeFork syscall.runtime_BeforeFork
//go:nosplit
func syscall_runtime_BeforeFork() {
	gp := getg().m.curg

	// Block signals during a fork, so that the child does not run
	// a signal handler before exec if a signal is sent to the process
	// group. See issue #18600.
	gp.m.locks++
	sigsave(&gp.m.sigmask)
	sigblock(false)

	// This function is called before fork in syscall package.
	// Code between fork and exec must not allocate memory nor even try to grow stack.
	// Here we spoil g.stackguard0 to reliably detect any attempts to grow stack.
	// runtime_AfterFork will undo this in parent process, but not in child.
	gp.stackguard0 = stackFork
}

// Called from syscall package after fork in parent.
//
//go:linkname syscall_runtime_AfterFork syscall.runtime_AfterFork
//go:nosplit
func syscall_runtime_AfterFork() {
	gp := getg().m.curg

	// See the comments in beforefork.
	gp.stackguard0 = gp.stack.lo + stackGuard

	msigrestore(gp.m.sigmask)

	gp.m.locks--
}

// inForkedChild is true while manipulating signals in the child process.
// This is used to avoid calling libc functions in case we are using vfork.
var inForkedChild bool

// Called from syscall package after fork in child.
// It resets non-sigignored signals to the default handler, and
// restores the signal mask in preparation for the exec.
//
// Because this might be called during a vfork, and therefore may be
// temporarily sharing address space with the parent process, this must
// not change any global variables or calling into C code that may do so.
//
//go:linkname syscall_runtime_AfterForkInChild syscall.runtime_AfterForkInChild
//go:nosplit
//go:nowritebarrierrec
func syscall_runtime_AfterForkInChild() {
	// It's OK to change the global variable inForkedChild here
	// because we are going to change it back. There is no race here,
	// because if we are sharing address space with the parent process,
	// then the parent process can not be running concurrently.
	inForkedChild = true

	clearSignalHandlers()

	// When we are the child we are the only thread running,
	// so we know that nothing else has changed gp.m.sigmask.
	msigrestore(getg().m.sigmask)

	inForkedChild = false
}

// pendingPreemptSignals is the number of preemption signals
// that have been sent but not received. This is only used on Darwin.
// For #41702.
var pendingPreemptSignals atomic.Int32

// Called from syscall package before Exec.
//
//go:linkname syscall_runtime_BeforeExec syscall.runtime_BeforeExec
func syscall_runtime_BeforeExec() {
	// Prevent thread creation during exec.
	execLock.lock()

	// On Darwin, wait for all pending preemption signals to
	// be received. See issue #41702.
	if GOOS == "darwin" || GOOS == "ios" {
		for pendingPreemptSignals.Load() > 0 {
			osyield()
		}
	}
}

// Called from syscall package after Exec.
//
//go:linkname syscall_runtime_AfterExec syscall.runtime_AfterExec
func syscall_runtime_AfterExec() {
	execLock.unlock()
}

// Allocate a new g, with a stack big enough for stacksize bytes.
func malg(stacksize int32) *g {
	newg := new(g)
	if stacksize >= 0 {
		stacksize = round2(stackSystem + stacksize)
		systemstack(func() {
			newg.stack = stackalloc(uint32(stacksize))
		})
		newg.stackguard0 = newg.stack.lo + stackGuard
		newg.stackguard1 = ^uintptr(0)
		// Clear the bottom word of the stack. We record g
		// there on gsignal stack during VDSO on ARM and ARM64.
		*(*uintptr)(unsafe.Pointer(newg.stack.lo)) = 0
	}
	return newg
}

// Create a new g running fn.
// Put it on the queue of g's waiting to run.
// The compiler turns a go statement into a call to this.
func newproc(fn *funcval) {
	gp := getg()
	pc := getcallerpc()
	systemstack(func() {
		newg := newproc1(fn, gp, pc)

		pp := getg().m.p.ptr()
		runqput(pp, newg, true)

		if mainStarted {
			wakep()
		}
	})
}

// Create a new g in state _Grunnable, starting at fn. callerpc is the
// address of the go statement that created this. The caller is responsible
// for adding the new g to the scheduler.
func newproc1(fn *funcval, callergp *g, callerpc uintptr) *g {
	if fn == nil {
		fatal("go of nil func value")
	}

	mp := acquirem() // disable preemption because we hold M and P in local vars.
	pp := mp.p.ptr()
	newg := gfget(pp)
	if newg == nil {
		newg = malg(stackMin)
		casgstatus(newg, _Gidle, _Gdead)
		allgadd(newg) // publishes with a g->status of Gdead so GC scanner doesn't look at uninitialized stack.
	}
	if newg.stack.hi == 0 {
		throw("newproc1: newg missing stack")
	}

	if readgstatus(newg) != _Gdead {
		throw("newproc1: new g is not Gdead")
	}

	totalSize := uintptr(4*goarch.PtrSize + sys.MinFrameSize) // extra space in case of reads slightly beyond frame
	totalSize = alignUp(totalSize, sys.StackAlign)
	sp := newg.stack.hi - totalSize
	if usesLR {
		// caller's LR
		*(*uintptr)(unsafe.Pointer(sp)) = 0
		prepGoExitFrame(sp)
	}
	if GOARCH == "arm64" {
		// caller's FP
		*(*uintptr)(unsafe.Pointer(sp - goarch.PtrSize)) = 0
	}

	memclrNoHeapPointers(unsafe.Pointer(&newg.sched), unsafe.Sizeof(newg.sched))
	newg.sched.sp = sp
	newg.stktopsp = sp
	newg.sched.pc = abi.FuncPCABI0(goexit) + sys.PCQuantum // +PCQuantum so that previous instruction is in same function
	newg.sched.g = guintptr(unsafe.Pointer(newg))
	gostartcallfn(&newg.sched, fn)
	newg.parentGoid = callergp.goid
	newg.gopc = callerpc
	newg.ancestors = saveAncestors(callergp)
	newg.startpc = fn.fn
	if isSystemGoroutine(newg, false) {
		sched.ngsys.Add(1)
	} else {
		// Only user goroutines inherit pprof labels.
		if mp.curg != nil {
			newg.labels = mp.curg.labels
		}
		if goroutineProfile.active {
			// A concurrent goroutine profile is running. It should include
			// exactly the set of goroutines that were alive when the goroutine
			// profiler first stopped the world. That does not include newg, so
			// mark it as not needing a profile before transitioning it from
			// _Gdead.
			newg.goroutineProfiled.Store(goroutineProfileSatisfied)
		}
	}
	// Track initial transition?
	newg.trackingSeq = uint8(cheaprand())
	if newg.trackingSeq%gTrackingPeriod == 0 {
		newg.tracking = true
	}
	gcController.addScannableStack(pp, int64(newg.stack.hi-newg.stack.lo))

	// Get a goid and switch to runnable. Make all this atomic to the tracer.
	trace := traceAcquire()
	casgstatus(newg, _Gdead, _Grunnable)
	if pp.goidcache == pp.goidcacheend {
		// Sched.goidgen is the last allocated id,
		// this batch must be [sched.goidgen+1, sched.goidgen+GoidCacheBatch].
		// At startup sched.goidgen=0, so main goroutine receives goid=1.
		pp.goidcache = sched.goidgen.Add(_GoidCacheBatch)
		pp.goidcache -= _GoidCacheBatch - 1
		pp.goidcacheend = pp.goidcache + _GoidCacheBatch
	}
	newg.goid = pp.goidcache
	pp.goidcache++
	newg.trace.reset()
	if trace.ok() {
		trace.GoCreate(newg, newg.startpc)
		traceRelease(trace)
	}

	// Set up race context.
	if raceenabled {
		newg.racectx = racegostart(callerpc)
		newg.raceignore = 0
		if newg.labels != nil {
			// See note in proflabel.go on labelSync's role in synchronizing
			// with the reads in the signal handler.
			racereleasemergeg(newg, unsafe.Pointer(&labelSync))
		}
	}
	releasem(mp)

	return newg
}

// saveAncestors copies previous ancestors of the given caller g and
// includes info for the current caller into a new set of tracebacks for
// a g being created.
func saveAncestors(callergp *g) *[]ancestorInfo {
	// Copy all prior info, except for the root goroutine (goid 0).
	if debug.tracebackancestors <= 0 || callergp.goid == 0 {
		return nil
	}
	var callerAncestors []ancestorInfo
	if callergp.ancestors != nil {
		callerAncestors = *callergp.ancestors
	}
	n := int32(len(callerAncestors)) + 1
	if n > debug.tracebackancestors {
		n = debug.tracebackancestors
	}
	ancestors := make([]ancestorInfo, n)
	copy(ancestors[1:], callerAncestors)

	var pcs [tracebackInnerFrames]uintptr
	npcs := gcallers(callergp, 0, pcs[:])
	ipcs := make([]uintptr, npcs)
	copy(ipcs, pcs[:])
	ancestors[0] = ancestorInfo{
		pcs:  ipcs,
		goid: callergp.goid,
		gopc: callergp.gopc,
	}

	ancestorsp := new([]ancestorInfo)
	*ancestorsp = ancestors
	return ancestorsp
}

// Put on gfree list.
// If local list is too long, transfer a batch to the global list.
func gfput(pp *p, gp *g) {
	if readgstatus(gp) != _Gdead {
		throw("gfput: bad status (not Gdead)")
	}

	stksize := gp.stack.hi - gp.stack.lo

	if stksize != uintptr(startingStackSize) {
		// non-standard stack size - free it.
		stackfree(gp.stack)
		gp.stack.lo = 0
		gp.stack.hi = 0
		gp.stackguard0 = 0
	}

	pp.gFree.push(gp)
	pp.gFree.n++
	if pp.gFree.n >= 64 {
		var (
			inc      int32
			stackQ   gQueue
			noStackQ gQueue
		)
		for pp.gFree.n >= 32 {
			gp := pp.gFree.pop()
			pp.gFree.n--
			if gp.stack.lo == 0 {
				noStackQ.push(gp)
			} else {
				stackQ.push(gp)
			}
			inc++
		}
		lock(&sched.gFree.lock)
		sched.gFree.noStack.pushAll(noStackQ)
		sched.gFree.stack.pushAll(stackQ)
		sched.gFree.n += inc
		unlock(&sched.gFree.lock)
	}
}

// Get from gfree list.
// If local list is empty, grab a batch from global list.
func gfget(pp *p) *g {
retry:
	if pp.gFree.empty() && (!sched.gFree.stack.empty() || !sched.gFree.noStack.empty()) {
		lock(&sched.gFree.lock)
		// Move a batch of free Gs to the P.
		for pp.gFree.n < 32 {
			// Prefer Gs with stacks.
			gp := sched.gFree.stack.pop()
			if gp == nil {
				gp = sched.gFree.noStack.pop()
				if gp == nil {
					break
				}
			}
			sched.gFree.n--
			pp.gFree.push(gp)
			pp.gFree.n++
		}
		unlock(&sched.gFree.lock)
		goto retry
	}
	gp := pp.gFree.pop()
	if gp == nil {
		return nil
	}
	pp.gFree.n--
	if gp.stack.lo != 0 && gp.stack.hi-gp.stack.lo != uintptr(startingStackSize) {
		// Deallocate old stack. We kept it in gfput because it was the
		// right size when the goroutine was put on the free list, but
		// the right size has changed since then.
		systemstack(func() {
			stackfree(gp.stack)
			gp.stack.lo = 0
			gp.stack.hi = 0
			gp.stackguard0 = 0
		})
	}
	if gp.stack.lo == 0 {
		// Stack was deallocated in gfput or just above. Allocate a new one.
		systemstack(func() {
			gp.stack = stackalloc(startingStackSize)
		})
		gp.stackguard0 = gp.stack.lo + stackGuard
	} else {
		if raceenabled {
			racemalloc(unsafe.Pointer(gp.stack.lo), gp.stack.hi-gp.stack.lo)
		}
		if msanenabled {
			msanmalloc(unsafe.Pointer(gp.stack.lo), gp.stack.hi-gp.stack.lo)
		}
		if asanenabled {
			asanunpoison(unsafe.Pointer(gp.stack.lo), gp.stack.hi-gp.stack.lo)
		}
	}
	return gp
}

// Purge all cached G's from gfree list to the global list.
func gfpurge(pp *p) {
	var (
		inc      int32
		stackQ   gQueue
		noStackQ gQueue
	)
	for !pp.gFree.empty() {
		gp := pp.gFree.pop()
		pp.gFree.n--
		if gp.stack.lo == 0 {
			noStackQ.push(gp)
		} else {
			stackQ.push(gp)
		}
		inc++
	}
	lock(&sched.gFree.lock)
	sched.gFree.noStack.pushAll(noStackQ)
	sched.gFree.stack.pushAll(stackQ)
	sched.gFree.n += inc
	unlock(&sched.gFree.lock)
}

// Breakpoint executes a breakpoint trap.
func Breakpoint() {
	breakpoint()
}

// dolockOSThread is called by LockOSThread and lockOSThread below
// after they modify m.locked. Do not allow preemption during this call,
// or else the m might be different in this function than in the caller.
//
//go:nosplit
func dolockOSThread() {
	if GOARCH == "wasm" {
		return // no threads on wasm yet
	}
	gp := getg()
	gp.m.lockedg.set(gp)
	gp.lockedm.set(gp.m)
}

// LockOSThread wires the calling goroutine to its current operating system thread.
// The calling goroutine will always execute in that thread,
// and no other goroutine will execute in it,
// until the calling goroutine has made as many calls to
// [UnlockOSThread] as to LockOSThread.
// If the calling goroutine exits without unlocking the thread,
// the thread will be terminated.
//
// All init functions are run on the startup thread. Calling LockOSThread
// from an init function will cause the main function to be invoked on
// that thread.
//
// A goroutine should call LockOSThread before calling OS services or
// non-Go library functions that depend on per-thread state.
//
//go:nosplit
func LockOSThread() {
	if atomic.Load(&newmHandoff.haveTemplateThread) == 0 && GOOS != "plan9" {
		// If we need to start a new thread from the locked
		// thread, we need the template thread. Start it now
		// while we're in a known-good state.
		startTemplateThread()
	}
	gp := getg()
	gp.m.lockedExt++
	if gp.m.lockedExt == 0 {
		gp.m.lockedExt--
		panic("LockOSThread nesting overflow")
	}
	dolockOSThread()
}

//go:nosplit
func lockOSThread() {
	getg().m.lockedInt++
	dolockOSThread()
}

// dounlockOSThread is called by UnlockOSThread and unlockOSThread below
// after they update m->locked. Do not allow preemption during this call,
// or else the m might be in different in this function than in the caller.
//
//go:nosplit
func dounlockOSThread() {
	if GOARCH == "wasm" {
		return // no threads on wasm yet
	}
	gp := getg()
	if gp.m.lockedInt != 0 || gp.m.lockedExt != 0 {
		return
	}
	gp.m.lockedg = 0
	gp.lockedm = 0
}

// UnlockOSThread undoes an earlier call to LockOSThread.
// If this drops the number of active LockOSThread calls on the
// calling goroutine to zero, it unwires the calling goroutine from
// its fixed operating system thread.
// If there are no active LockOSThread calls, this is a no-op.
//
// Before calling UnlockOSThread, the caller must ensure that the OS
// thread is suitable for running other goroutines. If the caller made
// any permanent changes to the state of the thread that would affect
// other goroutines, it should not call this function and thus leave
// the goroutine locked to the OS thread until the goroutine (and
// hence the thread) exits.
//
//go:nosplit
func UnlockOSThread() {
	gp := getg()
	if gp.m.lockedExt == 0 {
		return
	}
	gp.m.lockedExt--
	dounlockOSThread()
}

//go:nosplit
func unlockOSThread() {
	gp := getg()
	if gp.m.lockedInt == 0 {
		systemstack(badunlockosthread)
	}
	gp.m.lockedInt--
	dounlockOSThread()
}

func badunlockosthread() {
	throw("runtime: internal error: misuse of lockOSThread/unlockOSThread")
}

func gcount() int32 {
	n := int32(atomic.Loaduintptr(&allglen)) - sched.gFree.n - sched.ngsys.Load()
	for _, pp := range allp {
		n -= pp.gFree.n
	}

	// All these variables can be changed concurrently, so the result can be inconsistent.
	// But at least the current goroutine is running.
	if n < 1 {
		n = 1
	}
	return n
}

func mcount() int32 {
	return int32(sched.mnext - sched.nmfreed)
}

var prof struct {
	signalLock atomic.Uint32

	// Must hold signalLock to write. Reads may be lock-free, but
	// signalLock should be taken to synchronize with changes.
	hz atomic.Int32
}

func _System()                    { _System() }
func _ExternalCode()              { _ExternalCode() }
func _LostExternalCode()          { _LostExternalCode() }
func _GC()                        { _GC() }
func _LostSIGPROFDuringAtomic64() { _LostSIGPROFDuringAtomic64() }
func _LostContendedRuntimeLock()  { _LostContendedRuntimeLock() }
func _VDSO()                      { _VDSO() }

// Called if we receive a SIGPROF signal.
// Called by the signal handler, may run during STW.
//
//go:nowritebarrierrec
func sigprof(pc, sp, lr uintptr, gp *g, mp *m) {
	if prof.hz.Load() == 0 {
		return
	}

	// If mp.profilehz is 0, then profiling is not enabled for this thread.
	// We must check this to avoid a deadlock between setcpuprofilerate
	// and the call to cpuprof.add, below.
	if mp != nil && mp.profilehz == 0 {
		return
	}

	// On mips{,le}/arm, 64bit atomics are emulated with spinlocks, in
	// runtime/internal/atomic. If SIGPROF arrives while the program is inside
	// the critical section, it creates a deadlock (when writing the sample).
	// As a workaround, create a counter of SIGPROFs while in critical section
	// to store the count, and pass it to sigprof.add() later when SIGPROF is
	// received from somewhere else (with _LostSIGPROFDuringAtomic64 as pc).
	if GOARCH == "mips" || GOARCH == "mipsle" || GOARCH == "arm" {
		if f := findfunc(pc); f.valid() {
			if hasPrefix(funcname(f), "runtime/internal/atomic") {
				cpuprof.lostAtomic++
				return
			}
		}
		if GOARCH == "arm" && goarm < 7 && GOOS == "linux" && pc&0xffff0000 == 0xffff0000 {
			// runtime/internal/atomic functions call into kernel
			// helpers on arm < 7. See
			// runtime/internal/atomic/sys_linux_arm.s.
			cpuprof.lostAtomic++
			return
		}
	}

	// Profiling runs concurrently with GC, so it must not allocate.
	// Set a trap in case the code does allocate.
	// Note that on windows, one thread takes profiles of all the
	// other threads, so mp is usually not getg().m.
	// In fact mp may not even be stopped.
	// See golang.org/issue/17165.
	getg().m.mallocing++

	var u unwinder
	var stk [maxCPUProfStack]uintptr
	n := 0
	if mp.ncgo > 0 && mp.curg != nil && mp.curg.syscallpc != 0 && mp.curg.syscallsp != 0 {
		cgoOff := 0
		// Check cgoCallersUse to make sure that we are not
		// interrupting other code that is fiddling with
		// cgoCallers.  We are running in a signal handler
		// with all signals blocked, so we don't have to worry
		// about any other code interrupting us.
		if mp.cgoCallersUse.Load() == 0 && mp.cgoCallers != nil && mp.cgoCallers[0] != 0 {
			for cgoOff < len(mp.cgoCallers) && mp.cgoCallers[cgoOff] != 0 {
				cgoOff++
			}
			n += copy(stk[:], mp.cgoCallers[:cgoOff])
			mp.cgoCallers[0] = 0
		}

		// Collect Go stack that leads to the cgo call.
		u.initAt(mp.curg.syscallpc, mp.curg.syscallsp, 0, mp.curg, unwindSilentErrors)
	} else if usesLibcall() && mp.libcallg != 0 && mp.libcallpc != 0 && mp.libcallsp != 0 {
		// Libcall, i.e. runtime syscall on windows.
		// Collect Go stack that leads to the call.
		u.initAt(mp.libcallpc, mp.libcallsp, 0, mp.libcallg.ptr(), unwindSilentErrors)
	} else if mp != nil && mp.vdsoSP != 0 {
		// VDSO call, e.g. nanotime1 on Linux.
		// Collect Go stack that leads to the call.
		u.initAt(mp.vdsoPC, mp.vdsoSP, 0, gp, unwindSilentErrors|unwindJumpStack)
	} else {
		u.initAt(pc, sp, lr, gp, unwindSilentErrors|unwindTrap|unwindJumpStack)
	}
	n += tracebackPCs(&u, 0, stk[n:])

	if n <= 0 {
		// Normal traceback is impossible or has failed.
		// Account it against abstract "System" or "GC".
		n = 2
		if inVDSOPage(pc) {
			pc = abi.FuncPCABIInternal(_VDSO) + sys.PCQuantum
		} else if pc > firstmoduledata.etext {
			// "ExternalCode" is better than "etext".
			pc = abi.FuncPCABIInternal(_ExternalCode) + sys.PCQuantum
		}
		stk[0] = pc
		if mp.preemptoff != "" {
			stk[1] = abi.FuncPCABIInternal(_GC) + sys.PCQuantum
		} else {
			stk[1] = abi.FuncPCABIInternal(_System) + sys.PCQuantum
		}
	}

	if prof.hz.Load() != 0 {
		// Note: it can happen on Windows that we interrupted a system thread
		// with no g, so gp could nil. The other nil checks are done out of
		// caution, but not expected to be nil in practice.
		var tagPtr *unsafe.Pointer
		if gp != nil && gp.m != nil && gp.m.curg != nil {
			tagPtr = &gp.m.curg.labels
		}
		cpuprof.add(tagPtr, stk[:n])

		gprof := gp
		var mp *m
		var pp *p
		if gp != nil && gp.m != nil {
			if gp.m.curg != nil {
				gprof = gp.m.curg
			}
			mp = gp.m
			pp = gp.m.p.ptr()
		}
		traceCPUSample(gprof, mp, pp, stk[:n])
	}
	getg().m.mallocing--
}

// setcpuprofilerate sets the CPU profiling rate to hz times per second.
// If hz <= 0, setcpuprofilerate turns off CPU profiling.
func setcpuprofilerate(hz int32) {
	// Force sane arguments.
	if hz < 0 {
		hz = 0
	}

	// Disable preemption, otherwise we can be rescheduled to another thread
	// that has profiling enabled.
	gp := getg()
	gp.m.locks++

	// Stop profiler on this thread so that it is safe to lock prof.
	// if a profiling signal came in while we had prof locked,
	// it would deadlock.
	setThreadCPUProfiler(0)

	for !prof.signalLock.CompareAndSwap(0, 1) {
		osyield()
	}
	if prof.hz.Load() != hz {
		setProcessCPUProfiler(hz)
		prof.hz.Store(hz)
	}
	prof.signalLock.Store(0)

	lock(&sched.lock)
	sched.profilehz = hz
	unlock(&sched.lock)

	if hz != 0 {
		setThreadCPUProfiler(hz)
	}

	gp.m.locks--
}

// init initializes pp, which may be a freshly allocated p or a
// previously destroyed p, and transitions it to status _Pgcstop.
func (pp *p) init(id int32) {
	pp.id = id
	pp.status = _Pgcstop
	pp.sudogcache = pp.sudogbuf[:0]
	pp.deferpool = pp.deferpoolbuf[:0]
	pp.wbBuf.reset()
	if pp.mcache == nil {
		if id == 0 {
			if mcache0 == nil {
				throw("missing mcache?")
			}
			// Use the bootstrap mcache0. Only one P will get
			// mcache0: the one with ID 0.
			pp.mcache = mcache0
		} else {
			pp.mcache = allocmcache()
		}
	}
	if raceenabled && pp.raceprocctx == 0 {
		if id == 0 {
			pp.raceprocctx = raceprocctx0
			raceprocctx0 = 0 // bootstrap
		} else {
			pp.raceprocctx = raceproccreate()
		}
	}
	lockInit(&pp.timersLock, lockRankTimers)

	// This P may get timers when it starts running. Set the mask here
	// since the P may not go through pidleget (notably P 0 on startup).
	timerpMask.set(id)
	// Similarly, we may not go through pidleget before this P starts
	// running if it is P 0 on startup.
	idlepMask.clear(id)
}

// destroy releases all of the resources associated with pp and
// transitions it to status _Pdead.
//
// sched.lock must be held and the world must be stopped.
func (pp *p) destroy() {
	assertLockHeld(&sched.lock)
	assertWorldStopped()

	// Move all runnable goroutines to the global queue
	for pp.runqhead != pp.runqtail {
		// Pop from tail of local queue
		pp.runqtail--
		gp := pp.runq[pp.runqtail%uint32(len(pp.runq))].ptr()
		// Push onto head of global queue
		globrunqputhead(gp)
	}
	if pp.runnext != 0 {
		globrunqputhead(pp.runnext.ptr())
		pp.runnext = 0
	}
	if len(pp.timers) > 0 {
		plocal := getg().m.p.ptr()
		// The world is stopped, but we acquire timersLock to
		// protect against sysmon calling timeSleepUntil.
		// This is the only case where we hold the timersLock of
		// more than one P, so there are no deadlock concerns.
		lock(&plocal.timersLock)
		lock(&pp.timersLock)
		moveTimers(plocal, pp.timers)
		pp.timers = nil
		pp.numTimers.Store(0)
		pp.deletedTimers.Store(0)
		pp.timer0When.Store(0)
		unlock(&pp.timersLock)
		unlock(&plocal.timersLock)
	}
	// Flush p's write barrier buffer.
	if gcphase != _GCoff {
		wbBufFlush1(pp)
		pp.gcw.dispose()
	}
	for i := range pp.sudogbuf {
		pp.sudogbuf[i] = nil
	}
	pp.sudogcache = pp.sudogbuf[:0]
	pp.pinnerCache = nil
	for j := range pp.deferpoolbuf {
		pp.deferpoolbuf[j] = nil
	}
	pp.deferpool = pp.deferpoolbuf[:0]
	systemstack(func() {
		for i := 0; i < pp.mspancache.len; i++ {
			// Safe to call since the world is stopped.
			mheap_.spanalloc.free(unsafe.Pointer(pp.mspancache.buf[i]))
		}
		pp.mspancache.len = 0
		lock(&mheap_.lock)
		pp.pcache.flush(&mheap_.pages)
		unlock(&mheap_.lock)
	})
	freemcache(pp.mcache)
	pp.mcache = nil
	gfpurge(pp)
	traceProcFree(pp)
	if raceenabled {
		if pp.timerRaceCtx != 0 {
			// The race detector code uses a callback to fetch
			// the proc context, so arrange for that callback
			// to see the right thing.
			// This hack only works because we are the only
			// thread running.
			mp := getg().m
			phold := mp.p.ptr()
			mp.p.set(pp)

			racectxend(pp.timerRaceCtx)
			pp.timerRaceCtx = 0

			mp.p.set(phold)
		}
		raceprocdestroy(pp.raceprocctx)
		pp.raceprocctx = 0
	}
	pp.gcAssistTime = 0
	pp.status = _Pdead
}

// Change number of processors.
//
// sched.lock must be held, and the world must be stopped.
//
// gcworkbufs must not be being modified by either the GC or the write barrier
// code, so the GC must not be running if the number of Ps actually changes.
//
// Returns list of Ps with local work, they need to be scheduled by the caller.
func procresize(nprocs int32) *p {
	assertLockHeld(&sched.lock)
	assertWorldStopped()

	old := gomaxprocs
	if old < 0 || nprocs <= 0 {
		throw("procresize: invalid arg")
	}
	trace := traceAcquire()
	if trace.ok() {
		trace.Gomaxprocs(nprocs)
		traceRelease(trace)
	}

	// update statistics
	now := nanotime()
	if sched.procresizetime != 0 {
		sched.totaltime += int64(old) * (now - sched.procresizetime)
	}
	sched.procresizetime = now

	maskWords := (nprocs + 31) / 32

	// Grow allp if necessary.
	if nprocs > int32(len(allp)) {
		// Synchronize with retake, which could be running
		// concurrently since it doesn't run on a P.
		lock(&allpLock)
		if nprocs <= int32(cap(allp)) {
			allp = allp[:nprocs]
		} else {
			nallp := make([]*p, nprocs)
			// Copy everything up to allp's cap so we
			// never lose old allocated Ps.
			copy(nallp, allp[:cap(allp)])
			allp = nallp
		}

		if maskWords <= int32(cap(idlepMask)) {
			idlepMask = idlepMask[:maskWords]
			timerpMask = timerpMask[:maskWords]
		} else {
			nidlepMask := make([]uint32, maskWords)
			// No need to copy beyond len, old Ps are irrelevant.
			copy(nidlepMask, idlepMask)
			idlepMask = nidlepMask

			ntimerpMask := make([]uint32, maskWords)
			copy(ntimerpMask, timerpMask)
			timerpMask = ntimerpMask
		}
		unlock(&allpLock)
	}

	// initialize new P's
	for i := old; i < nprocs; i++ {
		pp := allp[i]
		if pp == nil {
			pp = new(p)
		}
		pp.init(i)
		atomicstorep(unsafe.Pointer(&allp[i]), unsafe.Pointer(pp))
	}

	gp := getg()
	if gp.m.p != 0 && gp.m.p.ptr().id < nprocs {
		// continue to use the current P
		gp.m.p.ptr().status = _Prunning
		gp.m.p.ptr().mcache.prepareForSweep()
	} else {
		// release the current P and acquire allp[0].
		//
		// We must do this before destroying our current P
		// because p.destroy itself has write barriers, so we
		// need to do that from a valid P.
		if gp.m.p != 0 {
			trace := traceAcquire()
			if trace.ok() {
				// Pretend that we were descheduled
				// and then scheduled again to keep
				// the trace sane.
				trace.GoSched()
				trace.ProcStop(gp.m.p.ptr())
				traceRelease(trace)
			}
			gp.m.p.ptr().m = 0
		}
		gp.m.p = 0
		pp := allp[0]
		pp.m = 0
		pp.status = _Pidle
		acquirep(pp)
		trace := traceAcquire()
		if trace.ok() {
			trace.GoStart()
			traceRelease(trace)
		}
	}

	// g.m.p is now set, so we no longer need mcache0 for bootstrapping.
	mcache0 = nil

	// release resources from unused P's
	for i := nprocs; i < old; i++ {
		pp := allp[i]
		pp.destroy()
		// can't free P itself because it can be referenced by an M in syscall
	}

	// Trim allp.
	if int32(len(allp)) != nprocs {
		lock(&allpLock)
		allp = allp[:nprocs]
		idlepMask = idlepMask[:maskWords]
		timerpMask = timerpMask[:maskWords]
		unlock(&allpLock)
	}

	var runnablePs *p
	for i := nprocs - 1; i >= 0; i-- {
		pp := allp[i]
		if gp.m.p.ptr() == pp {
			continue
		}
		pp.status = _Pidle
		if runqempty(pp) {
			pidleput(pp, now)
		} else {
			pp.m.set(mget())
			pp.link.set(runnablePs)
			runnablePs = pp
		}
	}
	stealOrder.reset(uint32(nprocs))
	var int32p *int32 = &gomaxprocs // make compiler check that gomaxprocs is an int32
	atomic.Store((*uint32)(unsafe.Pointer(int32p)), uint32(nprocs))
	if old != nprocs {
		// Notify the limiter that the amount of procs has changed.
		gcCPULimiter.resetCapacity(now, nprocs)
	}
	return runnablePs
}

// Associate p and the current m.
//
// This function is allowed to have write barriers even if the caller
// isn't because it immediately acquires pp.
//
//go:yeswritebarrierrec
func acquirep(pp *p) {
	// Do the part that isn't allowed to have write barriers.
	wirep(pp)

	// Have p; write barriers now allowed.

	// Perform deferred mcache flush before this P can allocate
	// from a potentially stale mcache.
	pp.mcache.prepareForSweep()

	trace := traceAcquire()
	if trace.ok() {
		trace.ProcStart()
		traceRelease(trace)
	}
}

// wirep is the first step of acquirep, which actually associates the
// current M to pp. This is broken out so we can disallow write
// barriers for this part, since we don't yet have a P.
//
//go:nowritebarrierrec
//go:nosplit
func wirep(pp *p) {
	gp := getg()

	if gp.m.p != 0 {
		// Call on the systemstack to avoid a nosplit overflow build failure
		// on some platforms when built with -N -l. See #64113.
		systemstack(func() {
			throw("wirep: already in go")
		})
	}
	if pp.m != 0 || pp.status != _Pidle {
		// Call on the systemstack to avoid a nosplit overflow build failure
		// on some platforms when built with -N -l. See #64113.
		systemstack(func() {
			id := int64(0)
			if pp.m != 0 {
				id = pp.m.ptr().id
			}
			print("wirep: p->m=", pp.m, "(", id, ") p->status=", pp.status, "\n")
			throw("wirep: invalid p state")
		})
	}
	gp.m.p.set(pp)
	pp.m.set(gp.m)
	pp.status = _Prunning
}

// Disassociate p and the current m.
func releasep() *p {
	trace := traceAcquire()
	if trace.ok() {
		trace.ProcStop(getg().m.p.ptr())
		traceRelease(trace)
	}
	return releasepNoTrace()
}

// Disassociate p and the current m without tracing an event.
func releasepNoTrace() *p {
	gp := getg()

	if gp.m.p == 0 {
		throw("releasep: invalid arg")
	}
	pp := gp.m.p.ptr()
	if pp.m.ptr() != gp.m || pp.status != _Prunning {
		print("releasep: m=", gp.m, " m->p=", gp.m.p.ptr(), " p->m=", hex(pp.m), " p->status=", pp.status, "\n")
		throw("releasep: invalid p state")
	}
	gp.m.p = 0
	pp.m = 0
	pp.status = _Pidle
	return pp
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
// sched.lock must be held.
func checkdead() {
	assertLockHeld(&sched.lock)

	// For -buildmode=c-shared or -buildmode=c-archive it's OK if
	// there are no running goroutines. The calling program is
	// assumed to be running.
	if islibrary || isarchive {
		return
	}

	// If we are dying because of a signal caught on an already idle thread,
	// freezetheworld will cause all running threads to block.
	// And runtime will essentially enter into deadlock state,
	// except that there is a thread that will call exit soon.
	if panicking.Load() > 0 {
		return
	}

	// If we are not running under cgo, but we have an extra M then account
	// for it. (It is possible to have an extra M on Windows without cgo to
	// accommodate callbacks created by syscall.NewCallback. See issue #6751
	// for details.)
	var run0 int32
	if !iscgo && cgoHasExtraM && extraMLength.Load() > 0 {
		run0 = 1
	}

	run := mcount() - sched.nmidle - sched.nmidlelocked - sched.nmsys
	if run > run0 {
		return
	}
	if run < 0 {
		print("runtime: checkdead: nmidle=", sched.nmidle, " nmidlelocked=", sched.nmidlelocked, " mcount=", mcount(), " nmsys=", sched.nmsys, "\n")
		unlock(&sched.lock)
		throw("checkdead: inconsistent counts")
	}

	grunning := 0
	forEachG(func(gp *g) {
		if isSystemGoroutine(gp, false) {
			return
		}
		s := readgstatus(gp)
		switch s &^ _Gscan {
		case _Gwaiting,
			_Gpreempted:
			grunning++
		case _Grunnable,
			_Grunning,
			_Gsyscall:
			print("runtime: checkdead: find g ", gp.goid, " in status ", s, "\n")
			unlock(&sched.lock)
			throw("checkdead: runnable g")
		}
	})
	if grunning == 0 { // possible if main goroutine calls runtime·Goexit()
		unlock(&sched.lock) // unlock so that GODEBUG=scheddetail=1 doesn't hang
		fatal("no goroutines (main called runtime.Goexit) - deadlock!")
	}

	// Maybe jump time forward for playground.
	if faketime != 0 {
		if when := timeSleepUntil(); when < maxWhen {
			faketime = when

			// Start an M to steal the timer.
			pp, _ := pidleget(faketime)
			if pp == nil {
				// There should always be a free P since
				// nothing is running.
				unlock(&sched.lock)
				throw("checkdead: no p for timer")
			}
			mp := mget()
			if mp == nil {
				// There should always be a free M since
				// nothing is running.
				unlock(&sched.lock)
				throw("checkdead: no m for timer")
			}
			// M must be spinning to steal. We set this to be
			// explicit, but since this is the only M it would
			// become spinning on its own anyways.
			sched.nmspinning.Add(1)
			mp.spinning = true
			mp.nextp.set(pp)
			notewakeup(&mp.park)
			return
		}
	}

	// There are no goroutines running, so we can look at the P's.
	for _, pp := range allp {
		if len(pp.timers) > 0 {
			return
		}
	}

	unlock(&sched.lock) // unlock so that GODEBUG=scheddetail=1 doesn't hang
	fatal("all goroutines are asleep - deadlock!")
}

// forcegcperiod is the maximum time in nanoseconds between garbage
// collections. If we go this long without a garbage collection, one
// is forced to run.
//
// This is a variable for testing purposes. It normally doesn't change.
var forcegcperiod int64 = 2 * 60 * 1e9

// needSysmonWorkaround is true if the workaround for
// golang.org/issue/42515 is needed on NetBSD.
var needSysmonWorkaround bool = false

// haveSysmon indicates whether there is sysmon thread support.
//
// No threads on wasm yet, so no sysmon.
const haveSysmon = GOARCH != "wasm"

// Always runs without a P, so write barriers are not allowed.
//
//go:nowritebarrierrec
func sysmon() {
	lock(&sched.lock)
	sched.nmsys++
	checkdead()
	unlock(&sched.lock)

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

		// sysmon should not enter deep sleep if schedtrace is enabled so that
		// it can print that information at the right time.
		//
		// It should also not enter deep sleep if there are any active P's so
		// that it can retake P's from syscalls, preempt long running G's, and
		// poll the network if all P's are busy for long stretches.
		//
		// It should wakeup from deep sleep if any P's become active either due
		// to exiting a syscall or waking up due to a timer expiring so that it
		// can resume performing those duties. If it wakes from a syscall it
		// resets idle and delay as a bet that since it had retaken a P from a
		// syscall before, it may need to do it again shortly after the
		// application starts work again. It does not reset idle when waking
		// from a timer to avoid adding system load to applications that spend
		// most of their time sleeping.
		now := nanotime()
		if debug.schedtrace <= 0 && (sched.gcwaiting.Load() || sched.npidle.Load() == gomaxprocs) {
			lock(&sched.lock)
			if sched.gcwaiting.Load() || sched.npidle.Load() == gomaxprocs {
				syscallWake := false
				next := timeSleepUntil()
				if next > now {
					sched.sysmonwait.Store(true)
					unlock(&sched.lock)
					// Make wake-up period small enough
					// for the sampling to be correct.
					sleep := forcegcperiod / 2
					if next-now < sleep {
						sleep = next - now
					}
					shouldRelax := sleep >= osRelaxMinNS
					if shouldRelax {
						osRelax(true)
					}
					syscallWake = notetsleep(&sched.sysmonnote, sleep)
					if shouldRelax {
						osRelax(false)
					}
					lock(&sched.lock)
					sched.sysmonwait.Store(false)
					noteclear(&sched.sysmonnote)
				}
				if syscallWake {
					idle = 0
					delay = 20
				}
			}
			unlock(&sched.lock)
		}

		lock(&sched.sysmonlock)
		// Update now in case we blocked on sysmonnote or spent a long time
		// blocked on schedlock or sysmonlock above.
		now = nanotime()

		// trigger libc interceptors if needed
		if *cgo_yield != nil {
			asmcgocall(*cgo_yield, nil)
		}
		// poll network if not polled for more than 10ms
		lastpoll := sched.lastpoll.Load()
		if netpollinited() && lastpoll != 0 && lastpoll+10*1000*1000 < now {
			sched.lastpoll.CompareAndSwap(lastpoll, now)
			list, delta := netpoll(0) // non-blocking - returns list of goroutines
			if !list.empty() {
				// Need to decrement number of idle locked M's
				// (pretending that one more is running) before injectglist.
				// Otherwise it can lead to the following situation:
				// injectglist grabs all P's but before it starts M's to run the P's,
				// another M returns from syscall, finishes running its G,
				// observes that there is no work to do and no other running M's
				// and reports deadlock.
				incidlelocked(-1)
				injectglist(&list)
				incidlelocked(1)
				netpollAdjustWaiters(delta)
			}
		}
		if GOOS == "netbsd" && needSysmonWorkaround {
			// netpoll is responsible for waiting for timer
			// expiration, so we typically don't have to worry
			// about starting an M to service timers. (Note that
			// sleep for timeSleepUntil above simply ensures sysmon
			// starts running again when that timer expiration may
			// cause Go code to run again).
			//
			// However, netbsd has a kernel bug that sometimes
			// misses netpollBreak wake-ups, which can lead to
			// unbounded delays servicing timers. If we detect this
			// overrun, then startm to get something to handle the
			// timer.
			//
			// See issue 42515 and
			// https://gnats.netbsd.org/cgi-bin/query-pr-single.pl?number=50094.
			if next := timeSleepUntil(); next < now {
				startm(nil, false, false)
			}
		}
		if scavenger.sysmonWake.Load() != 0 {
			// Kick the scavenger awake if someone requested it.
			scavenger.wake()
		}
		// retake P's blocked in syscalls
		// and preempt long running G's
		if retake(now) != 0 {
			idle = 0
		} else {
			idle++
		}
		// check if we need to force a GC
		if t := (gcTrigger{kind: gcTriggerTime, now: now}); t.test() && forcegc.idle.Load() {
			lock(&forcegc.lock)
			forcegc.idle.Store(false)
			var list gList
			list.push(forcegc.g)
			injectglist(&list)
			unlock(&forcegc.lock)
		}
		if debug.schedtrace > 0 && lasttrace+int64(debug.schedtrace)*1000000 <= now {
			lasttrace = now
			schedtrace(debug.scheddetail > 0)
		}
		unlock(&sched.sysmonlock)
	}
}

type sysmontick struct {
	schedtick   uint32
	schedwhen   int64
	syscalltick uint32
	syscallwhen int64
}

// forcePreemptNS is the time slice given to a G before it is
// preempted.
const forcePreemptNS = 10 * 1000 * 1000 // 10ms

func retake(now int64) uint32 {
	n := 0
	// Prevent allp slice changes. This lock will be completely
	// uncontended unless we're already stopping the world.
	lock(&allpLock)
	// We can't use a range loop over allp because we may
	// temporarily drop the allpLock. Hence, we need to re-fetch
	// allp each time around the loop.
	for i := 0; i < len(allp); i++ {
		pp := allp[i]
		if pp == nil {
			// This can happen if procresize has grown
			// allp but not yet created new Ps.
			continue
		}
		pd := &pp.sysmontick
		s := pp.status
		sysretake := false
		if s == _Prunning || s == _Psyscall {
			// Preempt G if it's running on the same schedtick for
			// too long. This could be from a single long-running
			// goroutine or a sequence of goroutines run via
			// runnext, which share a single schedtick time slice.
			t := int64(pp.schedtick)
			if int64(pd.schedtick) != t {
				pd.schedtick = uint32(t)
				pd.schedwhen = now
			} else if pd.schedwhen+forcePreemptNS <= now {
				preemptone(pp)
				// In case of syscall, preemptone() doesn't
				// work, because there is no M wired to P.
				sysretake = true
			}
		}
		if s == _Psyscall {
			// Retake P from syscall if it's there for more than 1 sysmon tick (at least 20us).
			t := int64(pp.syscalltick)
			if !sysretake && int64(pd.syscalltick) != t {
				pd.syscalltick = uint32(t)
				pd.syscallwhen = now
				continue
			}
			// On the one hand we don't want to retake Ps if there is no other work to do,
			// but on the other hand we want to retake them eventually
			// because they can prevent the sysmon thread from deep sleep.
			if runqempty(pp) && sched.nmspinning.Load()+sched.npidle.Load() > 0 && pd.syscallwhen+10*1000*1000 > now {
				continue
			}
			// Drop allpLock so we can take sched.lock.
			unlock(&allpLock)
			// Need to decrement number of idle locked M's
			// (pretending that one more is running) before the CAS.
			// Otherwise the M from which we retake can exit the syscall,
			// increment nmidle and report deadlock.
			incidlelocked(-1)
			trace := traceAcquire()
			if atomic.Cas(&pp.status, s, _Pidle) {
				if trace.ok() {
					trace.GoSysBlock(pp)
					trace.ProcSteal(pp, false)
					traceRelease(trace)
				}
				n++
				pp.syscalltick++
				handoffp(pp)
			} else if trace.ok() {
				traceRelease(trace)
			}
			incidlelocked(1)
			lock(&allpLock)
		}
	}
	unlock(&allpLock)
	return uint32(n)
}

// Tell all goroutines that they have been preempted and they should stop.
// This function is purely best-effort. It can fail to inform a goroutine if a
// processor just started running it.
// No locks need to be held.
// Returns true if preemption request was issued to at least one goroutine.
func preemptall() bool {
	res := false
	for _, pp := range allp {
		if pp.status != _Prunning {
			continue
		}
		if preemptone(pp) {
			res = true
		}
	}
	return res
}

// Tell the goroutine running on processor P to stop.
// This function is purely best-effort. It can incorrectly fail to inform the
// goroutine. It can inform the wrong goroutine. Even if it informs the
// correct goroutine, that goroutine might ignore the request if it is
// simultaneously executing newstack.
// No lock needs to be held.
// Returns true if preemption request was issued.
// The actual preemption will happen at some point in the future
// and will be indicated by the gp->status no longer being
// Grunning
func preemptone(pp *p) bool {
	mp := pp.m.ptr()
	if mp == nil || mp == getg().m {
		return false
	}
	gp := mp.curg
	if gp == nil || gp == mp.g0 {
		return false
	}

	gp.preempt = true

	// Every call in a goroutine checks for stack overflow by
	// comparing the current stack pointer to gp->stackguard0.
	// Setting gp->stackguard0 to StackPreempt folds
	// preemption into the normal stack overflow check.
	gp.stackguard0 = stackPreempt

	// Request an async preemption of this P.
	if preemptMSupported && debug.asyncpreemptoff == 0 {
		pp.preempt = true
		preemptM(mp)
	}

	return true
}

var starttime int64

func schedtrace(detailed bool) {
	now := nanotime()
	if starttime == 0 {
		starttime = now
	}

	lock(&sched.lock)
	print("SCHED ", (now-starttime)/1e6, "ms: gomaxprocs=", gomaxprocs, " idleprocs=", sched.npidle.Load(), " threads=", mcount(), " spinningthreads=", sched.nmspinning.Load(), " needspinning=", sched.needspinning.Load(), " idlethreads=", sched.nmidle, " runqueue=", sched.runqsize)
	if detailed {
		print(" gcwaiting=", sched.gcwaiting.Load(), " nmidlelocked=", sched.nmidlelocked, " stopwait=", sched.stopwait, " sysmonwait=", sched.sysmonwait.Load(), "\n")
	}
	// We must be careful while reading data from P's, M's and G's.
	// Even if we hold schedlock, most data can be changed concurrently.
	// E.g. (p->m ? p->m->id : -1) can crash if p->m changes from non-nil to nil.
	for i, pp := range allp {
		mp := pp.m.ptr()
		h := atomic.Load(&pp.runqhead)
		t := atomic.Load(&pp.runqtail)
		if detailed {
			print("  P", i, ": status=", pp.status, " schedtick=", pp.schedtick, " syscalltick=", pp.syscalltick, " m=")
			if mp != nil {
				print(mp.id)
			} else {
				print("nil")
			}
			print(" runqsize=", t-h, " gfreecnt=", pp.gFree.n, " timerslen=", len(pp.timers), "\n")
		} else {
			// In non-detailed mode format lengths of per-P run queues as:
			// [len1 len2 len3 len4]
			print(" ")
			if i == 0 {
				print("[")
			}
			print(t - h)
			if i == len(allp)-1 {
				print("]\n")
			}
		}
	}

	if !detailed {
		unlock(&sched.lock)
		return
	}

	for mp := allm; mp != nil; mp = mp.alllink {
		pp := mp.p.ptr()
		print("  M", mp.id, ": p=")
		if pp != nil {
			print(pp.id)
		} else {
			print("nil")
		}
		print(" curg=")
		if mp.curg != nil {
			print(mp.curg.goid)
		} else {
			print("nil")
		}
		print(" mallocing=", mp.mallocing, " throwing=", mp.throwing, " preemptoff=", mp.preemptoff, " locks=", mp.locks, " dying=", mp.dying, " spinning=", mp.spinning, " blocked=", mp.blocked, " lockedg=")
		if lockedg := mp.lockedg.ptr(); lockedg != nil {
			print(lockedg.goid)
		} else {
			print("nil")
		}
		print("\n")
	}

	forEachG(func(gp *g) {
		print("  G", gp.goid, ": status=", readgstatus(gp), "(", gp.waitreason.String(), ") m=")
		if gp.m != nil {
			print(gp.m.id)
		} else {
			print("nil")
		}
		print(" lockedm=")
		if lockedm := gp.lockedm.ptr(); lockedm != nil {
			print(lockedm.id)
		} else {
			print("nil")
		}
		print("\n")
	})
	unlock(&sched.lock)
}

// schedEnableUser enables or disables the scheduling of user
// goroutines.
//
// This does not stop already running user goroutines, so the caller
// should first stop the world when disabling user goroutines.
func schedEnableUser(enable bool) {
	lock(&sched.lock)
	if sched.disable.user == !enable {
		unlock(&sched.lock)
		return
	}
	sched.disable.user = !enable
	if enable {
		n := sched.disable.n
		sched.disable.n = 0
		globrunqputbatch(&sched.disable.runnable, n)
		unlock(&sched.lock)
		for ; n != 0 && sched.npidle.Load() != 0; n-- {
			startm(nil, false, false)
		}
	} else {
		unlock(&sched.lock)
	}
}

// schedEnabled reports whether gp should be scheduled. It returns
// false is scheduling of gp is disabled.
//
// sched.lock must be held.
func schedEnabled(gp *g) bool {
	assertLockHeld(&sched.lock)

	if sched.disable.user {
		return isSystemGoroutine(gp, true)
	}
	return true
}

// Put mp on midle list.
// sched.lock must be held.
// May run during STW, so write barriers are not allowed.
//
//go:nowritebarrierrec
func mput(mp *m) {
	assertLockHeld(&sched.lock)

	mp.schedlink = sched.midle
	sched.midle.set(mp)
	sched.nmidle++
	checkdead()
}

// Try to get an m from midle list.
// sched.lock must be held.
// May run during STW, so write barriers are not allowed.
//
//go:nowritebarrierrec
func mget() *m {
	assertLockHeld(&sched.lock)

	mp := sched.midle.ptr()
	if mp != nil {
		sched.midle = mp.schedlink
		sched.nmidle--
	}
	return mp
}

// Put gp on the global runnable queue.
// sched.lock must be held.
// May run during STW, so write barriers are not allowed.
//
//go:nowritebarrierrec
func globrunqput(gp *g) {
	assertLockHeld(&sched.lock)

	sched.runq.pushBack(gp)
	sched.runqsize++
}

// Put gp at the head of the global runnable queue.
// sched.lock must be held.
// May run during STW, so write barriers are not allowed.
//
//go:nowritebarrierrec
func globrunqputhead(gp *g) {
	assertLockHeld(&sched.lock)

	sched.runq.push(gp)
	sched.runqsize++
}

// Put a batch of runnable goroutines on the global runnable queue.
// This clears *batch.
// sched.lock must be held.
// May run during STW, so write barriers are not allowed.
//
//go:nowritebarrierrec
func globrunqputbatch(batch *gQueue, n int32) {
	assertLockHeld(&sched.lock)

	sched.runq.pushBackAll(*batch)
	sched.runqsize += n
	*batch = gQueue{}
}

// Try get a batch of G's from the global runnable queue.
// sched.lock must be held.
func globrunqget(pp *p, max int32) *g {
	assertLockHeld(&sched.lock)

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
	if n > int32(len(pp.runq))/2 {
		n = int32(len(pp.runq)) / 2
	}

	sched.runqsize -= n

	gp := sched.runq.pop()
	n--
	for ; n > 0; n-- {
		gp1 := sched.runq.pop()
		runqput(pp, gp1, false)
	}
	return gp
}

// pMask is an atomic bitstring with one bit per P.
type pMask []uint32

// read returns true if P id's bit is set.
func (p pMask) read(id uint32) bool {
	word := id / 32
	mask := uint32(1) << (id % 32)
	return (atomic.Load(&p[word]) & mask) != 0
}

// set sets P id's bit.
func (p pMask) set(id int32) {
	word := id / 32
	mask := uint32(1) << (id % 32)
	atomic.Or(&p[word], mask)
}

// clear clears P id's bit.
func (p pMask) clear(id int32) {
	word := id / 32
	mask := uint32(1) << (id % 32)
	atomic.And(&p[word], ^mask)
}

// updateTimerPMask clears pp's timer mask if it has no timers on its heap.
//
// Ideally, the timer mask would be kept immediately consistent on any timer
// operations. Unfortunately, updating a shared global data structure in the
// timer hot path adds too much overhead in applications frequently switching
// between no timers and some timers.
//
// As a compromise, the timer mask is updated only on pidleget / pidleput. A
// running P (returned by pidleget) may add a timer at any time, so its mask
// must be set. An idle P (passed to pidleput) cannot add new timers while
// idle, so if it has no timers at that time, its mask may be cleared.
//
// Thus, we get the following effects on timer-stealing in findrunnable:
//
//   - Idle Ps with no timers when they go idle are never checked in findrunnable
//     (for work- or timer-stealing; this is the ideal case).
//   - Running Ps must always be checked.
//   - Idle Ps whose timers are stolen must continue to be checked until they run
//     again, even after timer expiration.
//
// When the P starts running again, the mask should be set, as a timer may be
// added at any time.
//
// TODO(prattmic): Additional targeted updates may improve the above cases.
// e.g., updating the mask when stealing a timer.
func updateTimerPMask(pp *p) {
	if pp.numTimers.Load() > 0 {
		return
	}

	// Looks like there are no timers, however another P may transiently
	// decrement numTimers when handling a timerModified timer in
	// checkTimers. We must take timersLock to serialize with these changes.
	lock(&pp.timersLock)
	if pp.numTimers.Load() == 0 {
		timerpMask.clear(pp.id)
	}
	unlock(&pp.timersLock)
}

// pidleput puts p on the _Pidle list. now must be a relatively recent call
// to nanotime or zero. Returns now or the current time if now was zero.
//
// This releases ownership of p. Once sched.lock is released it is no longer
// safe to use p.
//
// sched.lock must be held.
//
// May run during STW, so write barriers are not allowed.
//
//go:nowritebarrierrec
func pidleput(pp *p, now int64) int64 {
	assertLockHeld(&sched.lock)

	if !runqempty(pp) {
		throw("pidleput: P has non-empty run queue")
	}
	if now == 0 {
		now = nanotime()
	}
	updateTimerPMask(pp) // clear if there are no timers.
	idlepMask.set(pp.id)
	pp.link = sched.pidle
	sched.pidle.set(pp)
	sched.npidle.Add(1)
	if !pp.limiterEvent.start(limiterEventIdle, now) {
		throw("must be able to track idle limiter event")
	}
	return now
}

// pidleget tries to get a p from the _Pidle list, acquiring ownership.
//
// sched.lock must be held.
//
// May run during STW, so write barriers are not allowed.
//
//go:nowritebarrierrec
func pidleget(now int64) (*p, int64) {
	assertLockHeld(&sched.lock)

	pp := sched.pidle.ptr()
	if pp != nil {
		// Timer may get added at any time now.
		if now == 0 {
			now = nanotime()
		}
		timerpMask.set(pp.id)
		idlepMask.clear(pp.id)
		sched.pidle = pp.link
		sched.npidle.Add(-1)
		pp.limiterEvent.stop(limiterEventIdle, now)
	}
	return pp, now
}

// pidlegetSpinning tries to get a p from the _Pidle list, acquiring ownership.
// This is called by spinning Ms (or callers than need a spinning M) that have
// found work. If no P is available, this must synchronized with non-spinning
// Ms that may be preparing to drop their P without discovering this work.
//
// sched.lock must be held.
//
// May run during STW, so write barriers are not allowed.
//
//go:nowritebarrierrec
func pidlegetSpinning(now int64) (*p, int64) {
	assertLockHeld(&sched.lock)

	pp, now := pidleget(now)
	if pp == nil {
		// See "Delicate dance" comment in findrunnable. We found work
		// that we cannot take, we must synchronize with non-spinning
		// Ms that may be preparing to drop their P.
		sched.needspinning.Store(1)
		return nil, now
	}

	return pp, now
}

// runqempty reports whether pp has no Gs on its local run queue.
// It never returns true spuriously.
func runqempty(pp *p) bool {
	// Defend against a race where 1) pp has G1 in runqnext but runqhead == runqtail,
	// 2) runqput on pp kicks G1 to the runq, 3) runqget on pp empties runqnext.
	// Simply observing that runqhead == runqtail and then observing that runqnext == nil
	// does not mean the queue is empty.
	for {
		head := atomic.Load(&pp.runqhead)
		tail := atomic.Load(&pp.runqtail)
		runnext := atomic.Loaduintptr((*uintptr)(unsafe.Pointer(&pp.runnext)))
		if tail == atomic.Load(&pp.runqtail) {
			return head == tail && runnext == 0
		}
	}
}

// To shake out latent assumptions about scheduling order,
// we introduce some randomness into scheduling decisions
// when running with the race detector.
// The need for this was made obvious by changing the
// (deterministic) scheduling order in Go 1.5 and breaking
// many poorly-written tests.
// With the randomness here, as long as the tests pass
// consistently with -race, they shouldn't have latent scheduling
// assumptions.
const randomizeScheduler = raceenabled

// runqput tries to put g on the local runnable queue.
// If next is false, runqput adds g to the tail of the runnable queue.
// If next is true, runqput puts g in the pp.runnext slot.
// If the run queue is full, runnext puts g on the global queue.
// Executed only by the owner P.
func runqput(pp *p, gp *g, next bool) {
	if !haveSysmon && next {
		// A runnext goroutine shares the same time slice as the
		// current goroutine (inheritTime from runqget). To prevent a
		// ping-pong pair of goroutines from starving all others, we
		// depend on sysmon to preempt "long-running goroutines". That
		// is, any set of goroutines sharing the same time slice.
		//
		// If there is no sysmon, we must avoid runnext entirely or
		// risk starvation.
		next = false
	}
	if randomizeScheduler && next && randn(2) == 0 {
		next = false
	}

	if next {
	retryNext:
		oldnext := pp.runnext
		if !pp.runnext.cas(oldnext, guintptr(unsafe.Pointer(gp))) {
			goto retryNext
		}
		if oldnext == 0 {
			return
		}
		// Kick the old runnext out to the regular run queue.
		gp = oldnext.ptr()
	}

retry:
	h := atomic.LoadAcq(&pp.runqhead) // load-acquire, synchronize with consumers
	t := pp.runqtail
	if t-h < uint32(len(pp.runq)) {
		pp.runq[t%uint32(len(pp.runq))].set(gp)
		atomic.StoreRel(&pp.runqtail, t+1) // store-release, makes the item available for consumption
		return
	}
	if runqputslow(pp, gp, h, t) {
		return
	}
	// the queue is not full, now the put above must succeed
	goto retry
}

// Put g and a batch of work from local runnable queue on global queue.
// Executed only by the owner P.
func runqputslow(pp *p, gp *g, h, t uint32) bool {
	var batch [len(pp.runq)/2 + 1]*g

	// First, grab a batch from local queue.
	n := t - h
	n = n / 2
	if n != uint32(len(pp.runq)/2) {
		throw("runqputslow: queue is not full")
	}
	for i := uint32(0); i < n; i++ {
		batch[i] = pp.runq[(h+i)%uint32(len(pp.runq))].ptr()
	}
	if !atomic.CasRel(&pp.runqhead, h, h+n) { // cas-release, commits consume
		return false
	}
	batch[n] = gp

	if randomizeScheduler {
		for i := uint32(1); i <= n; i++ {
			j := cheaprandn(i + 1)
			batch[i], batch[j] = batch[j], batch[i]
		}
	}

	// Link the goroutines.
	for i := uint32(0); i < n; i++ {
		batch[i].schedlink.set(batch[i+1])
	}
	var q gQueue
	q.head.set(batch[0])
	q.tail.set(batch[n])

	// Now put the batch on global queue.
	lock(&sched.lock)
	globrunqputbatch(&q, int32(n+1))
	unlock(&sched.lock)
	return true
}

// runqputbatch tries to put all the G's on q on the local runnable queue.
// If the queue is full, they are put on the global queue; in that case
// this will temporarily acquire the scheduler lock.
// Executed only by the owner P.
func runqputbatch(pp *p, q *gQueue, qsize int) {
	h := atomic.LoadAcq(&pp.runqhead)
	t := pp.runqtail
	n := uint32(0)
	for !q.empty() && t-h < uint32(len(pp.runq)) {
		gp := q.pop()
		pp.runq[t%uint32(len(pp.runq))].set(gp)
		t++
		n++
	}
	qsize -= int(n)

	if randomizeScheduler {
		off := func(o uint32) uint32 {
			return (pp.runqtail + o) % uint32(len(pp.runq))
		}
		for i := uint32(1); i < n; i++ {
			j := cheaprandn(i + 1)
			pp.runq[off(i)], pp.runq[off(j)] = pp.runq[off(j)], pp.runq[off(i)]
		}
	}

	atomic.StoreRel(&pp.runqtail, t)
	if !q.empty() {
		lock(&sched.lock)
		globrunqputbatch(q, int32(qsize))
		unlock(&sched.lock)
	}
}

// Get g from local runnable queue.
// If inheritTime is true, gp should inherit the remaining time in the
// current time slice. Otherwise, it should start a new time slice.
// Executed only by the owner P.
func runqget(pp *p) (gp *g, inheritTime bool) {
	// If there's a runnext, it's the next G to run.
	next := pp.runnext
	// If the runnext is non-0 and the CAS fails, it could only have been stolen by another P,
	// because other Ps can race to set runnext to 0, but only the current P can set it to non-0.
	// Hence, there's no need to retry this CAS if it fails.
	if next != 0 && pp.runnext.cas(next, 0) {
		return next.ptr(), true
	}

	for {
		h := atomic.LoadAcq(&pp.runqhead) // load-acquire, synchronize with other consumers
		t := pp.runqtail
		if t == h {
			return nil, false
		}
		gp := pp.runq[h%uint32(len(pp.runq))].ptr()
		if atomic.CasRel(&pp.runqhead, h, h+1) { // cas-release, commits consume
			return gp, false
		}
	}
}

// runqdrain drains the local runnable queue of pp and returns all goroutines in it.
// Executed only by the owner P.
func runqdrain(pp *p) (drainQ gQueue, n uint32) {
	oldNext := pp.runnext
	if oldNext != 0 && pp.runnext.cas(oldNext, 0) {
		drainQ.pushBack(oldNext.ptr())
		n++
	}

retry:
	h := atomic.LoadAcq(&pp.runqhead) // load-acquire, synchronize with other consumers
	t := pp.runqtail
	qn := t - h
	if qn == 0 {
		return
	}
	if qn > uint32(len(pp.runq)) { // read inconsistent h and t
		goto retry
	}

	if !atomic.CasRel(&pp.runqhead, h, h+qn) { // cas-release, commits consume
		goto retry
	}

	// We've inverted the order in which it gets G's from the local P's runnable queue
	// and then advances the head pointer because we don't want to mess up the statuses of G's
	// while runqdrain() and runqsteal() are running in parallel.
	// Thus we should advance the head pointer before draining the local P into a gQueue,
	// so that we can update any gp.schedlink only after we take the full ownership of G,
	// meanwhile, other P's can't access to all G's in local P's runnable queue and steal them.
	// See https://groups.google.com/g/golang-dev/c/0pTKxEKhHSc/m/6Q85QjdVBQAJ for more details.
	for i := uint32(0); i < qn; i++ {
		gp := pp.runq[(h+i)%uint32(len(pp.runq))].ptr()
		drainQ.pushBack(gp)
		n++
	}
	return
}

// Grabs a batch of goroutines from pp's runnable queue into batch.
// Batch is a ring buffer starting at batchHead.
// Returns number of grabbed goroutines.
// Can be executed by any P.
func runqgrab(pp *p, batch *[256]guintptr, batchHead uint32, stealRunNextG bool) uint32 {
	for {
		h := atomic.LoadAcq(&pp.runqhead) // load-acquire, synchronize with other consumers
		t := atomic.LoadAcq(&pp.runqtail) // load-acquire, synchronize with the producer
		n := t - h
		n = n - n/2
		if n == 0 {
			if stealRunNextG {
				// Try to steal from pp.runnext.
				if next := pp.runnext; next != 0 {
					if pp.status == _Prunning {
						// Sleep to ensure that pp isn't about to run the g
						// we are about to steal.
						// The important use case here is when the g running
						// on pp ready()s another g and then almost
						// immediately blocks. Instead of stealing runnext
						// in this window, back off to give pp a chance to
						// schedule runnext. This will avoid thrashing gs
						// between different Ps.
						// A sync chan send/recv takes ~50ns as of time of
						// writing, so 3us gives ~50x overshoot.
						if !osHasLowResTimer {
							usleep(3)
						} else {
							// On some platforms system timer granularity is
							// 1-15ms, which is way too much for this
							// optimization. So just yield.
							osyield()
						}
					}
					if !pp.runnext.cas(next, 0) {
						continue
					}
					batch[batchHead%uint32(len(batch))] = next
					return 1
				}
			}
			return 0
		}
		if n > uint32(len(pp.runq)/2) { // read inconsistent h and t
			continue
		}
		for i := uint32(0); i < n; i++ {
			g := pp.runq[(h+i)%uint32(len(pp.runq))]
			batch[(batchHead+i)%uint32(len(batch))] = g
		}
		if atomic.CasRel(&pp.runqhead, h, h+n) { // cas-release, commits consume
			return n
		}
	}
}

// Steal half of elements from local runnable queue of p2
// and put onto local runnable queue of p.
// Returns one of the stolen elements (or nil if failed).
func runqsteal(pp, p2 *p, stealRunNextG bool) *g {
	t := pp.runqtail
	n := runqgrab(p2, &pp.runq, t, stealRunNextG)
	if n == 0 {
		return nil
	}
	n--
	gp := pp.runq[(t+n)%uint32(len(pp.runq))].ptr()
	if n == 0 {
		return gp
	}
	h := atomic.LoadAcq(&pp.runqhead) // load-acquire, synchronize with consumers
	if t-h+n >= uint32(len(pp.runq)) {
		throw("runqsteal: runq overflow")
	}
	atomic.StoreRel(&pp.runqtail, t+n) // store-release, makes the item available for consumption
	return gp
}

// A gQueue is a dequeue of Gs linked through g.schedlink. A G can only
// be on one gQueue or gList at a time.
type gQueue struct {
	head guintptr
	tail guintptr
}

// empty reports whether q is empty.
func (q *gQueue) empty() bool {
	return q.head == 0
}

// push adds gp to the head of q.
func (q *gQueue) push(gp *g) {
	gp.schedlink = q.head
	q.head.set(gp)
	if q.tail == 0 {
		q.tail.set(gp)
	}
}

// pushBack adds gp to the tail of q.
func (q *gQueue) pushBack(gp *g) {
	gp.schedlink = 0
	if q.tail != 0 {
		q.tail.ptr().schedlink.set(gp)
	} else {
		q.head.set(gp)
	}
	q.tail.set(gp)
}

// pushBackAll adds all Gs in q2 to the tail of q. After this q2 must
// not be used.
func (q *gQueue) pushBackAll(q2 gQueue) {
	if q2.tail == 0 {
		return
	}
	q2.tail.ptr().schedlink = 0
	if q.tail != 0 {
		q.tail.ptr().schedlink = q2.head
	} else {
		q.head = q2.head
	}
	q.tail = q2.tail
}

// pop removes and returns the head of queue q. It returns nil if
// q is empty.
func (q *gQueue) pop() *g {
	gp := q.head.ptr()
	if gp != nil {
		q.head = gp.schedlink
		if q.head == 0 {
			q.tail = 0
		}
	}
	return gp
}

// popList takes all Gs in q and returns them as a gList.
func (q *gQueue) popList() gList {
	stack := gList{q.head}
	*q = gQueue{}
	return stack
}

// A gList is a list of Gs linked through g.schedlink. A G can only be
// on one gQueue or gList at a time.
type gList struct {
	head guintptr
}

// empty reports whether l is empty.
func (l *gList) empty() bool {
	return l.head == 0
}

// push adds gp to the head of l.
func (l *gList) push(gp *g) {
	gp.schedlink = l.head
	l.head.set(gp)
}

// pushAll prepends all Gs in q to l.
func (l *gList) pushAll(q gQueue) {
	if !q.empty() {
		q.tail.ptr().schedlink = l.head
		l.head = q.head
	}
}

// pop removes and returns the head of l. If l is empty, it returns nil.
func (l *gList) pop() *g {
	gp := l.head.ptr()
	if gp != nil {
		l.head = gp.schedlink
	}
	return gp
}

//go:linkname setMaxThreads runtime/debug.setMaxThreads
func setMaxThreads(in int) (out int) {
	lock(&sched.lock)
	out = int(sched.maxmcount)
	if in > 0x7fffffff { // MaxInt32
		sched.maxmcount = 0x7fffffff
	} else {
		sched.maxmcount = int32(in)
	}
	checkmcount()
	unlock(&sched.lock)
	return
}

//go:nosplit
func procPin() int {
	gp := getg()
	mp := gp.m

	mp.locks++
	return int(mp.p.ptr().id)
}

//go:nosplit
func procUnpin() {
	gp := getg()
	gp.m.locks--
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

// Active spinning for sync.Mutex.
//
//go:linkname sync_runtime_canSpin sync.runtime_canSpin
//go:nosplit
func sync_runtime_canSpin(i int) bool {
	// sync.Mutex is cooperative, so we are conservative with spinning.
	// Spin only few times and only if running on a multicore machine and
	// GOMAXPROCS>1 and there is at least one other running P and local runq is empty.
	// As opposed to runtime mutex we don't do passive spinning here,
	// because there can be work on global runq or on other Ps.
	if i >= active_spin || ncpu <= 1 || gomaxprocs <= sched.npidle.Load()+sched.nmspinning.Load()+1 {
		return false
	}
	if p := getg().m.p.ptr(); !runqempty(p) {
		return false
	}
	return true
}

//go:linkname sync_runtime_doSpin sync.runtime_doSpin
//go:nosplit
func sync_runtime_doSpin() {
	procyield(active_spin_cnt)
}

var stealOrder randomOrder

// randomOrder/randomEnum are helper types for randomized work stealing.
// They allow to enumerate all Ps in different pseudo-random orders without repetitions.
// The algorithm is based on the fact that if we have X such that X and GOMAXPROCS
// are coprime, then a sequences of (i + X) % GOMAXPROCS gives the required enumeration.
type randomOrder struct {
	count    uint32
	coprimes []uint32
}

type randomEnum struct {
	i     uint32
	count uint32
	pos   uint32
	inc   uint32
}

func (ord *randomOrder) reset(count uint32) {
	ord.count = count
	ord.coprimes = ord.coprimes[:0]
	for i := uint32(1); i <= count; i++ {
		if gcd(i, count) == 1 {
			ord.coprimes = append(ord.coprimes, i)
		}
	}
}

func (ord *randomOrder) start(i uint32) randomEnum {
	return randomEnum{
		count: ord.count,
		pos:   i % ord.count,
		inc:   ord.coprimes[i/ord.count%uint32(len(ord.coprimes))],
	}
}

func (enum *randomEnum) done() bool {
	return enum.i == enum.count
}

func (enum *randomEnum) next() {
	enum.i++
	enum.pos = (enum.pos + enum.inc) % enum.count
}

func (enum *randomEnum) position() uint32 {
	return enum.pos
}

func gcd(a, b uint32) uint32 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// An initTask represents the set of initializations that need to be done for a package.
// Keep in sync with ../../test/noinit.go:initTask
type initTask struct {
	state uint32 // 0 = uninitialized, 1 = in progress, 2 = done
	nfns  uint32
	// followed by nfns pcs, uintptr sized, one per init function to run
}

// inittrace stores statistics for init functions which are
// updated by malloc and newproc when active is true.
var inittrace tracestat

type tracestat struct {
	active bool   // init tracing activation status
	id     uint64 // init goroutine id
	allocs uint64 // heap allocations
	bytes  uint64 // heap allocated bytes
}

func doInit(ts []*initTask) {
	for _, t := range ts {
		doInit1(t)
	}
}

func doInit1(t *initTask) {
	switch t.state {
	case 2: // fully initialized
		return
	case 1: // initialization in progress
		throw("recursive call during initialization - linker skew")
	default: // not initialized yet
		t.state = 1 // initialization in progress

		var (
			start  int64
			before tracestat
		)

		if inittrace.active {
			start = nanotime()
			// Load stats non-atomically since tracinit is updated only by this init goroutine.
			before = inittrace
		}

		if t.nfns == 0 {
			// We should have pruned all of these in the linker.
			throw("inittask with no functions")
		}

		firstFunc := add(unsafe.Pointer(t), 8)
		for i := uint32(0); i < t.nfns; i++ {
			p := add(firstFunc, uintptr(i)*goarch.PtrSize)
			f := *(*func())(unsafe.Pointer(&p))
			f()
		}

		if inittrace.active {
			end := nanotime()
			// Load stats non-atomically since tracinit is updated only by this init goroutine.
			after := inittrace

			f := *(*func())(unsafe.Pointer(&firstFunc))
			pkg := funcpkgpath(findfunc(abi.FuncPCABIInternal(f)))

			var sbuf [24]byte
			print("init ", pkg, " @")
			print(string(fmtNSAsMS(sbuf[:], uint64(start-runtimeInitTime))), " ms, ")
			print(string(fmtNSAsMS(sbuf[:], uint64(end-start))), " ms clock, ")
			print(string(itoa(sbuf[:], after.bytes-before.bytes)), " bytes, ")
			print(string(itoa(sbuf[:], after.allocs-before.allocs)), " allocs")
			print("\n")
		}

		t.state = 2 // initialization done
	}
}
