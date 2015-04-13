// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

//go:linkname runtime_init runtime.init
func runtime_init()

//go:linkname main_init main.init
func main_init()

// main_init_done is a signal used by cgocallbackg that initialization
// has been completed. It is made before _cgo_notify_runtime_init_done,
// so all cgo calls can rely on it existing. When main_init is complete,
// it is closed, meaning cgocallbackg can reliably receive from it.
var main_init_done chan bool

//go:linkname main_main main.main
func main_main()

// runtimeInitTime is the nanotime() at which the runtime started.
var runtimeInitTime int64

// The main goroutine.
func main() {
	g := getg()

	// Racectx of m0->g0 is used only as the parent of the main goroutine.
	// It must not be used for anything else.
	g.m.g0.racectx = 0

	// Max stack size is 1 GB on 64-bit, 250 MB on 32-bit.
	// Using decimal instead of binary GB and MB because
	// they look nicer in the stack overflow failure message.
	if ptrSize == 8 {
		maxstacksize = 1000000000
	} else {
		maxstacksize = 250000000
	}

	// Record when the world started.
	runtimeInitTime = nanotime()

	systemstack(func() {
		newm(sysmon, nil)
	})

	// Lock the main goroutine onto this, the main OS thread,
	// during initialization.  Most programs won't care, but a few
	// do require certain calls to be made by the main thread.
	// Those can arrange for main.main to run in the main thread
	// by calling runtime.LockOSThread during initialization
	// to preserve the lock.
	lockOSThread()

	if g.m != &m0 {
		throw("runtime.main not on m0")
	}

	runtime_init() // must be before defer

	// Defer unlock so that runtime.Goexit during init does the unlock too.
	needUnlock := true
	defer func() {
		if needUnlock {
			unlockOSThread()
		}
	}()

	gcenable()

	if islibrary {
		// Allocate new M as main_main() is expected to block forever.
		systemstack(newextram)
	}
	main_init_done = make(chan bool)
	if iscgo {
		if _cgo_thread_start == nil {
			throw("_cgo_thread_start missing")
		}
		if _cgo_malloc == nil {
			throw("_cgo_malloc missing")
		}
		if _cgo_free == nil {
			throw("_cgo_free missing")
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
		cgocall(_cgo_notify_runtime_init_done, nil)
	}

	main_init()
	close(main_init_done)

	needUnlock = false
	unlockOSThread()

	if isarchive {
		// A program compiled with -buildmode=c-archive has a main,
		// but it is not executed.
		return
	}
	main_main()
	if raceenabled {
		racefini()
	}

	// Make racy client program work: if panicking on
	// another goroutine at the same time as main returns,
	// let the other goroutine finish printing the panic trace.
	// Once it does, it will exit. See issue 3934.
	if panicking != 0 {
		gopark(nil, nil, "panicwait", traceEvGoStop, 1)
	}

	exit(0)
	for {
		var x *int32
		*x = 0
	}
}

// os_beforeExit is called from os.Exit(0).
//go:linkname os_beforeExit os.runtime_beforeExit
func os_beforeExit() {
	if raceenabled {
		racefini()
	}
}

// start forcegc helper goroutine
func init() {
	go forcegchelper()
}

func forcegchelper() {
	forcegc.g = getg()
	for {
		lock(&forcegc.lock)
		if forcegc.idle != 0 {
			throw("forcegc: phase error")
		}
		atomicstore(&forcegc.idle, 1)
		goparkunlock(&forcegc.lock, "force gc (idle)", traceEvGoBlock, 1)
		// this goroutine is explicitly resumed by sysmon
		if debug.gctrace > 0 {
			println("GC forced")
		}
		startGC(gcForceMode)
	}
}

//go:nosplit

// Gosched yields the processor, allowing other goroutines to run.  It does not
// suspend the current goroutine, so execution resumes automatically.
func Gosched() {
	mcall(gosched_m)
}

// Puts the current goroutine into a waiting state and calls unlockf.
// If unlockf returns false, the goroutine is resumed.
func gopark(unlockf func(*g, unsafe.Pointer) bool, lock unsafe.Pointer, reason string, traceEv byte, traceskip int) {
	mp := acquirem()
	gp := mp.curg
	status := readgstatus(gp)
	if status != _Grunning && status != _Gscanrunning {
		throw("gopark: bad g status")
	}
	mp.waitlock = lock
	mp.waitunlockf = *(*unsafe.Pointer)(unsafe.Pointer(&unlockf))
	gp.waitreason = reason
	mp.waittraceev = traceEv
	mp.waittraceskip = traceskip
	releasem(mp)
	// can't do anything that might move the G between Ms here.
	mcall(park_m)
}

// Puts the current goroutine into a waiting state and unlocks the lock.
// The goroutine can be made runnable again by calling goready(gp).
func goparkunlock(lock *mutex, reason string, traceEv byte, traceskip int) {
	gopark(parkunlock_c, unsafe.Pointer(lock), reason, traceEv, traceskip)
}

func goready(gp *g, traceskip int) {
	systemstack(func() {
		ready(gp, traceskip)
	})
}

//go:nosplit
func acquireSudog() *sudog {
	// Delicate dance: the semaphore implementation calls
	// acquireSudog, acquireSudog calls new(sudog),
	// new calls malloc, malloc can call the garbage collector,
	// and the garbage collector calls the semaphore implementation
	// in stoptheworld.
	// Break the cycle by doing acquirem/releasem around new(sudog).
	// The acquirem/releasem increments m.locks during new(sudog),
	// which keeps the garbage collector from being invoked.
	mp := acquirem()
	pp := mp.p
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
	if s.selectdone != nil {
		throw("runtime: sudog with non-nil selectdone")
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
	gp := getg()
	if gp.param != nil {
		throw("runtime: releaseSudog with non-nil gp.param")
	}
	mp := acquirem() // avoid rescheduling to another P
	pp := mp.p
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

// funcPC returns the entry PC of the function f.
// It assumes that f is a func value. Otherwise the behavior is undefined.
//go:nosplit
func funcPC(f interface{}) uintptr {
	return **(**uintptr)(add(unsafe.Pointer(&f), ptrSize))
}

// called from assembly
func badmcall(fn func(*g)) {
	throw("runtime: mcall called on m->g0 stack")
}

func badmcall2(fn func(*g)) {
	throw("runtime: mcall function returned")
}

func badreflectcall() {
	panic("runtime: arg size to reflect.call more than 1GB")
}

func lockedOSThread() bool {
	gp := getg()
	return gp.lockedm != nil && gp.m.lockedg != nil
}

var (
	allgs    []*g
	allglock mutex
)

func allgadd(gp *g) {
	if readgstatus(gp) == _Gidle {
		throw("allgadd: bad status Gidle")
	}

	lock(&allglock)
	allgs = append(allgs, gp)
	allg = &allgs[0]
	allglen = uintptr(len(allgs))
	unlock(&allglock)
}
