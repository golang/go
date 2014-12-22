// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

//go:linkname runtime_init runtime.init
func runtime_init()

//go:linkname main_init main.init
func main_init()

//go:linkname main_main main.main
func main_main()

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

	systemstack(newsysmon)

	// Lock the main goroutine onto this, the main OS thread,
	// during initialization.  Most programs won't care, but a few
	// do require certain calls to be made by the main thread.
	// Those can arrange for main.main to run in the main thread
	// by calling runtime.LockOSThread during initialization
	// to preserve the lock.
	lockOSThread()

	if g.m != &m0 {
		gothrow("runtime.main not on m0")
	}

	runtime_init() // must be before defer

	// Defer unlock so that runtime.Goexit during init does the unlock too.
	needUnlock := true
	defer func() {
		if needUnlock {
			unlockOSThread()
		}
	}()

	memstats.enablegc = true // now that runtime is initialized, GC is okay

	if iscgo {
		if _cgo_thread_start == nil {
			gothrow("_cgo_thread_start missing")
		}
		if _cgo_malloc == nil {
			gothrow("_cgo_malloc missing")
		}
		if _cgo_free == nil {
			gothrow("_cgo_free missing")
		}
		if GOOS != "windows" {
			if _cgo_setenv == nil {
				gothrow("_cgo_setenv missing")
			}
			if _cgo_unsetenv == nil {
				gothrow("_cgo_unsetenv missing")
			}
		}
	}

	main_init()

	needUnlock = false
	unlockOSThread()

	main_main()
	if raceenabled {
		racefini()
	}

	// Make racy client program work: if panicking on
	// another goroutine at the same time as main returns,
	// let the other goroutine finish printing the panic trace.
	// Once it does, it will exit. See issue 3934.
	if panicking != 0 {
		gopark(nil, nil, "panicwait")
	}

	exit(0)
	for {
		var x *int32
		*x = 0
	}
}

// start forcegc helper goroutine
func init() {
	go forcegchelper()
}

func forcegchelper() {
	forcegc.g = getg()
	forcegc.g.issystem = true
	for {
		lock(&forcegc.lock)
		if forcegc.idle != 0 {
			gothrow("forcegc: phase error")
		}
		atomicstore(&forcegc.idle, 1)
		goparkunlock(&forcegc.lock, "force gc (idle)")
		// this goroutine is explicitly resumed by sysmon
		if debug.gctrace > 0 {
			println("GC forced")
		}
		gogc(1)
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
func gopark(unlockf func(*g, unsafe.Pointer) bool, lock unsafe.Pointer, reason string) {
	mp := acquirem()
	gp := mp.curg
	status := readgstatus(gp)
	if status != _Grunning && status != _Gscanrunning {
		gothrow("gopark: bad g status")
	}
	mp.waitlock = lock
	mp.waitunlockf = *(*unsafe.Pointer)(unsafe.Pointer(&unlockf))
	gp.waitreason = reason
	releasem(mp)
	// can't do anything that might move the G between Ms here.
	mcall(park_m)
}

// Puts the current goroutine into a waiting state and unlocks the lock.
// The goroutine can be made runnable again by calling goready(gp).
func goparkunlock(lock *mutex, reason string) {
	gopark(parkunlock_c, unsafe.Pointer(lock), reason)
}

func goready(gp *g) {
	systemstack(func() {
		ready(gp)
	})
}

//go:nosplit
func acquireSudog() *sudog {
	c := gomcache()
	s := c.sudogcache
	if s != nil {
		if s.elem != nil {
			gothrow("acquireSudog: found s.elem != nil in cache")
		}
		c.sudogcache = s.next
		s.next = nil
		return s
	}

	// Delicate dance: the semaphore implementation calls
	// acquireSudog, acquireSudog calls new(sudog),
	// new calls malloc, malloc can call the garbage collector,
	// and the garbage collector calls the semaphore implementation
	// in stoptheworld.
	// Break the cycle by doing acquirem/releasem around new(sudog).
	// The acquirem/releasem increments m.locks during new(sudog),
	// which keeps the garbage collector from being invoked.
	mp := acquirem()
	p := new(sudog)
	if p.elem != nil {
		gothrow("acquireSudog: found p.elem != nil after new")
	}
	releasem(mp)
	return p
}

//go:nosplit
func releaseSudog(s *sudog) {
	if s.elem != nil {
		gothrow("runtime: sudog with non-nil elem")
	}
	if s.selectdone != nil {
		gothrow("runtime: sudog with non-nil selectdone")
	}
	if s.next != nil {
		gothrow("runtime: sudog with non-nil next")
	}
	if s.prev != nil {
		gothrow("runtime: sudog with non-nil prev")
	}
	if s.waitlink != nil {
		gothrow("runtime: sudog with non-nil waitlink")
	}
	gp := getg()
	if gp.param != nil {
		gothrow("runtime: releaseSudog with non-nil gp.param")
	}
	c := gomcache()
	s.next = c.sudogcache
	c.sudogcache = s
}

// funcPC returns the entry PC of the function f.
// It assumes that f is a func value. Otherwise the behavior is undefined.
//go:nosplit
func funcPC(f interface{}) uintptr {
	return **(**uintptr)(add(unsafe.Pointer(&f), ptrSize))
}

// called from assembly
func badmcall(fn func(*g)) {
	gothrow("runtime: mcall called on m->g0 stack")
}

func badmcall2(fn func(*g)) {
	gothrow("runtime: mcall function returned")
}

func badreflectcall() {
	panic("runtime: arg size to reflect.call more than 1GB")
}

func lockedOSThread() bool {
	gp := getg()
	return gp.lockedm != nil && gp.m.lockedg != nil
}

func newP() *p {
	return new(p)
}

func newM() *m {
	return new(m)
}

func newG() *g {
	return new(g)
}

var (
	allgs    []*g
	allglock mutex
)

func allgadd(gp *g) {
	if readgstatus(gp) == _Gidle {
		gothrow("allgadd: bad status Gidle")
	}

	lock(&allglock)
	allgs = append(allgs, gp)
	allg = &allgs[0]
	allglen = uintptr(len(allgs))
	unlock(&allglock)
}
