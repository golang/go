// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// This is not mechanically generated
// so be very careful and refer to runtime.h
// for the definitive enum.
const (
	gStatusidle = iota
	gStatusRunnable
	gStatusRunning
	gStatusSyscall
	gStatusWaiting
	gStatusMoribundUnused
	gStatusDead
	gStatusEnqueue
	gStatusCopystack
	gStatusScan         = 0x1000
	gStatusScanRunnable = gStatusScan + gStatusRunnable
	gStatusScanRunning  = gStatusScan + gStatusRunning
	gStatusScanSyscall  = gStatusScan + gStatusSyscall
	gStatusScanWaiting  = gStatusScan + gStatusWaiting
	gStatusScanEnqueue  = gStatusScan + gStatusEnqueue
)

var parkunlock_c byte

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

// Gosched yields the processor, allowing other goroutines to run.  It does not
// suspend the current goroutine, so execution resumes automatically.
func Gosched() {
	mcall(&gosched_m)
}

func readgStatus(gp *g) uint32 {
	//return atomic.LoadUint32(&gp.atomicstatus) // TODO: add bootstrap code to provide.
	return gp.atomicstatus
}

// Puts the current goroutine into a waiting state and calls unlockf.
// If unlockf returns false, the goroutine is resumed.
func gopark(unlockf unsafe.Pointer, lock unsafe.Pointer, reason string) {
	mp := acquirem()
	gp := mp.curg
	status := readgStatus(gp)
	if status != gStatusRunning && status != gStatusScanRunning {
		gothrow("gopark: bad g status")
	}
	mp.waitlock = lock
	mp.waitunlockf = unlockf
	gp.waitreason = reason
	releasem(mp)
	// can't do anything that might move the G between Ms here.
	mcall(&park_m)
}

// Puts the current goroutine into a waiting state and unlocks the lock.
// The goroutine can be made runnable again by calling goready(gp).
func goparkunlock(lock *mutex, reason string) {
	gopark(unsafe.Pointer(&parkunlock_c), unsafe.Pointer(lock), reason)
}

func goready(gp *g) {
	mp := acquirem()
	mp.ptrarg[0] = unsafe.Pointer(gp)
	onM(&ready_m)
	releasem(mp)
}

//go:nosplit
func acquireSudog() *sudog {
	c := gomcache()
	s := c.sudogcache
	if s != nil {
		c.sudogcache = s.next
		return s
	}
	return new(sudog)
}

//go:nosplit
func releaseSudog(s *sudog) {
	c := gomcache()
	s.next = c.sudogcache
	c.sudogcache = s
}

// funcPC returns the entry PC of the function f.
// It assumes that f is a func value. Otherwise the behavior is undefined.
func funcPC(f interface{}) uintptr {
	return **(**uintptr)(add(unsafe.Pointer(&f), ptrSize))
}
