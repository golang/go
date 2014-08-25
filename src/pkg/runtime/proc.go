// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	gStatusidle = iota
	gStatusRunnable
	gStatusRunning
	gStatusSyscall
	gStatusWaiting
	gStatusMoribundUnused
	gStatusDead
)

var parkunlock_c byte

// Gosched yields the processor, allowing other goroutines to run.  It does not
// suspend the current goroutine, so execution resumes automatically.
func Gosched() {
	mcall(&gosched_m)
}

// Puts the current goroutine into a waiting state and calls unlockf.
// If unlockf returns false, the goroutine is resumed.
func gopark(unlockf unsafe.Pointer, lock unsafe.Pointer, reason string) {
	mp := acquirem()
	gp := mp.curg
	if gp.status != gStatusRunning {
		gothrow("gopark: bad g status")
	}
	mp.waitlock = lock
	mp.waitunlockf = *(*func(*g, unsafe.Pointer) uint8)(unsafe.Pointer(&unlockf))
	gp.waitreason = reason
	releasem(mp)
	// can't do anything that might move the G between Ms here.
	mcall(&park_m)
}

// Puts the current goroutine into a waiting state and unlocks the lock.
// The goroutine can be made runnable again by calling goready(gp).
func goparkunlock(lock *lock, reason string) {
	gopark(unsafe.Pointer(&parkunlock_c), unsafe.Pointer(lock), reason)
}

func goready(gp *g) {
	mp := acquirem()
	mp.ptrarg[0] = unsafe.Pointer(gp)
	onM(&ready_m)
	releasem(mp)
}

func goblockevent(cycles int64, skip int32) {
	// TODO: convert to Go when we do mprof.goc
	mp := acquirem()
	mp.scalararg[0] = uint(uint32(cycles))
	mp.scalararg[1] = uint(cycles >> 32)
	mp.scalararg[2] = uint(skip)
	onM(&blockevent_m)
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
