// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (aix || darwin || netbsd || openbsd || plan9 || solaris || windows) && !goexperiment.spinbitmutex

package runtime

import (
	"internal/runtime/atomic"
	"unsafe"
)

// This implementation depends on OS-specific implementations of
//
//	func semacreate(mp *m)
//		Create a semaphore for mp, if it does not already have one.
//
//	func semasleep(ns int64) int32
//		If ns < 0, acquire m's semaphore and return 0.
//		If ns >= 0, try to acquire m's semaphore for at most ns nanoseconds.
//		Return 0 if the semaphore was acquired, -1 if interrupted or timed out.
//
//	func semawakeup(mp *m)
//		Wake up mp, which is or will soon be sleeping on its semaphore.
const (
	active_spin     = 4
	active_spin_cnt = 30
	passive_spin    = 1
)

// mWaitList is part of the M struct, and holds the list of Ms that are waiting
// for a particular runtime.mutex.
//
// When an M is unable to immediately obtain a lock, it adds itself to the list
// of Ms waiting for the lock. It does that via this struct's next field,
// forming a singly-linked list with the mutex's key field pointing to the head
// of the list.
type mWaitList struct {
	next muintptr // next m waiting for lock
}

func lockVerifyMSize() {}

func mutexContended(l *mutex) bool {
	return atomic.Loaduintptr(&l.key) > locked
}

func lock(l *mutex) {
	lockWithRank(l, getLockRank(l))
}

func lock2(l *mutex) {
	gp := getg()
	if gp.m.locks < 0 {
		throw("runtime·lock: lock count")
	}
	gp.m.locks++

	// Speculative grab for lock.
	if atomic.Casuintptr(&l.key, 0, locked) {
		return
	}
	semacreate(gp.m)

	timer := &lockTimer{lock: l}
	timer.begin()
	// On uniprocessor's, no point spinning.
	// On multiprocessors, spin for ACTIVE_SPIN attempts.
	spin := 0
	if ncpu > 1 {
		spin = active_spin
	}
Loop:
	for i := 0; ; i++ {
		v := atomic.Loaduintptr(&l.key)
		if v&locked == 0 {
			// Unlocked. Try to lock.
			if atomic.Casuintptr(&l.key, v, v|locked) {
				timer.end()
				return
			}
			i = 0
		}
		if i < spin {
			procyield(active_spin_cnt)
		} else if i < spin+passive_spin {
			osyield()
		} else {
			// Someone else has it.
			// l.key points to a linked list of M's waiting
			// for this lock, chained through m.mWaitList.next.
			// Queue this M.
			for {
				gp.m.mWaitList.next = muintptr(v &^ locked)
				if atomic.Casuintptr(&l.key, v, uintptr(unsafe.Pointer(gp.m))|locked) {
					break
				}
				v = atomic.Loaduintptr(&l.key)
				if v&locked == 0 {
					continue Loop
				}
			}
			if v&locked != 0 {
				// Queued. Wait.
				semasleep(-1)
				i = 0
			}
		}
	}
}

func unlock(l *mutex) {
	unlockWithRank(l)
}

// We might not be holding a p in this code.
//
//go:nowritebarrier
func unlock2(l *mutex) {
	gp := getg()
	var mp *m
	for {
		v := atomic.Loaduintptr(&l.key)
		if v == locked {
			if atomic.Casuintptr(&l.key, locked, 0) {
				break
			}
		} else {
			// Other M's are waiting for the lock.
			// Dequeue an M.
			mp = muintptr(v &^ locked).ptr()
			if atomic.Casuintptr(&l.key, v, uintptr(mp.mWaitList.next)) {
				// Dequeued an M.  Wake it.
				semawakeup(mp) // no use of mp after this point; it's awake
				break
			}
		}
	}
	gp.m.mLockProfile.recordUnlock(l)
	gp.m.locks--
	if gp.m.locks < 0 {
		throw("runtime·unlock: lock count")
	}
	if gp.m.locks == 0 && gp.preempt { // restore the preemption request in case we've cleared it in newstack
		gp.stackguard0 = stackPreempt
	}
}
