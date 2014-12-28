// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin nacl netbsd openbsd plan9 solaris windows

package runtime

import "unsafe"

// This implementation depends on OS-specific implementations of
//
//	uintptr runtime·semacreate(void)
//		Create a semaphore, which will be assigned to m->waitsema.
//		The zero value is treated as absence of any semaphore,
//		so be sure to return a non-zero value.
//
//	int32 runtime·semasleep(int64 ns)
//		If ns < 0, acquire m->waitsema and return 0.
//		If ns >= 0, try to acquire m->waitsema for at most ns nanoseconds.
//		Return 0 if the semaphore was acquired, -1 if interrupted or timed out.
//
//	int32 runtime·semawakeup(M *mp)
//		Wake up mp, which is or will soon be sleeping on mp->waitsema.
//
const (
	locked uintptr = 1

	active_spin     = 4
	active_spin_cnt = 30
	passive_spin    = 1
)

func lock(l *mutex) {
	gp := getg()
	if gp.m.locks < 0 {
		throw("runtime·lock: lock count")
	}
	gp.m.locks++

	// Speculative grab for lock.
	if casuintptr(&l.key, 0, locked) {
		return
	}
	if gp.m.waitsema == 0 {
		gp.m.waitsema = semacreate()
	}

	// On uniprocessor's, no point spinning.
	// On multiprocessors, spin for ACTIVE_SPIN attempts.
	spin := 0
	if ncpu > 1 {
		spin = active_spin
	}
Loop:
	for i := 0; ; i++ {
		v := atomicloaduintptr(&l.key)
		if v&locked == 0 {
			// Unlocked. Try to lock.
			if casuintptr(&l.key, v, v|locked) {
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
			// l->waitm points to a linked list of M's waiting
			// for this lock, chained through m->nextwaitm.
			// Queue this M.
			for {
				gp.m.nextwaitm = (*m)((unsafe.Pointer)(v &^ locked))
				if casuintptr(&l.key, v, uintptr(unsafe.Pointer(gp.m))|locked) {
					break
				}
				v = atomicloaduintptr(&l.key)
				if v&locked == 0 {
					continue Loop
				}
			}
			if v&locked != 0 {
				// Queued.  Wait.
				semasleep(-1)
				i = 0
			}
		}
	}
}

func unlock(l *mutex) {
	gp := getg()
	var mp *m
	for {
		v := atomicloaduintptr(&l.key)
		if v == locked {
			if casuintptr(&l.key, locked, 0) {
				break
			}
		} else {
			// Other M's are waiting for the lock.
			// Dequeue an M.
			mp = (*m)((unsafe.Pointer)(v &^ locked))
			if casuintptr(&l.key, v, uintptr(unsafe.Pointer(mp.nextwaitm))) {
				// Dequeued an M.  Wake it.
				semawakeup(mp)
				break
			}
		}
	}
	gp.m.locks--
	if gp.m.locks < 0 {
		throw("runtime·unlock: lock count")
	}
	if gp.m.locks == 0 && gp.preempt { // restore the preemption request in case we've cleared it in newstack
		gp.stackguard0 = stackPreempt
	}
}

// One-time notifications.
func noteclear(n *note) {
	n.key = 0
}

func notewakeup(n *note) {
	var v uintptr
	for {
		v = atomicloaduintptr(&n.key)
		if casuintptr(&n.key, v, locked) {
			break
		}
	}

	// Successfully set waitm to locked.
	// What was it before?
	switch {
	case v == 0:
		// Nothing was waiting. Done.
	case v == locked:
		// Two notewakeups!  Not allowed.
		throw("notewakeup - double wakeup")
	default:
		// Must be the waiting m.  Wake it up.
		semawakeup((*m)(unsafe.Pointer(v)))
	}
}

func notesleep(n *note) {
	gp := getg()
	if gp != gp.m.g0 {
		throw("notesleep not on g0")
	}
	if gp.m.waitsema == 0 {
		gp.m.waitsema = semacreate()
	}
	if !casuintptr(&n.key, 0, uintptr(unsafe.Pointer(gp.m))) {
		// Must be locked (got wakeup).
		if n.key != locked {
			throw("notesleep - waitm out of sync")
		}
		return
	}
	// Queued.  Sleep.
	gp.m.blocked = true
	semasleep(-1)
	gp.m.blocked = false
}

//go:nosplit
func notetsleep_internal(n *note, ns int64, gp *g, deadline int64) bool {
	// gp and deadline are logically local variables, but they are written
	// as parameters so that the stack space they require is charged
	// to the caller.
	// This reduces the nosplit footprint of notetsleep_internal.
	gp = getg()

	// Register for wakeup on n->waitm.
	if !casuintptr(&n.key, 0, uintptr(unsafe.Pointer(gp.m))) {
		// Must be locked (got wakeup).
		if n.key != locked {
			throw("notetsleep - waitm out of sync")
		}
		return true
	}
	if ns < 0 {
		// Queued.  Sleep.
		gp.m.blocked = true
		semasleep(-1)
		gp.m.blocked = false
		return true
	}

	deadline = nanotime() + ns
	for {
		// Registered.  Sleep.
		gp.m.blocked = true
		if semasleep(ns) >= 0 {
			gp.m.blocked = false
			// Acquired semaphore, semawakeup unregistered us.
			// Done.
			return true
		}
		gp.m.blocked = false
		// Interrupted or timed out.  Still registered.  Semaphore not acquired.
		ns = deadline - nanotime()
		if ns <= 0 {
			break
		}
		// Deadline hasn't arrived.  Keep sleeping.
	}

	// Deadline arrived.  Still registered.  Semaphore not acquired.
	// Want to give up and return, but have to unregister first,
	// so that any notewakeup racing with the return does not
	// try to grant us the semaphore when we don't expect it.
	for {
		v := atomicloaduintptr(&n.key)
		switch v {
		case uintptr(unsafe.Pointer(gp.m)):
			// No wakeup yet; unregister if possible.
			if casuintptr(&n.key, v, 0) {
				return false
			}
		case locked:
			// Wakeup happened so semaphore is available.
			// Grab it to avoid getting out of sync.
			gp.m.blocked = true
			if semasleep(-1) < 0 {
				throw("runtime: unable to acquire - semaphore out of sync")
			}
			gp.m.blocked = false
			return true
		default:
			throw("runtime: unexpected waitm - semaphore out of sync")
		}
	}
}

func notetsleep(n *note, ns int64) bool {
	gp := getg()
	if gp != gp.m.g0 && gp.m.gcing == 0 {
		throw("notetsleep not on g0")
	}
	if gp.m.waitsema == 0 {
		gp.m.waitsema = semacreate()
	}
	return notetsleep_internal(n, ns, nil, 0)
}

// same as runtime·notetsleep, but called on user g (not g0)
// calls only nosplit functions between entersyscallblock/exitsyscall
func notetsleepg(n *note, ns int64) bool {
	gp := getg()
	if gp == gp.m.g0 {
		throw("notetsleepg on g0")
	}
	if gp.m.waitsema == 0 {
		gp.m.waitsema = semacreate()
	}
	entersyscallblock(0)
	ok := notetsleep_internal(n, ns, nil, 0)
	exitsyscall(0)
	return ok
}
