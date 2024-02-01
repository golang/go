// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.staticlockranking

package runtime

import (
	"internal/runtime/atomic"
	"unsafe"
)

const staticLockRanking = true

// worldIsStopped is accessed atomically to track world-stops. 1 == world
// stopped.
var worldIsStopped atomic.Uint32

// lockRankStruct is embedded in mutex
type lockRankStruct struct {
	// static lock ranking of the lock
	rank lockRank
	// pad field to make sure lockRankStruct is a multiple of 8 bytes, even on
	// 32-bit systems.
	pad int
}

// lockInit(l *mutex, rank int) sets the rank of lock before it is used.
// If there is no clear place to initialize a lock, then the rank of a lock can be
// specified during the lock call itself via lockWithRank(l *mutex, rank int).
func lockInit(l *mutex, rank lockRank) {
	l.rank = rank
}

func getLockRank(l *mutex) lockRank {
	return l.rank
}

// lockWithRank is like lock(l), but allows the caller to specify a lock rank
// when acquiring a non-static lock.
//
// Note that we need to be careful about stack splits:
//
// This function is not nosplit, thus it may split at function entry. This may
// introduce a new edge in the lock order, but it is no different from any
// other (nosplit) call before this call (including the call to lock() itself).
//
// However, we switch to the systemstack to record the lock held to ensure that
// we record an accurate lock ordering. e.g., without systemstack, a stack
// split on entry to lock2() would record stack split locks as taken after l,
// even though l is not actually locked yet.
func lockWithRank(l *mutex, rank lockRank) {
	if l == &debuglock || l == &paniclk || l == &raceFiniLock {
		// debuglock is only used for println/printlock(). Don't do lock
		// rank recording for it, since print/println are used when
		// printing out a lock ordering problem below.
		//
		// paniclk is only used for fatal throw/panic. Don't do lock
		// ranking recording for it, since we throw after reporting a
		// lock ordering problem. Additionally, paniclk may be taken
		// after effectively any lock (anywhere we might panic), which
		// the partial order doesn't cover.
		//
		// raceFiniLock is held while exiting when running
		// the race detector. Don't do lock rank recording for it,
		// since we are exiting.
		lock2(l)
		return
	}
	if rank == 0 {
		rank = lockRankLeafRank
	}
	gp := getg()
	// Log the new class.
	systemstack(func() {
		i := gp.m.locksHeldLen
		if i >= len(gp.m.locksHeld) {
			throw("too many locks held concurrently for rank checking")
		}
		gp.m.locksHeld[i].rank = rank
		gp.m.locksHeld[i].lockAddr = uintptr(unsafe.Pointer(l))
		gp.m.locksHeldLen++

		// i is the index of the lock being acquired
		if i > 0 {
			checkRanks(gp, gp.m.locksHeld[i-1].rank, rank)
		}
		lock2(l)
	})
}

// nosplit to ensure it can be called in as many contexts as possible.
//
//go:nosplit
func printHeldLocks(gp *g) {
	if gp.m.locksHeldLen == 0 {
		println("<none>")
		return
	}

	for j, held := range gp.m.locksHeld[:gp.m.locksHeldLen] {
		println(j, ":", held.rank.String(), held.rank, unsafe.Pointer(gp.m.locksHeld[j].lockAddr))
	}
}

// acquireLockRank acquires a rank which is not associated with a mutex lock
//
// This function may be called in nosplit context and thus must be nosplit.
//
//go:nosplit
func acquireLockRank(rank lockRank) {
	gp := getg()
	// Log the new class. See comment on lockWithRank.
	systemstack(func() {
		i := gp.m.locksHeldLen
		if i >= len(gp.m.locksHeld) {
			throw("too many locks held concurrently for rank checking")
		}
		gp.m.locksHeld[i].rank = rank
		gp.m.locksHeld[i].lockAddr = 0
		gp.m.locksHeldLen++

		// i is the index of the lock being acquired
		if i > 0 {
			checkRanks(gp, gp.m.locksHeld[i-1].rank, rank)
		}
	})
}

// checkRanks checks if goroutine g, which has mostly recently acquired a lock
// with rank 'prevRank', can now acquire a lock with rank 'rank'.
//
//go:systemstack
func checkRanks(gp *g, prevRank, rank lockRank) {
	rankOK := false
	if rank < prevRank {
		// If rank < prevRank, then we definitely have a rank error
		rankOK = false
	} else if rank == lockRankLeafRank {
		// If new lock is a leaf lock, then the preceding lock can
		// be anything except another leaf lock.
		rankOK = prevRank < lockRankLeafRank
	} else {
		// We've now verified the total lock ranking, but we
		// also enforce the partial ordering specified by
		// lockPartialOrder as well. Two locks with the same rank
		// can only be acquired at the same time if explicitly
		// listed in the lockPartialOrder table.
		list := lockPartialOrder[rank]
		for _, entry := range list {
			if entry == prevRank {
				rankOK = true
				break
			}
		}
	}
	if !rankOK {
		printlock()
		println(gp.m.procid, " ======")
		printHeldLocks(gp)
		throw("lock ordering problem")
	}
}

// See comment on lockWithRank regarding stack splitting.
func unlockWithRank(l *mutex) {
	if l == &debuglock || l == &paniclk || l == &raceFiniLock {
		// See comment at beginning of lockWithRank.
		unlock2(l)
		return
	}
	gp := getg()
	systemstack(func() {
		found := false
		for i := gp.m.locksHeldLen - 1; i >= 0; i-- {
			if gp.m.locksHeld[i].lockAddr == uintptr(unsafe.Pointer(l)) {
				found = true
				copy(gp.m.locksHeld[i:gp.m.locksHeldLen-1], gp.m.locksHeld[i+1:gp.m.locksHeldLen])
				gp.m.locksHeldLen--
				break
			}
		}
		if !found {
			println(gp.m.procid, ":", l.rank.String(), l.rank, l)
			throw("unlock without matching lock acquire")
		}
		unlock2(l)
	})
}

// releaseLockRank releases a rank which is not associated with a mutex lock
//
// This function may be called in nosplit context and thus must be nosplit.
//
//go:nosplit
func releaseLockRank(rank lockRank) {
	gp := getg()
	systemstack(func() {
		found := false
		for i := gp.m.locksHeldLen - 1; i >= 0; i-- {
			if gp.m.locksHeld[i].rank == rank && gp.m.locksHeld[i].lockAddr == 0 {
				found = true
				copy(gp.m.locksHeld[i:gp.m.locksHeldLen-1], gp.m.locksHeld[i+1:gp.m.locksHeldLen])
				gp.m.locksHeldLen--
				break
			}
		}
		if !found {
			println(gp.m.procid, ":", rank.String(), rank)
			throw("lockRank release without matching lockRank acquire")
		}
	})
}

// nosplit because it may be called from nosplit contexts.
//
//go:nosplit
func lockWithRankMayAcquire(l *mutex, rank lockRank) {
	gp := getg()
	if gp.m.locksHeldLen == 0 {
		// No possibility of lock ordering problem if no other locks held
		return
	}

	systemstack(func() {
		i := gp.m.locksHeldLen
		if i >= len(gp.m.locksHeld) {
			throw("too many locks held concurrently for rank checking")
		}
		// Temporarily add this lock to the locksHeld list, so
		// checkRanks() will print out list, including this lock, if there
		// is a lock ordering problem.
		gp.m.locksHeld[i].rank = rank
		gp.m.locksHeld[i].lockAddr = uintptr(unsafe.Pointer(l))
		gp.m.locksHeldLen++
		checkRanks(gp, gp.m.locksHeld[i-1].rank, rank)
		gp.m.locksHeldLen--
	})
}

// nosplit to ensure it can be called in as many contexts as possible.
//
//go:nosplit
func checkLockHeld(gp *g, l *mutex) bool {
	for i := gp.m.locksHeldLen - 1; i >= 0; i-- {
		if gp.m.locksHeld[i].lockAddr == uintptr(unsafe.Pointer(l)) {
			return true
		}
	}
	return false
}

// assertLockHeld throws if l is not held by the caller.
//
// nosplit to ensure it can be called in as many contexts as possible.
//
//go:nosplit
func assertLockHeld(l *mutex) {
	gp := getg()

	held := checkLockHeld(gp, l)
	if held {
		return
	}

	// Crash from system stack to avoid splits that may cause
	// additional issues.
	systemstack(func() {
		printlock()
		print("caller requires lock ", l, " (rank ", l.rank.String(), "), holding:\n")
		printHeldLocks(gp)
		throw("not holding required lock!")
	})
}

// assertRankHeld throws if a mutex with rank r is not held by the caller.
//
// This is less precise than assertLockHeld, but can be used in places where a
// pointer to the exact mutex is not available.
//
// nosplit to ensure it can be called in as many contexts as possible.
//
//go:nosplit
func assertRankHeld(r lockRank) {
	gp := getg()

	for i := gp.m.locksHeldLen - 1; i >= 0; i-- {
		if gp.m.locksHeld[i].rank == r {
			return
		}
	}

	// Crash from system stack to avoid splits that may cause
	// additional issues.
	systemstack(func() {
		printlock()
		print("caller requires lock with rank ", r.String(), "), holding:\n")
		printHeldLocks(gp)
		throw("not holding required lock!")
	})
}

// worldStopped notes that the world is stopped.
//
// Caller must hold worldsema.
//
// nosplit to ensure it can be called in as many contexts as possible.
//
//go:nosplit
func worldStopped() {
	if stopped := worldIsStopped.Add(1); stopped != 1 {
		systemstack(func() {
			print("world stop count=", stopped, "\n")
			throw("recursive world stop")
		})
	}
}

// worldStarted that the world is starting.
//
// Caller must hold worldsema.
//
// nosplit to ensure it can be called in as many contexts as possible.
//
//go:nosplit
func worldStarted() {
	if stopped := worldIsStopped.Add(-1); stopped != 0 {
		systemstack(func() {
			print("world stop count=", stopped, "\n")
			throw("released non-stopped world stop")
		})
	}
}

// nosplit to ensure it can be called in as many contexts as possible.
//
//go:nosplit
func checkWorldStopped() bool {
	stopped := worldIsStopped.Load()
	if stopped > 1 {
		systemstack(func() {
			print("inconsistent world stop count=", stopped, "\n")
			throw("inconsistent world stop count")
		})
	}

	return stopped == 1
}

// assertWorldStopped throws if the world is not stopped. It does not check
// which M stopped the world.
//
// nosplit to ensure it can be called in as many contexts as possible.
//
//go:nosplit
func assertWorldStopped() {
	if checkWorldStopped() {
		return
	}

	throw("world not stopped")
}

// assertWorldStoppedOrLockHeld throws if the world is not stopped and the
// passed lock is not held.
//
// nosplit to ensure it can be called in as many contexts as possible.
//
//go:nosplit
func assertWorldStoppedOrLockHeld(l *mutex) {
	if checkWorldStopped() {
		return
	}

	gp := getg()
	held := checkLockHeld(gp, l)
	if held {
		return
	}

	// Crash from system stack to avoid splits that may cause
	// additional issues.
	systemstack(func() {
		printlock()
		print("caller requires world stop or lock ", l, " (rank ", l.rank.String(), "), holding:\n")
		println("<no world stop>")
		printHeldLocks(gp)
		throw("no world stop or required lock!")
	})
}
