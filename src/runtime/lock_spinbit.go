// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !wasm

package runtime

import (
	"internal/goarch"
	"internal/runtime/atomic"
	"internal/runtime/gc"
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

// The mutex state consists of four flags and a pointer. The flag at bit 0,
// mutexLocked, represents the lock itself. Bit 1, mutexSleeping, is a hint that
// the pointer is non-nil. The fast paths for locking and unlocking the mutex
// are based on atomic 8-bit swap operations on the low byte; bits 2 through 7
// are unused.
//
// Bit 8, mutexSpinning, is a try-lock that grants a waiting M permission to
// spin on the state word. Most other Ms must attempt to spend their time
// sleeping to reduce traffic on the cache line. This is the "spin bit" for
// which the implementation is named. (The anti-starvation mechanism also grants
// temporary permission for an M to spin.)
//
// Bit 9, mutexStackLocked, is a try-lock that grants an unlocking M permission
// to inspect the list of waiting Ms and to pop an M off of that stack.
//
// The upper bits hold a (partial) pointer to the M that most recently went to
// sleep. The sleeping Ms form a stack linked by their mWaitList.next fields.
// Because the fast paths use an 8-bit swap on the low byte of the state word,
// we'll need to reconstruct the full M pointer from the bits we have. Most Ms
// are allocated on the heap, and have a known alignment and base offset. (The
// offset is due to mallocgc's allocation headers.) The main program thread uses
// a static M value, m0. We check for m0 specifically and add a known offset
// otherwise.

const (
	active_spin     = 4  // referenced in proc.go for sync.Mutex implementation
	active_spin_cnt = 30 // referenced in proc.go for sync.Mutex implementation
)

const (
	mutexLocked      = 0x001
	mutexSleeping    = 0x002
	mutexSpinning    = 0x100
	mutexStackLocked = 0x200
	mutexMMask       = 0x3FF
	mutexMOffset     = gc.MallocHeaderSize // alignment of heap-allocated Ms (those other than m0)

	mutexActiveSpinCount  = 4
	mutexActiveSpinSize   = 30
	mutexPassiveSpinCount = 1

	mutexTailWakePeriod = 16
)

//go:nosplit
func key8(p *uintptr) *uint8 {
	if goarch.BigEndian {
		return &(*[8]uint8)(unsafe.Pointer(p))[goarch.PtrSize/1-1]
	}
	return &(*[8]uint8)(unsafe.Pointer(p))[0]
}

// mWaitList is part of the M struct, and holds the list of Ms that are waiting
// for a particular runtime.mutex.
//
// When an M is unable to immediately obtain a lock, it adds itself to the list
// of Ms waiting for the lock. It does that via this struct's next field,
// forming a singly-linked list with the mutex's key field pointing to the head
// of the list.
type mWaitList struct {
	next       muintptr // next m waiting for lock
	startTicks int64    // when this m started waiting for the current lock holder, in cputicks
}

// lockVerifyMSize confirms that we can recreate the low bits of the M pointer.
func lockVerifyMSize() {
	size := roundupsize(unsafe.Sizeof(mPadded{}), false) + gc.MallocHeaderSize
	if size&mutexMMask != 0 {
		print("M structure uses sizeclass ", size, "/", hex(size), " bytes; ",
			"incompatible with mutex flag mask ", hex(mutexMMask), "\n")
		throw("runtime.m memory alignment too small for spinbit mutex")
	}
}

// mutexWaitListHead recovers a full muintptr that was missing its low bits.
// With the exception of the static m0 value, it requires allocating runtime.m
// values in a size class with a particular minimum alignment. The 2048-byte
// size class allows recovering the full muintptr value even after overwriting
// the low 11 bits with flags. We can use those 11 bits as 3 flags and an
// atomically-swapped byte.
//
//go:nosplit
func mutexWaitListHead(v uintptr) muintptr {
	if highBits := v &^ mutexMMask; highBits == 0 {
		return 0
	} else if m0bits := muintptr(unsafe.Pointer(&m0)); highBits == uintptr(m0bits)&^mutexMMask {
		return m0bits
	} else {
		return muintptr(highBits + mutexMOffset)
	}
}

// mutexPreferLowLatency reports if this mutex prefers low latency at the risk
// of performance collapse. If so, we can allow all waiting threads to spin on
// the state word rather than go to sleep.
//
// TODO: We could have the waiting Ms each spin on their own private cache line,
// especially if we can put a bound on the on-CPU time that would consume.
//
// TODO: If there's a small set of mutex values with special requirements, they
// could make use of a more specialized lock2/unlock2 implementation. Otherwise,
// we're constrained to what we can fit within a single uintptr with no
// additional storage on the M for each lock held.
//
//go:nosplit
func mutexPreferLowLatency(l *mutex) bool {
	switch l {
	default:
		return false
	case &sched.lock:
		// We often expect sched.lock to pass quickly between Ms in a way that
		// each M has unique work to do: for instance when we stop-the-world
		// (bringing each P to idle) or add new netpoller-triggered work to the
		// global run queue.
		return true
	}
}

func mutexContended(l *mutex) bool {
	return atomic.Loaduintptr(&l.key)&^mutexMMask != 0
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

	k8 := key8(&l.key)

	// Speculative grab for lock.
	v8 := atomic.Xchg8(k8, mutexLocked)
	if v8&mutexLocked == 0 {
		if v8&mutexSleeping != 0 {
			atomic.Or8(k8, mutexSleeping)
		}
		return
	}
	semacreate(gp.m)

	var startTime int64
	// On uniprocessors, no point spinning.
	// On multiprocessors, spin for mutexActiveSpinCount attempts.
	spin := 0
	if numCPUStartup > 1 {
		spin = mutexActiveSpinCount
	}

	var weSpin, atTail, haveTimers bool
	v := atomic.Loaduintptr(&l.key)
tryAcquire:
	for i := 0; ; i++ {
		if v&mutexLocked == 0 {
			if weSpin {
				next := (v &^ mutexSpinning) | mutexSleeping | mutexLocked
				if next&^mutexMMask == 0 {
					// The fast-path Xchg8 may have cleared mutexSleeping. Fix
					// the hint so unlock2 knows when to use its slow path.
					next = next &^ mutexSleeping
				}
				if atomic.Casuintptr(&l.key, v, next) {
					gp.m.mLockProfile.end(startTime)
					return
				}
			} else {
				prev8 := atomic.Xchg8(k8, mutexLocked|mutexSleeping)
				if prev8&mutexLocked == 0 {
					gp.m.mLockProfile.end(startTime)
					return
				}
			}
			v = atomic.Loaduintptr(&l.key)
			continue tryAcquire
		}

		if !weSpin && v&mutexSpinning == 0 && atomic.Casuintptr(&l.key, v, v|mutexSpinning) {
			v |= mutexSpinning
			weSpin = true
		}

		if weSpin || atTail || mutexPreferLowLatency(l) {
			if i < spin {
				procyield(mutexActiveSpinSize)
				v = atomic.Loaduintptr(&l.key)
				continue tryAcquire
			} else if i < spin+mutexPassiveSpinCount {
				osyield() // TODO: Consider removing this step. See https://go.dev/issue/69268.
				v = atomic.Loaduintptr(&l.key)
				continue tryAcquire
			}
		}

		// Go to sleep
		if v&mutexLocked == 0 {
			throw("runtime·lock: sleeping while lock is available")
		}

		// Collect times for mutex profile (seen in unlock2 only via mWaitList),
		// and for "/sync/mutex/wait/total:seconds" metric (to match).
		if !haveTimers {
			gp.m.mWaitList.startTicks = cputicks()
			startTime = gp.m.mLockProfile.start()
			haveTimers = true
		}
		// Store the current head of the list of sleeping Ms in our gp.m.mWaitList.next field
		gp.m.mWaitList.next = mutexWaitListHead(v)

		// Pack a (partial) pointer to this M with the current lock state bits
		next := (uintptr(unsafe.Pointer(gp.m)) &^ mutexMMask) | v&mutexMMask | mutexSleeping
		if weSpin { // If we were spinning, prepare to retire
			next = next &^ mutexSpinning
		}

		if atomic.Casuintptr(&l.key, v, next) {
			weSpin = false
			// We've pushed ourselves onto the stack of waiters. Wait.
			semasleep(-1)
			atTail = gp.m.mWaitList.next == 0 // we were at risk of starving
			i = 0
		}

		gp.m.mWaitList.next = 0
		v = atomic.Loaduintptr(&l.key)
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

	var prev8 uint8
	var haveStackLock bool
	var endTicks int64
	if !mutexSampleContention() {
		// Not collecting a sample for the contention profile, do the quick release
		prev8 = atomic.Xchg8(key8(&l.key), 0)
	} else {
		// If there's contention, we'll sample it. Don't allow another
		// lock2/unlock2 pair to finish before us and take our blame. Prevent
		// that by trading for the stack lock with a CAS.
		v := atomic.Loaduintptr(&l.key)
		for {
			if v&^mutexMMask == 0 || v&mutexStackLocked != 0 {
				// No contention, or (stack lock unavailable) no way to calculate it
				prev8 = atomic.Xchg8(key8(&l.key), 0)
				endTicks = 0
				break
			}

			// There's contention, the stack lock appeared to be available, and
			// we'd like to collect a sample for the contention profile.
			if endTicks == 0 {
				// Read the time before releasing the lock. The profile will be
				// strictly smaller than what other threads would see by timing
				// their lock calls.
				endTicks = cputicks()
			}
			next := (v | mutexStackLocked) &^ (mutexLocked | mutexSleeping)
			if atomic.Casuintptr(&l.key, v, next) {
				haveStackLock = true
				prev8 = uint8(v)
				// The fast path of lock2 may have cleared mutexSleeping.
				// Restore it so we're sure to call unlock2Wake below.
				prev8 |= mutexSleeping
				break
			}
			v = atomic.Loaduintptr(&l.key)
		}
	}
	if prev8&mutexLocked == 0 {
		throw("unlock of unlocked lock")
	}

	if prev8&mutexSleeping != 0 {
		unlock2Wake(l, haveStackLock, endTicks)
	}

	gp.m.mLockProfile.store()
	gp.m.locks--
	if gp.m.locks < 0 {
		throw("runtime·unlock: lock count")
	}
	if gp.m.locks == 0 && gp.preempt { // restore the preemption request in case we've cleared it in newstack
		gp.stackguard0 = stackPreempt
	}
}

// mutexSampleContention returns whether the current mutex operation should
// report any contention it discovers.
func mutexSampleContention() bool {
	if rate := int64(atomic.Load64(&mutexprofilerate)); rate <= 0 {
		return false
	} else {
		// TODO: have SetMutexProfileFraction do the clamping
		rate32 := uint32(rate)
		if int64(rate32) != rate {
			rate32 = ^uint32(0)
		}
		return cheaprandn(rate32) == 0
	}
}

// unlock2Wake updates the list of Ms waiting on l, waking an M if necessary.
//
//go:nowritebarrier
func unlock2Wake(l *mutex, haveStackLock bool, endTicks int64) {
	v := atomic.Loaduintptr(&l.key)

	// On occasion, seek out and wake the M at the bottom of the stack so it
	// doesn't starve.
	antiStarve := cheaprandn(mutexTailWakePeriod) == 0

	if haveStackLock {
		goto useStackLock
	}

	if !(antiStarve || // avoiding starvation may require a wake
		v&mutexSpinning == 0 || // no spinners means we must wake
		mutexPreferLowLatency(l)) { // prefer waiters be awake as much as possible
		return
	}

	for {
		if v&^mutexMMask == 0 || v&mutexStackLocked != 0 {
			// No waiting Ms means nothing to do.
			//
			// If the stack lock is unavailable, its owner would make the same
			// wake decisions that we would, so there's nothing for us to do.
			//
			// Although: This thread may have a different call stack, which
			// would result in a different entry in the mutex contention profile
			// (upon completion of go.dev/issue/66999). That could lead to weird
			// results if a slow critical section ends but another thread
			// quickly takes the lock, finishes its own critical section,
			// releases the lock, and then grabs the stack lock. That quick
			// thread would then take credit (blame) for the delay that this
			// slow thread caused. The alternative is to have more expensive
			// atomic operations (a CAS) on the critical path of unlock2.
			return
		}
		// Other M's are waiting for the lock.
		// Obtain the stack lock, and pop off an M.
		next := v | mutexStackLocked
		if atomic.Casuintptr(&l.key, v, next) {
			break
		}
		v = atomic.Loaduintptr(&l.key)
	}

	// We own the mutexStackLocked flag. New Ms may push themselves onto the
	// stack concurrently, but we're now the only thread that can remove or
	// modify the Ms that are sleeping in the list.
useStackLock:

	if endTicks != 0 {
		// Find the M at the bottom of the stack of waiters, which has been
		// asleep for the longest. Take the average of its wait time and the
		// head M's wait time for the mutex contention profile, matching the
		// estimate we do in semrelease1 (for sync.Mutex contention).
		//
		// We don't keep track of the tail node (we don't need it often), so do
		// an O(N) walk on the list of sleeping Ms to find it.
		head := mutexWaitListHead(v).ptr()
		for node, n := head, 0; ; {
			n++
			next := node.mWaitList.next.ptr()
			if next == nil {
				cycles := ((endTicks - head.mWaitList.startTicks) + (endTicks - node.mWaitList.startTicks)) / 2
				node.mWaitList.startTicks = endTicks
				head.mWaitList.startTicks = endTicks
				getg().m.mLockProfile.recordUnlock(cycles * int64(n))
				break
			}
			node = next
		}
	}

	var committed *m // If we choose an M within the stack, we've made a promise to wake it
	for {
		headM := v &^ mutexMMask
		flags := v & (mutexMMask &^ mutexStackLocked) // preserve low bits, but release stack lock

		mp := mutexWaitListHead(v).ptr()
		wakem := committed
		if committed == nil {
			if v&mutexSpinning == 0 || mutexPreferLowLatency(l) {
				wakem = mp
			}
			if antiStarve {
				// Wake the M at the bottom of the stack of waiters. (This is
				// O(N) with the number of waiters.)
				wakem = mp
				prev := mp
				for {
					next := wakem.mWaitList.next.ptr()
					if next == nil {
						break
					}
					prev, wakem = wakem, next
				}
				if wakem != mp {
					committed = wakem
					prev.mWaitList.next = wakem.mWaitList.next
					// An M sets its own startTicks when it first goes to sleep.
					// When an unlock operation is sampled for the mutex
					// contention profile, it takes blame for the entire list of
					// waiting Ms but only updates the startTicks value at the
					// tail. Copy any updates to the next-oldest M.
					prev.mWaitList.startTicks = wakem.mWaitList.startTicks
				}
			}
		}

		if wakem == mp {
			headM = uintptr(mp.mWaitList.next) &^ mutexMMask
		}

		next := headM | flags
		if atomic.Casuintptr(&l.key, v, next) {
			if wakem != nil {
				// Claimed an M. Wake it.
				semawakeup(wakem)
			}
			return
		}

		v = atomic.Loaduintptr(&l.key)
	}
}
