// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Semaphore implementation exposed to Go.
// Intended use is provide a sleep and wakeup
// primitive that can be used in the contended case
// of other synchronization primitives.
// Thus it targets the same goal as Linux's futex,
// but it has much simpler semantics.
//
// That is, don't think of these as semaphores.
// Think of them as a way to implement sleep and wakeup
// such that every sleep is paired with a single wakeup,
// even if, due to races, the wakeup happens before the sleep.
//
// See Mullender and Cox, ``Semaphores in Plan 9,''
// https://swtch.com/semaphore.pdf

package runtime

import (
	"internal/cpu"
	"internal/runtime/atomic"
	"unsafe"
)

// Asynchronous semaphore for sync.Mutex.

// A semaRoot holds a balanced tree of waitingG with distinct addresses (wg.elem).
// Each of those waitingG may in turn point (through wg.waitlink) to a list
// of other waitingGs waiting on the same address.
// The operations on the inner lists of waitingGs with the same address
// are all O(1). The scanning of the top-level semaRoot list is O(log n),
// where n is the number of distinct addresses with goroutines blocked
// on them that hash to the given semaRoot.
// See golang.org/issue/17953 for a program that worked badly
// before we introduced the second level of list, and
// BenchmarkSemTable/OneAddrCollision/* for a benchmark that exercises this.
type semaRoot struct {
	lock  mutex
	treap *waitingG     // root of balanced tree of unique waiters.
	nwait atomic.Uint32 // Number of waiters. Read w/o the lock.
}

var semtable semTable

// Prime to not correlate with any user patterns.
const semTabSize = 251

type semTable [semTabSize]struct {
	root semaRoot
	pad  [cpu.CacheLinePadSize - unsafe.Sizeof(semaRoot{})]byte
}

func (t *semTable) rootFor(addr *uint32) *semaRoot {
	return &t[(uintptr(unsafe.Pointer(addr))>>3)%semTabSize].root
}

// sync_runtime_Semacquire should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - gvisor.dev/gvisor
//   - github.com/sagernet/gvisor
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname sync_runtime_Semacquire sync.runtime_Semacquire
func sync_runtime_Semacquire(addr *uint32) {
	semacquire1(addr, false, semaBlockProfile, 0, waitReasonSemacquire)
}

//go:linkname poll_runtime_Semacquire internal/poll.runtime_Semacquire
func poll_runtime_Semacquire(addr *uint32) {
	semacquire1(addr, false, semaBlockProfile, 0, waitReasonSemacquire)
}

// sync_runtime_Semrelease should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - gvisor.dev/gvisor
//   - github.com/sagernet/gvisor
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname sync_runtime_Semrelease sync.runtime_Semrelease
func sync_runtime_Semrelease(addr *uint32, handoff bool, skipframes int) {
	semrelease1(addr, handoff, skipframes)
}

//go:linkname internal_sync_runtime_SemacquireMutex internal/sync.runtime_SemacquireMutex
func internal_sync_runtime_SemacquireMutex(addr *uint32, lifo bool, skipframes int) {
	semacquire1(addr, lifo, semaBlockProfile|semaMutexProfile, skipframes, waitReasonSyncMutexLock)
}

//go:linkname sync_runtime_SemacquireRWMutexR sync.runtime_SemacquireRWMutexR
func sync_runtime_SemacquireRWMutexR(addr *uint32, lifo bool, skipframes int) {
	semacquire1(addr, lifo, semaBlockProfile|semaMutexProfile, skipframes, waitReasonSyncRWMutexRLock)
}

//go:linkname sync_runtime_SemacquireRWMutex sync.runtime_SemacquireRWMutex
func sync_runtime_SemacquireRWMutex(addr *uint32, lifo bool, skipframes int) {
	semacquire1(addr, lifo, semaBlockProfile|semaMutexProfile, skipframes, waitReasonSyncRWMutexLock)
}

//go:linkname sync_runtime_SemacquireWaitGroup sync.runtime_SemacquireWaitGroup
func sync_runtime_SemacquireWaitGroup(addr *uint32, synctestDurable bool) {
	reason := waitReasonSyncWaitGroupWait
	if synctestDurable {
		reason = waitReasonSynctestWaitGroupWait
	}
	semacquire1(addr, false, semaBlockProfile, 0, reason)
}

//go:linkname poll_runtime_Semrelease internal/poll.runtime_Semrelease
func poll_runtime_Semrelease(addr *uint32) {
	semrelease(addr)
}

//go:linkname internal_sync_runtime_Semrelease internal/sync.runtime_Semrelease
func internal_sync_runtime_Semrelease(addr *uint32, handoff bool, skipframes int) {
	semrelease1(addr, handoff, skipframes)
}

func readyWithTime(wg *waitingG, traceskip int) {
	if wg.releasetime != 0 {
		wg.releasetime = cputicks()
	}
	goready(wg.g, traceskip)
}

type semaProfileFlags int

const (
	semaBlockProfile semaProfileFlags = 1 << iota
	semaMutexProfile
)

// Called from runtime.
func semacquire(addr *uint32) {
	semacquire1(addr, false, 0, 0, waitReasonSemacquire)
}

func semacquire1(addr *uint32, lifo bool, profile semaProfileFlags, skipframes int, reason waitReason) {
	gp := getg()
	if gp != gp.m.curg {
		throw("semacquire not on the G stack")
	}

	// Easy case.
	if cansemacquire(addr) {
		return
	}

	// Harder case:
	//	increment waiter count
	//	try cansemacquire one more time, return if succeeded
	//	enqueue itself as a waiter
	//	sleep
	//	(waiter descriptor is dequeued by signaler)
	wg := acquireWaitingG(waitingGTypeSema)
	root := semtable.rootFor(addr)
	t0 := int64(0)
	wg.releasetime = 0
	wg.acquiretime = 0
	wg.ticket = 0
	if profile&semaBlockProfile != 0 && blockprofilerate > 0 {
		t0 = cputicks()
		wg.releasetime = -1
	}
	if profile&semaMutexProfile != 0 && mutexprofilerate > 0 {
		if t0 == 0 {
			t0 = cputicks()
		}
		wg.acquiretime = t0
	}
	for {
		lockWithRank(&root.lock, lockRankRoot)
		// Add ourselves to nwait to disable "easy case" in semrelease.
		root.nwait.Add(1)
		// Check cansemacquire to avoid missed wakeup.
		if cansemacquire(addr) {
			root.nwait.Add(-1)
			unlock(&root.lock)
			break
		}
		// Any semrelease after the cansemacquire knows we're waiting
		// (we set nwait above), so go to sleep.
		root.queue(addr, wg, lifo)
		goparkunlock(&root.lock, reason, traceBlockSync, 4+skipframes)
		if wg.ticket != 0 || cansemacquire(addr) {
			break
		}
	}
	if wg.releasetime > 0 {
		blockevent(wg.releasetime-t0, 3+skipframes)
	}
	releaseWaitingG(wg)
}

func semrelease(addr *uint32) {
	semrelease1(addr, false, 0)
}

func semrelease1(addr *uint32, handoff bool, skipframes int) {
	root := semtable.rootFor(addr)
	atomic.Xadd(addr, 1)

	// Easy case: no waiters?
	// This check must happen after the xadd, to avoid a missed wakeup
	// (see loop in semacquire).
	if root.nwait.Load() == 0 {
		return
	}

	// Harder case: search for a waiter and wake it.
	lockWithRank(&root.lock, lockRankRoot)
	if root.nwait.Load() == 0 {
		// The count is already consumed by another goroutine,
		// so no need to wake up another goroutine.
		unlock(&root.lock)
		return
	}
	wg, t0, tailtime := root.dequeue(addr)
	if wg != nil {
		root.nwait.Add(-1)
	}
	unlock(&root.lock)
	if wg != nil { // May be slow or even yield, so unlock first
		acquiretime := wg.acquiretime
		if acquiretime != 0 {
			// Charge contention that this (delayed) unlock caused.
			// If there are N more goroutines waiting beyond the
			// one that's waking up, charge their delay as well, so that
			// contention holding up many goroutines shows up as
			// more costly than contention holding up a single goroutine.
			// It would take O(N) time to calculate how long each goroutine
			// has been waiting, so instead we charge avg(head-wait, tail-wait)*N.
			// head-wait is the longest wait and tail-wait is the shortest.
			// (When we do a lifo insertion, we preserve this property by
			// copying the old head's acquiretime into the inserted new head.
			// In that case the overall average may be slightly high, but that's fine:
			// the average of the ends is only an approximation to the actual
			// average anyway.)
			// The root.dequeue above changed the head and tail acquiretime
			// to the current time, so the next unlock will not re-count this contention.
			dt0 := t0 - acquiretime
			dt := dt0
			if wg.waiters != 0 {
				dtail := t0 - tailtime
				dt += (dtail + dt0) / 2 * int64(wg.waiters)
			}
			mutexevent(dt, 3+skipframes)
		}
		if wg.ticket != 0 {
			throw("corrupted semaphore ticket")
		}
		if handoff && cansemacquire(addr) {
			wg.ticket = 1
		}
		readyWithTime(wg, 5+skipframes)
		if wg.ticket == 1 && getg().m.locks == 0 && getg() != getg().m.g0 {
			// Direct G handoff
			//
			// readyWithTime has added the waiter G as runnext in the
			// current P; we now call the scheduler so that we start running
			// the waiter G immediately.
			//
			// Note that waiter inherits our time slice: this is desirable
			// to avoid having a highly contended semaphore hog the P
			// indefinitely. goyield is like Gosched, but it emits a
			// "preempted" trace event instead and, more importantly, puts
			// the current G on the local runq instead of the global one.
			// We only do this in the starving regime (handoff=true), as in
			// the non-starving case it is possible for a different waiter
			// to acquire the semaphore while we are yielding/scheduling,
			// and this would be wasteful. We wait instead to enter starving
			// regime, and then we start to do direct handoffs of ticket and P.
			//
			// See issue 33747 for discussion.
			//
			// We don't handoff directly if we're holding locks or on the
			// system stack, since it's not safe to enter the scheduler.
			goyield()
		}
	}
}

func cansemacquire(addr *uint32) bool {
	for {
		v := atomic.Load(addr)
		if v == 0 {
			return false
		}
		if atomic.Cas(addr, v, v-1) {
			return true
		}
	}
}

// queue adds wg to the blocked goroutines in semaRoot.
func (root *semaRoot) queue(addr *uint32, wg *waitingG, lifo bool) {
	wg.g = getg()
	wg.elem.set(unsafe.Pointer(addr))
	// Storing this pointer so that we can trace the semaphore address
	// from the blocked goroutine when checking for goroutine leaks.
	wg.g.waiting = wg
	wg.next = nil
	wg.prev = nil
	wg.waiters = 0

	var last *waitingG
	pt := &root.treap
	for t := *pt; t != nil; t = *pt {
		if uintptr(unsafe.Pointer(addr)) == t.elem.uintptr() {
			// Already have addr in list.
			if lifo {
				// Substitute wg in t's place in treap.
				*pt = wg
				wg.ticket = t.ticket
				wg.acquiretime = t.acquiretime // preserve head acquiretime as oldest time
				wg.parent = t.parent
				wg.prev = t.prev
				wg.next = t.next
				if wg.prev != nil {
					wg.prev.parent = wg
				}
				if wg.next != nil {
					wg.next.parent = wg
				}
				// Add t first in wg's wait list.
				wg.waitlink = t
				wg.waittail = t.waittail
				if wg.waittail == nil {
					wg.waittail = t
				}
				wg.waiters = t.waiters
				if wg.waiters+1 != 0 {
					wg.waiters++
				}
				t.parent = nil
				t.prev = nil
				t.next = nil
				t.waittail = nil
			} else {
				// Add wg to end of t's wait list.
				if t.waittail == nil {
					t.waitlink = wg
				} else {
					t.waittail.waitlink = wg
				}
				t.waittail = wg
				wg.waitlink = nil
				if t.waiters+1 != 0 {
					t.waiters++
				}
			}
			return
		}
		last = t
		if uintptr(unsafe.Pointer(addr)) < t.elem.uintptr() {
			pt = &t.prev
		} else {
			pt = &t.next
		}
	}

	// Add wg as new leaf in tree of unique addrs.
	// The balanced tree is a treap using ticket as the random heap priority.
	// That is, it is a binary tree ordered according to the elem addresses,
	// but then among the space of possible binary trees respecting those
	// addresses, it is kept balanced on average by maintaining a heap ordering
	// on the ticket: wg.ticket <= both wg.prev.ticket and wg.next.ticket.
	// https://en.wikipedia.org/wiki/Treap
	// https://faculty.washington.edu/aragon/pubs/rst89.pdf
	//
	// wg.ticket compared with zero in couple of places, therefore set lowest bit.
	// It will not affect treap's quality noticeably.
	wg.ticket = cheaprand() | 1
	wg.parent = last
	*pt = wg

	// Rotate up into tree according to ticket (priority).
	for wg.parent != nil && wg.parent.ticket > wg.ticket {
		if wg.parent.prev == wg {
			root.rotateRight(wg.parent)
		} else {
			if wg.parent.next != wg {
				panic("semaRoot queue")
			}
			root.rotateLeft(wg.parent)
		}
	}
}

// dequeue searches for and finds the first goroutine
// in semaRoot blocked on addr.
// If the waitingG was being profiled, dequeue returns the time
// at which it was woken up as now. Otherwise now is 0.
// If there are additional entries in the wait list, dequeue
// returns tailtime set to the last entry's acquiretime.
// Otherwise tailtime is found.acquiretime.
func (root *semaRoot) dequeue(addr *uint32) (found *waitingG, now, tailtime int64) {
	pwg := &root.treap
	wg := *pwg

	for ; wg != nil; wg = *pwg {
		if uintptr(unsafe.Pointer(addr)) == wg.elem.uintptr() {
			goto Found
		}

		if uintptr(unsafe.Pointer(addr)) < wg.elem.uintptr() {
			pwg = &wg.prev
		} else {
			pwg = &wg.next
		}
	}
	return nil, 0, 0

Found:
	now = int64(0)
	if wg.acquiretime != 0 {
		now = cputicks()
	}
	if t := wg.waitlink; t != nil {
		// Substitute t, also waiting on addr, for wg in root tree of unique addrs.
		*pwg = t
		t.ticket = wg.ticket
		t.parent = wg.parent
		t.prev = wg.prev
		if t.prev != nil {
			t.prev.parent = t
		}
		t.next = wg.next
		if t.next != nil {
			t.next.parent = t
		}
		if t.waitlink != nil {
			t.waittail = wg.waittail
		} else {
			t.waittail = nil
		}
		t.waiters = wg.waiters
		if t.waiters > 1 {
			t.waiters--
		}
		// Set head and tail acquire time to 'now',
		// because the caller will take care of charging
		// the delays before now for all entries in the list.
		t.acquiretime = now
		tailtime = wg.waittail.acquiretime
		wg.waittail.acquiretime = now
		wg.waitlink = nil
		wg.waittail = nil
	} else {
		// Rotate wg down to be leaf of tree for removal, respecting priorities.
		for wg.next != nil || wg.prev != nil {
			if wg.next == nil || wg.prev != nil && wg.prev.ticket < wg.next.ticket {
				root.rotateRight(wg)
			} else {
				root.rotateLeft(wg)
			}
		}
		// Remove wg, now a leaf.
		if wg.parent != nil {
			if wg.parent.prev == wg {
				wg.parent.prev = nil
			} else {
				wg.parent.next = nil
			}
		} else {
			root.treap = nil
		}
		tailtime = wg.acquiretime
	}
	// Goroutine is no longer blocked. Clear the waiting pointer.
	wg.g.waiting = nil
	wg.parent = nil
	wg.elem.set(nil)
	wg.next = nil
	wg.prev = nil
	wg.ticket = 0
	return wg, now, tailtime
}

// rotateLeft rotates the tree rooted at node x.
// turning (x a (y b c)) into (y (x a b) c).
func (root *semaRoot) rotateLeft(x *waitingG) {
	// p -> (x a (y b c))
	p := x.parent
	y := x.next
	b := y.prev

	y.prev = x
	x.parent = y
	x.next = b
	if b != nil {
		b.parent = x
	}

	y.parent = p
	if p == nil {
		root.treap = y
	} else if p.prev == x {
		p.prev = y
	} else {
		if p.next != x {
			throw("semaRoot rotateLeft")
		}
		p.next = y
	}
}

// rotateRight rotates the tree rooted at node y.
// turning (y (x a b) c) into (x a (y b c)).
func (root *semaRoot) rotateRight(y *waitingG) {
	// p -> (y (x a b) c)
	p := y.parent
	x := y.prev
	b := x.next

	x.next = y
	y.parent = x
	y.prev = b
	if b != nil {
		b.parent = y
	}

	x.parent = p
	if p == nil {
		root.treap = x
	} else if p.prev == y {
		p.prev = x
	} else {
		if p.next != y {
			throw("semaRoot rotateRight")
		}
		p.next = x
	}
}

// notifyList is a ticket-based notification list used to implement sync.Cond.
//
// It must be kept in sync with the sync package.
type notifyList struct {
	// wait is the ticket number of the next waiter. It is atomically
	// incremented outside the lock.
	wait atomic.Uint32

	// notify is the ticket number of the next waiter to be notified. It can
	// be read outside the lock, but is only written to with lock held.
	//
	// Both wait & notify can wrap around, and such cases will be correctly
	// handled as long as their "unwrapped" difference is bounded by 2^31.
	// For this not to be the case, we'd need to have 2^31+ goroutines
	// blocked on the same condvar, which is currently not possible.
	notify uint32

	// List of parked waiters.
	lock mutex
	head *waitingG
	tail *waitingG
}

// less checks if a < b, considering a & b running counts that may overflow the
// 32-bit range, and that their "unwrapped" difference is always less than 2^31.
func less(a, b uint32) bool {
	return int32(a-b) < 0
}

// notifyListAdd adds the caller to a notify list such that it can receive
// notifications. The caller must eventually call notifyListWait to wait for
// such a notification, passing the returned ticket number.
//
//go:linkname notifyListAdd sync.runtime_notifyListAdd
func notifyListAdd(l *notifyList) uint32 {
	// This may be called concurrently, for example, when called from
	// sync.Cond.Wait while holding a RWMutex in read mode.
	return l.wait.Add(1) - 1
}

// notifyListWait waits for a notification. If one has been sent since
// notifyListAdd was called, it returns immediately. Otherwise, it blocks.
//
//go:linkname notifyListWait sync.runtime_notifyListWait
func notifyListWait(l *notifyList, t uint32) {
	lockWithRank(&l.lock, lockRankNotifyList)

	// Return right away if this ticket has already been notified.
	if less(t, l.notify) {
		unlock(&l.lock)
		return
	}

	// Enqueue itself.
	wg := acquireWaitingG(waitingGTypeCond)
	wg.g = getg()
	// Storing this pointer so that we can trace the condvar address
	// from the blocked goroutine when checking for goroutine leaks.
	wg.elem.set(unsafe.Pointer(l))
	wg.g.waiting = wg
	wg.ticket = t
	wg.releasetime = 0
	t0 := int64(0)
	if blockprofilerate > 0 {
		t0 = cputicks()
		wg.releasetime = -1
	}
	if l.tail == nil {
		l.head = wg
	} else {
		l.tail.next = wg
	}
	l.tail = wg
	goparkunlock(&l.lock, waitReasonSyncCondWait, traceBlockCondWait, 3)
	if t0 != 0 {
		blockevent(wg.releasetime-t0, 2)
	}
	// Goroutine is no longer blocked. Clear up its waiting pointer,
	// and clean up the waitingG before releasing it.
	wg.g.waiting = nil
	wg.elem.set(nil)
	releaseWaitingG(wg)
}

// notifyListNotifyAll notifies all entries in the list.
//
//go:linkname notifyListNotifyAll sync.runtime_notifyListNotifyAll
func notifyListNotifyAll(l *notifyList) {
	// Fast-path: if there are no new waiters since the last notification
	// we don't need to acquire the lock.
	if l.wait.Load() == atomic.Load(&l.notify) {
		return
	}

	// Pull the list out into a local variable, waiters will be readied
	// outside the lock.
	lockWithRank(&l.lock, lockRankNotifyList)
	wg := l.head
	l.head = nil
	l.tail = nil

	// Update the next ticket to be notified. We can set it to the current
	// value of wait because any previous waiters are already in the list
	// or will notice that they have already been notified when trying to
	// add themselves to the list.
	atomic.Store(&l.notify, l.wait.Load())
	unlock(&l.lock)

	// Go through the local list and ready all waiters.
	for wg != nil {
		next := wg.next
		wg.next = nil
		if wg.g.bubble != nil && getg().bubble != wg.g.bubble {
			println("semaphore wake of synctest goroutine", wg.g.goid, "from outside bubble")
			fatal("semaphore wake of synctest goroutine from outside bubble")
		}
		readyWithTime(wg, 4)
		wg = next
	}
}

// notifyListNotifyOne notifies one entry in the list.
//
//go:linkname notifyListNotifyOne sync.runtime_notifyListNotifyOne
func notifyListNotifyOne(l *notifyList) {
	// Fast-path: if there are no new waiters since the last notification
	// we don't need to acquire the lock at all.
	if l.wait.Load() == atomic.Load(&l.notify) {
		return
	}

	lockWithRank(&l.lock, lockRankNotifyList)

	// Re-check under the lock if we need to do anything.
	t := l.notify
	if t == l.wait.Load() {
		unlock(&l.lock)
		return
	}

	// Update the next notify ticket number.
	atomic.Store(&l.notify, t+1)

	// Try to find the g that needs to be notified.
	// If it hasn't made it to the list yet we won't find it,
	// but it won't park itself once it sees the new notify number.
	//
	// This scan looks linear but essentially always stops quickly.
	// Because g's queue separately from taking numbers,
	// there may be minor reorderings in the list, but we
	// expect the g we're looking for to be near the front.
	// The g has others in front of it on the list only to the
	// extent that it lost the race, so the iteration will not
	// be too long. This applies even when the g is missing:
	// it hasn't yet gotten to sleep and has lost the race to
	// the (few) other g's that we find on the list.
	for p, wg := (*waitingG)(nil), l.head; wg != nil; p, wg = wg, wg.next {
		if wg.ticket == t {
			n := wg.next
			if p != nil {
				p.next = n
			} else {
				l.head = n
			}
			if n == nil {
				l.tail = p
			}
			unlock(&l.lock)
			wg.next = nil
			if wg.g.bubble != nil && getg().bubble != wg.g.bubble {
				println("semaphore wake of synctest goroutine", wg.g.goid, "from outside bubble")
				fatal("semaphore wake of synctest goroutine from outside bubble")
			}
			readyWithTime(wg, 4)
			return
		}
	}
	unlock(&l.lock)
}

//go:linkname notifyListCheck sync.runtime_notifyListCheck
func notifyListCheck(sz uintptr) {
	if sz != unsafe.Sizeof(notifyList{}) {
		print("runtime: bad notifyList size - sync=", sz, " runtime=", unsafe.Sizeof(notifyList{}), "\n")
		throw("bad notifyList size")
	}
}

//go:linkname internal_sync_nanotime internal/sync.runtime_nanotime
func internal_sync_nanotime() int64 {
	return nanotime()
}
