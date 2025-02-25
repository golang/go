// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/cpu"
	"internal/goarch"
	"internal/runtime/atomic"
	"internal/runtime/sys"
	"unsafe"
)

// AddCleanup attaches a cleanup function to ptr. Some time after ptr is no longer
// reachable, the runtime will call cleanup(arg) in a separate goroutine.
//
// A typical use is that ptr is an object wrapping an underlying resource (e.g.,
// a File object wrapping an OS file descriptor), arg is the underlying resource
// (e.g., the OS file descriptor), and the cleanup function releases the underlying
// resource (e.g., by calling the close system call).
//
// There are few constraints on ptr. In particular, multiple cleanups may be
// attached to the same pointer, or to different pointers within the same
// allocation.
//
// If ptr is reachable from cleanup or arg, ptr will never be collected
// and the cleanup will never run. As a protection against simple cases of this,
// AddCleanup panics if arg is equal to ptr.
//
// There is no specified order in which cleanups will run.
// In particular, if several objects point to each other and all become
// unreachable at the same time, their cleanups all become eligible to run
// and can run in any order. This is true even if the objects form a cycle.
//
// Cleanups run concurrently with any user-created goroutines.
// Cleanups may also run concurrently with one another (unlike finalizers).
// If a cleanup function must run for a long time, it should create a new goroutine
// to avoid blocking the execution of other cleanups.
//
// If ptr has both a cleanup and a finalizer, the cleanup will only run once
// it has been finalized and becomes unreachable without an associated finalizer.
//
// The cleanup(arg) call is not always guaranteed to run; in particular it is not
// guaranteed to run before program exit.
//
// Cleanups are not guaranteed to run if the size of T is zero bytes, because
// it may share same address with other zero-size objects in memory. See
// https://go.dev/ref/spec#Size_and_alignment_guarantees.
//
// It is not guaranteed that a cleanup will run for objects allocated
// in initializers for package-level variables. Such objects may be
// linker-allocated, not heap-allocated.
//
// Note that because cleanups may execute arbitrarily far into the future
// after an object is no longer referenced, the runtime is allowed to perform
// a space-saving optimization that batches objects together in a single
// allocation slot. The cleanup for an unreferenced object in such an
// allocation may never run if it always exists in the same batch as a
// referenced object. Typically, this batching only happens for tiny
// (on the order of 16 bytes or less) and pointer-free objects.
//
// A cleanup may run as soon as an object becomes unreachable.
// In order to use cleanups correctly, the program must ensure that
// the object is reachable until it is safe to run its cleanup.
// Objects stored in global variables, or that can be found by tracing
// pointers from a global variable, are reachable. A function argument or
// receiver may become unreachable at the last point where the function
// mentions it. To ensure a cleanup does not get called prematurely,
// pass the object to the [KeepAlive] function after the last point
// where the object must remain reachable.
func AddCleanup[T, S any](ptr *T, cleanup func(S), arg S) Cleanup {
	// Explicitly force ptr to escape to the heap.
	ptr = abi.Escape(ptr)

	// The pointer to the object must be valid.
	if ptr == nil {
		panic("runtime.AddCleanup: ptr is nil")
	}
	usptr := uintptr(unsafe.Pointer(ptr))

	// Check that arg is not equal to ptr.
	if kind := abi.TypeOf(arg).Kind(); kind == abi.Pointer || kind == abi.UnsafePointer {
		if unsafe.Pointer(ptr) == *((*unsafe.Pointer)(unsafe.Pointer(&arg))) {
			panic("runtime.AddCleanup: ptr is equal to arg, cleanup will never run")
		}
	}
	if inUserArenaChunk(usptr) {
		// Arena-allocated objects are not eligible for cleanup.
		panic("runtime.AddCleanup: ptr is arena-allocated")
	}
	if debug.sbrk != 0 {
		// debug.sbrk never frees memory, so no cleanup will ever run
		// (and we don't have the data structures to record them).
		// Return a noop cleanup.
		return Cleanup{}
	}

	fn := func() {
		cleanup(arg)
	}
	// Closure must escape.
	fv := *(**funcval)(unsafe.Pointer(&fn))
	fv = abi.Escape(fv)

	// Find the containing object.
	base, _, _ := findObject(usptr, 0, 0)
	if base == 0 {
		if isGoPointerWithoutSpan(unsafe.Pointer(ptr)) {
			// Cleanup is a noop.
			return Cleanup{}
		}
		panic("runtime.AddCleanup: ptr not in allocated block")
	}

	// Create another G if necessary.
	if gcCleanups.needG() {
		gcCleanups.createGs()
	}

	id := addCleanup(unsafe.Pointer(ptr), fv)
	return Cleanup{
		id:  id,
		ptr: usptr,
	}
}

// Cleanup is a handle to a cleanup call for a specific object.
type Cleanup struct {
	// id is the unique identifier for the cleanup within the arena.
	id uint64
	// ptr contains the pointer to the object.
	ptr uintptr
}

// Stop cancels the cleanup call. Stop will have no effect if the cleanup call
// has already been queued for execution (because ptr became unreachable).
// To guarantee that Stop removes the cleanup function, the caller must ensure
// that the pointer that was passed to AddCleanup is reachable across the call to Stop.
func (c Cleanup) Stop() {
	if c.id == 0 {
		// id is set to zero when the cleanup is a noop.
		return
	}

	// The following block removes the Special record of type cleanup for the object c.ptr.
	span := spanOfHeap(uintptr(unsafe.Pointer(c.ptr)))
	if span == nil {
		return
	}
	// Ensure that the span is swept.
	// Sweeping accesses the specials list w/o locks, so we have
	// to synchronize with it. And it's just much safer.
	mp := acquirem()
	span.ensureSwept()

	offset := uintptr(unsafe.Pointer(c.ptr)) - span.base()

	var found *special
	lock(&span.speciallock)

	iter, exists := span.specialFindSplicePoint(offset, _KindSpecialCleanup)
	if exists {
		for {
			s := *iter
			if s == nil {
				// Reached the end of the linked list. Stop searching at this point.
				break
			}
			if offset == uintptr(s.offset) && _KindSpecialCleanup == s.kind &&
				(*specialCleanup)(unsafe.Pointer(s)).id == c.id {
				// The special is a cleanup and contains a matching cleanup id.
				*iter = s.next
				found = s
				break
			}
			if offset < uintptr(s.offset) || (offset == uintptr(s.offset) && _KindSpecialCleanup < s.kind) {
				// The special is outside the region specified for that kind of
				// special. The specials are sorted by kind.
				break
			}
			// Try the next special.
			iter = &s.next
		}
	}
	if span.specials == nil {
		spanHasNoSpecials(span)
	}
	unlock(&span.speciallock)
	releasem(mp)

	if found == nil {
		return
	}
	lock(&mheap_.speciallock)
	mheap_.specialCleanupAlloc.free(unsafe.Pointer(found))
	unlock(&mheap_.speciallock)
}

const cleanupBlockSize = 512

// cleanupBlock is an block of cleanups to be executed.
//
// cleanupBlock is allocated from non-GC'd memory, so any heap pointers
// must be specially handled. The GC and cleanup queue currently assume
// that the cleanup queue does not grow during marking (but it can shrink).
type cleanupBlock struct {
	cleanupBlockHeader
	cleanups [(cleanupBlockSize - unsafe.Sizeof(cleanupBlockHeader{})) / goarch.PtrSize]*funcval
}

var cleanupBlockPtrMask [cleanupBlockSize / goarch.PtrSize / 8]byte

type cleanupBlockHeader struct {
	_ sys.NotInHeap
	lfnode
	alllink *cleanupBlock

	// n is sometimes accessed atomically.
	//
	// The invariant depends on what phase the garbage collector is in.
	// During the sweep phase (gcphase == _GCoff), each block has exactly
	// one owner, so it's always safe to update this without atomics.
	// But if this *could* be updated during the mark phase, it must be
	// updated atomically to synchronize with the garbage collector
	// scanning the block as a root.
	n uint32
}

// enqueue pushes a single cleanup function into the block.
//
// Returns if this enqueue call filled the block. This is odd,
// but we want to flush full blocks eagerly to get cleanups
// running as soon as possible.
//
// Must only be called if the GC is in the sweep phase (gcphase == _GCoff),
// because it does not synchronize with the garbage collector.
func (b *cleanupBlock) enqueue(fn *funcval) bool {
	b.cleanups[b.n] = fn
	b.n++
	return b.full()
}

// full returns true if the cleanup block is full.
func (b *cleanupBlock) full() bool {
	return b.n == uint32(len(b.cleanups))
}

// empty returns true if the cleanup block is empty.
func (b *cleanupBlock) empty() bool {
	return b.n == 0
}

// take moves as many cleanups as possible from b into a.
func (a *cleanupBlock) take(b *cleanupBlock) {
	dst := a.cleanups[a.n:]
	if uint32(len(dst)) >= b.n {
		// Take all.
		copy(dst, b.cleanups[:])
		a.n += b.n
		b.n = 0
	} else {
		// Partial take. Copy from the tail to avoid having
		// to move more memory around.
		copy(dst, b.cleanups[b.n-uint32(len(dst)):b.n])
		a.n = uint32(len(a.cleanups))
		b.n -= uint32(len(dst))
	}
}

// cleanupQueue is a queue of ready-to-run cleanup functions.
type cleanupQueue struct {
	// Stack of full cleanup blocks.
	full lfstack
	_    [cpu.CacheLinePadSize - unsafe.Sizeof(lfstack(0))]byte

	// Stack of free cleanup blocks.
	free lfstack

	// flushed indicates whether all local cleanupBlocks have been
	// flushed, and we're in a period of time where this condition is
	// stable (after the last sweeper, before the next sweep phase
	// begins).
	flushed atomic.Bool // Next to free because frequently accessed together.

	_ [cpu.CacheLinePadSize - unsafe.Sizeof(lfstack(0)) - 1]byte

	// Linked list of all cleanup blocks.
	all atomic.UnsafePointer // *cleanupBlock
	_   [cpu.CacheLinePadSize - unsafe.Sizeof(atomic.UnsafePointer{})]byte

	state cleanupSleep
	_     [cpu.CacheLinePadSize - unsafe.Sizeof(cleanupSleep{})]byte

	// Goroutine block state.
	//
	// lock protects sleeping and writes to ng. It is also the lock
	// used by cleanup goroutines to park atomically with updates to
	// sleeping and ng.
	lock     mutex
	sleeping gList
	running  atomic.Uint32
	ng       atomic.Uint32
	needg    atomic.Uint32
}

// cleanupSleep is an atomically-updatable cleanupSleepState.
type cleanupSleep struct {
	u atomic.Uint64 // cleanupSleepState
}

func (s *cleanupSleep) load() cleanupSleepState {
	return cleanupSleepState(s.u.Load())
}

// awaken indicates that N cleanup goroutines should be awoken.
func (s *cleanupSleep) awaken(n int) {
	s.u.Add(int64(n))
}

// sleep indicates that a cleanup goroutine is about to go to sleep.
func (s *cleanupSleep) sleep() {
	s.u.Add(1 << 32)
}

// take returns the number of goroutines to wake to handle
// the cleanup load, and also how many extra wake signals
// there were. The caller takes responsibility for waking
// up "wake" cleanup goroutines.
//
// The number of goroutines to wake is guaranteed to be
// bounded by the current sleeping goroutines, provided
// they call sleep before going to sleep, and all wakeups
// are preceded by a call to take.
func (s *cleanupSleep) take() (wake, extra uint32) {
	for {
		old := s.load()
		if old == 0 {
			return 0, 0
		}
		if old.wakes() > old.asleep() {
			wake = old.asleep()
			extra = old.wakes() - old.asleep()
		} else {
			wake = old.wakes()
			extra = 0
		}
		new := cleanupSleepState(old.asleep()-wake) << 32
		if s.u.CompareAndSwap(uint64(old), uint64(new)) {
			return
		}
	}
}

// cleanupSleepState consists of two fields: the number of
// goroutines currently asleep (equivalent to len(q.sleeping)), and
// the number of times a wakeup signal has been sent.
// These two fields are packed together in a uint64, such
// that they may be updated atomically as part of cleanupSleep.
// The top 32 bits is the number of sleeping goroutines,
// and the bottom 32 bits is the number of wakeup signals.
type cleanupSleepState uint64

func (s cleanupSleepState) asleep() uint32 {
	return uint32(s >> 32)
}

func (s cleanupSleepState) wakes() uint32 {
	return uint32(s)
}

// enqueue queues a single cleanup for execution.
//
// Called by the sweeper, and only the sweeper.
func (q *cleanupQueue) enqueue(fn *funcval) {
	mp := acquirem()
	pp := mp.p.ptr()
	b := pp.cleanups
	if b == nil {
		if q.flushed.Load() {
			q.flushed.Store(false)
		}
		b = (*cleanupBlock)(q.free.pop())
		if b == nil {
			b = (*cleanupBlock)(persistentalloc(cleanupBlockSize, tagAlign, &memstats.gcMiscSys))
			for {
				next := (*cleanupBlock)(q.all.Load())
				b.alllink = next
				if q.all.CompareAndSwap(unsafe.Pointer(next), unsafe.Pointer(b)) {
					break
				}
			}
		}
		pp.cleanups = b
	}
	if full := b.enqueue(fn); full {
		q.full.push(&b.lfnode)
		pp.cleanups = nil
		q.state.awaken(1)
	}
	releasem(mp)
}

// dequeue pops a block of cleanups from the queue. Blocks until one is available
// and never returns nil.
func (q *cleanupQueue) dequeue() *cleanupBlock {
	for {
		b := (*cleanupBlock)(q.full.pop())
		if b != nil {
			return b
		}
		lock(&q.lock)
		q.sleeping.push(getg())
		q.state.sleep()
		goparkunlock(&q.lock, waitReasonCleanupWait, traceBlockSystemGoroutine, 1)
	}
}

// tryDequeue is a non-blocking attempt to dequeue a block of cleanups.
// May return nil if there are no blocks to run.
func (q *cleanupQueue) tryDequeue() *cleanupBlock {
	return (*cleanupBlock)(q.full.pop())
}

// flush pushes all active cleanup blocks to the full list and wakes up cleanup
// goroutines to handle them.
//
// Must only be called at a point when we can guarantee that no more cleanups
// are being queued, such as after the final sweeper for the cycle is done
// but before the next mark phase.
func (q *cleanupQueue) flush() {
	mp := acquirem()
	flushed := 0
	emptied := 0
	missing := 0

	// Coalesce the partially-filled blocks to present a more accurate picture of demand.
	// We use the number of coalesced blocks to process as a signal for demand to create
	// new cleanup goroutines.
	var cb *cleanupBlock
	for _, pp := range allp {
		b := pp.cleanups
		if b == nil {
			missing++
			continue
		}
		pp.cleanups = nil
		if cb == nil {
			cb = b
			continue
		}
		// N.B. After take, either cb is full, b is empty, or both.
		cb.take(b)
		if cb.full() {
			q.full.push(&cb.lfnode)
			flushed++
			cb = b
			b = nil
		}
		if b != nil && b.empty() {
			q.free.push(&b.lfnode)
			emptied++
		}
	}
	if cb != nil {
		q.full.push(&cb.lfnode)
		flushed++
	}
	if flushed != 0 {
		q.state.awaken(flushed)
	}
	if flushed+emptied+missing != len(allp) {
		throw("failed to correctly flush all P-owned cleanup blocks")
	}
	q.flushed.Store(true)
	releasem(mp)
}

// needsWake returns true if cleanup goroutines need to be awoken or created to handle cleanup load.
func (q *cleanupQueue) needsWake() bool {
	s := q.state.load()
	return s.wakes() > 0 && (s.asleep() > 0 || q.ng.Load() < maxCleanupGs())
}

// wake wakes up one or more goroutines to process the cleanup queue. If there aren't
// enough sleeping goroutines to handle the demand, wake will arrange for new goroutines
// to be created.
func (q *cleanupQueue) wake() {
	wake, extra := q.state.take()
	if extra != 0 {
		newg := min(extra, maxCleanupGs()-q.ng.Load())
		if newg > 0 {
			q.needg.Add(int32(newg))
		}
	}
	if wake == 0 {
		return
	}

	// By calling 'take', we've taken ownership of waking 'wake' goroutines.
	// Nobody else will wake up these goroutines, so they're guaranteed
	// to be sitting on q.sleeping, waiting for us to wake them.
	//
	// Collect them and schedule them.
	var list gList
	lock(&q.lock)
	for range wake {
		list.push(q.sleeping.pop())
	}
	unlock(&q.lock)

	injectglist(&list)
	return
}

func (q *cleanupQueue) needG() bool {
	have := q.ng.Load()
	if have >= maxCleanupGs() {
		return false
	}
	if have == 0 {
		// Make sure we have at least one.
		return true
	}
	return q.needg.Load() > 0
}

func (q *cleanupQueue) createGs() {
	lock(&q.lock)
	have := q.ng.Load()
	need := min(q.needg.Swap(0), maxCleanupGs()-have)
	if have == 0 && need == 0 {
		// Make sure we have at least one.
		need = 1
	}
	if need > 0 {
		q.ng.Add(int32(need))
	}
	unlock(&q.lock)

	for range need {
		go runCleanups()
	}
}

func (q *cleanupQueue) beginRunningCleanups() {
	// Update runningCleanups and running atomically with respect
	// to goroutine profiles by disabling preemption.
	mp := acquirem()
	getg().runningCleanups.Store(true)
	q.running.Add(1)
	releasem(mp)
}

func (q *cleanupQueue) endRunningCleanups() {
	// Update runningCleanups and running atomically with respect
	// to goroutine profiles by disabling preemption.
	mp := acquirem()
	getg().runningCleanups.Store(false)
	q.running.Add(-1)
	releasem(mp)
}

func maxCleanupGs() uint32 {
	// N.B. Left as a function to make changing the policy easier.
	return uint32(max(gomaxprocs/4, 1))
}

// gcCleanups is the global cleanup queue.
var gcCleanups cleanupQueue

// runCleanups is the entrypoint for all cleanup-running goroutines.
func runCleanups() {
	for {
		b := gcCleanups.dequeue()
		if raceenabled {
			// Approximately: adds a happens-before edge between the cleanup
			// argument being mutated and the call to the cleanup below.
			racefingo()
		}

		gcCleanups.beginRunningCleanups()
		for i := 0; i < int(b.n); i++ {
			fn := b.cleanups[i]

			var racectx uintptr
			if raceenabled {
				// Enter a new race context so the race detector can catch
				// potential races between cleanups, even if they execute on
				// the same goroutine.
				//
				// Synchronize on fn. This would fail to find races on the
				// closed-over values in fn (suppose fn is passed to multiple
				// AddCleanup calls) if fn was not unique, but it is. Update
				// the synchronization on fn if you intend to optimize it
				// and store the cleanup function and cleanup argument on the
				// queue directly.
				racerelease(unsafe.Pointer(fn))
				racectx = raceEnterNewCtx()
				raceacquire(unsafe.Pointer(fn))
			}

			// Execute the next cleanup.
			cleanup := *(*func())(unsafe.Pointer(&fn))
			cleanup()
			b.cleanups[i] = nil

			if raceenabled {
				// Restore the old context.
				raceRestoreCtx(racectx)
			}
		}
		gcCleanups.endRunningCleanups()

		atomic.Store(&b.n, 0) // Synchronize with markroot. See comment in cleanupBlockHeader.
		gcCleanups.free.push(&b.lfnode)
	}
}

// blockUntilEmpty blocks until either the cleanup queue is emptied
// and the cleanups have been executed, or the timeout is reached.
// Returns true if the cleanup queue was emptied.
// This is used by the sync and unique tests.
func (q *cleanupQueue) blockUntilEmpty(timeout int64) bool {
	start := nanotime()
	for nanotime()-start < timeout {
		lock(&q.lock)
		// The queue is empty when there's no work left to do *and* all the cleanup goroutines
		// are asleep. If they're not asleep, they may be actively working on a block.
		if q.flushed.Load() && q.full.empty() && uint32(q.sleeping.size) == q.ng.Load() {
			unlock(&q.lock)
			return true
		}
		unlock(&q.lock)
		Gosched()
	}
	return false
}

//go:linkname unique_runtime_blockUntilEmptyCleanupQueue unique.runtime_blockUntilEmptyCleanupQueue
func unique_runtime_blockUntilEmptyCleanupQueue(timeout int64) bool {
	return gcCleanups.blockUntilEmpty(timeout)
}

//go:linkname sync_test_runtime_blockUntilEmptyCleanupQueue sync_test.runtime_blockUntilEmptyCleanupQueue
func sync_test_runtime_blockUntilEmptyCleanupQueue(timeout int64) bool {
	return gcCleanups.blockUntilEmpty(timeout)
}

// raceEnterNewCtx creates a new racectx and switches the current
// goroutine to it. Returns the old racectx.
//
// Must be running on a user goroutine. nosplit to match other race
// instrumentation.
//
//go:nosplit
func raceEnterNewCtx() uintptr {
	// We use the existing ctx as the spawn context, but gp.gopc
	// as the spawn PC to make the error output a little nicer
	// (pointing to AddCleanup, where the goroutines are created).
	//
	// We also need to carefully indicate to the race detector
	// that the goroutine stack will only be accessed by the new
	// race context, to avoid false positives on stack locations.
	// We do this by marking the stack as free in the first context
	// and then re-marking it as allocated in the second. Crucially,
	// there must be (1) no race operations and (2) no stack changes
	// in between. (1) is easy to avoid because we're in the runtime
	// so there's no implicit race instrumentation. To avoid (2) we
	// defensively become non-preemptible so the GC can't stop us,
	// and rely on the fact that racemalloc, racefreem, and racectx
	// are nosplit.
	mp := acquirem()
	gp := getg()
	ctx := getg().racectx
	racefree(unsafe.Pointer(gp.stack.lo), gp.stack.hi-gp.stack.lo)
	getg().racectx = racectxstart(gp.gopc, ctx)
	racemalloc(unsafe.Pointer(gp.stack.lo), gp.stack.hi-gp.stack.lo)
	releasem(mp)
	return ctx
}

// raceRestoreCtx restores ctx on the goroutine. It is the inverse of
// raceenternewctx and must be called with its result.
//
// Must be running on a user goroutine. nosplit to match other race
// instrumentation.
//
//go:nosplit
func raceRestoreCtx(ctx uintptr) {
	mp := acquirem()
	gp := getg()
	racefree(unsafe.Pointer(gp.stack.lo), gp.stack.hi-gp.stack.lo)
	racectxend(getg().racectx)
	racemalloc(unsafe.Pointer(gp.stack.lo), gp.stack.hi-gp.stack.lo)
	getg().racectx = ctx
	releasem(mp)
}
