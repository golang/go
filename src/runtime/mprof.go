// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Malloc profiling.
// Patterned after tcmalloc's algorithms; shorter code.

package runtime

import (
	"internal/abi"
	"internal/bytealg"
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// NOTE(rsc): Everything here could use cas if contention became an issue.
var (
	// profInsertLock protects changes to the start of all *bucket linked lists
	profInsertLock mutex
	// profBlockLock protects the contents of every blockRecord struct
	profBlockLock mutex
	// profMemActiveLock protects the active field of every memRecord struct
	profMemActiveLock mutex
	// profMemFutureLock is a set of locks that protect the respective elements
	// of the future array of every memRecord struct
	profMemFutureLock [len(memRecord{}.future)]mutex
)

// All memory allocations are local and do not escape outside of the profiler.
// The profiler is forbidden from referring to garbage-collected memory.

const (
	// profile types
	memProfile bucketType = 1 + iota
	blockProfile
	mutexProfile

	// size of bucket hash table
	buckHashSize = 179999

	// maxStack is the max depth of stack to record in bucket.
	// Note that it's only used internally as a guard against
	// wildly out-of-bounds slicing of the PCs that come after
	// a bucket struct, and it could increase in the future.
	maxStack = 32
)

type bucketType int

// A bucket holds per-call-stack profiling information.
// The representation is a bit sleazy, inherited from C.
// This struct defines the bucket header. It is followed in
// memory by the stack words and then the actual record
// data, either a memRecord or a blockRecord.
//
// Per-call-stack profiling information.
// Lookup by hashing call stack into a linked-list hash table.
//
// None of the fields in this bucket header are modified after
// creation, including its next and allnext links.
//
// No heap pointers.
type bucket struct {
	_       sys.NotInHeap
	next    *bucket
	allnext *bucket
	typ     bucketType // memBucket or blockBucket (includes mutexProfile)
	hash    uintptr
	size    uintptr
	nstk    uintptr
}

// A memRecord is the bucket data for a bucket of type memProfile,
// part of the memory profile.
type memRecord struct {
	// The following complex 3-stage scheme of stats accumulation
	// is required to obtain a consistent picture of mallocs and frees
	// for some point in time.
	// The problem is that mallocs come in real time, while frees
	// come only after a GC during concurrent sweeping. So if we would
	// naively count them, we would get a skew toward mallocs.
	//
	// Hence, we delay information to get consistent snapshots as
	// of mark termination. Allocations count toward the next mark
	// termination's snapshot, while sweep frees count toward the
	// previous mark termination's snapshot:
	//
	//              MT          MT          MT          MT
	//             .·|         .·|         .·|         .·|
	//          .·˙  |      .·˙  |      .·˙  |      .·˙  |
	//       .·˙     |   .·˙     |   .·˙     |   .·˙     |
	//    .·˙        |.·˙        |.·˙        |.·˙        |
	//
	//       alloc → ▲ ← free
	//               ┠┅┅┅┅┅┅┅┅┅┅┅P
	//       C+2     →    C+1    →  C
	//
	//                   alloc → ▲ ← free
	//                           ┠┅┅┅┅┅┅┅┅┅┅┅P
	//                   C+2     →    C+1    →  C
	//
	// Since we can't publish a consistent snapshot until all of
	// the sweep frees are accounted for, we wait until the next
	// mark termination ("MT" above) to publish the previous mark
	// termination's snapshot ("P" above). To do this, allocation
	// and free events are accounted to *future* heap profile
	// cycles ("C+n" above) and we only publish a cycle once all
	// of the events from that cycle must be done. Specifically:
	//
	// Mallocs are accounted to cycle C+2.
	// Explicit frees are accounted to cycle C+2.
	// GC frees (done during sweeping) are accounted to cycle C+1.
	//
	// After mark termination, we increment the global heap
	// profile cycle counter and accumulate the stats from cycle C
	// into the active profile.

	// active is the currently published profile. A profiling
	// cycle can be accumulated into active once its complete.
	active memRecordCycle

	// future records the profile events we're counting for cycles
	// that have not yet been published. This is ring buffer
	// indexed by the global heap profile cycle C and stores
	// cycles C, C+1, and C+2. Unlike active, these counts are
	// only for a single cycle; they are not cumulative across
	// cycles.
	//
	// We store cycle C here because there's a window between when
	// C becomes the active cycle and when we've flushed it to
	// active.
	future [3]memRecordCycle
}

// memRecordCycle
type memRecordCycle struct {
	allocs, frees           uintptr
	alloc_bytes, free_bytes uintptr
}

// add accumulates b into a. It does not zero b.
func (a *memRecordCycle) add(b *memRecordCycle) {
	a.allocs += b.allocs
	a.frees += b.frees
	a.alloc_bytes += b.alloc_bytes
	a.free_bytes += b.free_bytes
}

// A blockRecord is the bucket data for a bucket of type blockProfile,
// which is used in blocking and mutex profiles.
type blockRecord struct {
	count  float64
	cycles int64
}

var (
	mbuckets atomic.UnsafePointer // *bucket, memory profile buckets
	bbuckets atomic.UnsafePointer // *bucket, blocking profile buckets
	xbuckets atomic.UnsafePointer // *bucket, mutex profile buckets
	buckhash atomic.UnsafePointer // *buckhashArray

	mProfCycle mProfCycleHolder
)

type buckhashArray [buckHashSize]atomic.UnsafePointer // *bucket

const mProfCycleWrap = uint32(len(memRecord{}.future)) * (2 << 24)

// mProfCycleHolder holds the global heap profile cycle number (wrapped at
// mProfCycleWrap, stored starting at bit 1), and a flag (stored at bit 0) to
// indicate whether future[cycle] in all buckets has been queued to flush into
// the active profile.
type mProfCycleHolder struct {
	value atomic.Uint32
}

// read returns the current cycle count.
func (c *mProfCycleHolder) read() (cycle uint32) {
	v := c.value.Load()
	cycle = v >> 1
	return cycle
}

// setFlushed sets the flushed flag. It returns the current cycle count and the
// previous value of the flushed flag.
func (c *mProfCycleHolder) setFlushed() (cycle uint32, alreadyFlushed bool) {
	for {
		prev := c.value.Load()
		cycle = prev >> 1
		alreadyFlushed = (prev & 0x1) != 0
		next := prev | 0x1
		if c.value.CompareAndSwap(prev, next) {
			return cycle, alreadyFlushed
		}
	}
}

// increment increases the cycle count by one, wrapping the value at
// mProfCycleWrap. It clears the flushed flag.
func (c *mProfCycleHolder) increment() {
	// We explicitly wrap mProfCycle rather than depending on
	// uint wraparound because the memRecord.future ring does not
	// itself wrap at a power of two.
	for {
		prev := c.value.Load()
		cycle := prev >> 1
		cycle = (cycle + 1) % mProfCycleWrap
		next := cycle << 1
		if c.value.CompareAndSwap(prev, next) {
			break
		}
	}
}

// newBucket allocates a bucket with the given type and number of stack entries.
func newBucket(typ bucketType, nstk int) *bucket {
	size := unsafe.Sizeof(bucket{}) + uintptr(nstk)*unsafe.Sizeof(uintptr(0))
	switch typ {
	default:
		throw("invalid profile bucket type")
	case memProfile:
		size += unsafe.Sizeof(memRecord{})
	case blockProfile, mutexProfile:
		size += unsafe.Sizeof(blockRecord{})
	}

	b := (*bucket)(persistentalloc(size, 0, &memstats.buckhash_sys))
	b.typ = typ
	b.nstk = uintptr(nstk)
	return b
}

// stk returns the slice in b holding the stack.
func (b *bucket) stk() []uintptr {
	stk := (*[maxStack]uintptr)(add(unsafe.Pointer(b), unsafe.Sizeof(*b)))
	if b.nstk > maxStack {
		// prove that slicing works; otherwise a failure requires a P
		throw("bad profile stack count")
	}
	return stk[:b.nstk:b.nstk]
}

// mp returns the memRecord associated with the memProfile bucket b.
func (b *bucket) mp() *memRecord {
	if b.typ != memProfile {
		throw("bad use of bucket.mp")
	}
	data := add(unsafe.Pointer(b), unsafe.Sizeof(*b)+b.nstk*unsafe.Sizeof(uintptr(0)))
	return (*memRecord)(data)
}

// bp returns the blockRecord associated with the blockProfile bucket b.
func (b *bucket) bp() *blockRecord {
	if b.typ != blockProfile && b.typ != mutexProfile {
		throw("bad use of bucket.bp")
	}
	data := add(unsafe.Pointer(b), unsafe.Sizeof(*b)+b.nstk*unsafe.Sizeof(uintptr(0)))
	return (*blockRecord)(data)
}

// Return the bucket for stk[0:nstk], allocating new bucket if needed.
func stkbucket(typ bucketType, size uintptr, stk []uintptr, alloc bool) *bucket {
	bh := (*buckhashArray)(buckhash.Load())
	if bh == nil {
		lock(&profInsertLock)
		// check again under the lock
		bh = (*buckhashArray)(buckhash.Load())
		if bh == nil {
			bh = (*buckhashArray)(sysAlloc(unsafe.Sizeof(buckhashArray{}), &memstats.buckhash_sys))
			if bh == nil {
				throw("runtime: cannot allocate memory")
			}
			buckhash.StoreNoWB(unsafe.Pointer(bh))
		}
		unlock(&profInsertLock)
	}

	// Hash stack.
	var h uintptr
	for _, pc := range stk {
		h += pc
		h += h << 10
		h ^= h >> 6
	}
	// hash in size
	h += size
	h += h << 10
	h ^= h >> 6
	// finalize
	h += h << 3
	h ^= h >> 11

	i := int(h % buckHashSize)
	// first check optimistically, without the lock
	for b := (*bucket)(bh[i].Load()); b != nil; b = b.next {
		bstk := b.stk()
		if b.typ == typ && b.hash == h && b.size == size && bytealg.Equal(unsafe.Slice((*byte)((*slice)(unsafe.Pointer(&bstk)).array), len(bstk)*8), unsafe.Slice((*byte)(((*slice)(unsafe.Pointer(&stk))).array), len(stk)*8)) {
			return b
		}
	}

	if !alloc {
		return nil
	}

	lock(&profInsertLock)
	// check again under the insertion lock
	for b := (*bucket)(bh[i].Load()); b != nil; b = b.next {
		bstk := b.stk()
		if b.typ == typ && b.hash == h && b.size == size && bytealg.Equal(unsafe.Slice((*byte)((*slice)(unsafe.Pointer(&bstk)).array), len(bstk)*8), unsafe.Slice((*byte)(((*slice)(unsafe.Pointer(&stk))).array), len(stk)*8)) {
			unlock(&profInsertLock)
			return b
		}
	}

	// Create new bucket.
	b := newBucket(typ, len(stk))
	copy(b.stk(), stk)
	b.hash = h
	b.size = size

	var allnext *atomic.UnsafePointer
	if typ == memProfile {
		allnext = &mbuckets
	} else if typ == mutexProfile {
		allnext = &xbuckets
	} else {
		allnext = &bbuckets
	}

	b.next = (*bucket)(bh[i].Load())
	b.allnext = (*bucket)(allnext.Load())

	bh[i].StoreNoWB(unsafe.Pointer(b))
	allnext.StoreNoWB(unsafe.Pointer(b))

	unlock(&profInsertLock)
	return b
}

// mProf_NextCycle publishes the next heap profile cycle and creates a
// fresh heap profile cycle. This operation is fast and can be done
// during STW. The caller must call mProf_Flush before calling
// mProf_NextCycle again.
//
// This is called by mark termination during STW so allocations and
// frees after the world is started again count towards a new heap
// profiling cycle.
func mProf_NextCycle() {
	mProfCycle.increment()
}

// mProf_Flush flushes the events from the current heap profiling
// cycle into the active profile. After this it is safe to start a new
// heap profiling cycle with mProf_NextCycle.
//
// This is called by GC after mark termination starts the world. In
// contrast with mProf_NextCycle, this is somewhat expensive, but safe
// to do concurrently.
func mProf_Flush() {
	cycle, alreadyFlushed := mProfCycle.setFlushed()
	if alreadyFlushed {
		return
	}

	index := cycle % uint32(len(memRecord{}.future))
	lock(&profMemActiveLock)
	lock(&profMemFutureLock[index])
	mProf_FlushLocked(index)
	unlock(&profMemFutureLock[index])
	unlock(&profMemActiveLock)
}

// mProf_FlushLocked flushes the events from the heap profiling cycle at index
// into the active profile. The caller must hold the lock for the active profile
// (profMemActiveLock) and for the profiling cycle at index
// (profMemFutureLock[index]).
func mProf_FlushLocked(index uint32) {
	assertLockHeld(&profMemActiveLock)
	assertLockHeld(&profMemFutureLock[index])
	head := (*bucket)(mbuckets.Load())
	for b := head; b != nil; b = b.allnext {
		mp := b.mp()

		// Flush cycle C into the published profile and clear
		// it for reuse.
		mpc := &mp.future[index]
		mp.active.add(mpc)
		*mpc = memRecordCycle{}
	}
}

// mProf_PostSweep records that all sweep frees for this GC cycle have
// completed. This has the effect of publishing the heap profile
// snapshot as of the last mark termination without advancing the heap
// profile cycle.
func mProf_PostSweep() {
	// Flush cycle C+1 to the active profile so everything as of
	// the last mark termination becomes visible. *Don't* advance
	// the cycle, since we're still accumulating allocs in cycle
	// C+2, which have to become C+1 in the next mark termination
	// and so on.
	cycle := mProfCycle.read() + 1

	index := cycle % uint32(len(memRecord{}.future))
	lock(&profMemActiveLock)
	lock(&profMemFutureLock[index])
	mProf_FlushLocked(index)
	unlock(&profMemFutureLock[index])
	unlock(&profMemActiveLock)
}

// Called by malloc to record a profiled block.
func mProf_Malloc(p unsafe.Pointer, size uintptr) {
	var stk [maxStack]uintptr
	nstk := callers(4, stk[:])

	index := (mProfCycle.read() + 2) % uint32(len(memRecord{}.future))

	b := stkbucket(memProfile, size, stk[:nstk], true)
	mp := b.mp()
	mpc := &mp.future[index]

	lock(&profMemFutureLock[index])
	mpc.allocs++
	mpc.alloc_bytes += size
	unlock(&profMemFutureLock[index])

	// Setprofilebucket locks a bunch of other mutexes, so we call it outside of
	// the profiler locks. This reduces potential contention and chances of
	// deadlocks. Since the object must be alive during the call to
	// mProf_Malloc, it's fine to do this non-atomically.
	systemstack(func() {
		setprofilebucket(p, b)
	})
}

// Called when freeing a profiled block.
func mProf_Free(b *bucket, size uintptr) {
	index := (mProfCycle.read() + 1) % uint32(len(memRecord{}.future))

	mp := b.mp()
	mpc := &mp.future[index]

	lock(&profMemFutureLock[index])
	mpc.frees++
	mpc.free_bytes += size
	unlock(&profMemFutureLock[index])
}

var blockprofilerate uint64 // in CPU ticks

// SetBlockProfileRate controls the fraction of goroutine blocking events
// that are reported in the blocking profile. The profiler aims to sample
// an average of one blocking event per rate nanoseconds spent blocked.
//
// To include every blocking event in the profile, pass rate = 1.
// To turn off profiling entirely, pass rate <= 0.
func SetBlockProfileRate(rate int) {
	var r int64
	if rate <= 0 {
		r = 0 // disable profiling
	} else if rate == 1 {
		r = 1 // profile everything
	} else {
		// convert ns to cycles, use float64 to prevent overflow during multiplication
		r = int64(float64(rate) * float64(ticksPerSecond()) / (1000 * 1000 * 1000))
		if r == 0 {
			r = 1
		}
	}

	atomic.Store64(&blockprofilerate, uint64(r))
}

func blockevent(cycles int64, skip int) {
	if cycles <= 0 {
		cycles = 1
	}

	rate := int64(atomic.Load64(&blockprofilerate))
	if blocksampled(cycles, rate) {
		saveblockevent(cycles, rate, skip+1, blockProfile)
	}
}

// blocksampled returns true for all events where cycles >= rate. Shorter
// events have a cycles/rate random chance of returning true.
func blocksampled(cycles, rate int64) bool {
	if rate <= 0 || (rate > cycles && cheaprand64()%rate > cycles) {
		return false
	}
	return true
}

func saveblockevent(cycles, rate int64, skip int, which bucketType) {
	gp := getg()
	var nstk int
	var stk [maxStack]uintptr
	if gp.m.curg == nil || gp.m.curg == gp {
		nstk = callers(skip, stk[:])
	} else {
		nstk = gcallers(gp.m.curg, skip, stk[:])
	}

	saveBlockEventStack(cycles, rate, stk[:nstk], which)
}

// lockTimer assists with profiling contention on runtime-internal locks.
//
// There are several steps between the time that an M experiences contention and
// when that contention may be added to the profile. This comes from our
// constraints: We need to keep the critical section of each lock small,
// especially when those locks are contended. The reporting code cannot acquire
// new locks until the M has released all other locks, which means no memory
// allocations and encourages use of (temporary) M-local storage.
//
// The M will have space for storing one call stack that caused contention, and
// for the magnitude of that contention. It will also have space to store the
// magnitude of additional contention the M caused, since it only has space to
// remember one call stack and might encounter several contention events before
// it releases all of its locks and is thus able to transfer the local buffer
// into the profile.
//
// The M will collect the call stack when it unlocks the contended lock. That
// minimizes the impact on the critical section of the contended lock, and
// matches the mutex profile's behavior for contention in sync.Mutex: measured
// at the Unlock method.
//
// The profile for contention on sync.Mutex blames the caller of Unlock for the
// amount of contention experienced by the callers of Lock which had to wait.
// When there are several critical sections, this allows identifying which of
// them is responsible.
//
// Matching that behavior for runtime-internal locks will require identifying
// which Ms are blocked on the mutex. The semaphore-based implementation is
// ready to allow that, but the futex-based implementation will require a bit
// more work. Until then, we report contention on runtime-internal locks with a
// call stack taken from the unlock call (like the rest of the user-space
// "mutex" profile), but assign it a duration value based on how long the
// previous lock call took (like the user-space "block" profile).
//
// Thus, reporting the call stacks of runtime-internal lock contention is
// guarded by GODEBUG for now. Set GODEBUG=runtimecontentionstacks=1 to enable.
//
// TODO(rhysh): plumb through the delay duration, remove GODEBUG, update comment
//
// The M will track this by storing a pointer to the lock; lock/unlock pairs for
// runtime-internal locks are always on the same M.
//
// Together, that demands several steps for recording contention. First, when
// finally acquiring a contended lock, the M decides whether it should plan to
// profile that event by storing a pointer to the lock in its "to be profiled
// upon unlock" field. If that field is already set, it uses the relative
// magnitudes to weight a random choice between itself and the other lock, with
// the loser's time being added to the "additional contention" field. Otherwise
// if the M's call stack buffer is occupied, it does the comparison against that
// sample's magnitude.
//
// Second, having unlocked a mutex the M checks to see if it should capture the
// call stack into its local buffer. Finally, when the M unlocks its last mutex,
// it transfers the local buffer into the profile. As part of that step, it also
// transfers any "additional contention" time to the profile. Any lock
// contention that it experiences while adding samples to the profile will be
// recorded later as "additional contention" and not include a call stack, to
// avoid an echo.
type lockTimer struct {
	lock      *mutex
	timeRate  int64
	timeStart int64
	tickStart int64
}

func (lt *lockTimer) begin() {
	rate := int64(atomic.Load64(&mutexprofilerate))

	lt.timeRate = gTrackingPeriod
	if rate != 0 && rate < lt.timeRate {
		lt.timeRate = rate
	}
	if int64(cheaprand())%lt.timeRate == 0 {
		lt.timeStart = nanotime()
	}

	if rate > 0 && int64(cheaprand())%rate == 0 {
		lt.tickStart = cputicks()
	}
}

func (lt *lockTimer) end() {
	gp := getg()

	if lt.timeStart != 0 {
		nowTime := nanotime()
		gp.m.mLockProfile.waitTime.Add((nowTime - lt.timeStart) * lt.timeRate)
	}

	if lt.tickStart != 0 {
		nowTick := cputicks()
		gp.m.mLockProfile.recordLock(nowTick-lt.tickStart, lt.lock)
	}
}

type mLockProfile struct {
	waitTime   atomic.Int64      // total nanoseconds spent waiting in runtime.lockWithRank
	stack      [maxStack]uintptr // stack that experienced contention in runtime.lockWithRank
	pending    uintptr           // *mutex that experienced contention (to be traceback-ed)
	cycles     int64             // cycles attributable to "pending" (if set), otherwise to "stack"
	cyclesLost int64             // contention for which we weren't able to record a call stack
	disabled   bool              // attribute all time to "lost"
}

func (prof *mLockProfile) recordLock(cycles int64, l *mutex) {
	if cycles <= 0 {
		return
	}

	if prof.disabled {
		// We're experiencing contention while attempting to report contention.
		// Make a note of its magnitude, but don't allow it to be the sole cause
		// of another contention report.
		prof.cyclesLost += cycles
		return
	}

	if uintptr(unsafe.Pointer(l)) == prof.pending {
		// Optimization: we'd already planned to profile this same lock (though
		// possibly from a different unlock site).
		prof.cycles += cycles
		return
	}

	if prev := prof.cycles; prev > 0 {
		// We can only store one call stack for runtime-internal lock contention
		// on this M, and we've already got one. Decide which should stay, and
		// add the other to the report for runtime._LostContendedRuntimeLock.
		prevScore := uint64(cheaprand64()) % uint64(prev)
		thisScore := uint64(cheaprand64()) % uint64(cycles)
		if prevScore > thisScore {
			prof.cyclesLost += cycles
			return
		} else {
			prof.cyclesLost += prev
		}
	}
	// Saving the *mutex as a uintptr is safe because:
	//  - lockrank_on.go does this too, which gives it regular exercise
	//  - the lock would only move if it's stack allocated, which means it
	//      cannot experience multi-M contention
	prof.pending = uintptr(unsafe.Pointer(l))
	prof.cycles = cycles
}

// From unlock2, we might not be holding a p in this code.
//
//go:nowritebarrierrec
func (prof *mLockProfile) recordUnlock(l *mutex) {
	if uintptr(unsafe.Pointer(l)) == prof.pending {
		prof.captureStack()
	}
	if gp := getg(); gp.m.locks == 1 && gp.m.mLockProfile.cycles != 0 {
		prof.store()
	}
}

func (prof *mLockProfile) captureStack() {
	skip := 3 // runtime.(*mLockProfile).recordUnlock runtime.unlock2 runtime.unlockWithRank
	if staticLockRanking {
		// When static lock ranking is enabled, we'll always be on the system
		// stack at this point. There will be a runtime.unlockWithRank.func1
		// frame, and if the call to runtime.unlock took place on a user stack
		// then there'll also be a runtime.systemstack frame. To keep stack
		// traces somewhat consistent whether or not static lock ranking is
		// enabled, we'd like to skip those. But it's hard to tell how long
		// we've been on the system stack so accept an extra frame in that case,
		// with a leaf of "runtime.unlockWithRank runtime.unlock" instead of
		// "runtime.unlock".
		skip += 1 // runtime.unlockWithRank.func1
	}
	prof.pending = 0

	if debug.runtimeContentionStacks.Load() == 0 {
		prof.stack[0] = abi.FuncPCABIInternal(_LostContendedRuntimeLock) + sys.PCQuantum
		prof.stack[1] = 0
		return
	}

	var nstk int
	gp := getg()
	sp := getcallersp()
	pc := getcallerpc()
	systemstack(func() {
		var u unwinder
		u.initAt(pc, sp, 0, gp, unwindSilentErrors|unwindJumpStack)
		nstk = tracebackPCs(&u, skip, prof.stack[:])
	})
	if nstk < len(prof.stack) {
		prof.stack[nstk] = 0
	}
}

func (prof *mLockProfile) store() {
	// Report any contention we experience within this function as "lost"; it's
	// important that the act of reporting a contention event not lead to a
	// reportable contention event. This also means we can use prof.stack
	// without copying, since it won't change during this function.
	mp := acquirem()
	prof.disabled = true

	nstk := maxStack
	for i := 0; i < nstk; i++ {
		if pc := prof.stack[i]; pc == 0 {
			nstk = i
			break
		}
	}

	cycles, lost := prof.cycles, prof.cyclesLost
	prof.cycles, prof.cyclesLost = 0, 0

	rate := int64(atomic.Load64(&mutexprofilerate))
	saveBlockEventStack(cycles, rate, prof.stack[:nstk], mutexProfile)
	if lost > 0 {
		lostStk := [...]uintptr{
			abi.FuncPCABIInternal(_LostContendedRuntimeLock) + sys.PCQuantum,
		}
		saveBlockEventStack(lost, rate, lostStk[:], mutexProfile)
	}

	prof.disabled = false
	releasem(mp)
}

func saveBlockEventStack(cycles, rate int64, stk []uintptr, which bucketType) {
	b := stkbucket(which, 0, stk, true)
	bp := b.bp()

	lock(&profBlockLock)
	// We want to up-scale the count and cycles according to the
	// probability that the event was sampled. For block profile events,
	// the sample probability is 1 if cycles >= rate, and cycles / rate
	// otherwise. For mutex profile events, the sample probability is 1 / rate.
	// We scale the events by 1 / (probability the event was sampled).
	if which == blockProfile && cycles < rate {
		// Remove sampling bias, see discussion on http://golang.org/cl/299991.
		bp.count += float64(rate) / float64(cycles)
		bp.cycles += rate
	} else if which == mutexProfile {
		bp.count += float64(rate)
		bp.cycles += rate * cycles
	} else {
		bp.count++
		bp.cycles += cycles
	}
	unlock(&profBlockLock)
}

var mutexprofilerate uint64 // fraction sampled

// SetMutexProfileFraction controls the fraction of mutex contention events
// that are reported in the mutex profile. On average 1/rate events are
// reported. The previous rate is returned.
//
// To turn off profiling entirely, pass rate 0.
// To just read the current rate, pass rate < 0.
// (For n>1 the details of sampling may change.)
func SetMutexProfileFraction(rate int) int {
	if rate < 0 {
		return int(mutexprofilerate)
	}
	old := mutexprofilerate
	atomic.Store64(&mutexprofilerate, uint64(rate))
	return int(old)
}

//go:linkname mutexevent sync.event
func mutexevent(cycles int64, skip int) {
	if cycles < 0 {
		cycles = 0
	}
	rate := int64(atomic.Load64(&mutexprofilerate))
	if rate > 0 && cheaprand64()%rate == 0 {
		saveblockevent(cycles, rate, skip+1, mutexProfile)
	}
}

// Go interface to profile data.

// A StackRecord describes a single execution stack.
type StackRecord struct {
	Stack0 [32]uintptr // stack trace for this record; ends at first 0 entry
}

// Stack returns the stack trace associated with the record,
// a prefix of r.Stack0.
func (r *StackRecord) Stack() []uintptr {
	for i, v := range r.Stack0 {
		if v == 0 {
			return r.Stack0[0:i]
		}
	}
	return r.Stack0[0:]
}

// MemProfileRate controls the fraction of memory allocations
// that are recorded and reported in the memory profile.
// The profiler aims to sample an average of
// one allocation per MemProfileRate bytes allocated.
//
// To include every allocated block in the profile, set MemProfileRate to 1.
// To turn off profiling entirely, set MemProfileRate to 0.
//
// The tools that process the memory profiles assume that the
// profile rate is constant across the lifetime of the program
// and equal to the current value. Programs that change the
// memory profiling rate should do so just once, as early as
// possible in the execution of the program (for example,
// at the beginning of main).
var MemProfileRate int = 512 * 1024

// disableMemoryProfiling is set by the linker if runtime.MemProfile
// is not used and the link type guarantees nobody else could use it
// elsewhere.
var disableMemoryProfiling bool

// A MemProfileRecord describes the live objects allocated
// by a particular call sequence (stack trace).
type MemProfileRecord struct {
	AllocBytes, FreeBytes     int64       // number of bytes allocated, freed
	AllocObjects, FreeObjects int64       // number of objects allocated, freed
	Stack0                    [32]uintptr // stack trace for this record; ends at first 0 entry
}

// InUseBytes returns the number of bytes in use (AllocBytes - FreeBytes).
func (r *MemProfileRecord) InUseBytes() int64 { return r.AllocBytes - r.FreeBytes }

// InUseObjects returns the number of objects in use (AllocObjects - FreeObjects).
func (r *MemProfileRecord) InUseObjects() int64 {
	return r.AllocObjects - r.FreeObjects
}

// Stack returns the stack trace associated with the record,
// a prefix of r.Stack0.
func (r *MemProfileRecord) Stack() []uintptr {
	for i, v := range r.Stack0 {
		if v == 0 {
			return r.Stack0[0:i]
		}
	}
	return r.Stack0[0:]
}

// MemProfile returns a profile of memory allocated and freed per allocation
// site.
//
// MemProfile returns n, the number of records in the current memory profile.
// If len(p) >= n, MemProfile copies the profile into p and returns n, true.
// If len(p) < n, MemProfile does not change p and returns n, false.
//
// If inuseZero is true, the profile includes allocation records
// where r.AllocBytes > 0 but r.AllocBytes == r.FreeBytes.
// These are sites where memory was allocated, but it has all
// been released back to the runtime.
//
// The returned profile may be up to two garbage collection cycles old.
// This is to avoid skewing the profile toward allocations; because
// allocations happen in real time but frees are delayed until the garbage
// collector performs sweeping, the profile only accounts for allocations
// that have had a chance to be freed by the garbage collector.
//
// Most clients should use the runtime/pprof package or
// the testing package's -test.memprofile flag instead
// of calling MemProfile directly.
func MemProfile(p []MemProfileRecord, inuseZero bool) (n int, ok bool) {
	cycle := mProfCycle.read()
	// If we're between mProf_NextCycle and mProf_Flush, take care
	// of flushing to the active profile so we only have to look
	// at the active profile below.
	index := cycle % uint32(len(memRecord{}.future))
	lock(&profMemActiveLock)
	lock(&profMemFutureLock[index])
	mProf_FlushLocked(index)
	unlock(&profMemFutureLock[index])
	clear := true
	head := (*bucket)(mbuckets.Load())
	for b := head; b != nil; b = b.allnext {
		mp := b.mp()
		if inuseZero || mp.active.alloc_bytes != mp.active.free_bytes {
			n++
		}
		if mp.active.allocs != 0 || mp.active.frees != 0 {
			clear = false
		}
	}
	if clear {
		// Absolutely no data, suggesting that a garbage collection
		// has not yet happened. In order to allow profiling when
		// garbage collection is disabled from the beginning of execution,
		// accumulate all of the cycles, and recount buckets.
		n = 0
		for b := head; b != nil; b = b.allnext {
			mp := b.mp()
			for c := range mp.future {
				lock(&profMemFutureLock[c])
				mp.active.add(&mp.future[c])
				mp.future[c] = memRecordCycle{}
				unlock(&profMemFutureLock[c])
			}
			if inuseZero || mp.active.alloc_bytes != mp.active.free_bytes {
				n++
			}
		}
	}
	if n <= len(p) {
		ok = true
		idx := 0
		for b := head; b != nil; b = b.allnext {
			mp := b.mp()
			if inuseZero || mp.active.alloc_bytes != mp.active.free_bytes {
				record(&p[idx], b)
				idx++
			}
		}
	}
	unlock(&profMemActiveLock)
	return
}

// Write b's data to r.
func record(r *MemProfileRecord, b *bucket) {
	mp := b.mp()
	r.AllocBytes = int64(mp.active.alloc_bytes)
	r.FreeBytes = int64(mp.active.free_bytes)
	r.AllocObjects = int64(mp.active.allocs)
	r.FreeObjects = int64(mp.active.frees)
	if raceenabled {
		racewriterangepc(unsafe.Pointer(&r.Stack0[0]), unsafe.Sizeof(r.Stack0), getcallerpc(), abi.FuncPCABIInternal(MemProfile))
	}
	if msanenabled {
		msanwrite(unsafe.Pointer(&r.Stack0[0]), unsafe.Sizeof(r.Stack0))
	}
	if asanenabled {
		asanwrite(unsafe.Pointer(&r.Stack0[0]), unsafe.Sizeof(r.Stack0))
	}
	copy(r.Stack0[:], b.stk())
	for i := int(b.nstk); i < len(r.Stack0); i++ {
		r.Stack0[i] = 0
	}
}

func iterate_memprof(fn func(*bucket, uintptr, *uintptr, uintptr, uintptr, uintptr)) {
	lock(&profMemActiveLock)
	head := (*bucket)(mbuckets.Load())
	for b := head; b != nil; b = b.allnext {
		mp := b.mp()
		fn(b, b.nstk, &b.stk()[0], b.size, mp.active.allocs, mp.active.frees)
	}
	unlock(&profMemActiveLock)
}

// BlockProfileRecord describes blocking events originated
// at a particular call sequence (stack trace).
type BlockProfileRecord struct {
	Count  int64
	Cycles int64
	StackRecord
}

// BlockProfile returns n, the number of records in the current blocking profile.
// If len(p) >= n, BlockProfile copies the profile into p and returns n, true.
// If len(p) < n, BlockProfile does not change p and returns n, false.
//
// Most clients should use the runtime/pprof package or
// the testing package's -test.blockprofile flag instead
// of calling BlockProfile directly.
func BlockProfile(p []BlockProfileRecord) (n int, ok bool) {
	lock(&profBlockLock)
	head := (*bucket)(bbuckets.Load())
	for b := head; b != nil; b = b.allnext {
		n++
	}
	if n <= len(p) {
		ok = true
		for b := head; b != nil; b = b.allnext {
			bp := b.bp()
			r := &p[0]
			r.Count = int64(bp.count)
			// Prevent callers from having to worry about division by zero errors.
			// See discussion on http://golang.org/cl/299991.
			if r.Count == 0 {
				r.Count = 1
			}
			r.Cycles = bp.cycles
			if raceenabled {
				racewriterangepc(unsafe.Pointer(&r.Stack0[0]), unsafe.Sizeof(r.Stack0), getcallerpc(), abi.FuncPCABIInternal(BlockProfile))
			}
			if msanenabled {
				msanwrite(unsafe.Pointer(&r.Stack0[0]), unsafe.Sizeof(r.Stack0))
			}
			if asanenabled {
				asanwrite(unsafe.Pointer(&r.Stack0[0]), unsafe.Sizeof(r.Stack0))
			}
			i := copy(r.Stack0[:], b.stk())
			for ; i < len(r.Stack0); i++ {
				r.Stack0[i] = 0
			}
			p = p[1:]
		}
	}
	unlock(&profBlockLock)
	return
}

// MutexProfile returns n, the number of records in the current mutex profile.
// If len(p) >= n, MutexProfile copies the profile into p and returns n, true.
// Otherwise, MutexProfile does not change p, and returns n, false.
//
// Most clients should use the [runtime/pprof] package
// instead of calling MutexProfile directly.
func MutexProfile(p []BlockProfileRecord) (n int, ok bool) {
	lock(&profBlockLock)
	head := (*bucket)(xbuckets.Load())
	for b := head; b != nil; b = b.allnext {
		n++
	}
	if n <= len(p) {
		ok = true
		for b := head; b != nil; b = b.allnext {
			bp := b.bp()
			r := &p[0]
			r.Count = int64(bp.count)
			r.Cycles = bp.cycles
			i := copy(r.Stack0[:], b.stk())
			for ; i < len(r.Stack0); i++ {
				r.Stack0[i] = 0
			}
			p = p[1:]
		}
	}
	unlock(&profBlockLock)
	return
}

// ThreadCreateProfile returns n, the number of records in the thread creation profile.
// If len(p) >= n, ThreadCreateProfile copies the profile into p and returns n, true.
// If len(p) < n, ThreadCreateProfile does not change p and returns n, false.
//
// Most clients should use the runtime/pprof package instead
// of calling ThreadCreateProfile directly.
func ThreadCreateProfile(p []StackRecord) (n int, ok bool) {
	first := (*m)(atomic.Loadp(unsafe.Pointer(&allm)))
	for mp := first; mp != nil; mp = mp.alllink {
		n++
	}
	if n <= len(p) {
		ok = true
		i := 0
		for mp := first; mp != nil; mp = mp.alllink {
			p[i].Stack0 = mp.createstack
			i++
		}
	}
	return
}

//go:linkname runtime_goroutineProfileWithLabels runtime/pprof.runtime_goroutineProfileWithLabels
func runtime_goroutineProfileWithLabels(p []StackRecord, labels []unsafe.Pointer) (n int, ok bool) {
	return goroutineProfileWithLabels(p, labels)
}

// labels may be nil. If labels is non-nil, it must have the same length as p.
func goroutineProfileWithLabels(p []StackRecord, labels []unsafe.Pointer) (n int, ok bool) {
	if labels != nil && len(labels) != len(p) {
		labels = nil
	}

	return goroutineProfileWithLabelsConcurrent(p, labels)
}

var goroutineProfile = struct {
	sema    uint32
	active  bool
	offset  atomic.Int64
	records []StackRecord
	labels  []unsafe.Pointer
}{
	sema: 1,
}

// goroutineProfileState indicates the status of a goroutine's stack for the
// current in-progress goroutine profile. Goroutines' stacks are initially
// "Absent" from the profile, and end up "Satisfied" by the time the profile is
// complete. While a goroutine's stack is being captured, its
// goroutineProfileState will be "InProgress" and it will not be able to run
// until the capture completes and the state moves to "Satisfied".
//
// Some goroutines (the finalizer goroutine, which at various times can be
// either a "system" or a "user" goroutine, and the goroutine that is
// coordinating the profile, any goroutines created during the profile) move
// directly to the "Satisfied" state.
type goroutineProfileState uint32

const (
	goroutineProfileAbsent goroutineProfileState = iota
	goroutineProfileInProgress
	goroutineProfileSatisfied
)

type goroutineProfileStateHolder atomic.Uint32

func (p *goroutineProfileStateHolder) Load() goroutineProfileState {
	return goroutineProfileState((*atomic.Uint32)(p).Load())
}

func (p *goroutineProfileStateHolder) Store(value goroutineProfileState) {
	(*atomic.Uint32)(p).Store(uint32(value))
}

func (p *goroutineProfileStateHolder) CompareAndSwap(old, new goroutineProfileState) bool {
	return (*atomic.Uint32)(p).CompareAndSwap(uint32(old), uint32(new))
}

func goroutineProfileWithLabelsConcurrent(p []StackRecord, labels []unsafe.Pointer) (n int, ok bool) {
	semacquire(&goroutineProfile.sema)

	ourg := getg()

	stw := stopTheWorld(stwGoroutineProfile)
	// Using gcount while the world is stopped should give us a consistent view
	// of the number of live goroutines, minus the number of goroutines that are
	// alive and permanently marked as "system". But to make this count agree
	// with what we'd get from isSystemGoroutine, we need special handling for
	// goroutines that can vary between user and system to ensure that the count
	// doesn't change during the collection. So, check the finalizer goroutine
	// in particular.
	n = int(gcount())
	if fingStatus.Load()&fingRunningFinalizer != 0 {
		n++
	}

	if n > len(p) {
		// There's not enough space in p to store the whole profile, so (per the
		// contract of runtime.GoroutineProfile) we're not allowed to write to p
		// at all and must return n, false.
		startTheWorld(stw)
		semrelease(&goroutineProfile.sema)
		return n, false
	}

	// Save current goroutine.
	sp := getcallersp()
	pc := getcallerpc()
	systemstack(func() {
		saveg(pc, sp, ourg, &p[0])
	})
	if labels != nil {
		labels[0] = ourg.labels
	}
	ourg.goroutineProfiled.Store(goroutineProfileSatisfied)
	goroutineProfile.offset.Store(1)

	// Prepare for all other goroutines to enter the profile. Aside from ourg,
	// every goroutine struct in the allgs list has its goroutineProfiled field
	// cleared. Any goroutine created from this point on (while
	// goroutineProfile.active is set) will start with its goroutineProfiled
	// field set to goroutineProfileSatisfied.
	goroutineProfile.active = true
	goroutineProfile.records = p
	goroutineProfile.labels = labels
	// The finalizer goroutine needs special handling because it can vary over
	// time between being a user goroutine (eligible for this profile) and a
	// system goroutine (to be excluded). Pick one before restarting the world.
	if fing != nil {
		fing.goroutineProfiled.Store(goroutineProfileSatisfied)
		if readgstatus(fing) != _Gdead && !isSystemGoroutine(fing, false) {
			doRecordGoroutineProfile(fing)
		}
	}
	startTheWorld(stw)

	// Visit each goroutine that existed as of the startTheWorld call above.
	//
	// New goroutines may not be in this list, but we didn't want to know about
	// them anyway. If they do appear in this list (via reusing a dead goroutine
	// struct, or racing to launch between the world restarting and us getting
	// the list), they will already have their goroutineProfiled field set to
	// goroutineProfileSatisfied before their state transitions out of _Gdead.
	//
	// Any goroutine that the scheduler tries to execute concurrently with this
	// call will start by adding itself to the profile (before the act of
	// executing can cause any changes in its stack).
	forEachGRace(func(gp1 *g) {
		tryRecordGoroutineProfile(gp1, Gosched)
	})

	stw = stopTheWorld(stwGoroutineProfileCleanup)
	endOffset := goroutineProfile.offset.Swap(0)
	goroutineProfile.active = false
	goroutineProfile.records = nil
	goroutineProfile.labels = nil
	startTheWorld(stw)

	// Restore the invariant that every goroutine struct in allgs has its
	// goroutineProfiled field cleared.
	forEachGRace(func(gp1 *g) {
		gp1.goroutineProfiled.Store(goroutineProfileAbsent)
	})

	if raceenabled {
		raceacquire(unsafe.Pointer(&labelSync))
	}

	if n != int(endOffset) {
		// It's a big surprise that the number of goroutines changed while we
		// were collecting the profile. But probably better to return a
		// truncated profile than to crash the whole process.
		//
		// For instance, needm moves a goroutine out of the _Gdead state and so
		// might be able to change the goroutine count without interacting with
		// the scheduler. For code like that, the race windows are small and the
		// combination of features is uncommon, so it's hard to be (and remain)
		// sure we've caught them all.
	}

	semrelease(&goroutineProfile.sema)
	return n, true
}

// tryRecordGoroutineProfileWB asserts that write barriers are allowed and calls
// tryRecordGoroutineProfile.
//
//go:yeswritebarrierrec
func tryRecordGoroutineProfileWB(gp1 *g) {
	if getg().m.p.ptr() == nil {
		throw("no P available, write barriers are forbidden")
	}
	tryRecordGoroutineProfile(gp1, osyield)
}

// tryRecordGoroutineProfile ensures that gp1 has the appropriate representation
// in the current goroutine profile: either that it should not be profiled, or
// that a snapshot of its call stack and labels are now in the profile.
func tryRecordGoroutineProfile(gp1 *g, yield func()) {
	if readgstatus(gp1) == _Gdead {
		// Dead goroutines should not appear in the profile. Goroutines that
		// start while profile collection is active will get goroutineProfiled
		// set to goroutineProfileSatisfied before transitioning out of _Gdead,
		// so here we check _Gdead first.
		return
	}
	if isSystemGoroutine(gp1, true) {
		// System goroutines should not appear in the profile. (The finalizer
		// goroutine is marked as "already profiled".)
		return
	}

	for {
		prev := gp1.goroutineProfiled.Load()
		if prev == goroutineProfileSatisfied {
			// This goroutine is already in the profile (or is new since the
			// start of collection, so shouldn't appear in the profile).
			break
		}
		if prev == goroutineProfileInProgress {
			// Something else is adding gp1 to the goroutine profile right now.
			// Give that a moment to finish.
			yield()
			continue
		}

		// While we have gp1.goroutineProfiled set to
		// goroutineProfileInProgress, gp1 may appear _Grunnable but will not
		// actually be able to run. Disable preemption for ourselves, to make
		// sure we finish profiling gp1 right away instead of leaving it stuck
		// in this limbo.
		mp := acquirem()
		if gp1.goroutineProfiled.CompareAndSwap(goroutineProfileAbsent, goroutineProfileInProgress) {
			doRecordGoroutineProfile(gp1)
			gp1.goroutineProfiled.Store(goroutineProfileSatisfied)
		}
		releasem(mp)
	}
}

// doRecordGoroutineProfile writes gp1's call stack and labels to an in-progress
// goroutine profile. Preemption is disabled.
//
// This may be called via tryRecordGoroutineProfile in two ways: by the
// goroutine that is coordinating the goroutine profile (running on its own
// stack), or from the scheduler in preparation to execute gp1 (running on the
// system stack).
func doRecordGoroutineProfile(gp1 *g) {
	if readgstatus(gp1) == _Grunning {
		print("doRecordGoroutineProfile gp1=", gp1.goid, "\n")
		throw("cannot read stack of running goroutine")
	}

	offset := int(goroutineProfile.offset.Add(1)) - 1

	if offset >= len(goroutineProfile.records) {
		// Should be impossible, but better to return a truncated profile than
		// to crash the entire process at this point. Instead, deal with it in
		// goroutineProfileWithLabelsConcurrent where we have more context.
		return
	}

	// saveg calls gentraceback, which may call cgo traceback functions. When
	// called from the scheduler, this is on the system stack already so
	// traceback.go:cgoContextPCs will avoid calling back into the scheduler.
	//
	// When called from the goroutine coordinating the profile, we still have
	// set gp1.goroutineProfiled to goroutineProfileInProgress and so are still
	// preventing it from being truly _Grunnable. So we'll use the system stack
	// to avoid schedule delays.
	systemstack(func() { saveg(^uintptr(0), ^uintptr(0), gp1, &goroutineProfile.records[offset]) })

	if goroutineProfile.labels != nil {
		goroutineProfile.labels[offset] = gp1.labels
	}
}

func goroutineProfileWithLabelsSync(p []StackRecord, labels []unsafe.Pointer) (n int, ok bool) {
	gp := getg()

	isOK := func(gp1 *g) bool {
		// Checking isSystemGoroutine here makes GoroutineProfile
		// consistent with both NumGoroutine and Stack.
		return gp1 != gp && readgstatus(gp1) != _Gdead && !isSystemGoroutine(gp1, false)
	}

	stw := stopTheWorld(stwGoroutineProfile)

	// World is stopped, no locking required.
	n = 1
	forEachGRace(func(gp1 *g) {
		if isOK(gp1) {
			n++
		}
	})

	if n <= len(p) {
		ok = true
		r, lbl := p, labels

		// Save current goroutine.
		sp := getcallersp()
		pc := getcallerpc()
		systemstack(func() {
			saveg(pc, sp, gp, &r[0])
		})
		r = r[1:]

		// If we have a place to put our goroutine labelmap, insert it there.
		if labels != nil {
			lbl[0] = gp.labels
			lbl = lbl[1:]
		}

		// Save other goroutines.
		forEachGRace(func(gp1 *g) {
			if !isOK(gp1) {
				return
			}

			if len(r) == 0 {
				// Should be impossible, but better to return a
				// truncated profile than to crash the entire process.
				return
			}
			// saveg calls gentraceback, which may call cgo traceback functions.
			// The world is stopped, so it cannot use cgocall (which will be
			// blocked at exitsyscall). Do it on the system stack so it won't
			// call into the schedular (see traceback.go:cgoContextPCs).
			systemstack(func() { saveg(^uintptr(0), ^uintptr(0), gp1, &r[0]) })
			if labels != nil {
				lbl[0] = gp1.labels
				lbl = lbl[1:]
			}
			r = r[1:]
		})
	}

	if raceenabled {
		raceacquire(unsafe.Pointer(&labelSync))
	}

	startTheWorld(stw)
	return n, ok
}

// GoroutineProfile returns n, the number of records in the active goroutine stack profile.
// If len(p) >= n, GoroutineProfile copies the profile into p and returns n, true.
// If len(p) < n, GoroutineProfile does not change p and returns n, false.
//
// Most clients should use the [runtime/pprof] package instead
// of calling GoroutineProfile directly.
func GoroutineProfile(p []StackRecord) (n int, ok bool) {

	return goroutineProfileWithLabels(p, nil)
}

func saveg(pc, sp uintptr, gp *g, r *StackRecord) {
	var u unwinder
	u.initAt(pc, sp, 0, gp, unwindSilentErrors)
	n := tracebackPCs(&u, 0, r.Stack0[:])
	if n < len(r.Stack0) {
		r.Stack0[n] = 0
	}
}

// Stack formats a stack trace of the calling goroutine into buf
// and returns the number of bytes written to buf.
// If all is true, Stack formats stack traces of all other goroutines
// into buf after the trace for the current goroutine.
func Stack(buf []byte, all bool) int {
	var stw worldStop
	if all {
		stw = stopTheWorld(stwAllGoroutinesStack)
	}

	n := 0
	if len(buf) > 0 {
		gp := getg()
		sp := getcallersp()
		pc := getcallerpc()
		systemstack(func() {
			g0 := getg()
			// Force traceback=1 to override GOTRACEBACK setting,
			// so that Stack's results are consistent.
			// GOTRACEBACK is only about crash dumps.
			g0.m.traceback = 1
			g0.writebuf = buf[0:0:len(buf)]
			goroutineheader(gp)
			traceback(pc, sp, 0, gp)
			if all {
				tracebackothers(gp)
			}
			g0.m.traceback = 0
			n = len(g0.writebuf)
			g0.writebuf = nil
		})
	}

	if all {
		startTheWorld(stw)
	}
	return n
}

// Tracing of alloc/free/gc.

var tracelock mutex

func tracealloc(p unsafe.Pointer, size uintptr, typ *_type) {
	lock(&tracelock)
	gp := getg()
	gp.m.traceback = 2
	if typ == nil {
		print("tracealloc(", p, ", ", hex(size), ")\n")
	} else {
		print("tracealloc(", p, ", ", hex(size), ", ", toRType(typ).string(), ")\n")
	}
	if gp.m.curg == nil || gp == gp.m.curg {
		goroutineheader(gp)
		pc := getcallerpc()
		sp := getcallersp()
		systemstack(func() {
			traceback(pc, sp, 0, gp)
		})
	} else {
		goroutineheader(gp.m.curg)
		traceback(^uintptr(0), ^uintptr(0), 0, gp.m.curg)
	}
	print("\n")
	gp.m.traceback = 0
	unlock(&tracelock)
}

func tracefree(p unsafe.Pointer, size uintptr) {
	lock(&tracelock)
	gp := getg()
	gp.m.traceback = 2
	print("tracefree(", p, ", ", hex(size), ")\n")
	goroutineheader(gp)
	pc := getcallerpc()
	sp := getcallersp()
	systemstack(func() {
		traceback(pc, sp, 0, gp)
	})
	print("\n")
	gp.m.traceback = 0
	unlock(&tracelock)
}

func tracegc() {
	lock(&tracelock)
	gp := getg()
	gp.m.traceback = 2
	print("tracegc()\n")
	// running on m->g0 stack; show all non-g0 goroutines
	tracebackothers(gp)
	print("end tracegc\n")
	print("\n")
	gp.m.traceback = 0
	unlock(&tracelock)
}
