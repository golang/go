// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Malloc profiling.
// Patterned after tcmalloc's algorithms; shorter code.

package runtime

import (
	"unsafe"
)

// NOTE(rsc): Everything here could use cas if contention became an issue.
var proflock mutex

// All memory allocations are local and do not escape outside of the profiler.
// The profiler is forbidden from referring to garbage-collected memory.

const (
	// profile types
	memProfile bucketType = 1 + iota
	blockProfile

	// size of bucket hash table
	buckHashSize = 179999

	// max depth of stack to record in bucket
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
type bucket struct {
	next    *bucket
	allnext *bucket
	typ     bucketType // memBucket or blockBucket
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
	// Mallocs are accounted in recent stats.
	// Explicit frees are accounted in recent stats.
	// GC frees are accounted in prev stats.
	// After GC prev stats are added to final stats and
	// recent stats are moved into prev stats.
	allocs      uintptr
	frees       uintptr
	alloc_bytes uintptr
	free_bytes  uintptr

	// changes between next-to-last GC and last GC
	prev_allocs      uintptr
	prev_frees       uintptr
	prev_alloc_bytes uintptr
	prev_free_bytes  uintptr

	// changes since last GC
	recent_allocs      uintptr
	recent_frees       uintptr
	recent_alloc_bytes uintptr
	recent_free_bytes  uintptr
}

// A blockRecord is the bucket data for a bucket of type blockProfile,
// part of the blocking profile.
type blockRecord struct {
	count  int64
	cycles int64
}

var (
	mbuckets  *bucket // memory profile buckets
	bbuckets  *bucket // blocking profile buckets
	buckhash  *[179999]*bucket
	bucketmem uintptr
)

// newBucket allocates a bucket with the given type and number of stack entries.
func newBucket(typ bucketType, nstk int) *bucket {
	size := unsafe.Sizeof(bucket{}) + uintptr(nstk)*unsafe.Sizeof(uintptr(0))
	switch typ {
	default:
		gothrow("invalid profile bucket type")
	case memProfile:
		size += unsafe.Sizeof(memRecord{})
	case blockProfile:
		size += unsafe.Sizeof(blockRecord{})
	}

	b := (*bucket)(persistentalloc(size, 0, &memstats.buckhash_sys))
	bucketmem += size
	b.typ = typ
	b.nstk = uintptr(nstk)
	return b
}

// stk returns the slice in b holding the stack.
func (b *bucket) stk() []uintptr {
	stk := (*[maxStack]uintptr)(add(unsafe.Pointer(b), unsafe.Sizeof(*b)))
	return stk[:b.nstk:b.nstk]
}

// mp returns the memRecord associated with the memProfile bucket b.
func (b *bucket) mp() *memRecord {
	if b.typ != memProfile {
		gothrow("bad use of bucket.mp")
	}
	data := add(unsafe.Pointer(b), unsafe.Sizeof(*b)+b.nstk*unsafe.Sizeof(uintptr(0)))
	return (*memRecord)(data)
}

// bp returns the blockRecord associated with the blockProfile bucket b.
func (b *bucket) bp() *blockRecord {
	if b.typ != blockProfile {
		gothrow("bad use of bucket.bp")
	}
	data := add(unsafe.Pointer(b), unsafe.Sizeof(*b)+b.nstk*unsafe.Sizeof(uintptr(0)))
	return (*blockRecord)(data)
}

// Return the bucket for stk[0:nstk], allocating new bucket if needed.
func stkbucket(typ bucketType, size uintptr, stk []uintptr, alloc bool) *bucket {
	if buckhash == nil {
		buckhash = (*[buckHashSize]*bucket)(sysAlloc(unsafe.Sizeof(*buckhash), &memstats.buckhash_sys))
		if buckhash == nil {
			gothrow("runtime: cannot allocate memory")
		}
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
	for b := buckhash[i]; b != nil; b = b.next {
		if b.typ == typ && b.hash == h && b.size == size && eqslice(b.stk(), stk) {
			return b
		}
	}

	if !alloc {
		return nil
	}

	// Create new bucket.
	b := newBucket(typ, len(stk))
	copy(b.stk(), stk)
	b.hash = h
	b.size = size
	b.next = buckhash[i]
	buckhash[i] = b
	if typ == memProfile {
		b.allnext = mbuckets
		mbuckets = b
	} else {
		b.allnext = bbuckets
		bbuckets = b
	}
	return b
}

func sysAlloc(n uintptr, stat *uint64) unsafe.Pointer

func eqslice(x, y []uintptr) bool {
	if len(x) != len(y) {
		return false
	}
	for i, xi := range x {
		if xi != y[i] {
			return false
		}
	}
	return true
}

func mprof_GC() {
	for b := mbuckets; b != nil; b = b.allnext {
		mp := b.mp()
		mp.allocs += mp.prev_allocs
		mp.frees += mp.prev_frees
		mp.alloc_bytes += mp.prev_alloc_bytes
		mp.free_bytes += mp.prev_free_bytes

		mp.prev_allocs = mp.recent_allocs
		mp.prev_frees = mp.recent_frees
		mp.prev_alloc_bytes = mp.recent_alloc_bytes
		mp.prev_free_bytes = mp.recent_free_bytes

		mp.recent_allocs = 0
		mp.recent_frees = 0
		mp.recent_alloc_bytes = 0
		mp.recent_free_bytes = 0
	}
}

// Record that a gc just happened: all the 'recent' statistics are now real.
func mProf_GC() {
	lock(&proflock)
	mprof_GC()
	unlock(&proflock)
}

// Called by malloc to record a profiled block.
func mProf_Malloc(p unsafe.Pointer, size uintptr) {
	var stk [maxStack]uintptr
	nstk := callers(4, &stk[0], len(stk))
	lock(&proflock)
	b := stkbucket(memProfile, size, stk[:nstk], true)
	mp := b.mp()
	mp.recent_allocs++
	mp.recent_alloc_bytes += size
	unlock(&proflock)

	// Setprofilebucket locks a bunch of other mutexes, so we call it outside of proflock.
	// This reduces potential contention and chances of deadlocks.
	// Since the object must be alive during call to mProf_Malloc,
	// it's fine to do this non-atomically.
	setprofilebucket(p, b)
}

func setprofilebucket_m() // mheap.c

func setprofilebucket(p unsafe.Pointer, b *bucket) {
	g := getg()
	g.m.ptrarg[0] = p
	g.m.ptrarg[1] = unsafe.Pointer(b)
	onM(setprofilebucket_m)
}

// Called when freeing a profiled block.
func mProf_Free(b *bucket, size uintptr, freed bool) {
	lock(&proflock)
	mp := b.mp()
	if freed {
		mp.recent_frees++
		mp.recent_free_bytes += size
	} else {
		mp.prev_frees++
		mp.prev_free_bytes += size
	}
	unlock(&proflock)
}

var blockprofilerate uint64 // in CPU ticks

// SetBlockProfileRate controls the fraction of goroutine blocking events
// that are reported in the blocking profile.  The profiler aims to sample
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
		r = int64(float64(rate) * float64(tickspersecond()) / (1000 * 1000 * 1000))
		if r == 0 {
			r = 1
		}
	}

	atomicstore64(&blockprofilerate, uint64(r))
}

func blockevent(cycles int64, skip int) {
	if cycles <= 0 {
		cycles = 1
	}
	rate := int64(atomicload64(&blockprofilerate))
	if rate <= 0 || (rate > cycles && int64(fastrand1())%rate > cycles) {
		return
	}
	gp := getg()
	var nstk int
	var stk [maxStack]uintptr
	if gp.m.curg == nil || gp.m.curg == gp {
		nstk = callers(skip, &stk[0], len(stk))
	} else {
		nstk = gcallers(gp.m.curg, skip, &stk[0], len(stk))
	}
	lock(&proflock)
	b := stkbucket(blockProfile, 0, stk[:nstk], true)
	b.bp().count++
	b.bp().cycles += cycles
	unlock(&proflock)
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
// and equal to the current value.  Programs that change the
// memory profiling rate should do so just once, as early as
// possible in the execution of the program (for example,
// at the beginning of main).
var MemProfileRate int = 512 * 1024

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

// MemProfile returns n, the number of records in the current memory profile.
// If len(p) >= n, MemProfile copies the profile into p and returns n, true.
// If len(p) < n, MemProfile does not change p and returns n, false.
//
// If inuseZero is true, the profile includes allocation records
// where r.AllocBytes > 0 but r.AllocBytes == r.FreeBytes.
// These are sites where memory was allocated, but it has all
// been released back to the runtime.
//
// Most clients should use the runtime/pprof package or
// the testing package's -test.memprofile flag instead
// of calling MemProfile directly.
func MemProfile(p []MemProfileRecord, inuseZero bool) (n int, ok bool) {
	lock(&proflock)
	clear := true
	for b := mbuckets; b != nil; b = b.allnext {
		mp := b.mp()
		if inuseZero || mp.alloc_bytes != mp.free_bytes {
			n++
		}
		if mp.allocs != 0 || mp.frees != 0 {
			clear = false
		}
	}
	if clear {
		// Absolutely no data, suggesting that a garbage collection
		// has not yet happened. In order to allow profiling when
		// garbage collection is disabled from the beginning of execution,
		// accumulate stats as if a GC just happened, and recount buckets.
		mprof_GC()
		mprof_GC()
		n = 0
		for b := mbuckets; b != nil; b = b.allnext {
			mp := b.mp()
			if inuseZero || mp.alloc_bytes != mp.free_bytes {
				n++
			}
		}
	}
	if n <= len(p) {
		ok = true
		idx := 0
		for b := mbuckets; b != nil; b = b.allnext {
			mp := b.mp()
			if inuseZero || mp.alloc_bytes != mp.free_bytes {
				record(&p[idx], b)
				idx++
			}
		}
	}
	unlock(&proflock)
	return
}

// Write b's data to r.
func record(r *MemProfileRecord, b *bucket) {
	mp := b.mp()
	r.AllocBytes = int64(mp.alloc_bytes)
	r.FreeBytes = int64(mp.free_bytes)
	r.AllocObjects = int64(mp.allocs)
	r.FreeObjects = int64(mp.frees)
	copy(r.Stack0[:], b.stk())
	for i := int(b.nstk); i < len(r.Stack0); i++ {
		r.Stack0[i] = 0
	}
}

func iterate_memprof(fn func(*bucket, uintptr, *uintptr, uintptr, uintptr, uintptr)) {
	lock(&proflock)
	for b := mbuckets; b != nil; b = b.allnext {
		mp := b.mp()
		fn(b, uintptr(b.nstk), &b.stk()[0], b.size, mp.allocs, mp.frees)
	}
	unlock(&proflock)
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
	lock(&proflock)
	for b := bbuckets; b != nil; b = b.allnext {
		n++
	}
	if n <= len(p) {
		ok = true
		for b := bbuckets; b != nil; b = b.allnext {
			bp := b.bp()
			r := &p[0]
			r.Count = int64(bp.count)
			r.Cycles = int64(bp.cycles)
			i := copy(r.Stack0[:], b.stk())
			for ; i < len(r.Stack0); i++ {
				r.Stack0[i] = 0
			}
			p = p[1:]
		}
	}
	unlock(&proflock)
	return
}

// ThreadCreateProfile returns n, the number of records in the thread creation profile.
// If len(p) >= n, ThreadCreateProfile copies the profile into p and returns n, true.
// If len(p) < n, ThreadCreateProfile does not change p and returns n, false.
//
// Most clients should use the runtime/pprof package instead
// of calling ThreadCreateProfile directly.
func ThreadCreateProfile(p []StackRecord) (n int, ok bool) {
	first := (*m)(atomicloadp(unsafe.Pointer(&allm)))
	for mp := first; mp != nil; mp = mp.alllink {
		n++
	}
	if n <= len(p) {
		ok = true
		i := 0
		for mp := first; mp != nil; mp = mp.alllink {
			for s := range mp.createstack {
				p[i].Stack0[s] = uintptr(mp.createstack[s])
			}
			i++
		}
	}
	return
}

var allgs []*g // proc.c

// GoroutineProfile returns n, the number of records in the active goroutine stack profile.
// If len(p) >= n, GoroutineProfile copies the profile into p and returns n, true.
// If len(p) < n, GoroutineProfile does not change p and returns n, false.
//
// Most clients should use the runtime/pprof package instead
// of calling GoroutineProfile directly.
func GoroutineProfile(p []StackRecord) (n int, ok bool) {

	n = NumGoroutine()
	if n <= len(p) {
		gp := getg()
		semacquire(&worldsema, false)
		gp.m.gcing = 1
		onM(stoptheworld)

		n = NumGoroutine()
		if n <= len(p) {
			ok = true
			r := p
			sp := getcallersp(unsafe.Pointer(&p))
			pc := getcallerpc(unsafe.Pointer(&p))
			onM(func() {
				saveg(pc, sp, gp, &r[0])
			})
			r = r[1:]
			for _, gp1 := range allgs {
				if gp1 == gp || readgstatus(gp1) == _Gdead {
					continue
				}
				saveg(^uintptr(0), ^uintptr(0), gp1, &r[0])
				r = r[1:]
			}
		}

		gp.m.gcing = 0
		semrelease(&worldsema)
		onM(starttheworld)
	}

	return n, ok
}

func saveg(pc, sp uintptr, gp *g, r *StackRecord) {
	n := gentraceback(pc, sp, 0, gp, 0, &r.Stack0[0], len(r.Stack0), nil, nil, 0)
	if n < len(r.Stack0) {
		r.Stack0[n] = 0
	}
}

// Stack formats a stack trace of the calling goroutine into buf
// and returns the number of bytes written to buf.
// If all is true, Stack formats stack traces of all other goroutines
// into buf after the trace for the current goroutine.
func Stack(buf []byte, all bool) int {
	if all {
		semacquire(&worldsema, false)
		gp := getg()
		gp.m.gcing = 1
		onM(stoptheworld)
	}

	n := 0
	if len(buf) > 0 {
		gp := getg()
		sp := getcallersp(unsafe.Pointer(&buf))
		pc := getcallerpc(unsafe.Pointer(&buf))
		onM(func() {
			g0 := getg()
			g0.writebuf = buf[0:0:len(buf)]
			goroutineheader(gp)
			traceback(pc, sp, 0, gp)
			if all {
				tracebackothers(gp)
			}
			n = len(g0.writebuf)
			g0.writebuf = nil
		})
	}

	if all {
		gp := getg()
		gp.m.gcing = 0
		semrelease(&worldsema)
		onM(starttheworld)
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
		print("tracealloc(", p, ", ", hex(size), ", ", *typ._string, ")\n")
	}
	if gp.m.curg == nil || gp == gp.m.curg {
		goroutineheader(gp)
		pc := getcallerpc(unsafe.Pointer(&p))
		sp := getcallersp(unsafe.Pointer(&p))
		onM(func() {
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
	pc := getcallerpc(unsafe.Pointer(&p))
	sp := getcallersp(unsafe.Pointer(&p))
	onM(func() {
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
