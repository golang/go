// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

// Malloc profiling.
// Patterned after tcmalloc's algorithms; shorter code.

// NOTE(rsc): Everything here could use cas if contention became an issue.
var proflock mutex

// All memory allocations are local and do not escape outside of the profiler.
// The profiler is forbidden from referring to garbage-collected memory.

/*
enum { MProf, BProf }  // profile types
*/

/*
// Per-call-stack profiling information.
// Lookup by hashing call stack into a linked-list hash table.
struct Bucket
{
	Bucket	*next	// next in hash list
	Bucket	*allnext	// next in list of all mbuckets/bbuckets
	int32	typ
	// Generally unions can break precise GC,
	// this one is fine because it does not contain pointers.
	union
	{
		struct MProfRecord // typ == MProf
		{
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
			uintptr	allocs
			uintptr	frees
			uintptr	alloc_bytes
			uintptr	free_bytes

			uintptr	prev_allocs  // since last but one till last gc
			uintptr	prev_frees
			uintptr	prev_alloc_bytes
			uintptr	prev_free_bytes

			uintptr	recent_allocs  // since last gc till now
			uintptr	recent_frees
			uintptr	recent_alloc_bytes
			uintptr	recent_free_bytes

		} mp
		struct BProfRecord // typ == BProf
		{
			int64	count
			int64	cycles
		} bp
	} data
	uintptr	hash	// hash of size + stk
	uintptr	size
	uintptr	nstk
	uintptr	stk[1]
}
*/

var (
	mbuckets *bucket // memory profile buckets
	bbuckets *bucket // blocking profile buckets
)

/*
enum {
	BuckHashSize = 179999,
}
static Bucket **buckhash
static uintptr bucketmem
*/

/*
// Return the bucket for stk[0:nstk], allocating new bucket if needed.
static Bucket*
stkbucket(int32 typ, uintptr size, uintptr *stk, int32 nstk, bool alloc)
{
	int32 i
	uintptr h
	Bucket *b

	if(buckhash == nil) {
		buckhash = sysAlloc(BuckHashSize*sizeof buckhash[0], &mstats.buckhash_sys)
		if(buckhash == nil)
			throw("runtime: cannot allocate memory")
	}

	// Hash stack.
	h = 0
	for(i=0 i<nstk i++) {
		h += stk[i]
		h += h<<10
		h ^= h>>6
	}
	// hash in size
	h += size
	h += h<<10
	h ^= h>>6
	// finalize
	h += h<<3
	h ^= h>>11

	i = h%BuckHashSize
	for(b = buckhash[i] b b=b.next)
		if(b.typ == typ && b.hash == h && b.size == size && b.nstk == nstk &&
		   mcmp((byte*)b.stk, (byte*)stk, nstk*sizeof stk[0]) == 0)
			return b

	if(!alloc)
		return nil

	b = persistentalloc(sizeof *b + nstk*sizeof stk[0], 0, &mstats.buckhash_sys)
	bucketmem += sizeof *b + nstk*sizeof stk[0]
	memmove(b.stk, stk, nstk*sizeof stk[0])
	b.typ = typ
	b.hash = h
	b.size = size
	b.nstk = nstk
	b.next = buckhash[i]
	buckhash[i] = b
	if(typ == MProf) {
		b.allnext = mbuckets
		mbuckets = b
	} else {
		b.allnext = bbuckets
		bbuckets = b
	}
	return b
}
*/

func mprof_GC() {
	for b := mbuckets; b != nil; b = b.allnext {
		b.data.mp.allocs += b.data.mp.prev_allocs
		b.data.mp.frees += b.data.mp.prev_frees
		b.data.mp.alloc_bytes += b.data.mp.prev_alloc_bytes
		b.data.mp.free_bytes += b.data.mp.prev_free_bytes

		b.data.mp.prev_allocs = b.data.mp.recent_allocs
		b.data.mp.prev_frees = b.data.mp.recent_frees
		b.data.mp.prev_alloc_bytes = b.data.mp.recent_alloc_bytes
		b.data.mp.prev_free_bytes = b.data.mp.recent_free_bytes

		b.data.mp.recent_allocs = 0
		b.data.mp.recent_frees = 0
		b.data.mp.recent_alloc_bytes = 0
		b.data.mp.recent_free_bytes = 0
	}
}

/*
// Record that a gc just happened: all the 'recent' statistics are now real.
void
MProf_GC(void)
{
	lock(&proflock)
	MProf_GC()
	unlock(&proflock)
}
*/

/*
// Called by malloc to record a profiled block.
void
MProf_Malloc(void *p, uintptr size)
{
	uintptr stk[32]
	Bucket *b
	int32 nstk

	nstk = callers(1, stk, nelem(stk))
	lock(&proflock)
	b = stkbucket(MProf, size, stk, nstk, true)
	b.data.mp.recent_allocs++
	b.data.mp.recent_alloc_bytes += size
	unlock(&proflock)

	// Setprofilebucket locks a bunch of other mutexes, so we call it outside of proflock.
	// This reduces potential contention and chances of deadlocks.
	// Since the object must be alive during call to MProf_Malloc,
	// it's fine to do this non-atomically.
	setprofilebucket(p, b)
}
*/

/*
void
MProf_Free(Bucket *b, uintptr size, bool freed)
{
	lock(&proflock)
	if(freed) {
		b.data.mp.recent_frees++
		b.data.mp.recent_free_bytes += size
	} else {
		b.data.mp.prev_frees++
		b.data.mp.prev_free_bytes += size
	}
	unlock(&proflock)
}
*/

/*
int64 blockprofilerate  // in CPU ticks
*/

/*
void
SetBlockProfileRate(intgo rate)
{
	int64 r

	if(rate <= 0)
		r = 0  // disable profiling
	else {
		// convert ns to cycles, use float64 to prevent overflow during multiplication
		r = (float64)rate*tickspersecond()/(1000*1000*1000)
		if(r == 0)
			r = 1
	}
	atomicstore64((uint64*)&blockprofilerate, r)
}
*/

/*
void
blockevent(int64 cycles, int32 skip)
{
	int32 nstk
	int64 rate
	uintptr stk[32]
	Bucket *b

	if(cycles <= 0)
		return
	rate = atomicload64((uint64*)&blockprofilerate)
	if(rate <= 0 || (rate > cycles && fastrand1()%rate > cycles))
		return

	if(g.m.curg == nil || g.m.curg == g)
		nstk = callers(skip, stk, nelem(stk))
	else
		nstk = gcallers(g.m.curg, skip, stk, nelem(stk))
	lock(&proflock)
	b = stkbucket(BProf, 0, stk, nstk, true)
	b.data.bp.count++
	b.data.bp.cycles += cycles
	unlock(&proflock)
}
*/

// Go interface to profile data.

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
		if inuseZero || b.data.mp.alloc_bytes != b.data.mp.free_bytes {
			n++
		}
		if b.data.mp.allocs != 0 || b.data.mp.frees != 0 {
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
			if inuseZero || b.data.mp.alloc_bytes != b.data.mp.free_bytes {
				n++
			}
		}
	}
	if n <= len(p) {
		ok = true
		idx := 0
		for b := mbuckets; b != nil; b = b.allnext {
			if inuseZero || b.data.mp.alloc_bytes != b.data.mp.free_bytes {
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
	r.AllocBytes = int64(b.data.mp.alloc_bytes)
	r.FreeBytes = int64(b.data.mp.free_bytes)
	r.AllocObjects = int64(b.data.mp.allocs)
	r.FreeObjects = int64(b.data.mp.frees)
	for i := 0; uintptr(i) < b.nstk && i < len(r.Stack0); i++ {
		r.Stack0[i] = *(*uintptr)(add(unsafe.Pointer(&b.stk), uintptr(i)*ptrSize))
	}
	for i := b.nstk; i < uintptr(len(r.Stack0)); i++ {
		r.Stack0[i] = 0
	}
}

/*
void
iterate_memprof(void (*callback)(Bucket*, uintptr, uintptr*, uintptr, uintptr, uintptr))
{
	Bucket *b

	lock(&proflock)
	for(b=mbuckets b b=b.allnext) {
		callback(b, b.nstk, b.stk, b.size, b.data.mp.allocs, b.data.mp.frees)
	}
	unlock(&proflock)
}
*/

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
		idx := 0
		for b := bbuckets; b != nil; b = b.allnext {
			bp := (*bprofrecord)(unsafe.Pointer(&b.data))
			p[idx].Count = int64(bp.count)
			p[idx].Cycles = int64(bp.cycles)
			i := 0
			for uintptr(i) < b.nstk && i < len(p[idx].Stack0) {
				p[idx].Stack0[i] = *(*uintptr)(add(unsafe.Pointer(&b.stk), uintptr(i)*ptrSize))
				i++
			}
			for i < len(p[idx].Stack0) {
				p[idx].Stack0[i] = 0
				i++
			}
			idx++
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

/*
func GoroutineProfile(b Slice) (n int, ok bool) {
	uintptr pc, sp, i
	TRecord *r
	G *gp

	sp = getcallersp(&b)
	pc = (uintptr)getcallerpc(&b)

	ok = false
	n = gcount()
	if(n <= b.len) {
		semacquire(&worldsema, false)
		g.m.gcing = 1
		stoptheworld()

		n = gcount()
		if(n <= b.len) {
			ok = true
			r = (TRecord*)b.array
			saveg(pc, sp, g, r++)
			for(i = 0 i < allglen i++) {
				gp = allg[i]
				if(gp == g || readgstatus(gp) == Gdead)
					continue
				saveg(~(uintptr)0, ~(uintptr)0, gp, r++)
			}
		}

		g.m.gcing = 0
		semrelease(&worldsema)
		starttheworld()
	}
}
*/

/*
static void
saveg(uintptr pc, uintptr sp, G *gp, TRecord *r)
{
	int32 n

	n = gentraceback(pc, sp, 0, gp, 0, r.stk, nelem(r.stk), nil, nil, false)
	if(n < nelem(r.stk))
		r.stk[n] = 0
}
*/

// Stack formats a stack trace of the calling goroutine into buf
// and returns the number of bytes written to buf.
// If all is true, Stack formats stack traces of all other goroutines
// into buf after the trace for the current goroutine.
func Stack(buf []byte, all bool) int {
	sp := getcallersp(unsafe.Pointer(&buf))
	pc := getcallerpc(unsafe.Pointer(&buf))
	mp := acquirem()
	gp := mp.curg
	if all {
		semacquire(&worldsema, false)
		mp.gcing = 1
		releasem(mp)
		stoptheworld()
		if mp != acquirem() {
			gothrow("Stack: rescheduled")
		}
	}

	n := 0
	if len(buf) > 0 {
		gp.writebuf = buf
		goroutineheader(gp)
		traceback(pc, sp, 0, gp)
		if all {
			tracebackothers(gp)
		}
		n = len(buf) - len(gp.writebuf)
		gp.writebuf = nil
	}

	if all {
		mp.gcing = 0
		semrelease(&worldsema)
		starttheworld()
	}
	releasem(mp)
	return n
}

/*
// Tracing of alloc/free/gc.

static Mutex tracelock

void
tracealloc(void *p, uintptr size, Type *type)
{
	lock(&tracelock)
	g.m.traceback = 2
	if(type == nil)
		printf("tracealloc(%p, %p)\n", p, size)
	else
		printf("tracealloc(%p, %p, %S)\n", p, size, *type.string)
	if(g.m.curg == nil || g == g.m.curg) {
		goroutineheader(g)
		traceback((uintptr)getcallerpc(&p), (uintptr)getcallersp(&p), 0, g)
	} else {
		goroutineheader(g.m.curg)
		traceback(~(uintptr)0, ~(uintptr)0, 0, g.m.curg)
	}
	printf("\n")
	g.m.traceback = 0
	unlock(&tracelock)
}

void
tracefree(void *p, uintptr size)
{
	lock(&tracelock)
	g.m.traceback = 2
	printf("tracefree(%p, %p)\n", p, size)
	goroutineheader(g)
	traceback((uintptr)getcallerpc(&p), (uintptr)getcallersp(&p), 0, g)
	printf("\n")
	g.m.traceback = 0
	unlock(&tracelock)
}

void
tracegc(void)
{
	lock(&tracelock)
	g.m.traceback = 2
	printf("tracegc()\n")
	// running on m.g0 stack show all non-g0 goroutines
	tracebackothers(g)
	printf("end tracegc\n")
	printf("\n")
	g.m.traceback = 0
	unlock(&tracelock)
}
*/
