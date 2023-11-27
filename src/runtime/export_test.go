// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export guts for testing.

package runtime

import (
	"internal/goarch"
	"internal/goos"
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

var Fadd64 = fadd64
var Fsub64 = fsub64
var Fmul64 = fmul64
var Fdiv64 = fdiv64
var F64to32 = f64to32
var F32to64 = f32to64
var Fcmp64 = fcmp64
var Fintto64 = fintto64
var F64toint = f64toint

var Entersyscall = entersyscall
var Exitsyscall = exitsyscall
var LockedOSThread = lockedOSThread
var Xadduintptr = atomic.Xadduintptr

var Fastlog2 = fastlog2

var Atoi = atoi
var Atoi32 = atoi32
var ParseByteCount = parseByteCount

var Nanotime = nanotime
var NetpollBreak = netpollBreak
var Usleep = usleep

var PhysPageSize = physPageSize
var PhysHugePageSize = physHugePageSize

var NetpollGenericInit = netpollGenericInit

var Memmove = memmove
var MemclrNoHeapPointers = memclrNoHeapPointers

var LockPartialOrder = lockPartialOrder

type LockRank lockRank

func (l LockRank) String() string {
	return lockRank(l).String()
}

const PreemptMSupported = preemptMSupported

type LFNode struct {
	Next    uint64
	Pushcnt uintptr
}

func LFStackPush(head *uint64, node *LFNode) {
	(*lfstack)(head).push((*lfnode)(unsafe.Pointer(node)))
}

func LFStackPop(head *uint64) *LFNode {
	return (*LFNode)(unsafe.Pointer((*lfstack)(head).pop()))
}
func LFNodeValidate(node *LFNode) {
	lfnodeValidate((*lfnode)(unsafe.Pointer(node)))
}

func Netpoll(delta int64) {
	systemstack(func() {
		netpoll(delta)
	})
}

func GCMask(x any) (ret []byte) {
	systemstack(func() {
		ret = getgcmask(x)
	})
	return
}

func RunSchedLocalQueueTest() {
	pp := new(p)
	gs := make([]g, len(pp.runq))
	Escape(gs) // Ensure gs doesn't move, since we use guintptrs
	for i := 0; i < len(pp.runq); i++ {
		if g, _ := runqget(pp); g != nil {
			throw("runq is not empty initially")
		}
		for j := 0; j < i; j++ {
			runqput(pp, &gs[i], false)
		}
		for j := 0; j < i; j++ {
			if g, _ := runqget(pp); g != &gs[i] {
				print("bad element at iter ", i, "/", j, "\n")
				throw("bad element")
			}
		}
		if g, _ := runqget(pp); g != nil {
			throw("runq is not empty afterwards")
		}
	}
}

func RunSchedLocalQueueStealTest() {
	p1 := new(p)
	p2 := new(p)
	gs := make([]g, len(p1.runq))
	Escape(gs) // Ensure gs doesn't move, since we use guintptrs
	for i := 0; i < len(p1.runq); i++ {
		for j := 0; j < i; j++ {
			gs[j].sig = 0
			runqput(p1, &gs[j], false)
		}
		gp := runqsteal(p2, p1, true)
		s := 0
		if gp != nil {
			s++
			gp.sig++
		}
		for {
			gp, _ = runqget(p2)
			if gp == nil {
				break
			}
			s++
			gp.sig++
		}
		for {
			gp, _ = runqget(p1)
			if gp == nil {
				break
			}
			gp.sig++
		}
		for j := 0; j < i; j++ {
			if gs[j].sig != 1 {
				print("bad element ", j, "(", gs[j].sig, ") at iter ", i, "\n")
				throw("bad element")
			}
		}
		if s != i/2 && s != i/2+1 {
			print("bad steal ", s, ", want ", i/2, " or ", i/2+1, ", iter ", i, "\n")
			throw("bad steal")
		}
	}
}

func RunSchedLocalQueueEmptyTest(iters int) {
	// Test that runq is not spuriously reported as empty.
	// Runq emptiness affects scheduling decisions and spurious emptiness
	// can lead to underutilization (both runnable Gs and idle Ps coexist
	// for arbitrary long time).
	done := make(chan bool, 1)
	p := new(p)
	gs := make([]g, 2)
	Escape(gs) // Ensure gs doesn't move, since we use guintptrs
	ready := new(uint32)
	for i := 0; i < iters; i++ {
		*ready = 0
		next0 := (i & 1) == 0
		next1 := (i & 2) == 0
		runqput(p, &gs[0], next0)
		go func() {
			for atomic.Xadd(ready, 1); atomic.Load(ready) != 2; {
			}
			if runqempty(p) {
				println("next:", next0, next1)
				throw("queue is empty")
			}
			done <- true
		}()
		for atomic.Xadd(ready, 1); atomic.Load(ready) != 2; {
		}
		runqput(p, &gs[1], next1)
		runqget(p)
		<-done
		runqget(p)
	}
}

var (
	StringHash = stringHash
	BytesHash  = bytesHash
	Int32Hash  = int32Hash
	Int64Hash  = int64Hash
	MemHash    = memhash
	MemHash32  = memhash32
	MemHash64  = memhash64
	EfaceHash  = efaceHash
	IfaceHash  = ifaceHash
)

var UseAeshash = &useAeshash

func MemclrBytes(b []byte) {
	s := (*slice)(unsafe.Pointer(&b))
	memclrNoHeapPointers(s.array, uintptr(s.len))
}

const HashLoad = hashLoad

// entry point for testing
func GostringW(w []uint16) (s string) {
	systemstack(func() {
		s = gostringw(&w[0])
	})
	return
}

var Open = open
var Close = closefd
var Read = read
var Write = write

func Envs() []string     { return envs }
func SetEnvs(e []string) { envs = e }

// For benchmarking.

func BenchSetType(n int, x any) {
	e := *efaceOf(&x)
	t := e._type
	var size uintptr
	var p unsafe.Pointer
	switch t.kind & kindMask {
	case kindPtr:
		t = (*ptrtype)(unsafe.Pointer(t)).elem
		size = t.size
		p = e.data
	case kindSlice:
		slice := *(*struct {
			ptr      unsafe.Pointer
			len, cap uintptr
		})(e.data)
		t = (*slicetype)(unsafe.Pointer(t)).elem
		size = t.size * slice.len
		p = slice.ptr
	}
	allocSize := roundupsize(size)
	systemstack(func() {
		for i := 0; i < n; i++ {
			heapBitsSetType(uintptr(p), allocSize, size, t)
		}
	})
}

const PtrSize = goarch.PtrSize

var ForceGCPeriod = &forcegcperiod

// SetTracebackEnv is like runtime/debug.SetTraceback, but it raises
// the "environment" traceback level, so later calls to
// debug.SetTraceback (e.g., from testing timeouts) can't lower it.
func SetTracebackEnv(level string) {
	setTraceback(level)
	traceback_env = traceback_cache
}

var ReadUnaligned32 = readUnaligned32
var ReadUnaligned64 = readUnaligned64

func CountPagesInUse() (pagesInUse, counted uintptr) {
	stopTheWorld("CountPagesInUse")

	pagesInUse = uintptr(mheap_.pagesInUse.Load())

	for _, s := range mheap_.allspans {
		if s.state.get() == mSpanInUse {
			counted += s.npages
		}
	}

	startTheWorld()

	return
}

func Fastrand() uint32          { return fastrand() }
func Fastrand64() uint64        { return fastrand64() }
func Fastrandn(n uint32) uint32 { return fastrandn(n) }

type ProfBuf profBuf

func NewProfBuf(hdrsize, bufwords, tags int) *ProfBuf {
	return (*ProfBuf)(newProfBuf(hdrsize, bufwords, tags))
}

func (p *ProfBuf) Write(tag *unsafe.Pointer, now int64, hdr []uint64, stk []uintptr) {
	(*profBuf)(p).write(tag, now, hdr, stk)
}

const (
	ProfBufBlocking    = profBufBlocking
	ProfBufNonBlocking = profBufNonBlocking
)

func (p *ProfBuf) Read(mode profBufReadMode) ([]uint64, []unsafe.Pointer, bool) {
	return (*profBuf)(p).read(profBufReadMode(mode))
}

func (p *ProfBuf) Close() {
	(*profBuf)(p).close()
}

func ReadMetricsSlow(memStats *MemStats, samplesp unsafe.Pointer, len, cap int) {
	stopTheWorld("ReadMetricsSlow")

	// Initialize the metrics beforehand because this could
	// allocate and skew the stats.
	metricsLock()
	initMetrics()
	metricsUnlock()

	systemstack(func() {
		// Read memstats first. It's going to flush
		// the mcaches which readMetrics does not do, so
		// going the other way around may result in
		// inconsistent statistics.
		readmemstats_m(memStats)
	})

	// Read metrics off the system stack.
	//
	// The only part of readMetrics that could allocate
	// and skew the stats is initMetrics.
	readMetrics(samplesp, len, cap)

	startTheWorld()
}

var DoubleCheckReadMemStats = &doubleCheckReadMemStats

// ReadMemStatsSlow returns both the runtime-computed MemStats and
// MemStats accumulated by scanning the heap.
func ReadMemStatsSlow() (base, slow MemStats) {
	stopTheWorld("ReadMemStatsSlow")

	// Run on the system stack to avoid stack growth allocation.
	systemstack(func() {
		// Make sure stats don't change.
		getg().m.mallocing++

		readmemstats_m(&base)

		// Initialize slow from base and zero the fields we're
		// recomputing.
		slow = base
		slow.Alloc = 0
		slow.TotalAlloc = 0
		slow.Mallocs = 0
		slow.Frees = 0
		slow.HeapReleased = 0
		var bySize [_NumSizeClasses]struct {
			Mallocs, Frees uint64
		}

		// Add up current allocations in spans.
		for _, s := range mheap_.allspans {
			if s.state.get() != mSpanInUse {
				continue
			}
			if s.isUnusedUserArenaChunk() {
				continue
			}
			if sizeclass := s.spanclass.sizeclass(); sizeclass == 0 {
				slow.Mallocs++
				slow.Alloc += uint64(s.elemsize)
			} else {
				slow.Mallocs += uint64(s.allocCount)
				slow.Alloc += uint64(s.allocCount) * uint64(s.elemsize)
				bySize[sizeclass].Mallocs += uint64(s.allocCount)
			}
		}

		// Add in frees by just reading the stats for those directly.
		var m heapStatsDelta
		memstats.heapStats.unsafeRead(&m)

		// Collect per-sizeclass free stats.
		var smallFree uint64
		for i := 0; i < _NumSizeClasses; i++ {
			slow.Frees += uint64(m.smallFreeCount[i])
			bySize[i].Frees += uint64(m.smallFreeCount[i])
			bySize[i].Mallocs += uint64(m.smallFreeCount[i])
			smallFree += uint64(m.smallFreeCount[i]) * uint64(class_to_size[i])
		}
		slow.Frees += uint64(m.tinyAllocCount) + uint64(m.largeFreeCount)
		slow.Mallocs += slow.Frees

		slow.TotalAlloc = slow.Alloc + uint64(m.largeFree) + smallFree

		for i := range slow.BySize {
			slow.BySize[i].Mallocs = bySize[i].Mallocs
			slow.BySize[i].Frees = bySize[i].Frees
		}

		for i := mheap_.pages.start; i < mheap_.pages.end; i++ {
			chunk := mheap_.pages.tryChunkOf(i)
			if chunk == nil {
				continue
			}
			pg := chunk.scavenged.popcntRange(0, pallocChunkPages)
			slow.HeapReleased += uint64(pg) * pageSize
		}
		for _, p := range allp {
			pg := sys.OnesCount64(p.pcache.scav)
			slow.HeapReleased += uint64(pg) * pageSize
		}

		getg().m.mallocing--
	})

	startTheWorld()
	return
}

// BlockOnSystemStack switches to the system stack, prints "x\n" to
// stderr, and blocks in a stack containing
// "runtime.blockOnSystemStackInternal".
func BlockOnSystemStack() {
	systemstack(blockOnSystemStackInternal)
}

func blockOnSystemStackInternal() {
	print("x\n")
	lock(&deadlock)
	lock(&deadlock)
}

type RWMutex struct {
	rw rwmutex
}

func (rw *RWMutex) RLock() {
	rw.rw.rlock()
}

func (rw *RWMutex) RUnlock() {
	rw.rw.runlock()
}

func (rw *RWMutex) Lock() {
	rw.rw.lock()
}

func (rw *RWMutex) Unlock() {
	rw.rw.unlock()
}

const RuntimeHmapSize = unsafe.Sizeof(hmap{})

func MapBucketsCount(m map[int]int) int {
	h := *(**hmap)(unsafe.Pointer(&m))
	return 1 << h.B
}

func MapBucketsPointerIsNil(m map[int]int) bool {
	h := *(**hmap)(unsafe.Pointer(&m))
	return h.buckets == nil
}

func LockOSCounts() (external, internal uint32) {
	gp := getg()
	if gp.m.lockedExt+gp.m.lockedInt == 0 {
		if gp.lockedm != 0 {
			panic("lockedm on non-locked goroutine")
		}
	} else {
		if gp.lockedm == 0 {
			panic("nil lockedm on locked goroutine")
		}
	}
	return gp.m.lockedExt, gp.m.lockedInt
}

//go:noinline
func TracebackSystemstack(stk []uintptr, i int) int {
	if i == 0 {
		pc, sp := getcallerpc(), getcallersp()
		return gentraceback(pc, sp, 0, getg(), 0, &stk[0], len(stk), nil, nil, _TraceJumpStack)
	}
	n := 0
	systemstack(func() {
		n = TracebackSystemstack(stk, i-1)
	})
	return n
}

func KeepNArenaHints(n int) {
	hint := mheap_.arenaHints
	for i := 1; i < n; i++ {
		hint = hint.next
		if hint == nil {
			return
		}
	}
	hint.next = nil
}

// MapNextArenaHint reserves a page at the next arena growth hint,
// preventing the arena from growing there, and returns the range of
// addresses that are no longer viable.
//
// This may fail to reserve memory. If it fails, it still returns the
// address range it attempted to reserve.
func MapNextArenaHint() (start, end uintptr, ok bool) {
	hint := mheap_.arenaHints
	addr := hint.addr
	if hint.down {
		start, end = addr-heapArenaBytes, addr
		addr -= physPageSize
	} else {
		start, end = addr, addr+heapArenaBytes
	}
	got := sysReserve(unsafe.Pointer(addr), physPageSize)
	ok = (addr == uintptr(got))
	if !ok {
		// We were unable to get the requested reservation.
		// Release what we did get and fail.
		sysFreeOS(got, physPageSize)
	}
	return
}

func GetNextArenaHint() uintptr {
	return mheap_.arenaHints.addr
}

type G = g

type Sudog = sudog

func Getg() *G {
	return getg()
}

func GIsWaitingOnMutex(gp *G) bool {
	return readgstatus(gp) == _Gwaiting && gp.waitreason.isMutexWait()
}

var CasGStatusAlwaysTrack = &casgstatusAlwaysTrack

//go:noinline
func PanicForTesting(b []byte, i int) byte {
	return unexportedPanicForTesting(b, i)
}

//go:noinline
func unexportedPanicForTesting(b []byte, i int) byte {
	return b[i]
}

func G0StackOverflow() {
	systemstack(func() {
		stackOverflow(nil)
	})
}

func stackOverflow(x *byte) {
	var buf [256]byte
	stackOverflow(&buf[0])
}

func MapTombstoneCheck(m map[int]int) {
	// Make sure emptyOne and emptyRest are distributed correctly.
	// We should have a series of filled and emptyOne cells, followed by
	// a series of emptyRest cells.
	h := *(**hmap)(unsafe.Pointer(&m))
	i := any(m)
	t := *(**maptype)(unsafe.Pointer(&i))

	for x := 0; x < 1<<h.B; x++ {
		b0 := (*bmap)(add(h.buckets, uintptr(x)*uintptr(t.bucketsize)))
		n := 0
		for b := b0; b != nil; b = b.overflow(t) {
			for i := 0; i < bucketCnt; i++ {
				if b.tophash[i] != emptyRest {
					n++
				}
			}
		}
		k := 0
		for b := b0; b != nil; b = b.overflow(t) {
			for i := 0; i < bucketCnt; i++ {
				if k < n && b.tophash[i] == emptyRest {
					panic("early emptyRest")
				}
				if k >= n && b.tophash[i] != emptyRest {
					panic("late non-emptyRest")
				}
				if k == n-1 && b.tophash[i] == emptyOne {
					panic("last non-emptyRest entry is emptyOne")
				}
				k++
			}
		}
	}
}

func RunGetgThreadSwitchTest() {
	// Test that getg works correctly with thread switch.
	// With gccgo, if we generate getg inlined, the backend
	// may cache the address of the TLS variable, which
	// will become invalid after a thread switch. This test
	// checks that the bad caching doesn't happen.

	ch := make(chan int)
	go func(ch chan int) {
		ch <- 5
		LockOSThread()
	}(ch)

	g1 := getg()

	// Block on a receive. This is likely to get us a thread
	// switch. If we yield to the sender goroutine, it will
	// lock the thread, forcing us to resume on a different
	// thread.
	<-ch

	g2 := getg()
	if g1 != g2 {
		panic("g1 != g2")
	}

	// Also test getg after some control flow, as the
	// backend is sensitive to control flow.
	g3 := getg()
	if g1 != g3 {
		panic("g1 != g3")
	}
}

const (
	PageSize         = pageSize
	PallocChunkPages = pallocChunkPages
	PageAlloc64Bit   = pageAlloc64Bit
	PallocSumBytes   = pallocSumBytes
)

// Expose pallocSum for testing.
type PallocSum pallocSum

func PackPallocSum(start, max, end uint) PallocSum { return PallocSum(packPallocSum(start, max, end)) }
func (m PallocSum) Start() uint                    { return pallocSum(m).start() }
func (m PallocSum) Max() uint                      { return pallocSum(m).max() }
func (m PallocSum) End() uint                      { return pallocSum(m).end() }

// Expose pallocBits for testing.
type PallocBits pallocBits

func (b *PallocBits) Find(npages uintptr, searchIdx uint) (uint, uint) {
	return (*pallocBits)(b).find(npages, searchIdx)
}
func (b *PallocBits) AllocRange(i, n uint)       { (*pallocBits)(b).allocRange(i, n) }
func (b *PallocBits) Free(i, n uint)             { (*pallocBits)(b).free(i, n) }
func (b *PallocBits) Summarize() PallocSum       { return PallocSum((*pallocBits)(b).summarize()) }
func (b *PallocBits) PopcntRange(i, n uint) uint { return (*pageBits)(b).popcntRange(i, n) }

// SummarizeSlow is a slow but more obviously correct implementation
// of (*pallocBits).summarize. Used for testing.
func SummarizeSlow(b *PallocBits) PallocSum {
	var start, max, end uint

	const N = uint(len(b)) * 64
	for start < N && (*pageBits)(b).get(start) == 0 {
		start++
	}
	for end < N && (*pageBits)(b).get(N-end-1) == 0 {
		end++
	}
	run := uint(0)
	for i := uint(0); i < N; i++ {
		if (*pageBits)(b).get(i) == 0 {
			run++
		} else {
			run = 0
		}
		if run > max {
			max = run
		}
	}
	return PackPallocSum(start, max, end)
}

// Expose non-trivial helpers for testing.
func FindBitRange64(c uint64, n uint) uint { return findBitRange64(c, n) }

// Given two PallocBits, returns a set of bit ranges where
// they differ.
func DiffPallocBits(a, b *PallocBits) []BitRange {
	ba := (*pageBits)(a)
	bb := (*pageBits)(b)

	var d []BitRange
	base, size := uint(0), uint(0)
	for i := uint(0); i < uint(len(ba))*64; i++ {
		if ba.get(i) != bb.get(i) {
			if size == 0 {
				base = i
			}
			size++
		} else {
			if size != 0 {
				d = append(d, BitRange{base, size})
			}
			size = 0
		}
	}
	if size != 0 {
		d = append(d, BitRange{base, size})
	}
	return d
}

// StringifyPallocBits gets the bits in the bit range r from b,
// and returns a string containing the bits as ASCII 0 and 1
// characters.
func StringifyPallocBits(b *PallocBits, r BitRange) string {
	str := ""
	for j := r.I; j < r.I+r.N; j++ {
		if (*pageBits)(b).get(j) != 0 {
			str += "1"
		} else {
			str += "0"
		}
	}
	return str
}

// Expose pallocData for testing.
type PallocData pallocData

func (d *PallocData) FindScavengeCandidate(searchIdx uint, min, max uintptr) (uint, uint) {
	return (*pallocData)(d).findScavengeCandidate(searchIdx, min, max)
}
func (d *PallocData) AllocRange(i, n uint) { (*pallocData)(d).allocRange(i, n) }
func (d *PallocData) ScavengedSetRange(i, n uint) {
	(*pallocData)(d).scavenged.setRange(i, n)
}
func (d *PallocData) PallocBits() *PallocBits {
	return (*PallocBits)(&(*pallocData)(d).pallocBits)
}
func (d *PallocData) Scavenged() *PallocBits {
	return (*PallocBits)(&(*pallocData)(d).scavenged)
}

// Expose fillAligned for testing.
func FillAligned(x uint64, m uint) uint64 { return fillAligned(x, m) }

// Expose pageCache for testing.
type PageCache pageCache

const PageCachePages = pageCachePages

func NewPageCache(base uintptr, cache, scav uint64) PageCache {
	return PageCache(pageCache{base: base, cache: cache, scav: scav})
}
func (c *PageCache) Empty() bool   { return (*pageCache)(c).empty() }
func (c *PageCache) Base() uintptr { return (*pageCache)(c).base }
func (c *PageCache) Cache() uint64 { return (*pageCache)(c).cache }
func (c *PageCache) Scav() uint64  { return (*pageCache)(c).scav }
func (c *PageCache) Alloc(npages uintptr) (uintptr, uintptr) {
	return (*pageCache)(c).alloc(npages)
}
func (c *PageCache) Flush(s *PageAlloc) {
	cp := (*pageCache)(c)
	sp := (*pageAlloc)(s)

	systemstack(func() {
		// None of the tests need any higher-level locking, so we just
		// take the lock internally.
		lock(sp.mheapLock)
		cp.flush(sp)
		unlock(sp.mheapLock)
	})
}

// Expose chunk index type.
type ChunkIdx chunkIdx

// Expose pageAlloc for testing. Note that because pageAlloc is
// not in the heap, so is PageAlloc.
type PageAlloc pageAlloc

func (p *PageAlloc) Alloc(npages uintptr) (uintptr, uintptr) {
	pp := (*pageAlloc)(p)

	var addr, scav uintptr
	systemstack(func() {
		// None of the tests need any higher-level locking, so we just
		// take the lock internally.
		lock(pp.mheapLock)
		addr, scav = pp.alloc(npages)
		unlock(pp.mheapLock)
	})
	return addr, scav
}
func (p *PageAlloc) AllocToCache() PageCache {
	pp := (*pageAlloc)(p)

	var c PageCache
	systemstack(func() {
		// None of the tests need any higher-level locking, so we just
		// take the lock internally.
		lock(pp.mheapLock)
		c = PageCache(pp.allocToCache())
		unlock(pp.mheapLock)
	})
	return c
}
func (p *PageAlloc) Free(base, npages uintptr) {
	pp := (*pageAlloc)(p)

	systemstack(func() {
		// None of the tests need any higher-level locking, so we just
		// take the lock internally.
		lock(pp.mheapLock)
		pp.free(base, npages, true)
		unlock(pp.mheapLock)
	})
}
func (p *PageAlloc) Bounds() (ChunkIdx, ChunkIdx) {
	return ChunkIdx((*pageAlloc)(p).start), ChunkIdx((*pageAlloc)(p).end)
}
func (p *PageAlloc) Scavenge(nbytes uintptr) (r uintptr) {
	pp := (*pageAlloc)(p)
	systemstack(func() {
		r = pp.scavenge(nbytes, nil)
	})
	return
}
func (p *PageAlloc) InUse() []AddrRange {
	ranges := make([]AddrRange, 0, len(p.inUse.ranges))
	for _, r := range p.inUse.ranges {
		ranges = append(ranges, AddrRange{r})
	}
	return ranges
}

// Returns nil if the PallocData's L2 is missing.
func (p *PageAlloc) PallocData(i ChunkIdx) *PallocData {
	ci := chunkIdx(i)
	return (*PallocData)((*pageAlloc)(p).tryChunkOf(ci))
}

// AddrRange is a wrapper around addrRange for testing.
type AddrRange struct {
	addrRange
}

// MakeAddrRange creates a new address range.
func MakeAddrRange(base, limit uintptr) AddrRange {
	return AddrRange{makeAddrRange(base, limit)}
}

// Base returns the virtual base address of the address range.
func (a AddrRange) Base() uintptr {
	return a.addrRange.base.addr()
}

// Base returns the virtual address of the limit of the address range.
func (a AddrRange) Limit() uintptr {
	return a.addrRange.limit.addr()
}

// Equals returns true if the two address ranges are exactly equal.
func (a AddrRange) Equals(b AddrRange) bool {
	return a == b
}

// Size returns the size in bytes of the address range.
func (a AddrRange) Size() uintptr {
	return a.addrRange.size()
}

// testSysStat is the sysStat passed to test versions of various
// runtime structures. We do actually have to keep track of this
// because otherwise memstats.mappedReady won't actually line up
// with other stats in the runtime during tests.
var testSysStat = &memstats.other_sys

// AddrRanges is a wrapper around addrRanges for testing.
type AddrRanges struct {
	addrRanges
	mutable bool
}

// NewAddrRanges creates a new empty addrRanges.
//
// Note that this initializes addrRanges just like in the
// runtime, so its memory is persistentalloc'd. Call this
// function sparingly since the memory it allocates is
// leaked.
//
// This AddrRanges is mutable, so we can test methods like
// Add.
func NewAddrRanges() AddrRanges {
	r := addrRanges{}
	r.init(testSysStat)
	return AddrRanges{r, true}
}

// MakeAddrRanges creates a new addrRanges populated with
// the ranges in a.
//
// The returned AddrRanges is immutable, so methods like
// Add will fail.
func MakeAddrRanges(a ...AddrRange) AddrRanges {
	// Methods that manipulate the backing store of addrRanges.ranges should
	// not be used on the result from this function (e.g. add) since they may
	// trigger reallocation. That would normally be fine, except the new
	// backing store won't come from the heap, but from persistentalloc, so
	// we'll leak some memory implicitly.
	ranges := make([]addrRange, 0, len(a))
	total := uintptr(0)
	for _, r := range a {
		ranges = append(ranges, r.addrRange)
		total += r.Size()
	}
	return AddrRanges{addrRanges{
		ranges:     ranges,
		totalBytes: total,
		sysStat:    testSysStat,
	}, false}
}

// Ranges returns a copy of the ranges described by the
// addrRanges.
func (a *AddrRanges) Ranges() []AddrRange {
	result := make([]AddrRange, 0, len(a.addrRanges.ranges))
	for _, r := range a.addrRanges.ranges {
		result = append(result, AddrRange{r})
	}
	return result
}

// FindSucc returns the successor to base. See addrRanges.findSucc
// for more details.
func (a *AddrRanges) FindSucc(base uintptr) int {
	return a.findSucc(base)
}

// Add adds a new AddrRange to the AddrRanges.
//
// The AddrRange must be mutable (i.e. created by NewAddrRanges),
// otherwise this method will throw.
func (a *AddrRanges) Add(r AddrRange) {
	if !a.mutable {
		throw("attempt to mutate immutable AddrRanges")
	}
	a.add(r.addrRange)
}

// TotalBytes returns the totalBytes field of the addrRanges.
func (a *AddrRanges) TotalBytes() uintptr {
	return a.addrRanges.totalBytes
}

// BitRange represents a range over a bitmap.
type BitRange struct {
	I, N uint // bit index and length in bits
}

// NewPageAlloc creates a new page allocator for testing and
// initializes it with the scav and chunks maps. Each key in these maps
// represents a chunk index and each value is a series of bit ranges to
// set within each bitmap's chunk.
//
// The initialization of the pageAlloc preserves the invariant that if a
// scavenged bit is set the alloc bit is necessarily unset, so some
// of the bits described by scav may be cleared in the final bitmap if
// ranges in chunks overlap with them.
//
// scav is optional, and if nil, the scavenged bitmap will be cleared
// (as opposed to all 1s, which it usually is). Furthermore, every
// chunk index in scav must appear in chunks; ones that do not are
// ignored.
func NewPageAlloc(chunks, scav map[ChunkIdx][]BitRange) *PageAlloc {
	p := new(pageAlloc)

	// We've got an entry, so initialize the pageAlloc.
	p.init(new(mutex), testSysStat)
	lockInit(p.mheapLock, lockRankMheap)
	p.test = true
	for i, init := range chunks {
		addr := chunkBase(chunkIdx(i))

		// Mark the chunk's existence in the pageAlloc.
		systemstack(func() {
			lock(p.mheapLock)
			p.grow(addr, pallocChunkBytes)
			unlock(p.mheapLock)
		})

		// Initialize the bitmap and update pageAlloc metadata.
		chunk := p.chunkOf(chunkIndex(addr))

		// Clear all the scavenged bits which grow set.
		chunk.scavenged.clearRange(0, pallocChunkPages)

		// Apply scavenge state if applicable.
		if scav != nil {
			if scvg, ok := scav[i]; ok {
				for _, s := range scvg {
					// Ignore the case of s.N == 0. setRange doesn't handle
					// it and it's a no-op anyway.
					if s.N != 0 {
						chunk.scavenged.setRange(s.I, s.N)
					}
				}
			}
		}

		// Apply alloc state.
		for _, s := range init {
			// Ignore the case of s.N == 0. allocRange doesn't handle
			// it and it's a no-op anyway.
			if s.N != 0 {
				chunk.allocRange(s.I, s.N)
			}
		}

		// Make sure the scavenge index is updated.
		//
		// This is an inefficient way to do it, but it's also the simplest way.
		minPages := physPageSize / pageSize
		if minPages < 1 {
			minPages = 1
		}
		_, npages := chunk.findScavengeCandidate(pallocChunkPages-1, minPages, minPages)
		if npages != 0 {
			p.scav.index.mark(addr, addr+pallocChunkBytes)
		}

		// Update heap metadata for the allocRange calls above.
		systemstack(func() {
			lock(p.mheapLock)
			p.update(addr, pallocChunkPages, false, false)
			unlock(p.mheapLock)
		})
	}

	return (*PageAlloc)(p)
}

// FreePageAlloc releases hard OS resources owned by the pageAlloc. Once this
// is called the pageAlloc may no longer be used. The object itself will be
// collected by the garbage collector once it is no longer live.
func FreePageAlloc(pp *PageAlloc) {
	p := (*pageAlloc)(pp)

	// Free all the mapped space for the summary levels.
	if pageAlloc64Bit != 0 {
		for l := 0; l < summaryLevels; l++ {
			sysFreeOS(unsafe.Pointer(&p.summary[l][0]), uintptr(cap(p.summary[l]))*pallocSumBytes)
		}
		// Only necessary on 64-bit. This is a global on 32-bit.
		sysFreeOS(unsafe.Pointer(&p.scav.index.chunks[0]), uintptr(cap(p.scav.index.chunks)))
	} else {
		resSize := uintptr(0)
		for _, s := range p.summary {
			resSize += uintptr(cap(s)) * pallocSumBytes
		}
		sysFreeOS(unsafe.Pointer(&p.summary[0][0]), alignUp(resSize, physPageSize))
	}

	// Subtract back out whatever we mapped for the summaries.
	// sysUsed adds to p.sysStat and memstats.mappedReady no matter what
	// (and in anger should actually be accounted for), and there's no other
	// way to figure out how much we actually mapped.
	gcController.mappedReady.Add(-int64(p.summaryMappedReady))
	testSysStat.add(-int64(p.summaryMappedReady))

	// Free the mapped space for chunks.
	for i := range p.chunks {
		if x := p.chunks[i]; x != nil {
			p.chunks[i] = nil
			// This memory comes from sysAlloc and will always be page-aligned.
			sysFree(unsafe.Pointer(x), unsafe.Sizeof(*p.chunks[0]), testSysStat)
		}
	}
}

// BaseChunkIdx is a convenient chunkIdx value which works on both
// 64 bit and 32 bit platforms, allowing the tests to share code
// between the two.
//
// This should not be higher than 0x100*pallocChunkBytes to support
// mips and mipsle, which only have 31-bit address spaces.
var BaseChunkIdx = func() ChunkIdx {
	var prefix uintptr
	if pageAlloc64Bit != 0 {
		prefix = 0xc000
	} else {
		prefix = 0x100
	}
	baseAddr := prefix * pallocChunkBytes
	if goos.IsAix != 0 {
		baseAddr += arenaBaseOffset
	}
	return ChunkIdx(chunkIndex(baseAddr))
}()

// PageBase returns an address given a chunk index and a page index
// relative to that chunk.
func PageBase(c ChunkIdx, pageIdx uint) uintptr {
	return chunkBase(chunkIdx(c)) + uintptr(pageIdx)*pageSize
}

type BitsMismatch struct {
	Base      uintptr
	Got, Want uint64
}

func CheckScavengedBitsCleared(mismatches []BitsMismatch) (n int, ok bool) {
	ok = true

	// Run on the system stack to avoid stack growth allocation.
	systemstack(func() {
		getg().m.mallocing++

		// Lock so that we can safely access the bitmap.
		lock(&mheap_.lock)
	chunkLoop:
		for i := mheap_.pages.start; i < mheap_.pages.end; i++ {
			chunk := mheap_.pages.tryChunkOf(i)
			if chunk == nil {
				continue
			}
			for j := 0; j < pallocChunkPages/64; j++ {
				// Run over each 64-bit bitmap section and ensure
				// scavenged is being cleared properly on allocation.
				// If a used bit and scavenged bit are both set, that's
				// an error, and could indicate a larger problem, or
				// an accounting problem.
				want := chunk.scavenged[j] &^ chunk.pallocBits[j]
				got := chunk.scavenged[j]
				if want != got {
					ok = false
					if n >= len(mismatches) {
						break chunkLoop
					}
					mismatches[n] = BitsMismatch{
						Base: chunkBase(i) + uintptr(j)*64*pageSize,
						Got:  got,
						Want: want,
					}
					n++
				}
			}
		}
		unlock(&mheap_.lock)

		getg().m.mallocing--
	})
	return
}

func PageCachePagesLeaked() (leaked uintptr) {
	stopTheWorld("PageCachePagesLeaked")

	// Walk over destroyed Ps and look for unflushed caches.
	deadp := allp[len(allp):cap(allp)]
	for _, p := range deadp {
		// Since we're going past len(allp) we may see nil Ps.
		// Just ignore them.
		if p != nil {
			leaked += uintptr(sys.OnesCount64(p.pcache.cache))
		}
	}

	startTheWorld()
	return
}

var Semacquire = semacquire
var Semrelease1 = semrelease1

func SemNwait(addr *uint32) uint32 {
	root := semtable.rootFor(addr)
	return root.nwait.Load()
}

const SemTableSize = semTabSize

// SemTable is a wrapper around semTable exported for testing.
type SemTable struct {
	semTable
}

// Enqueue simulates enqueuing a waiter for a semaphore (or lock) at addr.
func (t *SemTable) Enqueue(addr *uint32) {
	s := acquireSudog()
	s.releasetime = 0
	s.acquiretime = 0
	s.ticket = 0
	t.semTable.rootFor(addr).queue(addr, s, false)
}

// Dequeue simulates dequeuing a waiter for a semaphore (or lock) at addr.
//
// Returns true if there actually was a waiter to be dequeued.
func (t *SemTable) Dequeue(addr *uint32) bool {
	s, _ := t.semTable.rootFor(addr).dequeue(addr)
	if s != nil {
		releaseSudog(s)
		return true
	}
	return false
}

// mspan wrapper for testing.
type MSpan mspan

// Allocate an mspan for testing.
func AllocMSpan() *MSpan {
	var s *mspan
	systemstack(func() {
		lock(&mheap_.lock)
		s = (*mspan)(mheap_.spanalloc.alloc())
		unlock(&mheap_.lock)
	})
	return (*MSpan)(s)
}

// Free an allocated mspan.
func FreeMSpan(s *MSpan) {
	systemstack(func() {
		lock(&mheap_.lock)
		mheap_.spanalloc.free(unsafe.Pointer(s))
		unlock(&mheap_.lock)
	})
}

func MSpanCountAlloc(ms *MSpan, bits []byte) int {
	s := (*mspan)(ms)
	s.nelems = uintptr(len(bits) * 8)
	s.gcmarkBits = (*gcBits)(unsafe.Pointer(&bits[0]))
	result := s.countAlloc()
	s.gcmarkBits = nil
	return result
}

const (
	TimeHistSubBucketBits = timeHistSubBucketBits
	TimeHistNumSubBuckets = timeHistNumSubBuckets
	TimeHistNumBuckets    = timeHistNumBuckets
	TimeHistMinBucketBits = timeHistMinBucketBits
	TimeHistMaxBucketBits = timeHistMaxBucketBits
)

type TimeHistogram timeHistogram

// Counts returns the counts for the given bucket, subBucket indices.
// Returns true if the bucket was valid, otherwise returns the counts
// for the overflow bucket if bucket > 0 or the underflow bucket if
// bucket < 0, and false.
func (th *TimeHistogram) Count(bucket, subBucket int) (uint64, bool) {
	t := (*timeHistogram)(th)
	if bucket < 0 {
		return t.underflow.Load(), false
	}
	i := bucket*TimeHistNumSubBuckets + subBucket
	if i >= len(t.counts) {
		return t.overflow.Load(), false
	}
	return t.counts[i].Load(), true
}

func (th *TimeHistogram) Record(duration int64) {
	(*timeHistogram)(th).record(duration)
}

var TimeHistogramMetricsBuckets = timeHistogramMetricsBuckets

func SetIntArgRegs(a int) int {
	lock(&finlock)
	old := intArgRegs
	if a >= 0 {
		intArgRegs = a
	}
	unlock(&finlock)
	return old
}

func FinalizerGAsleep() bool {
	return fingStatus.Load()&fingWait != 0
}

// For GCTestMoveStackOnNextCall, it's important not to introduce an
// extra layer of call, since then there's a return before the "real"
// next call.
var GCTestMoveStackOnNextCall = gcTestMoveStackOnNextCall

// For GCTestIsReachable, it's important that we do this as a call so
// escape analysis can see through it.
func GCTestIsReachable(ptrs ...unsafe.Pointer) (mask uint64) {
	return gcTestIsReachable(ptrs...)
}

// For GCTestPointerClass, it's important that we do this as a call so
// escape analysis can see through it.
//
// This is nosplit because gcTestPointerClass is.
//
//go:nosplit
func GCTestPointerClass(p unsafe.Pointer) string {
	return gcTestPointerClass(p)
}

const Raceenabled = raceenabled

const (
	GCBackgroundUtilization     = gcBackgroundUtilization
	GCGoalUtilization           = gcGoalUtilization
	DefaultHeapMinimum          = defaultHeapMinimum
	MemoryLimitHeapGoalHeadroom = memoryLimitHeapGoalHeadroom
)

type GCController struct {
	gcControllerState
}

func NewGCController(gcPercent int, memoryLimit int64) *GCController {
	// Force the controller to escape. We're going to
	// do 64-bit atomics on it, and if it gets stack-allocated
	// on a 32-bit architecture, it may get allocated unaligned
	// space.
	g := Escape(new(GCController))
	g.gcControllerState.test = true // Mark it as a test copy.
	g.init(int32(gcPercent), memoryLimit)
	return g
}

func (c *GCController) StartCycle(stackSize, globalsSize uint64, scannableFrac float64, gomaxprocs int) {
	trigger, _ := c.trigger()
	if c.heapMarked > trigger {
		trigger = c.heapMarked
	}
	c.maxStackScan.Store(stackSize)
	c.globalsScan.Store(globalsSize)
	c.heapLive.Store(trigger)
	c.heapScan.Add(int64(float64(trigger-c.heapMarked) * scannableFrac))
	c.startCycle(0, gomaxprocs, gcTrigger{kind: gcTriggerHeap})
}

func (c *GCController) AssistWorkPerByte() float64 {
	return c.assistWorkPerByte.Load()
}

func (c *GCController) HeapGoal() uint64 {
	return c.heapGoal()
}

func (c *GCController) HeapLive() uint64 {
	return c.heapLive.Load()
}

func (c *GCController) HeapMarked() uint64 {
	return c.heapMarked
}

func (c *GCController) Triggered() uint64 {
	return c.triggered
}

type GCControllerReviseDelta struct {
	HeapLive        int64
	HeapScan        int64
	HeapScanWork    int64
	StackScanWork   int64
	GlobalsScanWork int64
}

func (c *GCController) Revise(d GCControllerReviseDelta) {
	c.heapLive.Add(d.HeapLive)
	c.heapScan.Add(d.HeapScan)
	c.heapScanWork.Add(d.HeapScanWork)
	c.stackScanWork.Add(d.StackScanWork)
	c.globalsScanWork.Add(d.GlobalsScanWork)
	c.revise()
}

func (c *GCController) EndCycle(bytesMarked uint64, assistTime, elapsed int64, gomaxprocs int) {
	c.assistTime.Store(assistTime)
	c.endCycle(elapsed, gomaxprocs, false)
	c.resetLive(bytesMarked)
	c.commit(false)
}

func (c *GCController) AddIdleMarkWorker() bool {
	return c.addIdleMarkWorker()
}

func (c *GCController) NeedIdleMarkWorker() bool {
	return c.needIdleMarkWorker()
}

func (c *GCController) RemoveIdleMarkWorker() {
	c.removeIdleMarkWorker()
}

func (c *GCController) SetMaxIdleMarkWorkers(max int32) {
	c.setMaxIdleMarkWorkers(max)
}

var alwaysFalse bool
var escapeSink any

func Escape[T any](x T) T {
	if alwaysFalse {
		escapeSink = x
	}
	return x
}

// Acquirem blocks preemption.
func Acquirem() {
	acquirem()
}

func Releasem() {
	releasem(getg().m)
}

var Timediv = timediv

type PIController struct {
	piController
}

func NewPIController(kp, ti, tt, min, max float64) *PIController {
	return &PIController{piController{
		kp:  kp,
		ti:  ti,
		tt:  tt,
		min: min,
		max: max,
	}}
}

func (c *PIController) Next(input, setpoint, period float64) (float64, bool) {
	return c.piController.next(input, setpoint, period)
}

const (
	CapacityPerProc          = capacityPerProc
	GCCPULimiterUpdatePeriod = gcCPULimiterUpdatePeriod
)

type GCCPULimiter struct {
	limiter gcCPULimiterState
}

func NewGCCPULimiter(now int64, gomaxprocs int32) *GCCPULimiter {
	// Force the controller to escape. We're going to
	// do 64-bit atomics on it, and if it gets stack-allocated
	// on a 32-bit architecture, it may get allocated unaligned
	// space.
	l := Escape(new(GCCPULimiter))
	l.limiter.test = true
	l.limiter.resetCapacity(now, gomaxprocs)
	return l
}

func (l *GCCPULimiter) Fill() uint64 {
	return l.limiter.bucket.fill
}

func (l *GCCPULimiter) Capacity() uint64 {
	return l.limiter.bucket.capacity
}

func (l *GCCPULimiter) Overflow() uint64 {
	return l.limiter.overflow
}

func (l *GCCPULimiter) Limiting() bool {
	return l.limiter.limiting()
}

func (l *GCCPULimiter) NeedUpdate(now int64) bool {
	return l.limiter.needUpdate(now)
}

func (l *GCCPULimiter) StartGCTransition(enableGC bool, now int64) {
	l.limiter.startGCTransition(enableGC, now)
}

func (l *GCCPULimiter) FinishGCTransition(now int64) {
	l.limiter.finishGCTransition(now)
}

func (l *GCCPULimiter) Update(now int64) {
	l.limiter.update(now)
}

func (l *GCCPULimiter) AddAssistTime(t int64) {
	l.limiter.addAssistTime(t)
}

func (l *GCCPULimiter) ResetCapacity(now int64, nprocs int32) {
	l.limiter.resetCapacity(now, nprocs)
}

const ScavengePercent = scavengePercent

type Scavenger struct {
	Sleep      func(int64) int64
	Scavenge   func(uintptr) (uintptr, int64)
	ShouldStop func() bool
	GoMaxProcs func() int32

	released  atomic.Uintptr
	scavenger scavengerState
	stop      chan<- struct{}
	done      <-chan struct{}
}

func (s *Scavenger) Start() {
	if s.Sleep == nil || s.Scavenge == nil || s.ShouldStop == nil || s.GoMaxProcs == nil {
		panic("must populate all stubs")
	}

	// Install hooks.
	s.scavenger.sleepStub = s.Sleep
	s.scavenger.scavenge = s.Scavenge
	s.scavenger.shouldStop = s.ShouldStop
	s.scavenger.gomaxprocs = s.GoMaxProcs

	// Start up scavenger goroutine, and wait for it to be ready.
	stop := make(chan struct{})
	s.stop = stop
	done := make(chan struct{})
	s.done = done
	go func() {
		// This should match bgscavenge, loosely.
		s.scavenger.init()
		s.scavenger.park()
		for {
			select {
			case <-stop:
				close(done)
				return
			default:
			}
			released, workTime := s.scavenger.run()
			if released == 0 {
				s.scavenger.park()
				continue
			}
			s.released.Add(released)
			s.scavenger.sleep(workTime)
		}
	}()
	if !s.BlockUntilParked(1e9 /* 1 second */) {
		panic("timed out waiting for scavenger to get ready")
	}
}

// BlockUntilParked blocks until the scavenger parks, or until
// timeout is exceeded. Returns true if the scavenger parked.
//
// Note that in testing, parked means something slightly different.
// In anger, the scavenger parks to sleep, too, but in testing,
// it only parks when it actually has no work to do.
func (s *Scavenger) BlockUntilParked(timeout int64) bool {
	// Just spin, waiting for it to park.
	//
	// The actual parking process is racy with respect to
	// wakeups, which is fine, but for testing we need something
	// a bit more robust.
	start := nanotime()
	for nanotime()-start < timeout {
		lock(&s.scavenger.lock)
		parked := s.scavenger.parked
		unlock(&s.scavenger.lock)
		if parked {
			return true
		}
		Gosched()
	}
	return false
}

// Released returns how many bytes the scavenger released.
func (s *Scavenger) Released() uintptr {
	return s.released.Load()
}

// Wake wakes up a parked scavenger to keep running.
func (s *Scavenger) Wake() {
	s.scavenger.wake()
}

// Stop cleans up the scavenger's resources. The scavenger
// must be parked for this to work.
func (s *Scavenger) Stop() {
	lock(&s.scavenger.lock)
	parked := s.scavenger.parked
	unlock(&s.scavenger.lock)
	if !parked {
		panic("tried to clean up scavenger that is not parked")
	}
	close(s.stop)
	s.Wake()
	<-s.done
}

type ScavengeIndex struct {
	i scavengeIndex
}

func NewScavengeIndex(min, max ChunkIdx) *ScavengeIndex {
	s := new(ScavengeIndex)
	s.i.chunks = make([]atomic.Uint8, uintptr(1<<heapAddrBits/pallocChunkBytes/8))
	s.i.min.Store(int32(min / 8))
	s.i.max.Store(int32(max / 8))
	return s
}

func (s *ScavengeIndex) Find() (ChunkIdx, uint) {
	ci, off := s.i.find()
	return ChunkIdx(ci), off
}

func (s *ScavengeIndex) Mark(base, limit uintptr) {
	s.i.mark(base, limit)
}

func (s *ScavengeIndex) Clear(ci ChunkIdx) {
	s.i.clear(chunkIdx(ci))
}

const GTrackingPeriod = gTrackingPeriod

var ZeroBase = unsafe.Pointer(&zerobase)

const UserArenaChunkBytes = userArenaChunkBytes

type UserArena struct {
	arena *userArena
}

func NewUserArena() *UserArena {
	return &UserArena{newUserArena()}
}

func (a *UserArena) New(out *any) {
	i := efaceOf(out)
	typ := i._type
	if typ.kind&kindMask != kindPtr {
		panic("new result of non-ptr type")
	}
	typ = (*ptrtype)(unsafe.Pointer(typ)).elem
	i.data = a.arena.new(typ)
}

func (a *UserArena) Slice(sl any, cap int) {
	a.arena.slice(sl, cap)
}

func (a *UserArena) Free() {
	a.arena.free()
}

func GlobalWaitingArenaChunks() int {
	n := 0
	systemstack(func() {
		lock(&mheap_.lock)
		for s := mheap_.userArena.quarantineList.first; s != nil; s = s.next {
			n++
		}
		unlock(&mheap_.lock)
	})
	return n
}

func UserArenaClone[T any](s T) T {
	return arena_heapify(s).(T)
}

var AlignUp = alignUp

// BlockUntilEmptyFinalizerQueue blocks until either the finalizer
// queue is emptied (and the finalizers have executed) or the timeout
// is reached. Returns true if the finalizer queue was emptied.
func BlockUntilEmptyFinalizerQueue(timeout int64) bool {
	start := nanotime()
	for nanotime()-start < timeout {
		lock(&finlock)
		// We know the queue has been drained when both finq is nil
		// and the finalizer g has stopped executing.
		empty := finq == nil
		empty = empty && readgstatus(fing) == _Gwaiting && fing.waitreason == waitReasonFinalizerWait
		unlock(&finlock)
		if empty {
			return true
		}
		Gosched()
	}
	return false
}

func FrameStartLine(f *Frame) int {
	return f.startLine
}

// PersistentAlloc allocates some memory that lives outside the Go heap.
// This memory will never be freed; use sparingly.
func PersistentAlloc(n uintptr) unsafe.Pointer {
	return persistentalloc(n, 0, &memstats.other_sys)
}
