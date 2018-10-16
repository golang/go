// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export guts for testing.

package runtime

import (
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

var FuncPC = funcPC

var Fastlog2 = fastlog2

var Atoi = atoi
var Atoi32 = atoi32

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

func GCMask(x interface{}) (ret []byte) {
	systemstack(func() {
		ret = getgcmask(x)
	})
	return
}

func RunSchedLocalQueueTest() {
	_p_ := new(p)
	gs := make([]g, len(_p_.runq))
	for i := 0; i < len(_p_.runq); i++ {
		if g, _ := runqget(_p_); g != nil {
			throw("runq is not empty initially")
		}
		for j := 0; j < i; j++ {
			runqput(_p_, &gs[i], false)
		}
		for j := 0; j < i; j++ {
			if g, _ := runqget(_p_); g != &gs[i] {
				print("bad element at iter ", i, "/", j, "\n")
				throw("bad element")
			}
		}
		if g, _ := runqget(_p_); g != nil {
			throw("runq is not empty afterwards")
		}
	}
}

func RunSchedLocalQueueStealTest() {
	p1 := new(p)
	p2 := new(p)
	gs := make([]g, len(p1.runq))
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

var HashLoad = &hashLoad

// entry point for testing
func GostringW(w []uint16) (s string) {
	systemstack(func() {
		s = gostringw(&w[0])
	})
	return
}

type Uintreg sys.Uintreg

var Open = open
var Close = closefd
var Read = read
var Write = write

func Envs() []string     { return envs }
func SetEnvs(e []string) { envs = e }

var BigEndian = sys.BigEndian

// For benchmarking.

func BenchSetType(n int, x interface{}) {
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

const PtrSize = sys.PtrSize

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

	pagesInUse = uintptr(mheap_.pagesInUse)

	for _, s := range mheap_.allspans {
		if s.state == mSpanInUse {
			counted += s.npages
		}
	}

	startTheWorld()

	return
}

func Fastrand() uint32          { return fastrand() }
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
			if s.state != mSpanInUse {
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

		// Add in frees. readmemstats_m flushed the cached stats, so
		// these are up-to-date.
		var smallFree uint64
		slow.Frees = mheap_.nlargefree
		for i := range mheap_.nsmallfree {
			slow.Frees += mheap_.nsmallfree[i]
			bySize[i].Frees = mheap_.nsmallfree[i]
			bySize[i].Mallocs += mheap_.nsmallfree[i]
			smallFree += mheap_.nsmallfree[i] * uint64(class_to_size[i])
		}
		slow.Frees += memstats.tinyallocs
		slow.Mallocs += slow.Frees

		slow.TotalAlloc = slow.Alloc + mheap_.largefree + smallFree

		for i := range slow.BySize {
			slow.BySize[i].Mallocs = bySize[i].Mallocs
			slow.BySize[i].Frees = bySize[i].Frees
		}

		mheap_.scav.treap.walkTreap(func(tn *treapNode) {
			slow.HeapReleased += uint64(tn.spanKey.released())
		})

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
	g := getg()
	if g.m.lockedExt+g.m.lockedInt == 0 {
		if g.lockedm != 0 {
			panic("lockedm on non-locked goroutine")
		}
	} else {
		if g.lockedm == 0 {
			panic("nil lockedm on locked goroutine")
		}
	}
	return g.m.lockedExt, g.m.lockedInt
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
func MapNextArenaHint() (start, end uintptr) {
	hint := mheap_.arenaHints
	addr := hint.addr
	if hint.down {
		start, end = addr-heapArenaBytes, addr
		addr -= physPageSize
	} else {
		start, end = addr, addr+heapArenaBytes
	}
	sysReserve(unsafe.Pointer(addr), physPageSize)
	return
}

func GetNextArenaHint() uintptr {
	return mheap_.arenaHints.addr
}

type G = g

func Getg() *G {
	return getg()
}

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
	i := interface{}(m)
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
