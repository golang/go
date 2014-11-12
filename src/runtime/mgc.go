// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): The code having to do with the heap bitmap needs very serious cleanup.
// It has gotten completely out of control.

// Garbage collector (GC).
//
// GC is:
// - mark&sweep
// - mostly precise (with the exception of some C-allocated objects, assembly frames/arguments, etc)
// - parallel (up to MaxGcproc threads)
// - partially concurrent (mark is stop-the-world, while sweep is concurrent)
// - non-moving/non-compacting
// - full (non-partial)
//
// GC rate.
// Next GC is after we've allocated an extra amount of memory proportional to
// the amount already in use. The proportion is controlled by GOGC environment variable
// (100 by default). If GOGC=100 and we're using 4M, we'll GC again when we get to 8M
// (this mark is tracked in next_gc variable). This keeps the GC cost in linear
// proportion to the allocation cost. Adjusting GOGC just changes the linear constant
// (and also the amount of extra memory used).
//
// Concurrent sweep.
// The sweep phase proceeds concurrently with normal program execution.
// The heap is swept span-by-span both lazily (when a goroutine needs another span)
// and concurrently in a background goroutine (this helps programs that are not CPU bound).
// However, at the end of the stop-the-world GC phase we don't know the size of the live heap,
// and so next_gc calculation is tricky and happens as follows.
// At the end of the stop-the-world phase next_gc is conservatively set based on total
// heap size; all spans are marked as "needs sweeping".
// Whenever a span is swept, next_gc is decremented by GOGC*newly_freed_memory.
// The background sweeper goroutine simply sweeps spans one-by-one bringing next_gc
// closer to the target value. However, this is not enough to avoid over-allocating memory.
// Consider that a goroutine wants to allocate a new span for a large object and
// there are no free swept spans, but there are small-object unswept spans.
// If the goroutine naively allocates a new span, it can surpass the yet-unknown
// target next_gc value. In order to prevent such cases (1) when a goroutine needs
// to allocate a new small-object span, it sweeps small-object spans for the same
// object size until it frees at least one object; (2) when a goroutine needs to
// allocate large-object span from heap, it sweeps spans until it frees at least
// that many pages into heap. Together these two measures ensure that we don't surpass
// target next_gc value by a large margin. There is an exception: if a goroutine sweeps
// and frees two nonadjacent one-page spans to the heap, it will allocate a new two-page span,
// but there can still be other one-page unswept spans which could be combined into a two-page span.
// It's critical to ensure that no operations proceed on unswept spans (that would corrupt
// mark bits in GC bitmap). During GC all mcaches are flushed into the central cache,
// so they are empty. When a goroutine grabs a new span into mcache, it sweeps it.
// When a goroutine explicitly frees an object or sets a finalizer, it ensures that
// the span is swept (either by sweeping it, or by waiting for the concurrent sweep to finish).
// The finalizer goroutine is kicked off only when all spans are swept.
// When the next GC starts, it sweeps all not-yet-swept spans (if any).

package runtime

import "unsafe"

const (
	_DebugGC         = 0
	_DebugGCPtrs     = false // if true, print trace of every pointer load during GC
	_ConcurrentSweep = true

	_WorkbufSize     = 4 * 1024
	_FinBlockSize    = 4 * 1024
	_RootData        = 0
	_RootBss         = 1
	_RootFinalizers  = 2
	_RootSpans       = 3
	_RootFlushCaches = 4
	_RootCount       = 5
)

// ptrmask for an allocation containing a single pointer.
var oneptr = [...]uint8{bitsPointer}

// Initialized from $GOGC.  GOGC=off means no gc.
var gcpercent int32

// Holding worldsema grants an M the right to try to stop the world.
// The procedure is:
//
//	semacquire(&worldsema);
//	m.gcing = 1;
//	stoptheworld();
//
//	... do stuff ...
//
//	m.gcing = 0;
//	semrelease(&worldsema);
//	starttheworld();
//
var worldsema uint32 = 1

type workbuf struct {
	node lfnode // must be first
	nobj uintptr
	obj  [(_WorkbufSize - unsafe.Sizeof(lfnode{}) - ptrSize) / ptrSize]uintptr
}

var data, edata, bss, ebss, gcdata, gcbss struct{}

var finlock mutex  // protects the following variables
var fing *g        // goroutine that runs finalizers
var finq *finblock // list of finalizers that are to be executed
var finc *finblock // cache of free blocks
var finptrmask [_FinBlockSize / ptrSize / pointersPerByte]byte
var fingwait bool
var fingwake bool
var allfin *finblock // list of all blocks

var gcdatamask bitvector
var gcbssmask bitvector

var gclock mutex

var badblock [1024]uintptr
var nbadblock int32

type workdata struct {
	full    uint64                // lock-free list of full blocks
	empty   uint64                // lock-free list of empty blocks
	pad0    [_CacheLineSize]uint8 // prevents false-sharing between full/empty and nproc/nwait
	nproc   uint32
	tstart  int64
	nwait   uint32
	ndone   uint32
	alldone note
	markfor *parfor

	// Copy of mheap.allspans for marker or sweeper.
	spans []*mspan
}

var work workdata

//go:linkname weak_cgo_allocate go.weak.runtime._cgo_allocate_internal
var weak_cgo_allocate byte

// Is _cgo_allocate linked into the binary?
func have_cgo_allocate() bool {
	return &weak_cgo_allocate != nil
}

// scanblock scans a block of n bytes starting at pointer b for references
// to other objects, scanning any it finds recursively until there are no
// unscanned objects left.  Instead of using an explicit recursion, it keeps
// a work list in the Workbuf* structures and loops in the main function
// body.  Keeping an explicit work list is easier on the stack allocator and
// more efficient.
func scanblock(b, n uintptr, ptrmask *uint8) {
	// Cache memory arena parameters in local vars.
	arena_start := mheap_.arena_start
	arena_used := mheap_.arena_used

	wbuf := getempty(nil)
	nobj := wbuf.nobj
	wp := &wbuf.obj[nobj]
	keepworking := b == 0

	var ptrbitp unsafe.Pointer

	// ptrmask can have 2 possible values:
	// 1. nil - obtain pointer mask from GC bitmap.
	// 2. pointer to a compact mask (for stacks and data).
	goto_scanobj := b != 0

	for {
		if goto_scanobj {
			goto_scanobj = false
		} else {
			if nobj == 0 {
				// Out of work in workbuf.
				if !keepworking {
					putempty(wbuf)
					return
				}

				// Refill workbuf from global queue.
				wbuf = getfull(wbuf)
				if wbuf == nil {
					return
				}
				nobj = wbuf.nobj
				if nobj < uintptr(len(wbuf.obj)) {
					wp = &wbuf.obj[nobj]
				} else {
					wp = nil
				}
			}

			// If another proc wants a pointer, give it some.
			if work.nwait > 0 && nobj > 4 && work.full == 0 {
				wbuf.nobj = nobj
				wbuf = handoff(wbuf)
				nobj = wbuf.nobj
				if nobj < uintptr(len(wbuf.obj)) {
					wp = &wbuf.obj[nobj]
				} else {
					wp = nil
				}
			}

			nobj--
			wp = &wbuf.obj[nobj]
			b = *wp
			n = arena_used - uintptr(b)
			ptrmask = nil // use GC bitmap for pointer info
		}

		if _DebugGCPtrs {
			print("scanblock ", b, " +", hex(n), " ", ptrmask, "\n")
		}

		// Find bits of the beginning of the object.
		if ptrmask == nil {
			off := (uintptr(b) - arena_start) / ptrSize
			ptrbitp = unsafe.Pointer(arena_start - off/wordsPerBitmapByte - 1)
		}

		var i uintptr
		for i = 0; i < n; i += ptrSize {
			// Find bits for this word.
			var bits uintptr
			if ptrmask == nil {
				// Check if we have reached end of span.
				if (uintptr(b)+i)%_PageSize == 0 &&
					h_spans[(uintptr(b)-arena_start)>>_PageShift] != h_spans[(uintptr(b)+i-arena_start)>>_PageShift] {
					break
				}

				// Consult GC bitmap.
				bits = uintptr(*(*byte)(ptrbitp))

				if wordsPerBitmapByte != 2 {
					gothrow("alg doesn't work for wordsPerBitmapByte != 2")
				}
				j := (uintptr(b) + i) / ptrSize & 1
				ptrbitp = add(ptrbitp, -j)
				bits >>= gcBits * j

				if bits&bitBoundary != 0 && i != 0 {
					break // reached beginning of the next object
				}
				bits = (bits >> 2) & bitsMask
				if bits == bitsDead {
					break // reached no-scan part of the object
				}
			} else {
				// dense mask (stack or data)
				bits = (uintptr(*(*byte)(add(unsafe.Pointer(ptrmask), (i/ptrSize)/4))) >> (((i / ptrSize) % 4) * bitsPerPointer)) & bitsMask
			}

			if bits <= _BitsScalar { // BitsScalar || BitsDead
				continue
			}

			if bits != _BitsPointer {
				gothrow("unexpected garbage collection bits")
			}

			obj := *(*uintptr)(unsafe.Pointer(b + i))
			obj0 := obj

		markobj:
			var s *mspan
			var off, bitp, shift, xbits uintptr

			// At this point we have extracted the next potential pointer.
			// Check if it points into heap.
			if obj == 0 {
				continue
			}
			if obj < arena_start || arena_used <= obj {
				if uintptr(obj) < _PhysPageSize && invalidptr != 0 {
					s = nil
					goto badobj
				}
				continue
			}

			// Mark the object.
			obj &^= ptrSize - 1
			off = (obj - arena_start) / ptrSize
			bitp = arena_start - off/wordsPerBitmapByte - 1
			shift = (off % wordsPerBitmapByte) * gcBits
			xbits = uintptr(*(*byte)(unsafe.Pointer(bitp)))
			bits = (xbits >> shift) & bitMask
			if (bits & bitBoundary) == 0 {
				// Not a beginning of a block, consult span table to find the block beginning.
				k := pageID(obj >> _PageShift)
				x := k
				x -= pageID(arena_start >> _PageShift)
				s = h_spans[x]
				if s == nil || k < s.start || s.limit <= obj || s.state != mSpanInUse {
					// Stack pointers lie within the arena bounds but are not part of the GC heap.
					// Ignore them.
					if s != nil && s.state == _MSpanStack {
						continue
					}
					goto badobj
				}
				p := uintptr(s.start) << _PageShift
				if s.sizeclass != 0 {
					size := s.elemsize
					idx := (obj - p) / size
					p = p + idx*size
				}
				if p == obj {
					print("runtime: failed to find block beginning for ", hex(p), " s=", hex(s.start*_PageSize), " s.limit=", hex(s.limit), "\n")
					gothrow("failed to find block beginning")
				}
				obj = p
				goto markobj
			}

			if _DebugGCPtrs {
				print("scan *", hex(b+i), " = ", hex(obj0), " => base ", hex(obj), "\n")
			}

			if nbadblock > 0 && obj == badblock[nbadblock-1] {
				// Running garbage collection again because
				// we want to find the path from a root to a bad pointer.
				// Found possible next step; extend or finish path.
				for j := int32(0); j < nbadblock; j++ {
					if badblock[j] == b {
						goto AlreadyBad
					}
				}
				print("runtime: found *(", hex(b), "+", hex(i), ") = ", hex(obj0), "+", hex(obj-obj0), "\n")
				if ptrmask != nil {
					gothrow("bad pointer")
				}
				if nbadblock >= int32(len(badblock)) {
					gothrow("badblock trace too long")
				}
				badblock[nbadblock] = uintptr(b)
				nbadblock++
			AlreadyBad:
			}

			// Now we have bits, bitp, and shift correct for
			// obj pointing at the base of the object.
			// Only care about not marked objects.
			if bits&bitMarked != 0 {
				continue
			}

			// If obj size is greater than 8, then each byte of GC bitmap
			// contains info for at most one object. In such case we use
			// non-atomic byte store to mark the object. This can lead
			// to double enqueue of the object for scanning, but scanning
			// is an idempotent operation, so it is OK. This cannot lead
			// to bitmap corruption because the single marked bit is the
			// only thing that can change in the byte.
			// For 8-byte objects we use non-atomic store, if the other
			// quadruple is already marked. Otherwise we resort to CAS
			// loop for marking.
			if xbits&(bitMask|bitMask<<gcBits) != bitBoundary|bitBoundary<<gcBits || work.nproc == 1 {
				*(*byte)(unsafe.Pointer(bitp)) = uint8(xbits | bitMarked<<shift)
			} else {
				atomicor8((*byte)(unsafe.Pointer(bitp)), bitMarked<<shift)
			}

			if (xbits>>(shift+2))&bitsMask == bitsDead {
				continue // noscan object
			}

			// Queue the obj for scanning.
			// TODO: PREFETCH here.

			// If workbuf is full, obtain an empty one.
			if nobj >= uintptr(len(wbuf.obj)) {
				wbuf.nobj = nobj
				wbuf = getempty(wbuf)
				nobj = wbuf.nobj
				wp = &wbuf.obj[nobj]
			}
			*wp = obj
			nobj++
			if nobj < uintptr(len(wbuf.obj)) {
				wp = &wbuf.obj[nobj]
			} else {
				wp = nil
			}
			continue

		badobj:
			// If cgo_allocate is linked into the binary, it can allocate
			// memory as []unsafe.Pointer that may not contain actual
			// pointers and must be scanned conservatively.
			// In this case alone, allow the bad pointer.
			if have_cgo_allocate() && ptrmask == nil {
				continue
			}

			// Anything else indicates a bug somewhere.
			// If we're in the middle of chasing down a different bad pointer,
			// don't confuse the trace by printing about this one.
			if nbadblock > 0 {
				continue
			}

			print("runtime: garbage collector found invalid heap pointer *(", hex(b), "+", hex(i), ")=", hex(obj))
			if s == nil {
				print(" s=nil\n")
			} else {
				print(" span=", uintptr(s.start)<<_PageShift, "-", s.limit, "-", (uintptr(s.start)+s.npages)<<_PageShift, " state=", s.state, "\n")
			}
			if ptrmask != nil {
				gothrow("invalid heap pointer")
			}
			// Add to badblock list, which will cause the garbage collection
			// to keep repeating until it has traced the chain of pointers
			// leading to obj all the way back to a root.
			if nbadblock == 0 {
				badblock[nbadblock] = uintptr(b)
				nbadblock++
			}
		}
		if _DebugGCPtrs {
			print("end scanblock ", hex(b), " +", hex(n), " ", ptrmask, "\n")
		}
		if _DebugGC > 0 && ptrmask == nil {
			// For heap objects ensure that we did not overscan.
			var p, n uintptr
			if mlookup(b, &p, &n, nil) == 0 || b != p || i > n {
				print("runtime: scanned (", hex(b), "+", hex(i), "), heap object (", hex(p), "+", hex(n), ")\n")
				gothrow("scanblock: scanned invalid object")
			}
		}
	}
}

func markroot(desc *parfor, i uint32) {
	// Note: if you add a case here, please also update heapdump.c:dumproots.
	switch i {
	case _RootData:
		scanblock(uintptr(unsafe.Pointer(&data)), uintptr(unsafe.Pointer(&edata))-uintptr(unsafe.Pointer(&data)), gcdatamask.bytedata)

	case _RootBss:
		scanblock(uintptr(unsafe.Pointer(&bss)), uintptr(unsafe.Pointer(&ebss))-uintptr(unsafe.Pointer(&bss)), gcbssmask.bytedata)

	case _RootFinalizers:
		for fb := allfin; fb != nil; fb = fb.alllink {
			scanblock(uintptr(unsafe.Pointer(&fb.fin[0])), uintptr(fb.cnt)*unsafe.Sizeof(fb.fin[0]), &finptrmask[0])
		}

	case _RootSpans:
		// mark MSpan.specials
		sg := mheap_.sweepgen
		for spanidx := uint32(0); spanidx < uint32(len(work.spans)); spanidx++ {
			s := work.spans[spanidx]
			if s.state != mSpanInUse {
				continue
			}
			if s.sweepgen != sg {
				print("sweep ", s.sweepgen, " ", sg, "\n")
				gothrow("gc: unswept span")
			}
			for sp := s.specials; sp != nil; sp = sp.next {
				if sp.kind != _KindSpecialFinalizer {
					continue
				}
				// don't mark finalized object, but scan it so we
				// retain everything it points to.
				spf := (*specialfinalizer)(unsafe.Pointer(sp))
				// A finalizer can be set for an inner byte of an object, find object beginning.
				p := uintptr(s.start<<_PageShift) + uintptr(spf.special.offset)/s.elemsize*s.elemsize
				scanblock(p, s.elemsize, nil)
				scanblock(uintptr(unsafe.Pointer(&spf.fn)), ptrSize, &oneptr[0])
			}
		}

	case _RootFlushCaches:
		flushallmcaches()

	default:
		// the rest is scanning goroutine stacks
		if uintptr(i-_RootCount) >= allglen {
			gothrow("markroot: bad index")
		}
		gp := allgs[i-_RootCount]
		// remember when we've first observed the G blocked
		// needed only to output in traceback
		status := readgstatus(gp)
		if (status == _Gwaiting || status == _Gsyscall) && gp.waitsince == 0 {
			gp.waitsince = work.tstart
		}
		// Shrink a stack if not much of it is being used.
		shrinkstack(gp)
		if readgstatus(gp) == _Gdead {
			gp.gcworkdone = true
		} else {
			gp.gcworkdone = false
		}
		restart := stopg(gp)
		scanstack(gp)
		if restart {
			restartg(gp)
		}
	}
}

// Get an empty work buffer off the work.empty list,
// allocating new buffers as needed.
func getempty(b *workbuf) *workbuf {
	_g_ := getg()
	if b != nil {
		lfstackpush(&work.full, &b.node)
	}
	b = nil
	c := _g_.m.mcache
	if c.gcworkbuf != nil {
		b = (*workbuf)(c.gcworkbuf)
		c.gcworkbuf = nil
	}
	if b == nil {
		b = (*workbuf)(lfstackpop(&work.empty))
	}
	if b == nil {
		b = (*workbuf)(persistentalloc(unsafe.Sizeof(*b), _CacheLineSize, &memstats.gc_sys))
	}
	b.nobj = 0
	return b
}

func putempty(b *workbuf) {
	_g_ := getg()
	c := _g_.m.mcache
	if c.gcworkbuf == nil {
		c.gcworkbuf = (unsafe.Pointer)(b)
		return
	}
	lfstackpush(&work.empty, &b.node)
}

func gcworkbuffree(b unsafe.Pointer) {
	if b != nil {
		putempty((*workbuf)(b))
	}
}

// Get a full work buffer off the work.full list, or return nil.
func getfull(b *workbuf) *workbuf {
	if b != nil {
		lfstackpush(&work.empty, &b.node)
	}
	b = (*workbuf)(lfstackpop(&work.full))
	if b != nil || work.nproc == 1 {
		return b
	}

	xadd(&work.nwait, +1)
	for i := 0; ; i++ {
		if work.full != 0 {
			xadd(&work.nwait, -1)
			b = (*workbuf)(lfstackpop(&work.full))
			if b != nil {
				return b
			}
			xadd(&work.nwait, +1)
		}
		if work.nwait == work.nproc {
			return nil
		}
		_g_ := getg()
		if i < 10 {
			_g_.m.gcstats.nprocyield++
			procyield(20)
		} else if i < 20 {
			_g_.m.gcstats.nosyield++
			osyield()
		} else {
			_g_.m.gcstats.nsleep++
			usleep(100)
		}
	}
}

func handoff(b *workbuf) *workbuf {
	// Make new buffer with half of b's pointers.
	b1 := getempty(nil)
	n := b.nobj / 2
	b.nobj -= n
	b1.nobj = n
	memmove(unsafe.Pointer(&b1.obj[0]), unsafe.Pointer(&b.obj[b.nobj]), n*unsafe.Sizeof(b1.obj[0]))
	_g_ := getg()
	_g_.m.gcstats.nhandoff++
	_g_.m.gcstats.nhandoffcnt += uint64(n)

	// Put b on full list - let first half of b get stolen.
	lfstackpush(&work.full, &b.node)
	return b1
}

func stackmapdata(stkmap *stackmap, n int32) bitvector {
	if n < 0 || n >= stkmap.n {
		gothrow("stackmapdata: index out of range")
	}
	return bitvector{stkmap.nbit, (*byte)(add(unsafe.Pointer(&stkmap.bytedata), uintptr(n*((stkmap.nbit+31)/32*4))))}
}

// Scan a stack frame: local variables and function arguments/results.
func scanframe(frame *stkframe, unused unsafe.Pointer) bool {

	f := frame.fn
	targetpc := frame.continpc
	if targetpc == 0 {
		// Frame is dead.
		return true
	}
	if _DebugGC > 1 {
		print("scanframe ", gofuncname(f), "\n")
	}
	if targetpc != f.entry {
		targetpc--
	}
	pcdata := pcdatavalue(f, _PCDATA_StackMapIndex, targetpc)
	if pcdata == -1 {
		// We do not have a valid pcdata value but there might be a
		// stackmap for this function.  It is likely that we are looking
		// at the function prologue, assume so and hope for the best.
		pcdata = 0
	}

	// Scan local variables if stack frame has been allocated.
	size := frame.varp - frame.sp
	var minsize uintptr
	if thechar != '6' && thechar != '8' {
		minsize = ptrSize
	} else {
		minsize = 0
	}
	if size > minsize {
		stkmap := (*stackmap)(funcdata(f, _FUNCDATA_LocalsPointerMaps))
		if stkmap == nil || stkmap.n <= 0 {
			print("runtime: frame ", gofuncname(f), " untyped locals ", hex(frame.varp-size), "+", hex(size), "\n")
			gothrow("missing stackmap")
		}

		// Locals bitmap information, scan just the pointers in locals.
		if pcdata < 0 || pcdata >= stkmap.n {
			// don't know where we are
			print("runtime: pcdata is ", pcdata, " and ", stkmap.n, " locals stack map entries for ", gofuncname(f), " (targetpc=", targetpc, ")\n")
			gothrow("scanframe: bad symbol table")
		}
		bv := stackmapdata(stkmap, pcdata)
		size = (uintptr(bv.n) * ptrSize) / bitsPerPointer
		scanblock(frame.varp-size, uintptr(bv.n)/bitsPerPointer*ptrSize, bv.bytedata)
	}

	// Scan arguments.
	if frame.arglen > 0 {
		var bv bitvector
		if frame.argmap != nil {
			bv = *frame.argmap
		} else {
			stkmap := (*stackmap)(funcdata(f, _FUNCDATA_ArgsPointerMaps))
			if stkmap == nil || stkmap.n <= 0 {
				print("runtime: frame ", gofuncname(f), " untyped args ", hex(frame.argp), "+", hex(frame.arglen), "\n")
				gothrow("missing stackmap")
			}
			if pcdata < 0 || pcdata >= stkmap.n {
				// don't know where we are
				print("runtime: pcdata is ", pcdata, " and ", stkmap.n, " args stack map entries for ", gofuncname(f), " (targetpc=", targetpc, ")\n")
				gothrow("scanframe: bad symbol table")
			}
			bv = stackmapdata(stkmap, pcdata)
		}
		scanblock(frame.argp, uintptr(bv.n)/bitsPerPointer*ptrSize, bv.bytedata)
	}
	return true
}

func scanstack(gp *g) {
	// TODO(rsc): Due to a precedence error, this was never checked in the original C version.
	// If you enable the check, the gothrow happens.
	/*
		if readgstatus(gp)&_Gscan == 0 {
			print("runtime: gp=", gp, ", goid=", gp.goid, ", gp->atomicstatus=", readgstatus(gp), "\n")
			gothrow("mark - bad status")
		}
	*/

	switch readgstatus(gp) &^ _Gscan {
	default:
		print("runtime: gp=", gp, ", goid=", gp.goid, ", gp->atomicstatus=", readgstatus(gp), "\n")
		gothrow("mark - bad status")
	case _Gdead:
		return
	case _Grunning:
		print("runtime: gp=", gp, ", goid=", gp.goid, ", gp->atomicstatus=", readgstatus(gp), "\n")
		gothrow("mark - world not stopped")
	case _Grunnable, _Gsyscall, _Gwaiting:
		// ok
	}

	if gp == getg() {
		gothrow("can't scan our own stack")
	}
	mp := gp.m
	if mp != nil && mp.helpgc != 0 {
		gothrow("can't scan gchelper stack")
	}

	gentraceback(^uintptr(0), ^uintptr(0), 0, gp, 0, nil, 0x7fffffff, scanframe, nil, 0)
	tracebackdefers(gp, scanframe, nil)
}

// The gp has been moved to a gc safepoint. If there is gcphase specific
// work it is done here.
func gcphasework(gp *g) {
	switch gcphase {
	default:
		gothrow("gcphasework in bad gcphase")
	case _GCoff, _GCquiesce, _GCstw, _GCsweep:
		// No work for now.
	case _GCmark:
		// Disabled until concurrent GC is implemented
		// but indicate the scan has been done.
		// scanstack(gp);
	}
	gp.gcworkdone = true
}

var finalizer1 = [...]byte{
	// Each Finalizer is 5 words, ptr ptr uintptr ptr ptr.
	// Each byte describes 4 words.
	// Need 4 Finalizers described by 5 bytes before pattern repeats:
	//	ptr ptr uintptr ptr ptr
	//	ptr ptr uintptr ptr ptr
	//	ptr ptr uintptr ptr ptr
	//	ptr ptr uintptr ptr ptr
	// aka
	//	ptr ptr uintptr ptr
	//	ptr ptr ptr uintptr
	//	ptr ptr ptr ptr
	//	uintptr ptr ptr ptr
	//	ptr uintptr ptr ptr
	// Assumptions about Finalizer layout checked below.
	bitsPointer | bitsPointer<<2 | bitsScalar<<4 | bitsPointer<<6,
	bitsPointer | bitsPointer<<2 | bitsPointer<<4 | bitsScalar<<6,
	bitsPointer | bitsPointer<<2 | bitsPointer<<4 | bitsPointer<<6,
	bitsScalar | bitsPointer<<2 | bitsPointer<<4 | bitsPointer<<6,
	bitsPointer | bitsScalar<<2 | bitsPointer<<4 | bitsPointer<<6,
}

func queuefinalizer(p unsafe.Pointer, fn *funcval, nret uintptr, fint *_type, ot *ptrtype) {
	lock(&finlock)
	if finq == nil || finq.cnt == finq.cap {
		if finc == nil {
			finc = (*finblock)(persistentalloc(_FinBlockSize, 0, &memstats.gc_sys))
			finc.cap = int32((_FinBlockSize-unsafe.Sizeof(finblock{}))/unsafe.Sizeof(finalizer{}) + 1)
			finc.alllink = allfin
			allfin = finc
			if finptrmask[0] == 0 {
				// Build pointer mask for Finalizer array in block.
				// Check assumptions made in finalizer1 array above.
				if (unsafe.Sizeof(finalizer{}) != 5*ptrSize ||
					unsafe.Offsetof(finalizer{}.fn) != 0 ||
					unsafe.Offsetof(finalizer{}.arg) != ptrSize ||
					unsafe.Offsetof(finalizer{}.nret) != 2*ptrSize ||
					unsafe.Offsetof(finalizer{}.fint) != 3*ptrSize ||
					unsafe.Offsetof(finalizer{}.ot) != 4*ptrSize ||
					bitsPerPointer != 2) {
					gothrow("finalizer out of sync")
				}
				for i := range finptrmask {
					finptrmask[i] = finalizer1[i%len(finalizer1)]
				}
			}
		}
		block := finc
		finc = block.next
		block.next = finq
		finq = block
	}
	f := (*finalizer)(add(unsafe.Pointer(&finq.fin[0]), uintptr(finq.cnt)*unsafe.Sizeof(finq.fin[0])))
	finq.cnt++
	f.fn = fn
	f.nret = nret
	f.fint = fint
	f.ot = ot
	f.arg = p
	fingwake = true
	unlock(&finlock)
}

func iterate_finq(callback func(*funcval, unsafe.Pointer, uintptr, *_type, *ptrtype)) {
	for fb := allfin; fb != nil; fb = fb.alllink {
		for i := int32(0); i < fb.cnt; i++ {
			f := &fb.fin[i]
			callback(f.fn, f.arg, f.nret, f.fint, f.ot)
		}
	}
}

func mSpan_EnsureSwept(s *mspan) {
	// Caller must disable preemption.
	// Otherwise when this function returns the span can become unswept again
	// (if GC is triggered on another goroutine).
	_g_ := getg()
	if _g_.m.locks == 0 && _g_.m.mallocing == 0 && _g_ != _g_.m.g0 {
		gothrow("MSpan_EnsureSwept: m is not locked")
	}

	sg := mheap_.sweepgen
	if atomicload(&s.sweepgen) == sg {
		return
	}
	if cas(&s.sweepgen, sg-2, sg-1) {
		mSpan_Sweep(s, false)
		return
	}
	// unfortunate condition, and we don't have efficient means to wait
	for atomicload(&s.sweepgen) != sg {
		osyield()
	}
}

// Sweep frees or collects finalizers for blocks not marked in the mark phase.
// It clears the mark bits in preparation for the next GC round.
// Returns true if the span was returned to heap.
// If preserve=true, don't return it to heap nor relink in MCentral lists;
// caller takes care of it.
func mSpan_Sweep(s *mspan, preserve bool) bool {
	// It's critical that we enter this function with preemption disabled,
	// GC must not start while we are in the middle of this function.
	_g_ := getg()
	if _g_.m.locks == 0 && _g_.m.mallocing == 0 && _g_ != _g_.m.g0 {
		gothrow("MSpan_Sweep: m is not locked")
	}
	sweepgen := mheap_.sweepgen
	if s.state != mSpanInUse || s.sweepgen != sweepgen-1 {
		print("MSpan_Sweep: state=", s.state, " sweepgen=", s.sweepgen, " mheap.sweepgen=", sweepgen, "\n")
		gothrow("MSpan_Sweep: bad span state")
	}
	arena_start := mheap_.arena_start
	cl := s.sizeclass
	size := s.elemsize
	var n int32
	var npages int32
	if cl == 0 {
		n = 1
	} else {
		// Chunk full of small blocks.
		npages = class_to_allocnpages[cl]
		n = (npages << _PageShift) / int32(size)
	}
	res := false
	nfree := 0
	var head mlink
	end := &head
	c := _g_.m.mcache
	sweepgenset := false

	// Mark any free objects in this span so we don't collect them.
	for link := s.freelist; link != nil; link = link.next {
		off := (uintptr(unsafe.Pointer(link)) - arena_start) / ptrSize
		bitp := arena_start - off/wordsPerBitmapByte - 1
		shift := (off % wordsPerBitmapByte) * gcBits
		*(*byte)(unsafe.Pointer(bitp)) |= bitMarked << shift
	}

	// Unlink & free special records for any objects we're about to free.
	specialp := &s.specials
	special := *specialp
	for special != nil {
		// A finalizer can be set for an inner byte of an object, find object beginning.
		p := uintptr(s.start<<_PageShift) + uintptr(special.offset)/size*size
		off := (p - arena_start) / ptrSize
		bitp := arena_start - off/wordsPerBitmapByte - 1
		shift := (off % wordsPerBitmapByte) * gcBits
		bits := (*(*byte)(unsafe.Pointer(bitp)) >> shift) & bitMask
		if bits&bitMarked == 0 {
			// Find the exact byte for which the special was setup
			// (as opposed to object beginning).
			p := uintptr(s.start<<_PageShift) + uintptr(special.offset)
			// about to free object: splice out special record
			y := special
			special = special.next
			*specialp = special
			if !freespecial(y, unsafe.Pointer(p), size, false) {
				// stop freeing of object if it has a finalizer
				*(*byte)(unsafe.Pointer(bitp)) |= bitMarked << shift
			}
		} else {
			// object is still live: keep special record
			specialp = &special.next
			special = *specialp
		}
	}

	// Sweep through n objects of given size starting at p.
	// This thread owns the span now, so it can manipulate
	// the block bitmap without atomic operations.
	p := uintptr(s.start << _PageShift)
	off := (p - arena_start) / ptrSize
	bitp := arena_start - off/wordsPerBitmapByte - 1
	shift := uint(0)
	step := size / (ptrSize * wordsPerBitmapByte)
	// Rewind to the previous quadruple as we move to the next
	// in the beginning of the loop.
	bitp += step
	if step == 0 {
		// 8-byte objects.
		bitp++
		shift = gcBits
	}
	for ; n > 0; n, p = n-1, p+size {
		bitp -= step
		if step == 0 {
			if shift != 0 {
				bitp--
			}
			shift = gcBits - shift
		}

		xbits := *(*byte)(unsafe.Pointer(bitp))
		bits := (xbits >> shift) & bitMask

		// Allocated and marked object, reset bits to allocated.
		if bits&bitMarked != 0 {
			*(*byte)(unsafe.Pointer(bitp)) &^= bitMarked << shift
			continue
		}

		// At this point we know that we are looking at garbage object
		// that needs to be collected.
		if debug.allocfreetrace != 0 {
			tracefree(unsafe.Pointer(p), size)
		}

		// Reset to allocated+noscan.
		*(*byte)(unsafe.Pointer(bitp)) = uint8(uintptr(xbits&^((bitMarked|bitsMask<<2)<<shift)) | uintptr(bitsDead)<<(shift+2))
		if cl == 0 {
			// Free large span.
			if preserve {
				gothrow("can't preserve large span")
			}
			unmarkspan(p, s.npages<<_PageShift)
			s.needzero = 1

			// important to set sweepgen before returning it to heap
			atomicstore(&s.sweepgen, sweepgen)
			sweepgenset = true

			// NOTE(rsc,dvyukov): The original implementation of efence
			// in CL 22060046 used SysFree instead of SysFault, so that
			// the operating system would eventually give the memory
			// back to us again, so that an efence program could run
			// longer without running out of memory. Unfortunately,
			// calling SysFree here without any kind of adjustment of the
			// heap data structures means that when the memory does
			// come back to us, we have the wrong metadata for it, either in
			// the MSpan structures or in the garbage collection bitmap.
			// Using SysFault here means that the program will run out of
			// memory fairly quickly in efence mode, but at least it won't
			// have mysterious crashes due to confused memory reuse.
			// It should be possible to switch back to SysFree if we also
			// implement and then call some kind of MHeap_DeleteSpan.
			if debug.efence > 0 {
				s.limit = 0 // prevent mlookup from finding this span
				sysFault(unsafe.Pointer(p), size)
			} else {
				mHeap_Free(&mheap_, s, 1)
			}
			c.local_nlargefree++
			c.local_largefree += size
			xadd64(&memstats.next_gc, -int64(size)*int64(gcpercent+100)/100)
			res = true
		} else {
			// Free small object.
			if size > 2*ptrSize {
				*(*uintptr)(unsafe.Pointer(p + ptrSize)) = uintptrMask & 0xdeaddeaddeaddead // mark as "needs to be zeroed"
			} else if size > ptrSize {
				*(*uintptr)(unsafe.Pointer(p + ptrSize)) = 0
			}
			end.next = (*mlink)(unsafe.Pointer(p))
			end = end.next
			nfree++
		}
	}

	// We need to set s.sweepgen = h.sweepgen only when all blocks are swept,
	// because of the potential for a concurrent free/SetFinalizer.
	// But we need to set it before we make the span available for allocation
	// (return it to heap or mcentral), because allocation code assumes that a
	// span is already swept if available for allocation.
	if !sweepgenset && nfree == 0 {
		// The span must be in our exclusive ownership until we update sweepgen,
		// check for potential races.
		if s.state != mSpanInUse || s.sweepgen != sweepgen-1 {
			print("MSpan_Sweep: state=", s.state, " sweepgen=", s.sweepgen, " mheap.sweepgen=", sweepgen, "\n")
			gothrow("MSpan_Sweep: bad span state after sweep")
		}
		atomicstore(&s.sweepgen, sweepgen)
	}
	if nfree > 0 {
		c.local_nsmallfree[cl] += uintptr(nfree)
		c.local_cachealloc -= intptr(uintptr(nfree) * size)
		xadd64(&memstats.next_gc, -int64(nfree)*int64(size)*int64(gcpercent+100)/100)
		res = mCentral_FreeSpan(&mheap_.central[cl].mcentral, s, int32(nfree), head.next, end, preserve)
		// MCentral_FreeSpan updates sweepgen
	}
	return res
}

// State of background sweep.
// Protected by gclock.
type sweepdata struct {
	g       *g
	parked  bool
	started bool

	spanidx uint32 // background sweeper position

	nbgsweep    uint32
	npausesweep uint32
}

var sweep sweepdata

// sweeps one span
// returns number of pages returned to heap, or ^uintptr(0) if there is nothing to sweep
func sweepone() uintptr {
	_g_ := getg()

	// increment locks to ensure that the goroutine is not preempted
	// in the middle of sweep thus leaving the span in an inconsistent state for next GC
	_g_.m.locks++
	sg := mheap_.sweepgen
	for {
		idx := xadd(&sweep.spanidx, 1) - 1
		if idx >= uint32(len(work.spans)) {
			mheap_.sweepdone = 1
			_g_.m.locks--
			return ^uintptr(0)
		}
		s := work.spans[idx]
		if s.state != mSpanInUse {
			s.sweepgen = sg
			continue
		}
		if s.sweepgen != sg-2 || !cas(&s.sweepgen, sg-2, sg-1) {
			continue
		}
		npages := s.npages
		if !mSpan_Sweep(s, false) {
			npages = 0
		}
		_g_.m.locks--
		return npages
	}
}

func gosweepone() uintptr {
	var ret uintptr
	systemstack(func() {
		ret = sweepone()
	})
	return ret
}

func gosweepdone() bool {
	return mheap_.sweepdone != 0
}

func gchelper() {
	_g_ := getg()
	_g_.m.traceback = 2
	gchelperstart()

	// parallel mark for over gc roots
	parfordo(work.markfor)

	// help other threads scan secondary blocks
	scanblock(0, 0, nil)

	nproc := work.nproc // work.nproc can change right after we increment work.ndone
	if xadd(&work.ndone, +1) == nproc-1 {
		notewakeup(&work.alldone)
	}
	_g_.m.traceback = 0
}

func cachestats() {
	for i := 0; ; i++ {
		p := allp[i]
		if p == nil {
			break
		}
		c := p.mcache
		if c == nil {
			continue
		}
		purgecachedstats(c)
	}
}

func flushallmcaches() {
	for i := 0; ; i++ {
		p := allp[i]
		if p == nil {
			break
		}
		c := p.mcache
		if c == nil {
			continue
		}
		mCache_ReleaseAll(c)
		stackcache_clear(c)
	}
}

func updatememstats(stats *gcstats) {
	if stats != nil {
		*stats = gcstats{}
	}
	for mp := allm; mp != nil; mp = mp.alllink {
		if stats != nil {
			src := (*[unsafe.Sizeof(gcstats{}) / 8]uint64)(unsafe.Pointer(&mp.gcstats))
			dst := (*[unsafe.Sizeof(gcstats{}) / 8]uint64)(unsafe.Pointer(stats))
			for i, v := range src {
				dst[i] += v
			}
			mp.gcstats = gcstats{}
		}
	}

	memstats.mcache_inuse = uint64(mheap_.cachealloc.inuse)
	memstats.mspan_inuse = uint64(mheap_.spanalloc.inuse)
	memstats.sys = memstats.heap_sys + memstats.stacks_sys + memstats.mspan_sys +
		memstats.mcache_sys + memstats.buckhash_sys + memstats.gc_sys + memstats.other_sys

	// Calculate memory allocator stats.
	// During program execution we only count number of frees and amount of freed memory.
	// Current number of alive object in the heap and amount of alive heap memory
	// are calculated by scanning all spans.
	// Total number of mallocs is calculated as number of frees plus number of alive objects.
	// Similarly, total amount of allocated memory is calculated as amount of freed memory
	// plus amount of alive heap memory.
	memstats.alloc = 0
	memstats.total_alloc = 0
	memstats.nmalloc = 0
	memstats.nfree = 0
	for i := 0; i < len(memstats.by_size); i++ {
		memstats.by_size[i].nmalloc = 0
		memstats.by_size[i].nfree = 0
	}

	// Flush MCache's to MCentral.
	systemstack(flushallmcaches)

	// Aggregate local stats.
	cachestats()

	// Scan all spans and count number of alive objects.
	lock(&mheap_.lock)
	for i := uint32(0); i < mheap_.nspan; i++ {
		s := h_allspans[i]
		if s.state != mSpanInUse {
			continue
		}
		if s.sizeclass == 0 {
			memstats.nmalloc++
			memstats.alloc += uint64(s.elemsize)
		} else {
			memstats.nmalloc += uint64(s.ref)
			memstats.by_size[s.sizeclass].nmalloc += uint64(s.ref)
			memstats.alloc += uint64(s.ref) * uint64(s.elemsize)
		}
	}
	unlock(&mheap_.lock)

	// Aggregate by size class.
	smallfree := uint64(0)
	memstats.nfree = mheap_.nlargefree
	for i := 0; i < len(memstats.by_size); i++ {
		memstats.nfree += mheap_.nsmallfree[i]
		memstats.by_size[i].nfree = mheap_.nsmallfree[i]
		memstats.by_size[i].nmalloc += mheap_.nsmallfree[i]
		smallfree += uint64(mheap_.nsmallfree[i]) * uint64(class_to_size[i])
	}
	memstats.nfree += memstats.tinyallocs
	memstats.nmalloc += memstats.nfree

	// Calculate derived stats.
	memstats.total_alloc = uint64(memstats.alloc) + uint64(mheap_.largefree) + smallfree
	memstats.heap_alloc = memstats.alloc
	memstats.heap_objects = memstats.nmalloc - memstats.nfree
}

func gcinit() {
	if unsafe.Sizeof(workbuf{}) != _WorkbufSize {
		gothrow("runtime: size of Workbuf is suboptimal")
	}

	work.markfor = parforalloc(_MaxGcproc)
	gcpercent = readgogc()
	gcdatamask = unrollglobgcprog((*byte)(unsafe.Pointer(&gcdata)), uintptr(unsafe.Pointer(&edata))-uintptr(unsafe.Pointer(&data)))
	gcbssmask = unrollglobgcprog((*byte)(unsafe.Pointer(&gcbss)), uintptr(unsafe.Pointer(&ebss))-uintptr(unsafe.Pointer(&bss)))
}

func gc_m(start_time int64, eagersweep bool) {
	_g_ := getg()
	gp := _g_.m.curg
	casgstatus(gp, _Grunning, _Gwaiting)
	gp.waitreason = "garbage collection"

	gc(start_time, eagersweep)

	if nbadblock > 0 {
		// Work out path from root to bad block.
		for {
			gc(start_time, eagersweep)
			if nbadblock >= int32(len(badblock)) {
				gothrow("cannot find path to bad pointer")
			}
		}
	}

	casgstatus(gp, _Gwaiting, _Grunning)
}

func gc(start_time int64, eagersweep bool) {
	if _DebugGCPtrs {
		print("GC start\n")
	}

	if debug.allocfreetrace > 0 {
		tracegc()
	}

	_g_ := getg()
	_g_.m.traceback = 2
	t0 := start_time
	work.tstart = start_time

	var t1 int64
	if debug.gctrace > 0 {
		t1 = nanotime()
	}

	// Sweep what is not sweeped by bgsweep.
	for sweepone() != ^uintptr(0) {
		sweep.npausesweep++
	}

	// Cache runtime.mheap_.allspans in work.spans to avoid conflicts with
	// resizing/freeing allspans.
	// New spans can be created while GC progresses, but they are not garbage for
	// this round:
	//  - new stack spans can be created even while the world is stopped.
	//  - new malloc spans can be created during the concurrent sweep

	// Even if this is stop-the-world, a concurrent exitsyscall can allocate a stack from heap.
	lock(&mheap_.lock)
	// Free the old cached sweep array if necessary.
	if work.spans != nil && &work.spans[0] != &h_allspans[0] {
		sysFree(unsafe.Pointer(&work.spans[0]), uintptr(len(work.spans))*unsafe.Sizeof(work.spans[0]), &memstats.other_sys)
	}
	// Cache the current array for marking.
	mheap_.gcspans = mheap_.allspans
	work.spans = h_allspans
	unlock(&mheap_.lock)

	work.nwait = 0
	work.ndone = 0
	work.nproc = uint32(gcprocs())
	parforsetup(work.markfor, work.nproc, uint32(_RootCount+allglen), nil, false, markroot)
	if work.nproc > 1 {
		noteclear(&work.alldone)
		helpgc(int32(work.nproc))
	}

	var t2 int64
	if debug.gctrace > 0 {
		t2 = nanotime()
	}

	gchelperstart()
	parfordo(work.markfor)
	scanblock(0, 0, nil)

	var t3 int64
	if debug.gctrace > 0 {
		t3 = nanotime()
	}

	if work.nproc > 1 {
		notesleep(&work.alldone)
	}

	shrinkfinish()

	cachestats()
	// next_gc calculation is tricky with concurrent sweep since we don't know size of live heap
	// estimate what was live heap size after previous GC (for printing only)
	heap0 := memstats.next_gc * 100 / (uint64(gcpercent) + 100)
	// conservatively set next_gc to high value assuming that everything is live
	// concurrent/lazy sweep will reduce this number while discovering new garbage
	memstats.next_gc = memstats.heap_alloc + memstats.heap_alloc*uint64(gcpercent)/100

	t4 := nanotime()
	atomicstore64(&memstats.last_gc, uint64(unixnanotime())) // must be Unix time to make sense to user
	memstats.pause_ns[memstats.numgc%uint32(len(memstats.pause_ns))] = uint64(t4 - t0)
	memstats.pause_end[memstats.numgc%uint32(len(memstats.pause_end))] = uint64(t4)
	memstats.pause_total_ns += uint64(t4 - t0)
	memstats.numgc++
	if memstats.debuggc {
		print("pause ", t4-t0, "\n")
	}

	if debug.gctrace > 0 {
		heap1 := memstats.heap_alloc
		var stats gcstats
		updatememstats(&stats)
		if heap1 != memstats.heap_alloc {
			print("runtime: mstats skew: heap=", heap1, "/", memstats.heap_alloc, "\n")
			gothrow("mstats skew")
		}
		obj := memstats.nmalloc - memstats.nfree

		stats.nprocyield += work.markfor.nprocyield
		stats.nosyield += work.markfor.nosyield
		stats.nsleep += work.markfor.nsleep

		print("gc", memstats.numgc, "(", work.nproc, "): ",
			(t1-t0)/1000, "+", (t2-t1)/1000, "+", (t3-t2)/1000, "+", (t4-t3)/1000, " us, ",
			heap0>>20, " -> ", heap1>>20, " MB, ",
			obj, " (", memstats.nmalloc, "-", memstats.nfree, ") objects, ",
			gcount(), " goroutines, ",
			len(work.spans), "/", sweep.nbgsweep, "/", sweep.npausesweep, " sweeps, ",
			stats.nhandoff, "(", stats.nhandoffcnt, ") handoff, ",
			work.markfor.nsteal, "(", work.markfor.nstealcnt, ") steal, ",
			stats.nprocyield, "/", stats.nosyield, "/", stats.nsleep, " yields\n")
		sweep.nbgsweep = 0
		sweep.npausesweep = 0
	}

	// See the comment in the beginning of this function as to why we need the following.
	// Even if this is still stop-the-world, a concurrent exitsyscall can allocate a stack from heap.
	lock(&mheap_.lock)
	// Free the old cached mark array if necessary.
	if work.spans != nil && &work.spans[0] != &h_allspans[0] {
		sysFree(unsafe.Pointer(&work.spans[0]), uintptr(len(work.spans))*unsafe.Sizeof(work.spans[0]), &memstats.other_sys)
	}

	// Cache the current array for sweeping.
	mheap_.gcspans = mheap_.allspans
	mheap_.sweepgen += 2
	mheap_.sweepdone = 0
	work.spans = h_allspans
	sweep.spanidx = 0
	unlock(&mheap_.lock)

	if _ConcurrentSweep && !eagersweep {
		lock(&gclock)
		if !sweep.started {
			go bgsweep()
			sweep.started = true
		} else if sweep.parked {
			sweep.parked = false
			ready(sweep.g)
		}
		unlock(&gclock)
	} else {
		// Sweep all spans eagerly.
		for sweepone() != ^uintptr(0) {
			sweep.npausesweep++
		}
		// Do an additional mProf_GC, because all 'free' events are now real as well.
		mProf_GC()
	}

	mProf_GC()
	_g_.m.traceback = 0

	if _DebugGCPtrs {
		print("GC end\n")
	}
}

func readmemstats_m(stats *MemStats) {
	updatememstats(nil)

	// Size of the trailing by_size array differs between Go and C,
	// NumSizeClasses was changed, but we can not change Go struct because of backward compatibility.
	memmove(unsafe.Pointer(stats), unsafe.Pointer(&memstats), sizeof_C_MStats)

	// Stack numbers are part of the heap numbers, separate those out for user consumption
	stats.StackSys = stats.StackInuse
	stats.HeapInuse -= stats.StackInuse
	stats.HeapSys -= stats.StackInuse
}

//go:linkname readGCStats runtime/debug.readGCStats
func readGCStats(pauses *[]uint64) {
	systemstack(func() {
		readGCStats_m(pauses)
	})
}

func readGCStats_m(pauses *[]uint64) {
	p := *pauses
	// Calling code in runtime/debug should make the slice large enough.
	if cap(p) < len(memstats.pause_ns)+3 {
		gothrow("runtime: short slice passed to readGCStats")
	}

	// Pass back: pauses, pause ends, last gc (absolute time), number of gc, total pause ns.
	lock(&mheap_.lock)

	n := memstats.numgc
	if n > uint32(len(memstats.pause_ns)) {
		n = uint32(len(memstats.pause_ns))
	}

	// The pause buffer is circular. The most recent pause is at
	// pause_ns[(numgc-1)%len(pause_ns)], and then backward
	// from there to go back farther in time. We deliver the times
	// most recent first (in p[0]).
	p = p[:cap(p)]
	for i := uint32(0); i < n; i++ {
		j := (memstats.numgc - 1 - i) % uint32(len(memstats.pause_ns))
		p[i] = memstats.pause_ns[j]
		p[n+i] = memstats.pause_end[j]
	}

	p[n+n] = memstats.last_gc
	p[n+n+1] = uint64(memstats.numgc)
	p[n+n+2] = memstats.pause_total_ns
	unlock(&mheap_.lock)
	*pauses = p[:n+n+3]
}

func setGCPercent(in int32) (out int32) {
	lock(&mheap_.lock)
	out = gcpercent
	if in < 0 {
		in = -1
	}
	gcpercent = in
	unlock(&mheap_.lock)
	return out
}

func gchelperstart() {
	_g_ := getg()

	if _g_.m.helpgc < 0 || _g_.m.helpgc >= _MaxGcproc {
		gothrow("gchelperstart: bad m->helpgc")
	}
	if _g_ != _g_.m.g0 {
		gothrow("gchelper not running on g0 stack")
	}
}

func wakefing() *g {
	var res *g
	lock(&finlock)
	if fingwait && fingwake {
		fingwait = false
		fingwake = false
		res = fing
	}
	unlock(&finlock)
	return res
}

func addb(p *byte, n uintptr) *byte {
	return (*byte)(add(unsafe.Pointer(p), n))
}

// Recursively unrolls GC program in prog.
// mask is where to store the result.
// ppos is a pointer to position in mask, in bits.
// sparse says to generate 4-bits per word mask for heap (2-bits for data/bss otherwise).
func unrollgcprog1(maskp *byte, prog *byte, ppos *uintptr, inplace, sparse bool) *byte {
	arena_start := mheap_.arena_start
	pos := *ppos
	mask := (*[1 << 30]byte)(unsafe.Pointer(maskp))
	for {
		switch *prog {
		default:
			gothrow("unrollgcprog: unknown instruction")

		case insData:
			prog = addb(prog, 1)
			siz := int(*prog)
			prog = addb(prog, 1)
			p := (*[1 << 30]byte)(unsafe.Pointer(prog))
			for i := 0; i < siz; i++ {
				v := p[i/_PointersPerByte]
				v >>= (uint(i) % _PointersPerByte) * _BitsPerPointer
				v &= _BitsMask
				if inplace {
					// Store directly into GC bitmap.
					off := (uintptr(unsafe.Pointer(&mask[pos])) - arena_start) / ptrSize
					bitp := (*byte)(unsafe.Pointer(arena_start - off/wordsPerBitmapByte - 1))
					shift := (off % wordsPerBitmapByte) * gcBits
					if shift == 0 {
						*bitp = 0
					}
					*bitp |= v << (shift + 2)
					pos += ptrSize
				} else if sparse {
					// 4-bits per word
					v <<= (pos % 8) + 2
					mask[pos/8] |= v
					pos += gcBits
				} else {
					// 2-bits per word
					v <<= pos % 8
					mask[pos/8] |= v
					pos += _BitsPerPointer
				}
			}
			prog = addb(prog, round(uintptr(siz)*_BitsPerPointer, 8)/8)

		case insArray:
			prog = (*byte)(add(unsafe.Pointer(prog), 1))
			siz := uintptr(0)
			for i := uintptr(0); i < ptrSize; i++ {
				siz = (siz << 8) + uintptr(*(*byte)(add(unsafe.Pointer(prog), ptrSize-i-1)))
			}
			prog = (*byte)(add(unsafe.Pointer(prog), ptrSize))
			var prog1 *byte
			for i := uintptr(0); i < siz; i++ {
				prog1 = unrollgcprog1(&mask[0], prog, &pos, inplace, sparse)
			}
			if *prog1 != insArrayEnd {
				gothrow("unrollgcprog: array does not end with insArrayEnd")
			}
			prog = (*byte)(add(unsafe.Pointer(prog1), 1))

		case insArrayEnd, insEnd:
			*ppos = pos
			return prog
		}
	}
}

// Unrolls GC program prog for data/bss, returns dense GC mask.
func unrollglobgcprog(prog *byte, size uintptr) bitvector {
	masksize := round(round(size, ptrSize)/ptrSize*bitsPerPointer, 8) / 8
	mask := (*[1 << 30]byte)(persistentalloc(masksize+1, 0, &memstats.gc_sys))
	mask[masksize] = 0xa1
	pos := uintptr(0)
	prog = unrollgcprog1(&mask[0], prog, &pos, false, false)
	if pos != size/ptrSize*bitsPerPointer {
		print("unrollglobgcprog: bad program size, got ", pos, ", expect ", size/ptrSize*bitsPerPointer, "\n")
		gothrow("unrollglobgcprog: bad program size")
	}
	if *prog != insEnd {
		gothrow("unrollglobgcprog: program does not end with insEnd")
	}
	if mask[masksize] != 0xa1 {
		gothrow("unrollglobgcprog: overflow")
	}
	return bitvector{int32(masksize * 8), &mask[0]}
}

func unrollgcproginplace_m(v unsafe.Pointer, typ *_type, size, size0 uintptr) {
	pos := uintptr(0)
	prog := (*byte)(unsafe.Pointer(uintptr(typ.gc[1])))
	for pos != size0 {
		unrollgcprog1((*byte)(v), prog, &pos, true, true)
	}

	// Mark first word as bitAllocated.
	arena_start := mheap_.arena_start
	off := (uintptr(v) - arena_start) / ptrSize
	bitp := (*byte)(unsafe.Pointer(arena_start - off/wordsPerBitmapByte - 1))
	shift := (off % wordsPerBitmapByte) * gcBits
	*bitp |= bitBoundary << shift

	// Mark word after last as BitsDead.
	if size0 < size {
		off := (uintptr(v) + size0 - arena_start) / ptrSize
		bitp := (*byte)(unsafe.Pointer(arena_start - off/wordsPerBitmapByte - 1))
		shift := (off % wordsPerBitmapByte) * gcBits
		*bitp &= uint8(^(bitPtrMask << shift) | uintptr(bitsDead)<<(shift+2))
	}
}

var unroll mutex

// Unrolls GC program in typ.gc[1] into typ.gc[0]
func unrollgcprog_m(typ *_type) {
	lock(&unroll)
	mask := (*byte)(unsafe.Pointer(uintptr(typ.gc[0])))
	if *mask == 0 {
		pos := uintptr(8) // skip the unroll flag
		prog := (*byte)(unsafe.Pointer(uintptr(typ.gc[1])))
		prog = unrollgcprog1(mask, prog, &pos, false, true)
		if *prog != insEnd {
			gothrow("unrollgcprog: program does not end with insEnd")
		}
		if typ.size/ptrSize%2 != 0 {
			// repeat the program
			prog := (*byte)(unsafe.Pointer(uintptr(typ.gc[1])))
			unrollgcprog1(mask, prog, &pos, false, true)
		}
		// atomic way to say mask[0] = 1
		x := *(*uintptr)(unsafe.Pointer(mask))
		*(*byte)(unsafe.Pointer(&x)) = 1
		atomicstoreuintptr((*uintptr)(unsafe.Pointer(mask)), x)
	}
	unlock(&unroll)
}

// mark the span of memory at v as having n blocks of the given size.
// if leftover is true, there is left over space at the end of the span.
func markspan(v unsafe.Pointer, size uintptr, n uintptr, leftover bool) {
	if uintptr(v)+size*n > mheap_.arena_used || uintptr(v) < mheap_.arena_start {
		gothrow("markspan: bad pointer")
	}

	// Find bits of the beginning of the span.
	off := (uintptr(v) - uintptr(mheap_.arena_start)) / ptrSize
	if off%wordsPerBitmapByte != 0 {
		gothrow("markspan: unaligned length")
	}
	b := mheap_.arena_start - off/wordsPerBitmapByte - 1

	// Okay to use non-atomic ops here, because we control
	// the entire span, and each bitmap byte has bits for only
	// one span, so no other goroutines are changing these bitmap words.

	if size == ptrSize {
		// Possible only on 64-bits (minimal size class is 8 bytes).
		// Set memory to 0x11.
		if (bitBoundary|bitsDead)<<gcBits|bitBoundary|bitsDead != 0x11 {
			gothrow("markspan: bad bits")
		}
		if n%(wordsPerBitmapByte*ptrSize) != 0 {
			gothrow("markspan: unaligned length")
		}
		b = b - n/wordsPerBitmapByte + 1 // find first byte
		if b%ptrSize != 0 {
			gothrow("markspan: unaligned pointer")
		}
		for i := uintptr(0); i < n; i, b = i+wordsPerBitmapByte*ptrSize, b+ptrSize {
			*(*uintptr)(unsafe.Pointer(b)) = uintptrMask & 0x1111111111111111 // bitBoundary | bitsDead, repeated
		}
		return
	}

	if leftover {
		n++ // mark a boundary just past end of last block too
	}
	step := size / (ptrSize * wordsPerBitmapByte)
	for i := uintptr(0); i < n; i, b = i+1, b-step {
		*(*byte)(unsafe.Pointer(b)) = bitBoundary | bitsDead<<2
	}
}

// unmark the span of memory at v of length n bytes.
func unmarkspan(v, n uintptr) {
	if v+n > mheap_.arena_used || v < mheap_.arena_start {
		gothrow("markspan: bad pointer")
	}

	off := (v - mheap_.arena_start) / ptrSize // word offset
	if off%(ptrSize*wordsPerBitmapByte) != 0 {
		gothrow("markspan: unaligned pointer")
	}

	b := mheap_.arena_start - off/wordsPerBitmapByte - 1
	n /= ptrSize
	if n%(ptrSize*wordsPerBitmapByte) != 0 {
		gothrow("unmarkspan: unaligned length")
	}

	// Okay to use non-atomic ops here, because we control
	// the entire span, and each bitmap word has bits for only
	// one span, so no other goroutines are changing these
	// bitmap words.
	n /= wordsPerBitmapByte
	memclr(unsafe.Pointer(b-n+1), n)
}

func mHeap_MapBits(h *mheap) {
	// Caller has added extra mappings to the arena.
	// Add extra mappings of bitmap words as needed.
	// We allocate extra bitmap pieces in chunks of bitmapChunk.
	const bitmapChunk = 8192

	n := (h.arena_used - h.arena_start) / (ptrSize * wordsPerBitmapByte)
	n = round(n, bitmapChunk)
	n = round(n, _PhysPageSize)
	if h.bitmap_mapped >= n {
		return
	}

	sysMap(unsafe.Pointer(h.arena_start-n), n-h.bitmap_mapped, h.arena_reserved, &memstats.gc_sys)
	h.bitmap_mapped = n
}

func getgcmaskcb(frame *stkframe, ctxt unsafe.Pointer) bool {
	target := (*stkframe)(ctxt)
	if frame.sp <= target.sp && target.sp < frame.varp {
		*target = *frame
		return false
	}
	return true
}

// Returns GC type info for object p for testing.
func getgcmask(p unsafe.Pointer, t *_type, mask **byte, len *uintptr) {
	*mask = nil
	*len = 0

	// data
	if uintptr(unsafe.Pointer(&data)) <= uintptr(p) && uintptr(p) < uintptr(unsafe.Pointer(&edata)) {
		n := (*ptrtype)(unsafe.Pointer(t)).elem.size
		*len = n / ptrSize
		*mask = &make([]byte, *len)[0]
		for i := uintptr(0); i < n; i += ptrSize {
			off := (uintptr(p) + i - uintptr(unsafe.Pointer(&data))) / ptrSize
			bits := (*(*byte)(add(unsafe.Pointer(gcdatamask.bytedata), off/pointersPerByte)) >> ((off % pointersPerByte) * bitsPerPointer)) & bitsMask
			*(*byte)(add(unsafe.Pointer(*mask), i/ptrSize)) = bits
		}
		return
	}

	// bss
	if uintptr(unsafe.Pointer(&bss)) <= uintptr(p) && uintptr(p) < uintptr(unsafe.Pointer(&ebss)) {
		n := (*ptrtype)(unsafe.Pointer(t)).elem.size
		*len = n / ptrSize
		*mask = &make([]byte, *len)[0]
		for i := uintptr(0); i < n; i += ptrSize {
			off := (uintptr(p) + i - uintptr(unsafe.Pointer(&bss))) / ptrSize
			bits := (*(*byte)(add(unsafe.Pointer(gcbssmask.bytedata), off/pointersPerByte)) >> ((off % pointersPerByte) * bitsPerPointer)) & bitsMask
			*(*byte)(add(unsafe.Pointer(*mask), i/ptrSize)) = bits
		}
		return
	}

	// heap
	var n uintptr
	var base uintptr
	if mlookup(uintptr(p), &base, &n, nil) != 0 {
		*len = n / ptrSize
		*mask = &make([]byte, *len)[0]
		for i := uintptr(0); i < n; i += ptrSize {
			off := (uintptr(base) + i - mheap_.arena_start) / ptrSize
			b := mheap_.arena_start - off/wordsPerBitmapByte - 1
			shift := (off % wordsPerBitmapByte) * gcBits
			bits := (*(*byte)(unsafe.Pointer(b)) >> (shift + 2)) & bitsMask
			*(*byte)(add(unsafe.Pointer(*mask), i/ptrSize)) = bits
		}
		return
	}

	// stack
	var frame stkframe
	frame.sp = uintptr(p)
	_g_ := getg()
	gentraceback(_g_.m.curg.sched.pc, _g_.m.curg.sched.sp, 0, _g_.m.curg, 0, nil, 1000, getgcmaskcb, noescape(unsafe.Pointer(&frame)), 0)
	if frame.fn != nil {
		f := frame.fn
		targetpc := frame.continpc
		if targetpc == 0 {
			return
		}
		if targetpc != f.entry {
			targetpc--
		}
		pcdata := pcdatavalue(f, _PCDATA_StackMapIndex, targetpc)
		if pcdata == -1 {
			return
		}
		stkmap := (*stackmap)(funcdata(f, _FUNCDATA_LocalsPointerMaps))
		if stkmap == nil || stkmap.n <= 0 {
			return
		}
		bv := stackmapdata(stkmap, pcdata)
		size := uintptr(bv.n) / bitsPerPointer * ptrSize
		n := (*ptrtype)(unsafe.Pointer(t)).elem.size
		*len = n / ptrSize
		*mask = &make([]byte, *len)[0]
		for i := uintptr(0); i < n; i += ptrSize {
			off := (uintptr(p) + i - frame.varp + size) / ptrSize
			bits := ((*(*byte)(add(unsafe.Pointer(bv.bytedata), off*bitsPerPointer/8))) >> ((off * bitsPerPointer) % 8)) & bitsMask
			*(*byte)(add(unsafe.Pointer(*mask), i/ptrSize)) = bits
		}
	}
}

func unixnanotime() int64 {
	var now int64
	gc_unixnanotime(&now)
	return now
}
