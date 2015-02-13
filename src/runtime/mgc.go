// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): The code having to do with the heap bitmap needs very serious cleanup.
// It has gotten completely out of control.

// Garbage collector (GC).
//
// The GC runs concurrently with mutator threads, is type accurate (aka precise), allows multiple
// GC thread to run in parallel. It is a concurrent mark and sweep that uses a write barrier. It is
// non-generational and non-compacting. Allocation is done using size segregated per P allocation
// areas to minimize fragmentation while eliminating locks in the common case.
//
// The algorithm decomposes into several steps.
// This is a high level description of the algorithm being used. For an overview of GC a good
// place to start is Richard Jones' gchandbook.org.
//
// The algorithm's intellectual heritage includes Dijkstra's on-the-fly algorithm, see
// Edsger W. Dijkstra, Leslie Lamport, A. J. Martin, C. S. Scholten, and E. F. M. Steffens. 1978.
// On-the-fly garbage collection: an exercise in cooperation. Commun. ACM 21, 11 (November 1978),
// 966-975.
// For journal quality proofs that these steps are complete, correct, and terminate see
// Hudson, R., and Moss, J.E.B. Copying Garbage Collection without stopping the world.
// Concurrency and Computation: Practice and Experience 15(3-5), 2003.
//
//  0. Set phase = GCscan from GCoff.
//  1. Wait for all P's to acknowledge phase change.
//         At this point all goroutines have passed through a GC safepoint and
//         know we are in the GCscan phase.
//  2. GC scans all goroutine stacks, mark and enqueues all encountered pointers
//       (marking avoids most duplicate enqueuing but races may produce benign duplication).
//       Preempted goroutines are scanned before P schedules next goroutine.
//  3. Set phase = GCmark.
//  4. Wait for all P's to acknowledge phase change.
//  5. Now write barrier marks and enqueues black, grey, or white to white pointers.
//       Malloc still allocates white (non-marked) objects.
//  6. Meanwhile GC transitively walks the heap marking reachable objects.
//  7. When GC finishes marking heap, it preempts P's one-by-one and
//       retakes partial wbufs (filled by write barrier or during a stack scan of the goroutine
//       currently scheduled on the P).
//  8. Once the GC has exhausted all available marking work it sets phase = marktermination.
//  9. Wait for all P's to acknowledge phase change.
// 10. Malloc now allocates black objects, so number of unmarked reachable objects
//        monotonically decreases.
// 11. GC preempts P's one-by-one taking partial wbufs and marks all unmarked yet
//        reachable objects.
// 12. When GC completes a full cycle over P's and discovers no new grey
//         objects, (which means all reachable objects are marked) set phase = GCsweep.
// 13. Wait for all P's to acknowledge phase change.
// 14. Now malloc allocates white (but sweeps spans before use).
//         Write barrier becomes nop.
// 15. GC does background sweeping, see description below.
// 16. When sweeping is complete set phase to GCoff.
// 17. When sufficient allocation has taken place replay the sequence starting at 0 above,
//         see discussion of GC rate below.

// Changing phases.
// Phases are changed by setting the gcphase to the next phase and possibly calling ackgcphase.
// All phase action must be benign in the presence of a change.
// Starting with GCoff
// GCoff to GCscan
//     GSscan scans stacks and globals greying them and never marks an object black.
//     Once all the P's are aware of the new phase they will scan gs on preemption.
//     This means that the scanning of preempted gs can't start until all the Ps
//     have acknowledged.
// GCscan to GCmark
//     GCMark turns on the write barrier which also only greys objects. No scanning
//     of objects (making them black) can happen until all the Ps have acknowledged
//     the phase change.
// GCmark to GCmarktermination
//     The only change here is that we start allocating black so the Ps must acknowledge
//     the change before we begin the termination algorithm
// GCmarktermination to GSsweep
//     Object currently on the freelist must be marked black for this to work.
//     Are things on the free lists black or white? How does the sweep phase work?

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
// but there can still be other one-page unswept spans which could be combined into a
// two-page span.
// It's critical to ensure that no operations proceed on unswept spans (that would corrupt
// mark bits in GC bitmap). During GC all mcaches are flushed into the central cache,
// so they are empty. When a goroutine grabs a new span into mcache, it sweeps it.
// When a goroutine explicitly frees an object or sets a finalizer, it ensures that
// the span is swept (either by sweeping it, or by waiting for the concurrent sweep to finish).
// The finalizer goroutine is kicked off only when all spans are swept.
// When the next GC starts, it sweeps all not-yet-swept spans (if any).

// GC rate.
// Next GC is after we've allocated an extra amount of memory proportional to
// the amount already in use. The proportion is controlled by GOGC environment variable
// (100 by default). If GOGC=100 and we're using 4M, we'll GC again when we get to 8M
// (this mark is tracked in next_gc variable). This keeps the GC cost in linear
// proportion to the allocation cost. Adjusting GOGC just changes the linear constant
// (and also the amount of extra memory used).

package runtime

import "unsafe"

const (
	_DebugGC         = 0
	_DebugGCPtrs     = false // if true, print trace of every pointer load during GC
	_ConcurrentSweep = true
	_FinBlockSize    = 4 * 1024
	_RootData        = 0
	_RootBss         = 1
	_RootFinalizers  = 2
	_RootSpans       = 3
	_RootFlushCaches = 4
	_RootCount       = 5
)

// ptrmask for an allocation containing a single pointer.
var oneptr = [...]uint8{typePointer}

// Initialized from $GOGC.  GOGC=off means no GC.
var gcpercent int32

// Holding worldsema grants an M the right to try to stop the world.
// The procedure is:
//
//	semacquire(&worldsema);
//	m.preemptoff = "reason";
//	stoptheworld();
//
//	... do stuff ...
//
//	m.preemptoff = "";
//	semrelease(&worldsema);
//	starttheworld();
//
var worldsema uint32 = 1

var data, edata, bss, ebss, gcdata, gcbss struct{}

var gcdatamask bitvector
var gcbssmask bitvector

var gclock mutex

var badblock [1024]uintptr
var nbadblock int32

type workdata struct {
	full    uint64                // lock-free list of full blocks workbuf
	empty   uint64                // lock-free list of empty blocks workbuf
	partial uint64                // lock-free list of partially filled blocks workbuf
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
//go:nowritebarrier
func have_cgo_allocate() bool {
	return &weak_cgo_allocate != nil
}

// To help debug the concurrent GC we remark with the world
// stopped ensuring that any object encountered has their normal
// mark bit set. To do this we use an orthogonal bit
// pattern to indicate the object is marked. The following pattern
// uses the upper two bits in the object's bounday nibble.
// 01: scalar  not marked
// 10: pointer not marked
// 11: pointer     marked
// 00: scalar      marked
// Xoring with 01 will flip the pattern from marked to unmarked and vica versa.
// The higher bit is 1 for pointers and 0 for scalars, whether the object
// is marked or not.
// The first nibble no longer holds the typeDead pattern indicating that the
// there are no more pointers in the object. This information is held
// in the second nibble.

// When marking an object if the bool checkmarkphase is true one uses the above
// encoding, otherwise one uses the bitMarked bit in the lower two bits
// of the nibble.
var checkmarkphase = false

// inheap reports whether b is a pointer into a (potentially dead) heap object.
// It returns false for pointers into stack spans.
//go:nowritebarrier
func inheap(b uintptr) bool {
	if b == 0 || b < mheap_.arena_start || b >= mheap_.arena_used {
		return false
	}
	// Not a beginning of a block, consult span table to find the block beginning.
	k := b >> _PageShift
	x := k
	x -= mheap_.arena_start >> _PageShift
	s := h_spans[x]
	if s == nil || pageID(k) < s.start || b >= s.limit || s.state != mSpanInUse {
		return false
	}
	return true
}

// Slow for now as we serialize this, since this is on a debug path
// speed is not critical at this point.
var andlock mutex

//go:nowritebarrier
func atomicand8(src *byte, val byte) {
	lock(&andlock)
	*src &= val
	unlock(&andlock)
}

// When in GCmarkterminate phase we allocate black.
//go:nowritebarrier
func gcmarknewobject_m(obj uintptr) {
	if gcphase != _GCmarktermination {
		throw("marking new object while not in mark termination phase")
	}
	if checkmarkphase { // The world should be stopped so this should not happen.
		throw("gcmarknewobject called while doing checkmark")
	}

	heapBitsForAddr(obj).setMarked()
}

// obj is the start of an object with mark mbits.
// If it isn't already marked, mark it and enqueue into workbuf.
// Return possibly new workbuf to use.
// base and off are for debugging only and could be removed.
//go:nowritebarrier
func greyobject(obj, base, off uintptr, hbits heapBits, wbuf *workbuf) *workbuf {
	// obj should be start of allocation, and so must be at least pointer-aligned.
	if obj&(ptrSize-1) != 0 {
		throw("greyobject: obj not pointer-aligned")
	}

	if checkmarkphase {
		if !hbits.isMarked() {
			print("runtime:greyobject: checkmarks finds unexpected unmarked object obj=", hex(obj), "\n")
			print("runtime: found obj at *(", hex(base), "+", hex(off), ")\n")

			// Dump the source (base) object

			kb := base >> _PageShift
			xb := kb
			xb -= mheap_.arena_start >> _PageShift
			sb := h_spans[xb]
			printlock()
			print("runtime:greyobject Span: base=", hex(base), " kb=", hex(kb))
			if sb == nil {
				print(" sb=nil\n")
			} else {
				print(" sb.start*_PageSize=", hex(sb.start*_PageSize), " sb.limit=", hex(sb.limit), " sb.sizeclass=", sb.sizeclass, " sb.elemsize=", sb.elemsize, "\n")
				// base is (a pointer to) the source object holding the reference to object. Create a pointer to each of the fields
				// fields in base and print them out as hex values.
				for i := 0; i < int(sb.elemsize/ptrSize); i++ {
					print(" *(base+", i*ptrSize, ") = ", hex(*(*uintptr)(unsafe.Pointer(base + uintptr(i)*ptrSize))), "\n")
				}
			}

			// Dump the object

			k := obj >> _PageShift
			x := k
			x -= mheap_.arena_start >> _PageShift
			s := h_spans[x]
			print("runtime:greyobject Span: obj=", hex(obj), " k=", hex(k))
			if s == nil {
				print(" s=nil\n")
			} else {
				print(" s.start=", hex(s.start*_PageSize), " s.limit=", hex(s.limit), " s.sizeclass=", s.sizeclass, " s.elemsize=", s.elemsize, "\n")
				// NOTE(rsc): This code is using s.sizeclass as an approximation of the
				// number of pointer-sized words in an object. Perhaps not what was intended.
				for i := 0; i < int(s.sizeclass); i++ {
					print(" *(obj+", i*ptrSize, ") = ", hex(*(*uintptr)(unsafe.Pointer(obj + uintptr(i)*ptrSize))), "\n")
				}
			}
			throw("checkmark found unmarked object")
		}
		if !hbits.isCheckmarked() {
			return wbuf
		}
		hbits.setCheckmarked()
		if !hbits.isCheckmarked() {
			throw("setCheckmarked and isCheckmarked disagree")
		}
	} else {
		// If marked we have nothing to do.
		if hbits.isMarked() {
			return wbuf
		}

		// Each byte of GC bitmap holds info for two words.
		// Might be racing with other updates, so use atomic update always.
		// We used to be clever here and use a non-atomic update in certain
		// cases, but it's not worth the risk.
		hbits.setMarked()
	}

	if !checkmarkphase && hbits.typeBits() == typeDead {
		return wbuf // noscan object
	}

	// Queue the obj for scanning. The PREFETCH(obj) logic has been removed but
	// seems like a nice optimization that can be added back in.
	// There needs to be time between the PREFETCH and the use.
	// Previously we put the obj in an 8 element buffer that is drained at a rate
	// to give the PREFETCH time to do its work.
	// Use of PREFETCHNTA might be more appropriate than PREFETCH

	// If workbuf is full, obtain an empty one.
	if wbuf.nobj >= uintptr(len(wbuf.obj)) {
		putfull(wbuf, 358)
		wbuf = getempty(359)
	}

	wbuf.obj[wbuf.nobj] = obj
	wbuf.nobj++
	return wbuf
}

// Scan the object b of size n, adding pointers to wbuf.
// Return possibly new wbuf to use.
// If ptrmask != nil, it specifies where pointers are in b.
// If ptrmask == nil, the GC bitmap should be consulted.
// In this case, n may be an overestimate of the size; the GC bitmap
// must also be used to make sure the scan stops at the end of b.
//go:nowritebarrier
func scanobject(b, n uintptr, ptrmask *uint8, wbuf *workbuf) *workbuf {
	arena_start := mheap_.arena_start
	arena_used := mheap_.arena_used

	// Find bits of the beginning of the object.
	var hbits heapBits
	if ptrmask == nil {
		b, hbits = heapBitsForObject(b)
		if b == 0 {
			return wbuf
		}
		if n == 0 {
			n = mheap_.arena_used - b
		}
	}
	for i := uintptr(0); i < n; i += ptrSize {
		// Find bits for this word.
		var bits uintptr
		if ptrmask != nil {
			// dense mask (stack or data)
			bits = (uintptr(*(*byte)(add(unsafe.Pointer(ptrmask), (i/ptrSize)/4))) >> (((i / ptrSize) % 4) * typeBitsWidth)) & typeMask
		} else {
			// Check if we have reached end of span.
			// n is an overestimate of the size of the object.
			if (b+i)%_PageSize == 0 && h_spans[(b-arena_start)>>_PageShift] != h_spans[(b+i-arena_start)>>_PageShift] {
				break
			}

			bits = uintptr(hbits.typeBits())
			if i > 0 && (hbits.isBoundary() || bits == typeDead) {
				break // reached beginning of the next object
			}
			hbits = hbits.next()
		}

		if bits <= typeScalar { // typeScalar, typeDead, typeScalarMarked
			continue
		}

		if bits&typePointer != typePointer {
			print("gc checkmarkphase=", checkmarkphase, " b=", hex(b), " ptrmask=", ptrmask, "\n")
			throw("unexpected garbage collection bits")
		}

		obj := *(*uintptr)(unsafe.Pointer(b + i))

		// At this point we have extracted the next potential pointer.
		// Check if it points into heap.
		if obj == 0 || obj < arena_start || obj >= arena_used {
			continue
		}

		if mheap_.shadow_enabled && debug.wbshadow >= 2 && debug.gccheckmark > 0 && checkmarkphase {
			checkwbshadow((*uintptr)(unsafe.Pointer(b + i)))
		}

		// Mark the object.
		if obj, hbits := heapBitsForObject(obj); obj != 0 {
			wbuf = greyobject(obj, b, i, hbits, wbuf)
		}
	}
	return wbuf
}

// scanblock scans b as scanobject would.
// If the gcphase is GCscan, scanblock performs additional checks.
//go:nowritebarrier
func scanblock(b0, n0 uintptr, ptrmask *uint8, wbuf *workbuf) *workbuf {
	// Use local copies of original parameters, so that a stack trace
	// due to one of the throws below shows the original block
	// base and extent.
	b := b0
	n := n0

	// ptrmask can have 2 possible values:
	// 1. nil - obtain pointer mask from GC bitmap.
	// 2. pointer to a compact mask (for stacks and data).

	if wbuf == nil {
		wbuf = getpartialorempty(460) // no wbuf passed in.
	}
	wbuf = scanobject(b, n, ptrmask, wbuf)
	if gcphase == _GCscan {
		if inheap(b) && ptrmask == nil {
			// b is in heap, we are in GCscan so there should be a ptrmask.
			throw("scanblock: In GCscan phase and inheap is true.")
		}
	}
	return wbuf
}

// gcDrain scans objects in work buffers (starting with wbuf), blackening grey
// objects until all work buffers have been drained.
//go:nowritebarrier
func gcDrain(wbuf *workbuf) {
	if wbuf == nil {
		wbuf = getpartialorempty(472)
	}
	checknocurrentwbuf()
	if gcphase != _GCmark && gcphase != _GCmarktermination {
		throw("scanblock phase incorrect")
	}

	for {
		if wbuf.nobj == 0 {
			putempty(wbuf, 496)
			// Refill workbuf from global queue.
			wbuf = getfull(504)
			if wbuf == nil { // nil means out of work barrier reached
				break
			}
			wbuf.checknonempty()
		}

		// If another proc wants a pointer, give it some.
		if work.nwait > 0 && wbuf.nobj > 4 && work.full == 0 {
			wbuf = handoff(wbuf)
		}

		// This might be a good place to add prefetch code...
		// if(wbuf.nobj > 4) {
		//         PREFETCH(wbuf->obj[wbuf.nobj - 3];
		//  }
		wbuf.nobj--
		b := wbuf.obj[wbuf.nobj]
		// If the current wbuf is filled by the scan a new wbuf might be
		// returned that could possibly hold only a single object. This
		// could result in each iteration draining only a single object
		// out of the wbuf passed in + a single object placed
		// into an empty wbuf in scanobject so there could be
		// a performance hit as we keep fetching fresh wbufs.
		wbuf = scanobject(b, 0, nil, wbuf)
	}
	checknocurrentwbuf()
}

// gcDrainN scans n objects starting with those in wbuf, blackening
// grey objects.
//go:nowritebarrier
func gcDrainN(wbuf *workbuf, n uintptr) *workbuf {
	checknocurrentwbuf()
	for i := uintptr(0); i < n; i++ {
		if wbuf.nobj == 0 {
			putempty(wbuf, 544)
			wbuf = trygetfull(545)
			if wbuf == nil {
				return nil
			}
		}

		// This might be a good place to add prefetch code...
		// if(wbuf.nobj > 4) {
		//         PREFETCH(wbuf->obj[wbuf.nobj - 3];
		//  }
		wbuf.nobj--
		b := wbuf.obj[wbuf.nobj]
		wbuf = scanobject(b, 0, nil, wbuf)
	}
	return wbuf
}

//go:nowritebarrier
func markroot(desc *parfor, i uint32) {
	// Note: if you add a case here, please also update heapdump.c:dumproots.
	wbuf := (*workbuf)(unsafe.Pointer(xchguintptr(&getg().m.currentwbuf, 0)))
	switch i {
	case _RootData:
		wbuf = scanblock(uintptr(unsafe.Pointer(&data)), uintptr(unsafe.Pointer(&edata))-uintptr(unsafe.Pointer(&data)), gcdatamask.bytedata, wbuf)

	case _RootBss:
		wbuf = scanblock(uintptr(unsafe.Pointer(&bss)), uintptr(unsafe.Pointer(&ebss))-uintptr(unsafe.Pointer(&bss)), gcbssmask.bytedata, wbuf)

	case _RootFinalizers:
		for fb := allfin; fb != nil; fb = fb.alllink {
			wbuf = scanblock(uintptr(unsafe.Pointer(&fb.fin[0])), uintptr(fb.cnt)*unsafe.Sizeof(fb.fin[0]), &finptrmask[0], wbuf)
		}

	case _RootSpans:
		// mark MSpan.specials
		sg := mheap_.sweepgen
		for spanidx := uint32(0); spanidx < uint32(len(work.spans)); spanidx++ {
			s := work.spans[spanidx]
			if s.state != mSpanInUse {
				continue
			}
			if !checkmarkphase && s.sweepgen != sg {
				// sweepgen was updated (+2) during non-checkmark GC pass
				print("sweep ", s.sweepgen, " ", sg, "\n")
				throw("gc: unswept span")
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
				if gcphase != _GCscan {
					wbuf = scanblock(p, s.elemsize, nil, wbuf) // scanned during mark phase
				}
				wbuf = scanblock(uintptr(unsafe.Pointer(&spf.fn)), ptrSize, &oneptr[0], wbuf)
			}
		}

	case _RootFlushCaches:
		if gcphase != _GCscan { // Do not flush mcaches during GCscan phase.
			flushallmcaches()
		}

	default:
		// the rest is scanning goroutine stacks
		if uintptr(i-_RootCount) >= allglen {
			throw("markroot: bad index")
		}
		gp := allgs[i-_RootCount]

		// remember when we've first observed the G blocked
		// needed only to output in traceback
		status := readgstatus(gp) // We are not in a scan state
		if (status == _Gwaiting || status == _Gsyscall) && gp.waitsince == 0 {
			gp.waitsince = work.tstart
		}

		// Shrink a stack if not much of it is being used but not in the scan phase.
		if gcphase == _GCmarktermination {
			// Shrink during STW GCmarktermination phase thus avoiding
			// complications introduced by shrinking during
			// non-STW phases.
			shrinkstack(gp)
		}
		if readgstatus(gp) == _Gdead {
			gp.gcworkdone = true
		} else {
			gp.gcworkdone = false
		}
		restart := stopg(gp)

		// goroutine will scan its own stack when it stops running.
		// Wait until it has.
		for readgstatus(gp) == _Grunning && !gp.gcworkdone {
		}

		// scanstack(gp) is done as part of gcphasework
		// But to make sure we finished we need to make sure that
		// the stack traps have all responded so drop into
		// this while loop until they respond.
		for !gp.gcworkdone {
			status = readgstatus(gp)
			if status == _Gdead {
				gp.gcworkdone = true // scan is a noop
				break
			}
			if status == _Gwaiting || status == _Grunnable {
				restart = stopg(gp)
			}
		}
		if restart {
			restartg(gp)
		}
	}
	if wbuf == nil {
		return
	} else {
		putpartial(wbuf, 670)
	}
}

//go:nowritebarrier
func stackmapdata(stkmap *stackmap, n int32) bitvector {
	if n < 0 || n >= stkmap.n {
		throw("stackmapdata: index out of range")
	}
	return bitvector{stkmap.nbit, (*byte)(add(unsafe.Pointer(&stkmap.bytedata), uintptr(n*((stkmap.nbit+31)/32*4))))}
}

// Scan a stack frame: local variables and function arguments/results.
//go:nowritebarrier
func scanframeworker(frame *stkframe, unused unsafe.Pointer, wbuf *workbuf) *workbuf {

	f := frame.fn
	targetpc := frame.continpc
	if targetpc == 0 {
		// Frame is dead.
		return wbuf
	}
	if _DebugGC > 1 {
		print("scanframe ", funcname(f), "\n")
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
			print("runtime: frame ", funcname(f), " untyped locals ", hex(frame.varp-size), "+", hex(size), "\n")
			throw("missing stackmap")
		}

		// Locals bitmap information, scan just the pointers in locals.
		if pcdata < 0 || pcdata >= stkmap.n {
			// don't know where we are
			print("runtime: pcdata is ", pcdata, " and ", stkmap.n, " locals stack map entries for ", funcname(f), " (targetpc=", targetpc, ")\n")
			throw("scanframe: bad symbol table")
		}
		bv := stackmapdata(stkmap, pcdata)
		size = (uintptr(bv.n) / typeBitsWidth) * ptrSize
		wbuf = scanblock(frame.varp-size, size, bv.bytedata, wbuf)
	}

	// Scan arguments.
	if frame.arglen > 0 {
		var bv bitvector
		if frame.argmap != nil {
			bv = *frame.argmap
		} else {
			stkmap := (*stackmap)(funcdata(f, _FUNCDATA_ArgsPointerMaps))
			if stkmap == nil || stkmap.n <= 0 {
				print("runtime: frame ", funcname(f), " untyped args ", hex(frame.argp), "+", hex(frame.arglen), "\n")
				throw("missing stackmap")
			}
			if pcdata < 0 || pcdata >= stkmap.n {
				// don't know where we are
				print("runtime: pcdata is ", pcdata, " and ", stkmap.n, " args stack map entries for ", funcname(f), " (targetpc=", targetpc, ")\n")
				throw("scanframe: bad symbol table")
			}
			bv = stackmapdata(stkmap, pcdata)
		}
		wbuf = scanblock(frame.argp, uintptr(bv.n)/typeBitsWidth*ptrSize, bv.bytedata, wbuf)
	}
	return wbuf
}

//go:nowritebarrier
func scanstack(gp *g) {
	if gp.gcscanvalid {
		return
	}

	if readgstatus(gp)&_Gscan == 0 {
		print("runtime:scanstack: gp=", gp, ", goid=", gp.goid, ", gp->atomicstatus=", hex(readgstatus(gp)), "\n")
		throw("scanstack - bad status")
	}

	switch readgstatus(gp) &^ _Gscan {
	default:
		print("runtime: gp=", gp, ", goid=", gp.goid, ", gp->atomicstatus=", readgstatus(gp), "\n")
		throw("mark - bad status")
	case _Gdead:
		return
	case _Grunning:
		print("runtime: gp=", gp, ", goid=", gp.goid, ", gp->atomicstatus=", readgstatus(gp), "\n")
		throw("scanstack: goroutine not stopped")
	case _Grunnable, _Gsyscall, _Gwaiting:
		// ok
	}

	if gp == getg() {
		throw("can't scan our own stack")
	}
	mp := gp.m
	if mp != nil && mp.helpgc != 0 {
		throw("can't scan gchelper stack")
	}

	wbuf := (*workbuf)(unsafe.Pointer(xchguintptr(&getg().m.currentwbuf, 0)))
	scanframe := func(frame *stkframe, unused unsafe.Pointer) bool {
		// Pick up wbuf as free variable so gentraceback and friends can
		// keep the same signature.
		wbuf = scanframeworker(frame, unused, wbuf)
		return true
	}
	gentraceback(^uintptr(0), ^uintptr(0), 0, gp, 0, nil, 0x7fffffff, scanframe, nil, 0)
	tracebackdefers(gp, scanframe, nil)
	wbuf = (*workbuf)(unsafe.Pointer(xchguintptr(&getg().m.currentwbuf, uintptr(unsafe.Pointer(wbuf)))))
	if wbuf != nil {
		throw("wbuf not nil after stack scans")
	}
	gp.gcscanvalid = true
}

// Shade the object if it isn't already.
// The object is not nil and known to be in the heap.
//go:nowritebarrier
func shade(b uintptr) {
	var wbuf *workbuf

	if !inheap(b) {
		throw("shade: passed an address not in the heap")
	}
	if obj, hbits := heapBitsForObject(b); obj != 0 {
		// TODO: this would be a great place to put a check to see
		// if we are harvesting and if we are then we should
		// figure out why there is a call to shade when the
		// harvester thinks we are in a STW.
		// if atomicload(&harvestingwbufs) == uint32(1) {
		//	// Throw here to discover write barriers
		//	// being executed during a STW.
		// }

		wbuf = getpartialorempty(1181)
		wbuf := greyobject(obj, 0, 0, hbits, wbuf)
		checknocurrentwbuf()
		// This is part of the write barrier so put the wbuf back.
		if gcphase == _GCmarktermination {
			putpartial(wbuf, 1191) // Put on full???
		} else {
			wbuf = (*workbuf)(unsafe.Pointer(xchguintptr(&getg().m.currentwbuf, uintptr(unsafe.Pointer(wbuf)))))
			if wbuf != nil {
				throw("m.currentwbuf lost in shade")
			}
		}
	}
}

// gchelpwork does a small bounded amount of gc work. The purpose is to
// shorten the time (as measured by allocations) spent doing a concurrent GC.
// The number of mutator calls is roughly propotional to the number of allocations
// made by that mutator. This slows down the allocation while speeding up the GC.
//go:nowritebarrier
func gchelpwork() {
	switch gcphase {
	default:
		throw("gcphasework in bad gcphase")
	case _GCoff, _GCquiesce, _GCstw:
		// No work.
	case _GCsweep:
		// We could help by calling sweepone to sweep a single span.
		// _ = sweepone()
	case _GCscan:
		// scan the stack, mark the objects, put pointers in work buffers
		// hanging off the P where this is being run.
		// scanstack(gp)
	case _GCmark:
		// Get a full work buffer and empty it.
		m := getg().m
		// drain your own currentwbuf first in the hopes that it will
		// be more cache friendly.
		wbuf := (*workbuf)(unsafe.Pointer(xchguintptr(&m.currentwbuf, 0)))
		//		wbuf := (*workbuf)(unsafe.Pointer(m.currentwbuf))
		//		m.currentwbuf = 0
		if wbuf == nil {
			wbuf = trygetfull(1228)
		}
		if wbuf != nil {
			wbuf = gcDrainN(wbuf, uintptr(len(wbuf.obj))) // drain upto one buffer's worth of objects
			if wbuf != nil {
				if wbuf.nobj != 0 {
					putfull(wbuf, 1175)
				} else {
					putempty(wbuf, 1177)
				}
			}
		}
	case _GCmarktermination:
		// We should never be here since the world is stopped.
		// All available mark work will be emptied before returning.
		throw("gcphasework in bad gcphase")
	}
}

// The gp has been moved to a GC safepoint. GC phase specific
// work is done here.
//go:nowritebarrier
func gcphasework(gp *g) {
	switch gcphase {
	default:
		throw("gcphasework in bad gcphase")
	case _GCoff, _GCquiesce, _GCstw, _GCsweep:
		// No work.
	case _GCscan:
		// scan the stack, mark the objects, put pointers in work buffers
		// hanging off the P where this is being run.
		// Indicate that the scan is valid until the goroutine runs again
		scanstack(gp)
	case _GCmark:
		// No work.
	case _GCmarktermination:
		scanstack(gp)
		// All available mark work will be emptied before returning.
	}
	gp.gcworkdone = true
}

// Returns only when span s has been swept.
//go:nowritebarrier
func mSpan_EnsureSwept(s *mspan) {
	// Caller must disable preemption.
	// Otherwise when this function returns the span can become unswept again
	// (if GC is triggered on another goroutine).
	_g_ := getg()
	if _g_.m.locks == 0 && _g_.m.mallocing == 0 && _g_ != _g_.m.g0 {
		throw("MSpan_EnsureSwept: m is not locked")
	}

	sg := mheap_.sweepgen
	if atomicload(&s.sweepgen) == sg {
		return
	}
	// The caller must be sure that the span is a MSpanInUse span.
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
//TODO go:nowritebarrier
func mSpan_Sweep(s *mspan, preserve bool) bool {
	if checkmarkphase {
		throw("MSpan_Sweep: checkmark only runs in STW and after the sweep")
	}

	// It's critical that we enter this function with preemption disabled,
	// GC must not start while we are in the middle of this function.
	_g_ := getg()
	if _g_.m.locks == 0 && _g_.m.mallocing == 0 && _g_ != _g_.m.g0 {
		throw("MSpan_Sweep: m is not locked")
	}
	sweepgen := mheap_.sweepgen
	if s.state != mSpanInUse || s.sweepgen != sweepgen-1 {
		print("MSpan_Sweep: state=", s.state, " sweepgen=", s.sweepgen, " mheap.sweepgen=", sweepgen, "\n")
		throw("MSpan_Sweep: bad span state")
	}

	if trace.enabled {
		traceGCSweepStart()
	}

	cl := s.sizeclass
	size := s.elemsize
	res := false
	nfree := 0

	var head, end gclinkptr

	c := _g_.m.mcache
	sweepgenset := false

	// Mark any free objects in this span so we don't collect them.
	for link := s.freelist; link.ptr() != nil; link = link.ptr().next {
		heapBitsForAddr(uintptr(link)).setMarkedNonAtomic()
	}

	// Unlink & free special records for any objects we're about to free.
	specialp := &s.specials
	special := *specialp
	for special != nil {
		// A finalizer can be set for an inner byte of an object, find object beginning.
		p := uintptr(s.start<<_PageShift) + uintptr(special.offset)/size*size
		hbits := heapBitsForAddr(p)
		if !hbits.isMarked() {
			// Find the exact byte for which the special was setup
			// (as opposed to object beginning).
			p := uintptr(s.start<<_PageShift) + uintptr(special.offset)
			// about to free object: splice out special record
			y := special
			special = special.next
			*specialp = special
			if !freespecial(y, unsafe.Pointer(p), size, false) {
				// stop freeing of object if it has a finalizer
				hbits.setMarkedNonAtomic()
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

	size, n, _ := s.layout()
	heapBitsSweepSpan(s.base(), size, n, func(p uintptr) {
		// At this point we know that we are looking at garbage object
		// that needs to be collected.
		if debug.allocfreetrace != 0 {
			tracefree(unsafe.Pointer(p), size)
		}

		// Reset to allocated+noscan.
		if cl == 0 {
			// Free large span.
			if preserve {
				throw("can't preserve large span")
			}
			heapBitsForSpan(p).clearSpan(s.layout())
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
			reduction := int64(size) * int64(gcpercent+100) / 100
			if int64(memstats.next_gc)-reduction > int64(heapminimum) {
				xadd64(&memstats.next_gc, -reduction)
			} else {
				atomicstore64(&memstats.next_gc, heapminimum)
			}
			res = true
		} else {
			// Free small object.
			if size > 2*ptrSize {
				*(*uintptr)(unsafe.Pointer(p + ptrSize)) = uintptrMask & 0xdeaddeaddeaddead // mark as "needs to be zeroed"
			} else if size > ptrSize {
				*(*uintptr)(unsafe.Pointer(p + ptrSize)) = 0
			}
			if head.ptr() == nil {
				head = gclinkptr(p)
			} else {
				end.ptr().next = gclinkptr(p)
			}
			end = gclinkptr(p)
			end.ptr().next = gclinkptr(0x0bade5)
			nfree++
		}
	})

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
			throw("MSpan_Sweep: bad span state after sweep")
		}
		atomicstore(&s.sweepgen, sweepgen)
	}
	if nfree > 0 {
		c.local_nsmallfree[cl] += uintptr(nfree)
		c.local_cachealloc -= intptr(uintptr(nfree) * size)
		reduction := int64(nfree) * int64(size) * int64(gcpercent+100) / 100
		if int64(memstats.next_gc)-reduction > int64(heapminimum) {
			xadd64(&memstats.next_gc, -reduction)
		} else {
			atomicstore64(&memstats.next_gc, heapminimum)
		}
		res = mCentral_FreeSpan(&mheap_.central[cl].mcentral, s, int32(nfree), head, end, preserve)
		// MCentral_FreeSpan updates sweepgen
	}
	if trace.enabled {
		traceGCSweepDone()
		traceNextGC()
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

// State of the background concurrent GC goroutine.
var bggc struct {
	lock    mutex
	g       *g
	working uint
	started bool
}

// sweeps one span
// returns number of pages returned to heap, or ^uintptr(0) if there is nothing to sweep
//go:nowritebarrier
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

//go:nowritebarrier
func gosweepone() uintptr {
	var ret uintptr
	systemstack(func() {
		ret = sweepone()
	})
	return ret
}

//go:nowritebarrier
func gosweepdone() bool {
	return mheap_.sweepdone != 0
}

//go:nowritebarrier
func gchelper() {
	_g_ := getg()
	_g_.m.traceback = 2
	gchelperstart()

	if trace.enabled {
		traceGCScanStart()
	}

	// parallel mark for over GC roots
	parfordo(work.markfor)
	if gcphase != _GCscan {
		gcDrain(nil) // blocks in getfull
	}

	if trace.enabled {
		traceGCScanDone()
	}

	nproc := work.nproc // work.nproc can change right after we increment work.ndone
	if xadd(&work.ndone, +1) == nproc-1 {
		notewakeup(&work.alldone)
	}
	_g_.m.traceback = 0
}

//go:nowritebarrier
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

//go:nowritebarrier
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

//go:nowritebarrier
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

// heapminimum is the minimum number of bytes in the heap.
// This cleans up the corner case of where we have a very small live set but a lot
// of allocations and collecting every GOGC * live set is expensive.
var heapminimum = uint64(4 << 20)

func gcinit() {
	if unsafe.Sizeof(workbuf{}) != _WorkbufSize {
		throw("size of Workbuf is suboptimal")
	}

	work.markfor = parforalloc(_MaxGcproc)
	gcpercent = readgogc()
	gcdatamask = unrollglobgcprog((*byte)(unsafe.Pointer(&gcdata)), uintptr(unsafe.Pointer(&edata))-uintptr(unsafe.Pointer(&data)))
	gcbssmask = unrollglobgcprog((*byte)(unsafe.Pointer(&gcbss)), uintptr(unsafe.Pointer(&ebss))-uintptr(unsafe.Pointer(&bss)))
	memstats.next_gc = heapminimum
}

// Called from malloc.go using systemstack, stopping and starting the world handled in caller.
//go:nowritebarrier
func gc_m(start_time int64, eagersweep bool) {
	_g_ := getg()
	gp := _g_.m.curg
	casgstatus(gp, _Grunning, _Gwaiting)
	gp.waitreason = "garbage collection"

	gc(start_time, eagersweep)
	casgstatus(gp, _Gwaiting, _Grunning)
}

//go:nowritebarrier
func initCheckmarks() {
	for _, s := range work.spans {
		if s.state == _MSpanInUse {
			heapBitsForSpan(s.base()).initCheckmarkSpan(s.layout())
		}
	}
}

func clearCheckmarks() {
	for _, s := range work.spans {
		if s.state == _MSpanInUse {
			heapBitsForSpan(s.base()).clearCheckmarkSpan(s.layout())
		}
	}
}

// Called from malloc.go using systemstack.
// The world is stopped. Rerun the scan and mark phases
// using the bitMarkedCheck bit instead of the
// bitMarked bit. If the marking encounters an
// bitMarked bit that is not set then we throw.
//go:nowritebarrier
func gccheckmark_m(startTime int64, eagersweep bool) {
	if debug.gccheckmark == 0 {
		return
	}

	if checkmarkphase {
		throw("gccheckmark_m, entered with checkmarkphase already true")
	}

	checkmarkphase = true
	initCheckmarks()
	gc_m(startTime, eagersweep) // turns off checkmarkphase + calls clearcheckmarkbits
}

//go:nowritebarrier
func finishsweep_m() {
	// The world is stopped so we should be able to complete the sweeps
	// quickly.
	for sweepone() != ^uintptr(0) {
		sweep.npausesweep++
	}

	// There may be some other spans being swept concurrently that
	// we need to wait for. If finishsweep_m is done with the world stopped
	// this code is not required.
	sg := mheap_.sweepgen
	for _, s := range work.spans {
		if s.sweepgen != sg && s.state == _MSpanInUse {
			mSpan_EnsureSwept(s)
		}
	}
}

// Scan all of the stacks, greying (or graying if in America) the referents
// but not blackening them since the mark write barrier isn't installed.
//go:nowritebarrier
func gcscan_m() {
	_g_ := getg()

	// Grab the g that called us and potentially allow rescheduling.
	// This allows it to be scanned like other goroutines.
	mastergp := _g_.m.curg
	casgstatus(mastergp, _Grunning, _Gwaiting)
	mastergp.waitreason = "garbage collection scan"

	// Span sweeping has been done by finishsweep_m.
	// Long term we will want to make this goroutine runnable
	// by placing it onto a scanenqueue state and then calling
	// runtime·restartg(mastergp) to make it Grunnable.
	// At the bottom we will want to return this p back to the scheduler.

	// Prepare flag indicating that the scan has not been completed.
	lock(&allglock)
	local_allglen := allglen
	for i := uintptr(0); i < local_allglen; i++ {
		gp := allgs[i]
		gp.gcworkdone = false  // set to true in gcphasework
		gp.gcscanvalid = false // stack has not been scanned
	}
	unlock(&allglock)

	work.nwait = 0
	work.ndone = 0
	work.nproc = 1 // For now do not do this in parallel.
	//	ackgcphase is not needed since we are not scanning running goroutines.
	parforsetup(work.markfor, work.nproc, uint32(_RootCount+local_allglen), false, markroot)
	parfordo(work.markfor)

	lock(&allglock)
	// Check that gc work is done.
	for i := uintptr(0); i < local_allglen; i++ {
		gp := allgs[i]
		if !gp.gcworkdone {
			throw("scan missed a g")
		}
	}
	unlock(&allglock)

	casgstatus(mastergp, _Gwaiting, _Grunning)
	// Let the g that called us continue to run.
}

// Mark all objects that are known about.
// This is the concurrent mark phase.
//go:nowritebarrier
func gcmark_m() {
	gcDrain(nil)
	// TODO add another harvestwbuf and reset work.nwait=0, work.ndone=0, and work.nproc=1
	// and repeat the above gcDrain.
}

// For now this must be bracketed with a stoptheworld and a starttheworld to ensure
// all go routines see the new barrier.
//go:nowritebarrier
func gcinstalloffwb_m() {
	gcphase = _GCoff
}

// STW is in effect at this point.
//TODO go:nowritebarrier
func gc(start_time int64, eagersweep bool) {
	if _DebugGCPtrs {
		print("GC start\n")
	}

	gcphase = _GCmarktermination
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

	if !checkmarkphase {
		// TODO(austin) This is a noop beceause we should
		// already have swept everything to the current
		// sweepgen.
		finishsweep_m() // skip during checkmark debug phase.
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

	// World is stopped so allglen will not change.
	for i := uintptr(0); i < allglen; i++ {
		gp := allgs[i]
		gp.gcworkdone = false // set to true in gcphasework
	}

	if trace.enabled {
		traceGCScanStart()
	}

	parforsetup(work.markfor, work.nproc, uint32(_RootCount+allglen), false, markroot)
	if work.nproc > 1 {
		noteclear(&work.alldone)
		helpgc(int32(work.nproc))
	}

	var t2 int64
	if debug.gctrace > 0 {
		t2 = nanotime()
	}

	harvestwbufs() // move local workbufs onto global queues where the GC can find them
	gchelperstart()
	parfordo(work.markfor)
	gcDrain(nil)

	if work.full != 0 {
		throw("work.full != 0")
	}
	if work.partial != 0 {
		throw("work.partial != 0")
	}

	gcphase = _GCoff
	var t3 int64
	if debug.gctrace > 0 {
		t3 = nanotime()
	}

	if work.nproc > 1 {
		notesleep(&work.alldone)
	}

	if trace.enabled {
		traceGCScanDone()
	}

	shrinkfinish()

	cachestats()
	// next_gc calculation is tricky with concurrent sweep since we don't know size of live heap
	// estimate what was live heap size after previous GC (for printing only)
	heap0 := memstats.next_gc * 100 / (uint64(gcpercent) + 100)
	// conservatively set next_gc to high value assuming that everything is live
	// concurrent/lazy sweep will reduce this number while discovering new garbage
	memstats.next_gc = memstats.heap_alloc + memstats.heap_alloc*uint64(gcpercent)/100
	if memstats.next_gc < heapminimum {
		memstats.next_gc = heapminimum
	}

	if trace.enabled {
		traceNextGC()
	}

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
			throw("mstats skew")
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

	if debug.gccheckmark > 0 {
		if !checkmarkphase {
			// first half of two-pass; don't set up sweep
			unlock(&mheap_.lock)
			return
		}
		checkmarkphase = false // done checking marks
		clearCheckmarks()
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
		throw("short slice passed to readGCStats")
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
		throw("gchelperstart: bad m->helpgc")
	}
	if _g_ != _g_.m.g0 {
		throw("gchelper not running on g0 stack")
	}
}

func unixnanotime() int64 {
	sec, nsec := time_now()
	return sec*1e9 + int64(nsec)
}
