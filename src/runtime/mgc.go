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

// Initialized from $GOGC.  GOGC=off means no GC.
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

// It is a bug if bits does not have bitBoundary set but
// there are still some cases where this happens related
// to stack spans.
type markbits struct {
	bitp  *byte   // pointer to the byte holding xbits
	shift uintptr // bits xbits needs to be shifted to get bits
	xbits byte    // byte holding all the bits from *bitp
	bits  byte    // mark and boundary bits relevant to corresponding slot.
	tbits byte    // pointer||scalar bits relevant to corresponding slot.
}

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
	partial uint64                // lock-free list of partially filled blocks
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
// The first nibble no longer holds the bitsDead pattern indicating that the
// there are no more pointers in the object. This information is held
// in the second nibble.

// When marking an object if the bool checkmark is true one uses the above
// encoding, otherwise one uses the bitMarked bit in the lower two bits
// of the nibble.
var (
	checkmark         = false
	gccheckmarkenable = true
)

// Is address b in the known heap. If it doesn't have a valid gcmap
// returns false. For example pointers into stacks will return false.
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

// Given an address in the heap return the relevant byte from the gcmap. This routine
// can be used on addresses to the start of an object or to the interior of the an object.
//go:nowritebarrier
func slottombits(obj uintptr, mbits *markbits) {
	off := (obj&^(ptrSize-1) - mheap_.arena_start) / ptrSize
	*(*uintptr)(unsafe.Pointer(&mbits.bitp)) = mheap_.arena_start - off/wordsPerBitmapByte - 1
	mbits.shift = off % wordsPerBitmapByte * gcBits
	mbits.xbits = *mbits.bitp
	mbits.bits = (mbits.xbits >> mbits.shift) & bitMask
	mbits.tbits = ((mbits.xbits >> mbits.shift) & bitPtrMask) >> 2
}

// b is a pointer into the heap.
// Find the start of the object refered to by b.
// Set mbits to the associated bits from the bit map.
// If b is not a valid heap object return nil and
// undefined values in mbits.
//go:nowritebarrier
func objectstart(b uintptr, mbits *markbits) uintptr {
	obj := b &^ (ptrSize - 1)
	for {
		slottombits(obj, mbits)
		if mbits.bits&bitBoundary == bitBoundary {
			break
		}

		// Not a beginning of a block, consult span table to find the block beginning.
		k := b >> _PageShift
		x := k
		x -= mheap_.arena_start >> _PageShift
		s := h_spans[x]
		if s == nil || pageID(k) < s.start || b >= s.limit || s.state != mSpanInUse {
			if s != nil && s.state == _MSpanStack {
				return 0 // This is legit.
			}

			// The following ensures that we are rigorous about what data
			// structures hold valid pointers
			if false {
				// Still happens sometimes. We don't know why.
				printlock()
				print("runtime:objectstart Span weird: obj=", hex(obj), " k=", hex(k))
				if s == nil {
					print(" s=nil\n")
				} else {
					print(" s.start=", hex(s.start<<_PageShift), " s.limit=", hex(s.limit), " s.state=", s.state, "\n")
				}
				printunlock()
				throw("objectstart: bad pointer in unexpected span")
			}
			return 0
		}

		p := uintptr(s.start) << _PageShift
		if s.sizeclass != 0 {
			size := s.elemsize
			idx := (obj - p) / size
			p = p + idx*size
		}
		if p == obj {
			print("runtime: failed to find block beginning for ", hex(p), " s=", hex(s.start*_PageSize), " s.limit=", hex(s.limit), "\n")
			throw("failed to find block beginning")
		}
		obj = p
	}

	// if size(obj.firstfield) < PtrSize, the &obj.secondfield could map to the boundary bit
	// Clear any low bits to get to the start of the object.
	// greyobject depends on this.
	return obj
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

// Mark using the checkmark scheme.
//go:nowritebarrier
func docheckmark(mbits *markbits) {
	// xor 01 moves 01(scalar unmarked) to 00(scalar marked)
	// and 10(pointer unmarked) to 11(pointer marked)
	if mbits.tbits == _BitsScalar {
		atomicand8(mbits.bitp, ^byte(_BitsCheckMarkXor<<mbits.shift<<2))
	} else if mbits.tbits == _BitsPointer {
		atomicor8(mbits.bitp, byte(_BitsCheckMarkXor<<mbits.shift<<2))
	}

	// reload bits for ischeckmarked
	mbits.xbits = *mbits.bitp
	mbits.bits = (mbits.xbits >> mbits.shift) & bitMask
	mbits.tbits = ((mbits.xbits >> mbits.shift) & bitPtrMask) >> 2
}

// In the default scheme does mbits refer to a marked object.
//go:nowritebarrier
func ismarked(mbits *markbits) bool {
	if mbits.bits&bitBoundary != bitBoundary {
		throw("ismarked: bits should have boundary bit set")
	}
	return mbits.bits&bitMarked == bitMarked
}

// In the checkmark scheme does mbits refer to a marked object.
//go:nowritebarrier
func ischeckmarked(mbits *markbits) bool {
	if mbits.bits&bitBoundary != bitBoundary {
		throw("ischeckmarked: bits should have boundary bit set")
	}
	return mbits.tbits == _BitsScalarMarked || mbits.tbits == _BitsPointerMarked
}

// When in GCmarkterminate phase we allocate black.
//go:nowritebarrier
func gcmarknewobject_m(obj uintptr) {
	if gcphase != _GCmarktermination {
		throw("marking new object while not in mark termination phase")
	}
	if checkmark { // The world should be stopped so this should not happen.
		throw("gcmarknewobject called while doing checkmark")
	}

	var mbits markbits
	slottombits(obj, &mbits)
	if mbits.bits&bitMarked != 0 {
		return
	}

	// Each byte of GC bitmap holds info for two words.
	// If the current object is larger than two words, or if the object is one word
	// but the object it shares the byte with is already marked,
	// then all the possible concurrent updates are trying to set the same bit,
	// so we can use a non-atomic update.
	if mbits.xbits&(bitMask|(bitMask<<gcBits)) != bitBoundary|bitBoundary<<gcBits || work.nproc == 1 {
		*mbits.bitp = mbits.xbits | bitMarked<<mbits.shift
	} else {
		atomicor8(mbits.bitp, bitMarked<<mbits.shift)
	}
}

// obj is the start of an object with mark mbits.
// If it isn't already marked, mark it and enqueue into workbuf.
// Return possibly new workbuf to use.
//go:nowritebarrier
func greyobject(obj uintptr, mbits *markbits, wbuf *workbuf) *workbuf {
	// obj should be start of allocation, and so must be at least pointer-aligned.
	if obj&(ptrSize-1) != 0 {
		throw("greyobject: obj not pointer-aligned")
	}

	if checkmark {
		if !ismarked(mbits) {
			print("runtime:greyobject: checkmarks finds unexpected unmarked object obj=", hex(obj), ", mbits->bits=", hex(mbits.bits), " *mbits->bitp=", hex(*mbits.bitp), "\n")

			k := obj >> _PageShift
			x := k
			x -= mheap_.arena_start >> _PageShift
			s := h_spans[x]
			printlock()
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
		if ischeckmarked(mbits) {
			return wbuf
		}
		docheckmark(mbits)
		if !ischeckmarked(mbits) {
			print("mbits xbits=", hex(mbits.xbits), " bits=", hex(mbits.bits), " tbits=", hex(mbits.tbits), " shift=", mbits.shift, "\n")
			throw("docheckmark and ischeckmarked disagree")
		}
	} else {
		// If marked we have nothing to do.
		if mbits.bits&bitMarked != 0 {
			return wbuf
		}

		// Each byte of GC bitmap holds info for two words.
		// If the current object is larger than two words, or if the object is one word
		// but the object it shares the byte with is already marked,
		// then all the possible concurrent updates are trying to set the same bit,
		// so we can use a non-atomic update.
		if mbits.xbits&(bitMask|bitMask<<gcBits) != bitBoundary|bitBoundary<<gcBits || work.nproc == 1 {
			*mbits.bitp = mbits.xbits | bitMarked<<mbits.shift
		} else {
			atomicor8(mbits.bitp, bitMarked<<mbits.shift)
		}
	}

	if !checkmark && (mbits.xbits>>(mbits.shift+2))&_BitsMask == _BitsDead {
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
		wbuf = getempty(wbuf)
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
	var ptrbitp unsafe.Pointer
	var mbits markbits
	if ptrmask == nil {
		b = objectstart(b, &mbits)
		if b == 0 {
			return wbuf
		}
		ptrbitp = unsafe.Pointer(mbits.bitp)
	}
	for i := uintptr(0); i < n; i += ptrSize {
		// Find bits for this word.
		var bits uintptr
		if ptrmask != nil {
			// dense mask (stack or data)
			bits = (uintptr(*(*byte)(add(unsafe.Pointer(ptrmask), (i/ptrSize)/4))) >> (((i / ptrSize) % 4) * bitsPerPointer)) & bitsMask
		} else {
			// Check if we have reached end of span.
			// n is an overestimate of the size of the object.
			if (b+i)%_PageSize == 0 && h_spans[(b-arena_start)>>_PageShift] != h_spans[(b+i-arena_start)>>_PageShift] {
				break
			}

			// Consult GC bitmap.
			bits = uintptr(*(*byte)(ptrbitp))
			if wordsPerBitmapByte != 2 {
				throw("alg doesn't work for wordsPerBitmapByte != 2")
			}
			j := (uintptr(b) + i) / ptrSize & 1 // j indicates upper nibble or lower nibble
			bits >>= gcBits * j
			if i == 0 {
				bits &^= bitBoundary
			}
			ptrbitp = add(ptrbitp, -j)

			if bits&bitBoundary != 0 && i != 0 {
				break // reached beginning of the next object
			}
			bits = (bits & bitPtrMask) >> 2 // bits refer to the type bits.

			if i != 0 && bits == bitsDead { // BitsDead in first nibble not valid during checkmark
				break // reached no-scan part of the object
			}
		}

		if bits <= _BitsScalar { // _BitsScalar, _BitsDead, _BitsScalarMarked
			continue
		}

		if bits&_BitsPointer != _BitsPointer {
			print("gc checkmark=", checkmark, " b=", hex(b), " ptrmask=", ptrmask, " mbits.bitp=", mbits.bitp, " mbits.xbits=", hex(mbits.xbits), " bits=", hex(bits), "\n")
			throw("unexpected garbage collection bits")
		}

		obj := *(*uintptr)(unsafe.Pointer(b + i))

		// At this point we have extracted the next potential pointer.
		// Check if it points into heap.
		if obj == 0 || obj < arena_start || obj >= arena_used {
			continue
		}

		// Mark the object. return some important bits.
		// We we combine the following two rotines we don't have to pass mbits or obj around.
		var mbits markbits
		obj = objectstart(obj, &mbits)
		if obj == 0 {
			continue
		}
		wbuf = greyobject(obj, &mbits, wbuf)
	}
	return wbuf
}

// scanblock starts by scanning b as scanobject would.
// If the gcphase is GCscan, that's all scanblock does.
// Otherwise it traverses some fraction of the pointers it found in b, recursively.
// As a special case, scanblock(nil, 0, nil) means to scan previously queued work,
// stopping only when no work is left in the system.
//go:nowritebarrier
func scanblock(b, n uintptr, ptrmask *uint8) {
	wbuf := getpartialorempty()
	if b != 0 {
		wbuf = scanobject(b, n, ptrmask, wbuf)
		if gcphase == _GCscan {
			if inheap(b) && ptrmask == nil {
				// b is in heap, we are in GCscan so there should be a ptrmask.
				throw("scanblock: In GCscan phase and inheap is true.")
			}
			// GCscan only goes one level deep since mark wb not turned on.
			putpartial(wbuf)
			return
		}
	}
	if gcphase == _GCscan {
		throw("scanblock: In GCscan phase but no b passed in.")
	}

	keepworking := b == 0

	// ptrmask can have 2 possible values:
	// 1. nil - obtain pointer mask from GC bitmap.
	// 2. pointer to a compact mask (for stacks and data).
	for {
		if wbuf.nobj == 0 {
			if !keepworking {
				putempty(wbuf)
				return
			}
			// Refill workbuf from global queue.
			wbuf = getfull(wbuf)
			if wbuf == nil { // nil means out of work barrier reached
				return
			}

			if wbuf.nobj <= 0 {
				throw("runtime:scanblock getfull returns empty buffer")
			}
		}

		// If another proc wants a pointer, give it some.
		if work.nwait > 0 && wbuf.nobj > 4 && work.full == 0 {
			wbuf = handoff(wbuf)
		}

		// This might be a good place to add prefetch code...
		// if(wbuf->nobj > 4) {
		//         PREFETCH(wbuf->obj[wbuf->nobj - 3];
		//  }
		wbuf.nobj--
		b = wbuf.obj[wbuf.nobj]
		wbuf = scanobject(b, mheap_.arena_used-b, nil, wbuf)
	}
}

//go:nowritebarrier
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
			if !checkmark && s.sweepgen != sg {
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
					scanblock(p, s.elemsize, nil) // scanned during mark phase
				}
				scanblock(uintptr(unsafe.Pointer(&spf.fn)), ptrSize, &oneptr[0])
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
		if gcphase != _GCscan { // Do not shrink during GCscan phase.
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
}

// Get an empty work buffer off the work.empty list,
// allocating new buffers as needed.
//go:nowritebarrier
func getempty(b *workbuf) *workbuf {
	if b != nil {
		putfull(b)
		b = nil
	}
	if work.empty != 0 {
		b = (*workbuf)(lfstackpop(&work.empty))
	}
	if b != nil && b.nobj != 0 {
		_g_ := getg()
		print("m", _g_.m.id, ": getempty: popped b=", b, " with non-zero b.nobj=", b.nobj, "\n")
		throw("getempty: workbuffer not empty, b->nobj not 0")
	}
	if b == nil {
		b = (*workbuf)(persistentalloc(unsafe.Sizeof(*b), _CacheLineSize, &memstats.gc_sys))
		b.nobj = 0
	}
	return b
}

//go:nowritebarrier
func putempty(b *workbuf) {
	if b.nobj != 0 {
		throw("putempty: b->nobj not 0")
	}
	lfstackpush(&work.empty, &b.node)
}

//go:nowritebarrier
func putfull(b *workbuf) {
	if b.nobj <= 0 {
		throw("putfull: b->nobj <= 0")
	}
	lfstackpush(&work.full, &b.node)
}

// Get an partially empty work buffer
// if none are available get an empty one.
//go:nowritebarrier
func getpartialorempty() *workbuf {
	b := (*workbuf)(lfstackpop(&work.partial))
	if b == nil {
		b = getempty(nil)
	}
	return b
}

//go:nowritebarrier
func putpartial(b *workbuf) {
	if b.nobj == 0 {
		lfstackpush(&work.empty, &b.node)
	} else if b.nobj < uintptr(len(b.obj)) {
		lfstackpush(&work.partial, &b.node)
	} else if b.nobj == uintptr(len(b.obj)) {
		lfstackpush(&work.full, &b.node)
	} else {
		print("b=", b, " b.nobj=", b.nobj, " len(b.obj)=", len(b.obj), "\n")
		throw("putpartial: bad Workbuf b.nobj")
	}
}

// Get a full work buffer off the work.full or a partially
// filled one off the work.partial list. If nothing is available
// wait until all the other gc helpers have finished and then
// return nil.
// getfull acts as a barrier for work.nproc helpers. As long as one
// gchelper is actively marking objects it
// may create a workbuffer that the other helpers can work on.
// The for loop either exits when a work buffer is found
// or when _all_ of the work.nproc GC helpers are in the loop
// looking for work and thus not capable of creating new work.
// This is in fact the termination condition for the STW mark
// phase.
//go:nowritebarrier
func getfull(b *workbuf) *workbuf {
	if b != nil {
		putempty(b)
	}

	b = (*workbuf)(lfstackpop(&work.full))
	if b == nil {
		b = (*workbuf)(lfstackpop(&work.partial))
	}
	if b != nil || work.nproc == 1 {
		return b
	}

	xadd(&work.nwait, +1)
	for i := 0; ; i++ {
		if work.full != 0 {
			xadd(&work.nwait, -1)
			b = (*workbuf)(lfstackpop(&work.full))
			if b == nil {
				b = (*workbuf)(lfstackpop(&work.partial))
			}
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

//go:nowritebarrier
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

//go:nowritebarrier
func stackmapdata(stkmap *stackmap, n int32) bitvector {
	if n < 0 || n >= stkmap.n {
		throw("stackmapdata: index out of range")
	}
	return bitvector{stkmap.nbit, (*byte)(add(unsafe.Pointer(&stkmap.bytedata), uintptr(n*((stkmap.nbit+31)/32*4))))}
}

// Scan a stack frame: local variables and function arguments/results.
//go:nowritebarrier
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
			throw("missing stackmap")
		}

		// Locals bitmap information, scan just the pointers in locals.
		if pcdata < 0 || pcdata >= stkmap.n {
			// don't know where we are
			print("runtime: pcdata is ", pcdata, " and ", stkmap.n, " locals stack map entries for ", gofuncname(f), " (targetpc=", targetpc, ")\n")
			throw("scanframe: bad symbol table")
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
				throw("missing stackmap")
			}
			if pcdata < 0 || pcdata >= stkmap.n {
				// don't know where we are
				print("runtime: pcdata is ", pcdata, " and ", stkmap.n, " args stack map entries for ", gofuncname(f), " (targetpc=", targetpc, ")\n")
				throw("scanframe: bad symbol table")
			}
			bv = stackmapdata(stkmap, pcdata)
		}
		scanblock(frame.argp, uintptr(bv.n)/bitsPerPointer*ptrSize, bv.bytedata)
	}
	return true
}

//go:nowritebarrier
func scanstack(gp *g) {

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

	gentraceback(^uintptr(0), ^uintptr(0), 0, gp, 0, nil, 0x7fffffff, scanframe, nil, 0)
	tracebackdefers(gp, scanframe, nil)
}

// If the slot is grey or black return true, if white return false.
// If the slot is not in the known heap and thus does not have a valid GC bitmap then
// it is considered grey. Globals and stacks can hold such slots.
// The slot is grey if its mark bit is set and it is enqueued to be scanned.
// The slot is black if it has already been scanned.
// It is white if it has a valid mark bit and the bit is not set.
//go:nowritebarrier
func shaded(slot uintptr) bool {
	if !inheap(slot) { // non-heap slots considered grey
		return true
	}

	var mbits markbits
	valid := objectstart(slot, &mbits)
	if valid == 0 {
		return true
	}

	if checkmark {
		return ischeckmarked(&mbits)
	}

	return mbits.bits&bitMarked != 0
}

// Shade the object if it isn't already.
// The object is not nil and known to be in the heap.
//go:nowritebarrier
func shade(b uintptr) {
	if !inheap(b) {
		throw("shade: passed an address not in the heap")
	}

	wbuf := getpartialorempty()
	// Mark the object, return some important bits.
	// If we combine the following two rotines we don't have to pass mbits or obj around.
	var mbits markbits
	obj := objectstart(b, &mbits)
	if obj != 0 {
		wbuf = greyobject(obj, &mbits, wbuf) // augments the wbuf
	}
	putpartial(wbuf)
}

// This is the Dijkstra barrier coarsened to always shade the ptr (dst) object.
// The original Dijkstra barrier only shaded ptrs being placed in black slots.
//
// Shade indicates that it has seen a white pointer by adding the referent
// to wbuf as well as marking it.
//
// slot is the destination (dst) in go code
// ptr is the value that goes into the slot (src) in the go code
//
// Dijkstra pointed out that maintaining the no black to white
// pointers means that white to white pointers not need
// to be noted by the write barrier. Furthermore if either
// white object dies before it is reached by the
// GC then the object can be collected during this GC cycle
// instead of waiting for the next cycle. Unfortunately the cost of
// ensure that the object holding the slot doesn't concurrently
// change to black without the mutator noticing seems prohibitive.
//
// Consider the following example where the mutator writes into
// a slot and then loads the slot's mark bit while the GC thread
// writes to the slot's mark bit and then as part of scanning reads
// the slot.
//
// Initially both [slot] and [slotmark] are 0 (nil)
// Mutator thread          GC thread
// st [slot], ptr          st [slotmark], 1
//
// ld r1, [slotmark]       ld r2, [slot]
//
// This is a classic example of independent reads of independent writes,
// aka IRIW. The question is if r1==r2==0 is allowed and for most HW the
// answer is yes without inserting a memory barriers between the st and the ld.
// These barriers are expensive so we have decided that we will
// always grey the ptr object regardless of the slot's color.
//go:nowritebarrier
func gcmarkwb_m(slot *uintptr, ptr uintptr) {
	switch gcphase {
	default:
		throw("gcphasework in bad gcphase")

	case _GCoff, _GCquiesce, _GCstw, _GCsweep, _GCscan:
		// ok

	case _GCmark, _GCmarktermination:
		if ptr != 0 && inheap(ptr) {
			shade(ptr)
		}
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
		scanstack(gp)
	case _GCmark:
		// No work.
	case _GCmarktermination:
		scanstack(gp)
		// All available mark work will be emptied before returning.
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
	if finq == nil || finq.cnt == int32(len(finq.fin)) {
		if finc == nil {
			// Note: write barrier here, assigning to finc, but should be okay.
			finc = (*finblock)(persistentalloc(_FinBlockSize, 0, &memstats.gc_sys))
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
					throw("finalizer out of sync")
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
	f := &finq.fin[finq.cnt]
	finq.cnt++
	f.fn = fn
	f.nret = nret
	f.fint = fint
	f.ot = ot
	f.arg = p
	fingwake = true
	unlock(&finlock)
}

//go:nowritebarrier
func iterate_finq(callback func(*funcval, unsafe.Pointer, uintptr, *_type, *ptrtype)) {
	for fb := allfin; fb != nil; fb = fb.alllink {
		for i := int32(0); i < fb.cnt; i++ {
			f := &fb.fin[i]
			callback(f.fn, f.arg, f.nret, f.fint, f.ot)
		}
	}
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
	if checkmark {
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

	var head, end gclinkptr

	c := _g_.m.mcache
	sweepgenset := false

	// Mark any free objects in this span so we don't collect them.
	for link := s.freelist; link.ptr() != nil; link = link.ptr().next {
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
				throw("can't preserve large span")
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
			if head.ptr() == nil {
				head = gclinkptr(p)
			} else {
				end.ptr().next = gclinkptr(p)
			}
			end = gclinkptr(p)
			end.ptr().next = gclinkptr(0x0bade5)
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
			throw("MSpan_Sweep: bad span state after sweep")
		}
		atomicstore(&s.sweepgen, sweepgen)
	}
	if nfree > 0 {
		c.local_nsmallfree[cl] += uintptr(nfree)
		c.local_cachealloc -= intptr(uintptr(nfree) * size)
		xadd64(&memstats.next_gc, -int64(nfree)*int64(size)*int64(gcpercent+100)/100)
		res = mCentral_FreeSpan(&mheap_.central[cl].mcentral, s, int32(nfree), head, end, preserve)
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

	// parallel mark for over GC roots
	parfordo(work.markfor)
	if gcphase != _GCscan {
		scanblock(0, 0, nil) // blocks in getfull
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

func gcinit() {
	if unsafe.Sizeof(workbuf{}) != _WorkbufSize {
		throw("runtime: size of Workbuf is suboptimal")
	}

	work.markfor = parforalloc(_MaxGcproc)
	gcpercent = readgogc()
	gcdatamask = unrollglobgcprog((*byte)(unsafe.Pointer(&gcdata)), uintptr(unsafe.Pointer(&edata))-uintptr(unsafe.Pointer(&data)))
	gcbssmask = unrollglobgcprog((*byte)(unsafe.Pointer(&gcbss)), uintptr(unsafe.Pointer(&ebss))-uintptr(unsafe.Pointer(&bss)))
}

// Called from malloc.go using onM, stopping and starting the world handled in caller.
//go:nowritebarrier
func gc_m(start_time int64, eagersweep bool) {
	_g_ := getg()
	gp := _g_.m.curg
	casgstatus(gp, _Grunning, _Gwaiting)
	gp.waitreason = "garbage collection"

	gc(start_time, eagersweep)
	casgstatus(gp, _Gwaiting, _Grunning)
}

// Similar to clearcheckmarkbits but works on a single span.
// It preforms two tasks.
// 1. When used before the checkmark phase it converts BitsDead (00) to bitsScalar (01)
//    for nibbles with the BoundaryBit set.
// 2. When used after the checkmark phase it converts BitsPointerMark (11) to BitsPointer 10 and
//    BitsScalarMark (00) to BitsScalar (01), thus clearing the checkmark mark encoding.
// For the second case it is possible to restore the BitsDead pattern but since
// clearmark is a debug tool performance has a lower priority than simplicity.
// The span is MSpanInUse and the world is stopped.
//go:nowritebarrier
func clearcheckmarkbitsspan(s *mspan) {
	if s.state != _MSpanInUse {
		print("runtime:clearcheckmarkbitsspan: state=", s.state, "\n")
		throw("clearcheckmarkbitsspan: bad span state")
	}

	arena_start := mheap_.arena_start
	cl := s.sizeclass
	size := s.elemsize
	var n int32
	if cl == 0 {
		n = 1
	} else {
		// Chunk full of small blocks
		npages := class_to_allocnpages[cl]
		n = npages << _PageShift / int32(size)
	}

	// MSpan_Sweep has similar code but instead of overloading and
	// complicating that routine we do a simpler walk here.
	// Sweep through n objects of given size starting at p.
	// This thread owns the span now, so it can manipulate
	// the block bitmap without atomic operations.
	p := uintptr(s.start) << _PageShift

	// Find bits for the beginning of the span.
	off := (p - arena_start) / ptrSize
	bitp := (*byte)(unsafe.Pointer(arena_start - off/wordsPerBitmapByte - 1))
	step := size / (ptrSize * wordsPerBitmapByte)

	// The type bit values are:
	//	00 - BitsDead, for us BitsScalarMarked
	//	01 - BitsScalar
	//	10 - BitsPointer
	//	11 - unused, for us BitsPointerMarked
	//
	// When called to prepare for the checkmark phase (checkmark==1),
	// we change BitsDead to BitsScalar, so that there are no BitsScalarMarked
	// type bits anywhere.
	//
	// The checkmark phase marks by changing BitsScalar to BitsScalarMarked
	// and BitsPointer to BitsPointerMarked.
	//
	// When called to clean up after the checkmark phase (checkmark==0),
	// we unmark by changing BitsScalarMarked back to BitsScalar and
	// BitsPointerMarked back to BitsPointer.
	//
	// There are two problems with the scheme as just described.
	// First, the setup rewrites BitsDead to BitsScalar, but the type bits
	// following a BitsDead are uninitialized and must not be used.
	// Second, objects that are free are expected to have their type
	// bits zeroed (BitsDead), so in the cleanup we need to restore
	// any BitsDeads that were there originally.
	//
	// In a one-word object (8-byte allocation on 64-bit system),
	// there is no difference between BitsScalar and BitsDead, because
	// neither is a pointer and there are no more words in the object,
	// so using BitsScalar during the checkmark is safe and mapping
	// both back to BitsDead during cleanup is also safe.
	//
	// In a larger object, we need to be more careful. During setup,
	// if the type of the first word is BitsDead, we change it to BitsScalar
	// (as we must) but also initialize the type of the second
	// word to BitsDead, so that a scan during the checkmark phase
	// will still stop before seeing the uninitialized type bits in the
	// rest of the object. The sequence 'BitsScalar BitsDead' never
	// happens in real type bitmaps - BitsDead is always as early
	// as possible, so immediately after the last BitsPointer.
	// During cleanup, if we see a BitsScalar, we can check to see if it
	// is followed by BitsDead. If so, it was originally BitsDead and
	// we can change it back.

	if step == 0 {
		// updating top and bottom nibbles, all boundaries
		for i := int32(0); i < n/2; i, bitp = i+1, addb(bitp, uintptrMask&-1) {
			if *bitp&bitBoundary == 0 {
				throw("missing bitBoundary")
			}
			b := (*bitp & bitPtrMask) >> 2
			if !checkmark && (b == _BitsScalar || b == _BitsScalarMarked) {
				*bitp &^= 0x0c // convert to _BitsDead
			} else if b == _BitsScalarMarked || b == _BitsPointerMarked {
				*bitp &^= _BitsCheckMarkXor << 2
			}

			if (*bitp>>gcBits)&bitBoundary == 0 {
				throw("missing bitBoundary")
			}
			b = ((*bitp >> gcBits) & bitPtrMask) >> 2
			if !checkmark && (b == _BitsScalar || b == _BitsScalarMarked) {
				*bitp &^= 0xc0 // convert to _BitsDead
			} else if b == _BitsScalarMarked || b == _BitsPointerMarked {
				*bitp &^= _BitsCheckMarkXor << (2 + gcBits)
			}
		}
	} else {
		// updating bottom nibble for first word of each object
		for i := int32(0); i < n; i, bitp = i+1, addb(bitp, -step) {
			if *bitp&bitBoundary == 0 {
				throw("missing bitBoundary")
			}
			b := (*bitp & bitPtrMask) >> 2

			if checkmark && b == _BitsDead {
				// move BitsDead into second word.
				// set bits to BitsScalar in preparation for checkmark phase.
				*bitp &^= 0xc0
				*bitp |= _BitsScalar << 2
			} else if !checkmark && (b == _BitsScalar || b == _BitsScalarMarked) && *bitp&0xc0 == 0 {
				// Cleaning up after checkmark phase.
				// First word is scalar or dead (we forgot)
				// and second word is dead.
				// First word might as well be dead too.
				*bitp &^= 0x0c
			} else if b == _BitsScalarMarked || b == _BitsPointerMarked {
				*bitp ^= _BitsCheckMarkXor << 2
			}
		}
	}
}

// clearcheckmarkbits preforms two tasks.
// 1. When used before the checkmark phase it converts BitsDead (00) to bitsScalar (01)
//    for nibbles with the BoundaryBit set.
// 2. When used after the checkmark phase it converts BitsPointerMark (11) to BitsPointer 10 and
//    BitsScalarMark (00) to BitsScalar (01), thus clearing the checkmark mark encoding.
// This is a bit expensive but preserves the BitsDead encoding during the normal marking.
// BitsDead remains valid for every nibble except the ones with BitsBoundary set.
//go:nowritebarrier
func clearcheckmarkbits() {
	for _, s := range work.spans {
		if s.state == _MSpanInUse {
			clearcheckmarkbitsspan(s)
		}
	}
}

// Called from malloc.go using onM.
// The world is stopped. Rerun the scan and mark phases
// using the bitMarkedCheck bit instead of the
// bitMarked bit. If the marking encounters an
// bitMarked bit that is not set then we throw.
//go:nowritebarrier
func gccheckmark_m(startTime int64, eagersweep bool) {
	if !gccheckmarkenable {
		return
	}

	if checkmark {
		throw("gccheckmark_m, entered with checkmark already true")
	}

	checkmark = true
	clearcheckmarkbits()        // Converts BitsDead to BitsScalar.
	gc_m(startTime, eagersweep) // turns off checkmark
	// Work done, fixed up the GC bitmap to remove the checkmark bits.
	clearcheckmarkbits()
}

//go:nowritebarrier
func gccheckmarkenable_m() {
	gccheckmarkenable = true
}

//go:nowritebarrier
func gccheckmarkdisable_m() {
	gccheckmarkenable = false
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
	oldphase := gcphase

	// Prepare flag indicating that the scan has not been completed.
	lock(&allglock)
	local_allglen := allglen
	for i := uintptr(0); i < local_allglen; i++ {
		gp := allgs[i]
		gp.gcworkdone = false // set to true in gcphasework
	}
	unlock(&allglock)

	work.nwait = 0
	work.ndone = 0
	work.nproc = 1 // For now do not do this in parallel.
	gcphase = _GCscan
	//	ackgcphase is not needed since we are not scanning running goroutines.
	parforsetup(work.markfor, work.nproc, uint32(_RootCount+local_allglen), nil, false, markroot)
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

	gcphase = oldphase
	casgstatus(mastergp, _Gwaiting, _Grunning)
	// Let the g that called us continue to run.
}

// Mark all objects that are known about.
//go:nowritebarrier
func gcmark_m() {
	scanblock(0, 0, nil)
}

// For now this must be bracketed with a stoptheworld and a starttheworld to ensure
// all go routines see the new barrier.
//go:nowritebarrier
func gcinstallmarkwb_m() {
	gcphase = _GCmark
}

// For now this must be bracketed with a stoptheworld and a starttheworld to ensure
// all go routines see the new barrier.
//go:nowritebarrier
func gcinstalloffwb_m() {
	gcphase = _GCoff
}

//TODO go:nowritebarrier
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

	if !checkmark {
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
	oldphase := gcphase

	work.nwait = 0
	work.ndone = 0
	work.nproc = uint32(gcprocs())
	gcphase = _GCmarktermination

	// World is stopped so allglen will not change.
	for i := uintptr(0); i < allglen; i++ {
		gp := allgs[i]
		gp.gcworkdone = false // set to true in gcphasework
	}

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

	if work.full != 0 {
		throw("work.full != 0")
	}
	if work.partial != 0 {
		throw("work.partial != 0")
	}

	gcphase = oldphase
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

	if gccheckmarkenable {
		if !checkmark {
			// first half of two-pass; don't set up sweep
			unlock(&mheap_.lock)
			return
		}
		checkmark = false // done checking marks
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
		throw("runtime: short slice passed to readGCStats")
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

//go:nowritebarrier
func addb(p *byte, n uintptr) *byte {
	return (*byte)(add(unsafe.Pointer(p), n))
}

// Recursively unrolls GC program in prog.
// mask is where to store the result.
// ppos is a pointer to position in mask, in bits.
// sparse says to generate 4-bits per word mask for heap (2-bits for data/bss otherwise).
//go:nowritebarrier
func unrollgcprog1(maskp *byte, prog *byte, ppos *uintptr, inplace, sparse bool) *byte {
	arena_start := mheap_.arena_start
	pos := *ppos
	mask := (*[1 << 30]byte)(unsafe.Pointer(maskp))
	for {
		switch *prog {
		default:
			throw("unrollgcprog: unknown instruction")

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
				throw("unrollgcprog: array does not end with insArrayEnd")
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
		throw("unrollglobgcprog: bad program size")
	}
	if *prog != insEnd {
		throw("unrollglobgcprog: program does not end with insEnd")
	}
	if mask[masksize] != 0xa1 {
		throw("unrollglobgcprog: overflow")
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
//go:nowritebarrier
func unrollgcprog_m(typ *_type) {
	lock(&unroll)
	mask := (*byte)(unsafe.Pointer(uintptr(typ.gc[0])))
	if *mask == 0 {
		pos := uintptr(8) // skip the unroll flag
		prog := (*byte)(unsafe.Pointer(uintptr(typ.gc[1])))
		prog = unrollgcprog1(mask, prog, &pos, false, true)
		if *prog != insEnd {
			throw("unrollgcprog: program does not end with insEnd")
		}
		if typ.size/ptrSize%2 != 0 {
			// repeat the program
			prog := (*byte)(unsafe.Pointer(uintptr(typ.gc[1])))
			unrollgcprog1(mask, prog, &pos, false, true)
		}

		// atomic way to say mask[0] = 1
		atomicor8(mask, 1)
	}
	unlock(&unroll)
}

// mark the span of memory at v as having n blocks of the given size.
// if leftover is true, there is left over space at the end of the span.
//go:nowritebarrier
func markspan(v unsafe.Pointer, size uintptr, n uintptr, leftover bool) {
	if uintptr(v)+size*n > mheap_.arena_used || uintptr(v) < mheap_.arena_start {
		throw("markspan: bad pointer")
	}

	// Find bits of the beginning of the span.
	off := (uintptr(v) - uintptr(mheap_.arena_start)) / ptrSize
	if off%wordsPerBitmapByte != 0 {
		throw("markspan: unaligned length")
	}
	b := mheap_.arena_start - off/wordsPerBitmapByte - 1

	// Okay to use non-atomic ops here, because we control
	// the entire span, and each bitmap byte has bits for only
	// one span, so no other goroutines are changing these bitmap words.

	if size == ptrSize {
		// Possible only on 64-bits (minimal size class is 8 bytes).
		// Set memory to 0x11.
		if (bitBoundary|bitsDead)<<gcBits|bitBoundary|bitsDead != 0x11 {
			throw("markspan: bad bits")
		}
		if n%(wordsPerBitmapByte*ptrSize) != 0 {
			throw("markspan: unaligned length")
		}
		b = b - n/wordsPerBitmapByte + 1 // find first byte
		if b%ptrSize != 0 {
			throw("markspan: unaligned pointer")
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
//go:nowritebarrier
func unmarkspan(v, n uintptr) {
	if v+n > mheap_.arena_used || v < mheap_.arena_start {
		throw("markspan: bad pointer")
	}

	off := (v - mheap_.arena_start) / ptrSize // word offset
	if off%(ptrSize*wordsPerBitmapByte) != 0 {
		throw("markspan: unaligned pointer")
	}

	b := mheap_.arena_start - off/wordsPerBitmapByte - 1
	n /= ptrSize
	if n%(ptrSize*wordsPerBitmapByte) != 0 {
		throw("unmarkspan: unaligned length")
	}

	// Okay to use non-atomic ops here, because we control
	// the entire span, and each bitmap word has bits for only
	// one span, so no other goroutines are changing these
	// bitmap words.
	n /= wordsPerBitmapByte
	memclr(unsafe.Pointer(b-n+1), n)
}

//go:nowritebarrier
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
