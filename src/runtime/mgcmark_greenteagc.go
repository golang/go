// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Green Tea mark algorithm
//
// The core idea behind Green Tea is simple: achieve better locality during
// mark/scan by delaying scanning so that we can accumulate objects to scan
// within the same span, then scan the objects that have accumulated on the
// span all together.
//
// By batching objects this way, we increase the chance that adjacent objects
// will be accessed, amortize the cost of accessing object metadata, and create
// better opportunities for prefetching. We can take this even further and
// optimize the scan loop by size class (not yet completed) all the way to the
// point of applying SIMD techniques to really tear through the heap.
//
// Naturally, this depends on being able to create opportunties to batch objects
// together. The basic idea here is to have two sets of mark bits. One set is the
// regular set of mark bits ("marks"), while the other essentially says that the
// objects have been scanned already ("scans"). When we see a pointer for the first
// time we set its mark and enqueue its span. We track these spans in work queues
// with a FIFO policy, unlike workbufs which have a LIFO policy. Empirically, a
// FIFO policy appears to work best for accumulating objects to scan on a span.
// Later, when we dequeue the span, we find both the union and intersection of the
// mark and scan bitsets. The union is then written back into the scan bits, while
// the intersection is used to decide which objects need scanning, such that the GC
// is still precise.
//
// Below is the bulk of the implementation, focusing on the worst case
// for locality, small objects. Specifically, those that are smaller than
// a few cache lines in size and whose metadata is stored the same way (at the
// end of the span).

//go:build goexperiment.greenteagc

package runtime

import (
	"internal/cpu"
	"internal/goarch"
	"internal/runtime/atomic"
	"internal/runtime/gc"
	"internal/runtime/gc/scan"
	"internal/runtime/sys"
	"unsafe"
)

const doubleCheckGreenTea = false

// spanInlineMarkBits are mark bits that are inlined into the span
// itself. gcUsesSpanInlineMarkBits may be used to check if objects
// of a particular size use inline mark bits.
//
// Inline mark bits are a little bit more than just mark bits. They
// consist of two parts: scans and marks. Marks are like pre-mark
// bits. They're set once a pointer to an object is discovered for
// the first time. The marks allow us to scan many objects in bulk
// if we queue the whole span for scanning. Before we scan such objects
// in bulk, we copy the marks to the scans, computing a diff along the
// way. The resulting bitmap tells us which objects we should scan.
//
// The inlineMarkBits also hold state sufficient for scanning any
// object in the span, as well as state for acquiring ownership of
// the span for queuing. This avoids the need to look at the mspan when
// scanning.
type spanInlineMarkBits struct {
	scans [63]uint8         // scanned bits.
	owned spanScanOwnership // see the comment on spanScanOwnership.
	marks [63]uint8         // mark bits.
	class spanClass
}

// spanScanOwnership indicates whether some thread has acquired
// the span for scanning, and whether there has been one or more
// attempts to acquire the span. The latter information helps to
// fast-track span scans that only apply to a single mark, skipping
// the relatively costly merge-and-diff process for scans and marks
// by allowing one to just set the mark directly.
type spanScanOwnership uint8

const (
	spanScanUnowned  spanScanOwnership = 0         // Indicates the span is not acquired for scanning.
	spanScanOneMark                    = 1 << iota // Indicates that only one mark bit is set relative to the scan bits.
	spanScanManyMark                               // Indicates one or more scan bits may be set relative to the mark bits.
	// "ManyMark" need not be exactly the value it has. In practice we just
	// want to distinguish "none" from "one" from "many," so a comparison is
	// sufficient (as opposed to a bit test) to check between these cases.
)

// load atomically loads from a pointer to a spanScanOwnership.
func (o *spanScanOwnership) load() spanScanOwnership {
	return spanScanOwnership(atomic.Load8((*uint8)(unsafe.Pointer(o))))
}

func (o *spanScanOwnership) or(v spanScanOwnership) spanScanOwnership {
	// N.B. We round down the address and use Or32 because Or8 doesn't
	// return a result, and it's strictly necessary for this protocol.
	//
	// Making Or8 return a result, while making the code look nicer, would
	// not be strictly better on any supported platform, as an Or8 that
	// returns a result is not a common instruction. On many platforms it
	// would be implemented exactly as it is here, and since Or8 is
	// exclusively used in the runtime and a hot function, we want to keep
	// using its no-result version elsewhere for performance.
	o32 := (*uint32)(unsafe.Pointer(uintptr(unsafe.Pointer(o)) &^ 0b11))
	off := (uintptr(unsafe.Pointer(o)) & 0b11) * 8
	if goarch.BigEndian {
		off = 32 - off - 8
	}
	return spanScanOwnership(atomic.Or32(o32, uint32(v)<<off) >> off)
}

func (imb *spanInlineMarkBits) init(class spanClass, needzero bool) {
	if imb == nil {
		// This nil check and throw is almost pointless. Normally we would
		// expect imb to never be nil. However, this is called on potentially
		// freshly-allocated virtual memory. As of 2025, the compiler-inserted
		// nil check is not a branch but a memory read that we expect to fault
		// if the pointer really is nil.
		//
		// However, this causes a read of the page, and operating systems may
		// take it as a hint to back the accessed memory with a read-only zero
		// page. However, we immediately write to this memory, which can then
		// force operating systems to have to update the page table and flush
		// the TLB, causing a lot of churn for programs that are short-lived
		// and monotonically grow in size.
		//
		// This nil check is thus an explicit branch instead of what the compiler
		// would insert circa 2025, which is a memory read instruction.
		//
		// See go.dev/issue/74375 for details.
		throw("runtime: span inline mark bits nil?")
	}
	if needzero {
		// Use memclrNoHeapPointers to avoid having the compiler make a worse
		// decision. We know that imb is both aligned and a nice power-of-two
		// size that works well for wider SIMD instructions. The compiler likely
		// has no idea that imb is aligned to 128 bytes.
		memclrNoHeapPointers(unsafe.Pointer(imb), unsafe.Sizeof(spanInlineMarkBits{}))
	}
	imb.class = class
}

// tryAcquire attempts to acquire the span for scanning. On success, the caller
// must queue the span for scanning or scan the span immediately.
func (imb *spanInlineMarkBits) tryAcquire() bool {
	switch imb.owned.load() {
	case spanScanUnowned:
		// Try to mark the span as having only one object marked.
		if imb.owned.or(spanScanOneMark) == spanScanUnowned {
			return true
		}
		// If we didn't see an old value of spanScanUnowned, then we must
		// have raced with someone else and seen spanScanOneMark or greater.
		// Fall through and try to set spanScanManyMark.
		fallthrough
	case spanScanOneMark:
		// We may be the first to set *any* bit on owned. In such a case,
		// we still need to make sure the span is queued.
		return imb.owned.or(spanScanManyMark) == spanScanUnowned
	}
	return false
}

// release releases the span for scanning, allowing another thread to queue the span.
//
// Returns an upper bound on the number of mark bits set since the span was queued. The
// upper bound is described as "one" (spanScanOneMark) or "many" (spanScanManyMark, with or
// without spanScanOneMark). If the return value indicates only one mark bit was set, the
// caller can be certain that it was the same mark bit that caused the span to get queued.
// Take note of the fact that this is *only* an upper-bound. In particular, it may still
// turn out that only one mark bit was set, even if the return value indicates "many".
func (imb *spanInlineMarkBits) release() spanScanOwnership {
	return spanScanOwnership(atomic.Xchg8((*uint8)(unsafe.Pointer(&imb.owned)), uint8(spanScanUnowned)))
}

// spanInlineMarkBitsFromBase returns the spanInlineMarkBits for a span whose start address is base.
//
// The span must be gcUsesSpanInlineMarkBits(span.elemsize).
func spanInlineMarkBitsFromBase(base uintptr) *spanInlineMarkBits {
	return (*spanInlineMarkBits)(unsafe.Pointer(base + gc.PageSize - unsafe.Sizeof(spanInlineMarkBits{})))
}

// initInlineMarkBits initializes the inlineMarkBits stored at the end of the span.
func (s *mspan) initInlineMarkBits() {
	if doubleCheckGreenTea && !gcUsesSpanInlineMarkBits(s.elemsize) {
		throw("expected span with inline mark bits")
	}
	// Zeroing is only necessary if this span wasn't just freshly allocated from the OS.
	s.inlineMarkBits().init(s.spanclass, s.needzero != 0)
}

// moveInlineMarks merges the span's inline mark bits into dst and clears them.
//
// gcUsesSpanInlineMarkBits(s.elemsize) must be true.
func (s *mspan) moveInlineMarks(dst *gcBits) {
	if doubleCheckGreenTea && !gcUsesSpanInlineMarkBits(s.elemsize) {
		throw("expected span with inline mark bits")
	}
	bytes := divRoundUp(uintptr(s.nelems), 8)
	imb := s.inlineMarkBits()
	imbMarks := (*gc.ObjMask)(unsafe.Pointer(&imb.marks))
	for i := uintptr(0); i < bytes; i += goarch.PtrSize {
		marks := bswapIfBigEndian(imbMarks[i/goarch.PtrSize])
		if i/goarch.PtrSize == uintptr(len(imb.marks)+1)/goarch.PtrSize-1 {
			marks &^= 0xff << ((goarch.PtrSize - 1) * 8) // mask out class
		}
		*(*uintptr)(unsafe.Pointer(dst.bytep(i))) |= bswapIfBigEndian(marks)
	}
	if doubleCheckGreenTea && !s.spanclass.noscan() && imb.marks != imb.scans {
		throw("marks don't match scans for span with pointer")
	}

	// Reset the inline mark bits.
	imb.init(s.spanclass, true /* We know these bits are always dirty now. */)
}

// inlineMarkBits returns the inline mark bits for the span.
//
// gcUsesSpanInlineMarkBits(s.elemsize) must be true.
func (s *mspan) inlineMarkBits() *spanInlineMarkBits {
	if doubleCheckGreenTea && !gcUsesSpanInlineMarkBits(s.elemsize) {
		throw("expected span with inline mark bits")
	}
	return spanInlineMarkBitsFromBase(s.base())
}

func (s *mspan) markBitsForIndex(objIndex uintptr) (bits markBits) {
	if gcUsesSpanInlineMarkBits(s.elemsize) {
		bits.bytep = &s.inlineMarkBits().marks[objIndex/8]
	} else {
		bits.bytep = s.gcmarkBits.bytep(objIndex / 8)
	}
	bits.mask = uint8(1) << (objIndex % 8)
	bits.index = objIndex
	return
}

func (s *mspan) markBitsForBase() markBits {
	if gcUsesSpanInlineMarkBits(s.elemsize) {
		return markBits{&s.inlineMarkBits().marks[0], uint8(1), 0}
	}
	return markBits{&s.gcmarkBits.x, uint8(1), 0}
}

// scannedBitsForIndex returns a markBits representing the scanned bit
// for objIndex in the inline mark bits.
func (s *mspan) scannedBitsForIndex(objIndex uintptr) markBits {
	return markBits{&s.inlineMarkBits().scans[objIndex/8], uint8(1) << (objIndex % 8), objIndex}
}

// gcUsesSpanInlineMarkBits returns true if a span holding objects of a certain size
// has inline mark bits. size must be the span's elemsize.
//
// nosplit because this is called from gcmarknewobject, which is nosplit.
//
//go:nosplit
func gcUsesSpanInlineMarkBits(size uintptr) bool {
	return heapBitsInSpan(size) && size >= 16
}

// tryDeferToSpanScan tries to queue p on the span it points to, if it
// points to a small object span (gcUsesSpanQueue size).
func tryDeferToSpanScan(p uintptr, gcw *gcWork) bool {
	if useCheckmark {
		return false
	}

	// Quickly to see if this is a span that has inline mark bits.
	ha := heapArenaOf(p)
	if ha == nil {
		return false
	}
	pageIdx := ((p / pageSize) / 8) % uintptr(len(ha.pageInUse))
	pageMask := byte(1 << ((p / pageSize) % 8))
	if ha.pageUseSpanInlineMarkBits[pageIdx]&pageMask == 0 {
		return false
	}

	// Find the object's index from the span class info stored in the inline mark bits.
	base := alignDown(p, gc.PageSize)
	q := spanInlineMarkBitsFromBase(base)
	objIndex := uint16((uint64(p-base) * uint64(gc.SizeClassToDivMagic[q.class.sizeclass()])) >> 32)

	// Set mark bit.
	idx, mask := objIndex/8, uint8(1)<<(objIndex%8)
	if atomic.Load8(&q.marks[idx])&mask != 0 {
		return true
	}
	atomic.Or8(&q.marks[idx], mask)

	// Fast-track noscan objects.
	if q.class.noscan() {
		gcw.bytesMarked += uint64(gc.SizeClassToSize[q.class.sizeclass()])
		return true
	}

	// Queue up the pointer (as a representative for its span).
	if q.tryAcquire() {
		if gcw.spanq.put(makeObjPtr(base, objIndex)) {
			if gcphase == _GCmark {
				gcw.mayNeedWorker = true
			}
			gcw.flushedWork = true
		}
	}
	return true
}

// tryGetSpan attempts to get an entire span to scan.
func (w *gcWork) tryGetSpan(slow bool) objptr {
	if s := w.spanq.get(); s != 0 {
		return s
	}

	if slow {
		// Check the global span queue.
		if s := work.spanq.get(w); s != 0 {
			return s
		}

		// Attempt to steal spans to scan from other Ps.
		return spanQueueSteal(w)
	}
	return 0
}

// spanQueue is a concurrent safe queue of mspans. Each mspan is represented
// as an objptr whose spanBase is the base address of the span.
type spanQueue struct {
	avail atomic.Bool      // optimization to check emptiness w/o the lock
	_     cpu.CacheLinePad // prevents false-sharing between lock and avail
	lock  mutex
	q     mSpanQueue
}

func (q *spanQueue) empty() bool {
	return !q.avail.Load()
}

func (q *spanQueue) size() int {
	return q.q.n
}

// putBatch adds a whole batch of spans to the queue.
func (q *spanQueue) putBatch(batch []objptr) {
	var list mSpanQueue
	for _, p := range batch {
		s := spanOfUnchecked(p.spanBase())
		s.scanIdx = p.objIndex()
		list.push(s)
	}

	lock(&q.lock)
	if q.q.n == 0 {
		q.avail.Store(true)
	}
	q.q.takeAll(&list)
	unlock(&q.lock)
}

// get tries to take a span off the queue.
//
// Returns a non-zero objptr on success. Also, moves additional
// spans to gcw's local span queue.
func (q *spanQueue) get(gcw *gcWork) objptr {
	if q.empty() {
		return 0
	}
	lock(&q.lock)
	if q.q.n == 0 {
		unlock(&q.lock)
		return 0
	}
	n := q.q.n/int(gomaxprocs) + 1
	if n > q.q.n {
		n = q.q.n
	}
	if max := len(gcw.spanq.ring) / 2; n > max {
		n = max
	}
	newQ := q.q.popN(n)
	if q.q.n == 0 {
		q.avail.Store(false)
	}
	unlock(&q.lock)

	s := newQ.pop()
	for newQ.n > 0 {
		s := newQ.pop()
		gcw.spanq.put(makeObjPtr(s.base(), s.scanIdx))
	}
	return makeObjPtr(s.base(), s.scanIdx)
}

// localSpanQueue is a P-local ring buffer of objptrs that represent spans.
// Accessed without a lock.
//
// Multi-consumer, single-producer. The only producer is the P that owns this
// queue, but any other P may consume from it.
//
// This is based on the scheduler runqueues. If making changes there, consider
// also making them here.
type localSpanQueue struct {
	head atomic.Uint32
	tail atomic.Uint32
	ring [256]objptr
}

// put adds s to the queue. Returns true if put flushed to the global queue
// because it was full.
func (q *localSpanQueue) put(s objptr) (flushed bool) {
	for {
		h := q.head.Load() // synchronize with consumers
		t := q.tail.Load()
		if t-h < uint32(len(q.ring)) {
			q.ring[t%uint32(len(q.ring))] = s
			q.tail.Store(t + 1) // Makes the item avail for consumption.
			return false
		}
		if q.putSlow(s, h, t) {
			return true
		}
		// The queue is not full, now the put above must succeed.
	}
}

// putSlow is a helper for put to move spans to the global queue.
// Returns true on success, false on failure (nothing moved).
func (q *localSpanQueue) putSlow(s objptr, h, t uint32) bool {
	var batch [len(q.ring)/2 + 1]objptr

	// First, grab a batch from local queue.
	n := t - h
	n = n / 2
	if n != uint32(len(q.ring)/2) {
		throw("localSpanQueue.putSlow: queue is not full")
	}
	for i := uint32(0); i < n; i++ {
		batch[i] = q.ring[(h+i)%uint32(len(q.ring))]
	}
	if !q.head.CompareAndSwap(h, h+n) { // Commits consume.
		return false
	}
	batch[n] = s

	work.spanq.putBatch(batch[:])
	return true
}

// get attempts to take a span off the queue. Might fail if the
// queue is empty. May be called by multiple threads, but callers
// are better off using stealFrom to amortize the cost of stealing.
// This method is intended for use by the owner of this queue.
func (q *localSpanQueue) get() objptr {
	for {
		h := q.head.Load()
		t := q.tail.Load()
		if t == h {
			return 0
		}
		s := q.ring[h%uint32(len(q.ring))]
		if q.head.CompareAndSwap(h, h+1) {
			return s
		}
	}
}

func (q *localSpanQueue) empty() bool {
	h := q.head.Load()
	t := q.tail.Load()
	return t == h
}

// stealFrom takes spans from q2 and puts them into q1. One span is removed
// from the stolen spans and returned on success. Failure to steal returns a
// zero objptr.
func (q1 *localSpanQueue) stealFrom(q2 *localSpanQueue) objptr {
	writeHead := q1.tail.Load()

	var n uint32
	for {
		h := q2.head.Load() // load-acquire, synchronize with other consumers
		t := q2.tail.Load() // load-acquire, synchronize with the producer
		n = t - h
		n = n - n/2
		if n == 0 {
			return 0
		}
		if n > uint32(len(q2.ring)/2) { // read inconsistent h and t
			continue
		}
		for i := uint32(0); i < n; i++ {
			c := q2.ring[(h+i)%uint32(len(q2.ring))]
			q1.ring[(writeHead+i)%uint32(len(q1.ring))] = c
		}
		if q2.head.CompareAndSwap(h, h+n) {
			break
		}
	}
	n--
	c := q1.ring[(writeHead+n)%uint32(len(q1.ring))]
	if n == 0 {
		return c
	}
	h := q1.head.Load()
	if writeHead-h+n >= uint32(len(q1.ring)) {
		throw("localSpanQueue.stealFrom: queue overflow")
	}
	q1.tail.Store(writeHead + n)
	return c
}

// drain moves all spans in the queue to the global queue.
//
// Returns true if anything was moved.
func (q *localSpanQueue) drain() bool {
	var batch [len(q.ring)]objptr

	var n uint32
	for {
		var h uint32
		for {
			h = q.head.Load()
			t := q.tail.Load()
			n = t - h
			if n == 0 {
				return false
			}
			if n <= uint32(len(q.ring)) {
				break
			}
			// Read inconsistent h and t.
		}
		for i := uint32(0); i < n; i++ {
			batch[i] = q.ring[(h+i)%uint32(len(q.ring))]
		}
		if q.head.CompareAndSwap(h, h+n) { // Commits consume.
			break
		}
	}
	if !q.empty() {
		throw("drained local span queue, but not empty")
	}

	work.spanq.putBatch(batch[:n])
	return true
}

// spanQueueSteal attempts to steal a span from another P's local queue.
//
// Returns a non-zero objptr on success.
func spanQueueSteal(gcw *gcWork) objptr {
	pp := getg().m.p.ptr()

	for enum := stealOrder.start(cheaprand()); !enum.done(); enum.next() {
		p2 := allp[enum.position()]
		if pp == p2 {
			continue
		}
		if s := gcw.spanq.stealFrom(&p2.gcw.spanq); s != 0 {
			return s
		}
	}
	return 0
}

// objptr consists of a span base and the index of the object in the span.
type objptr uintptr

// makeObjPtr creates an objptr from a span base address and an object index.
func makeObjPtr(spanBase uintptr, objIndex uint16) objptr {
	if doubleCheckGreenTea && spanBase&((1<<gc.PageShift)-1) != 0 {
		throw("created objptr with address that is incorrectly aligned")
	}
	return objptr(spanBase | uintptr(objIndex))
}

func (p objptr) spanBase() uintptr {
	return uintptr(p) &^ ((1 << gc.PageShift) - 1)
}

func (p objptr) objIndex() uint16 {
	return uint16(p) & ((1 << gc.PageShift) - 1)
}

// scanSpan scans objects indicated marks&^scans and then scans those objects,
// queuing the resulting pointers into gcw.
func scanSpan(p objptr, gcw *gcWork) {
	spanBase := p.spanBase()
	imb := spanInlineMarkBitsFromBase(spanBase)
	spanclass := imb.class
	if spanclass.noscan() {
		throw("noscan object in scanSpan")
	}
	elemsize := uintptr(gc.SizeClassToSize[spanclass.sizeclass()])

	// Release span.
	if imb.release() == spanScanOneMark {
		// Nobody else set any mark bits on this span while it was acquired.
		// That means p is the sole object we need to handle. Fast-track it.
		objIndex := p.objIndex()
		bytep := &imb.scans[objIndex/8]
		mask := uint8(1) << (objIndex % 8)
		if atomic.Load8(bytep)&mask != 0 {
			return
		}
		atomic.Or8(bytep, mask)
		gcw.bytesMarked += uint64(elemsize)
		if debug.gctrace > 1 {
			gcw.stats[spanclass.sizeclass()].sparseObjsScanned++
		}
		b := spanBase + uintptr(objIndex)*elemsize
		scanObjectSmall(spanBase, b, elemsize, gcw)
		return
	}

	// Compute nelems.
	divMagic := uint64(gc.SizeClassToDivMagic[spanclass.sizeclass()])
	usableSpanSize := uint64(gc.PageSize - unsafe.Sizeof(spanInlineMarkBits{}))
	if !spanclass.noscan() {
		usableSpanSize -= gc.PageSize / goarch.PtrSize / 8
	}
	nelems := uint16((usableSpanSize * divMagic) >> 32)

	// Grey objects and return if there's nothing else to do.
	var toScan gc.ObjMask
	objsMarked := spanSetScans(spanBase, nelems, imb, &toScan)
	if objsMarked == 0 {
		return
	}
	gcw.bytesMarked += uint64(objsMarked) * uint64(elemsize)

	// Check if we have enough density to make a dartboard scan
	// worthwhile. If not, just do what scanobject does, but
	// localized to the span, using the dartboard.
	if !scan.HasFastScanSpanPacked() || objsMarked < int(nelems/8) {
		if debug.gctrace > 1 {
			gcw.stats[spanclass.sizeclass()].spansSparseScanned++
			gcw.stats[spanclass.sizeclass()].spanObjsSparseScanned += uint64(objsMarked)
		}
		scanObjectsSmall(spanBase, elemsize, nelems, gcw, &toScan)
		return
	}

	// Scan the span.
	//
	// N.B. Use gcw.ptrBuf as the output buffer. This is a bit different
	// from scanObjectsSmall, which puts addresses to dereference. ScanSpanPacked
	// on the other hand, fills gcw.ptrBuf with already dereferenced pointers.
	nptrs := scan.ScanSpanPacked(
		unsafe.Pointer(spanBase),
		&gcw.ptrBuf[0],
		&toScan,
		uintptr(spanclass.sizeclass()),
		spanPtrMaskUnsafe(spanBase),
	)
	gcw.heapScanWork += int64(objsMarked) * int64(elemsize)

	if debug.gctrace > 1 {
		// Write down some statistics.
		gcw.stats[spanclass.sizeclass()].spansDenseScanned++
		gcw.stats[spanclass.sizeclass()].spanObjsDenseScanned += uint64(objsMarked)
	}

	// Process all the pointers we just got.
	for _, p := range gcw.ptrBuf[:nptrs] {
		if !tryDeferToSpanScan(p, gcw) {
			if obj, span, objIndex := findObject(p, 0, 0); obj != 0 {
				greyobject(obj, 0, 0, span, gcw, objIndex)
			}
		}
	}
}

// spanSetScans sets any unset mark bits that have their mark bits set in the inline mark bits.
//
// toScan is populated with bits indicating whether a particular mark bit was set.
//
// Returns the number of objects marked, which could be zero.
func spanSetScans(spanBase uintptr, nelems uint16, imb *spanInlineMarkBits, toScan *gc.ObjMask) int {
	arena, pageIdx, pageMask := pageIndexOf(spanBase)
	if arena.pageMarks[pageIdx]&pageMask == 0 {
		atomic.Or8(&arena.pageMarks[pageIdx], pageMask)
	}

	bytes := divRoundUp(uintptr(nelems), 8)
	objsMarked := 0

	// Careful: these two structures alias since ObjMask is much bigger
	// than marks or scans. We do these unsafe shenanigans so that we can
	// access the marks and scans by uintptrs rather than by byte.
	imbMarks := (*gc.ObjMask)(unsafe.Pointer(&imb.marks))
	imbScans := (*gc.ObjMask)(unsafe.Pointer(&imb.scans))

	// Iterate over one uintptr-sized chunks at a time, computing both
	// the union and intersection of marks and scans. Store the union
	// into scans, and the intersection into toScan.
	for i := uintptr(0); i < bytes; i += goarch.PtrSize {
		scans := atomic.Loaduintptr(&imbScans[i/goarch.PtrSize])
		marks := imbMarks[i/goarch.PtrSize]
		scans = bswapIfBigEndian(scans)
		marks = bswapIfBigEndian(marks)
		if i/goarch.PtrSize == uintptr(len(imb.marks)+1)/goarch.PtrSize-1 {
			scans &^= 0xff << ((goarch.PtrSize - 1) * 8) // mask out owned
			marks &^= 0xff << ((goarch.PtrSize - 1) * 8) // mask out class
		}
		toGrey := marks &^ scans
		toScan[i/goarch.PtrSize] = toGrey

		// If there's anything left to grey, do it.
		if toGrey != 0 {
			toGrey = bswapIfBigEndian(toGrey)
			if goarch.PtrSize == 4 {
				atomic.Or32((*uint32)(unsafe.Pointer(&imbScans[i/goarch.PtrSize])), uint32(toGrey))
			} else {
				atomic.Or64((*uint64)(unsafe.Pointer(&imbScans[i/goarch.PtrSize])), uint64(toGrey))
			}
		}
		objsMarked += sys.OnesCount64(uint64(toGrey))
	}
	return objsMarked
}

func scanObjectSmall(spanBase, b, objSize uintptr, gcw *gcWork) {
	ptrBits := heapBitsSmallForAddrInline(spanBase, b, objSize)
	gcw.heapScanWork += int64(sys.Len64(uint64(ptrBits)) * goarch.PtrSize)
	nptrs := 0
	n := sys.OnesCount64(uint64(ptrBits))
	for range n {
		k := sys.TrailingZeros64(uint64(ptrBits))
		ptrBits &^= 1 << k
		addr := b + uintptr(k)*goarch.PtrSize

		// Prefetch addr since we're about to use it. This point for prefetching
		// was chosen empirically.
		sys.Prefetch(addr)

		// N.B. ptrBuf is always large enough to hold pointers for an entire 1-page span.
		gcw.ptrBuf[nptrs] = addr
		nptrs++
	}

	// Process all the pointers we just got.
	for _, p := range gcw.ptrBuf[:nptrs] {
		p = *(*uintptr)(unsafe.Pointer(p))
		if p == 0 {
			continue
		}
		if !tryDeferToSpanScan(p, gcw) {
			if obj, span, objIndex := findObject(p, 0, 0); obj != 0 {
				greyobject(obj, 0, 0, span, gcw, objIndex)
			}
		}
	}
}

func scanObjectsSmall(base, objSize uintptr, elems uint16, gcw *gcWork, scans *gc.ObjMask) {
	nptrs := 0
	for i, bits := range scans {
		if i*(goarch.PtrSize*8) > int(elems) {
			break
		}
		n := sys.OnesCount64(uint64(bits))
		for range n {
			j := sys.TrailingZeros64(uint64(bits))
			bits &^= 1 << j

			b := base + uintptr(i*(goarch.PtrSize*8)+j)*objSize
			ptrBits := heapBitsSmallForAddrInline(base, b, objSize)
			gcw.heapScanWork += int64(sys.Len64(uint64(ptrBits)) * goarch.PtrSize)

			n := sys.OnesCount64(uint64(ptrBits))
			for range n {
				k := sys.TrailingZeros64(uint64(ptrBits))
				ptrBits &^= 1 << k
				addr := b + uintptr(k)*goarch.PtrSize

				// Prefetch addr since we're about to use it. This point for prefetching
				// was chosen empirically.
				sys.Prefetch(addr)

				// N.B. ptrBuf is always large enough to hold pointers for an entire 1-page span.
				gcw.ptrBuf[nptrs] = addr
				nptrs++
			}
		}
	}

	// Process all the pointers we just got.
	for _, p := range gcw.ptrBuf[:nptrs] {
		p = *(*uintptr)(unsafe.Pointer(p))
		if p == 0 {
			continue
		}
		if !tryDeferToSpanScan(p, gcw) {
			if obj, span, objIndex := findObject(p, 0, 0); obj != 0 {
				greyobject(obj, 0, 0, span, gcw, objIndex)
			}
		}
	}
}

func heapBitsSmallForAddrInline(spanBase, addr, elemsize uintptr) uintptr {
	hbitsBase, _ := spanHeapBitsRange(spanBase, gc.PageSize, elemsize)
	hbits := (*byte)(unsafe.Pointer(hbitsBase))

	// These objects are always small enough that their bitmaps
	// fit in a single word, so just load the word or two we need.
	//
	// Mirrors mspan.writeHeapBitsSmall.
	//
	// We should be using heapBits(), but unfortunately it introduces
	// both bounds checks panics and throw which causes us to exceed
	// the nosplit limit in quite a few cases.
	i := (addr - spanBase) / goarch.PtrSize / ptrBits
	j := (addr - spanBase) / goarch.PtrSize % ptrBits
	bits := elemsize / goarch.PtrSize
	word0 := (*uintptr)(unsafe.Pointer(addb(hbits, goarch.PtrSize*(i+0))))
	word1 := (*uintptr)(unsafe.Pointer(addb(hbits, goarch.PtrSize*(i+1))))

	var read uintptr
	if j+bits > ptrBits {
		// Two reads.
		bits0 := ptrBits - j
		bits1 := bits - bits0
		read = *word0 >> j
		read |= (*word1 & ((1 << bits1) - 1)) << bits0
	} else {
		// One read.
		read = (*word0 >> j) & ((1 << bits) - 1)
	}
	return read
}

// spanPtrMaskUnsafe returns the pointer mask for a span with inline mark bits.
//
// The caller must ensure spanBase is the base of a span that:
// - 1 page in size,
// - Uses inline mark bits,
// - Contains pointers.
func spanPtrMaskUnsafe(spanBase uintptr) *gc.PtrMask {
	base := spanBase + gc.PageSize - unsafe.Sizeof(gc.PtrMask{}) - unsafe.Sizeof(spanInlineMarkBits{})
	return (*gc.PtrMask)(unsafe.Pointer(base))
}

type sizeClassScanStats struct {
	spansDenseScanned     uint64 // Spans scanned with ScanSpanPacked.
	spanObjsDenseScanned  uint64 // Objects scanned with ScanSpanPacked.
	spansSparseScanned    uint64 // Spans scanned with scanObjectsSmall.
	spanObjsSparseScanned uint64 // Objects scanned with scanObjectsSmall.
	sparseObjsScanned     uint64 // Objects scanned with scanobject or scanObjectSmall.
	// Note: sparseObjsScanned is sufficient for both cases because
	// a particular size class either uses scanobject or scanObjectSmall,
	// not both. In the latter case, we also know that there was one
	// object scanned per span, so no need for a span counter.
}

func dumpScanStats() {
	var (
		spansDenseScanned     uint64
		spanObjsDenseScanned  uint64
		spansSparseScanned    uint64
		spanObjsSparseScanned uint64
		sparseObjsScanned     uint64
	)
	for _, stats := range memstats.lastScanStats {
		spansDenseScanned += stats.spansDenseScanned
		spanObjsDenseScanned += stats.spanObjsDenseScanned
		spansSparseScanned += stats.spansSparseScanned
		spanObjsSparseScanned += stats.spanObjsSparseScanned
		sparseObjsScanned += stats.sparseObjsScanned
	}
	totalObjs := sparseObjsScanned + spanObjsSparseScanned + spanObjsDenseScanned
	totalSpans := spansSparseScanned + spansDenseScanned
	print("scan: total ", sparseObjsScanned, "+", spanObjsSparseScanned, "+", spanObjsDenseScanned, "=", totalObjs, " objs")
	print(", ", spansSparseScanned, "+", spansDenseScanned, "=", totalSpans, " spans\n")
	for i, stats := range memstats.lastScanStats {
		if stats == (sizeClassScanStats{}) {
			continue
		}
		totalObjs := stats.sparseObjsScanned + stats.spanObjsSparseScanned + stats.spanObjsDenseScanned
		totalSpans := stats.spansSparseScanned + stats.spansDenseScanned
		if i == 0 {
			print("scan: class L ")
		} else {
			print("scan: class ", gc.SizeClassToSize[i], "B ")
		}
		print(stats.sparseObjsScanned, "+", stats.spanObjsSparseScanned, "+", stats.spanObjsDenseScanned, "=", totalObjs, " objs")
		print(", ", stats.spansSparseScanned, "+", stats.spansDenseScanned, "=", totalSpans, " spans\n")
	}
}

func (w *gcWork) flushScanStats(dst *[gc.NumSizeClasses]sizeClassScanStats) {
	for i := range w.stats {
		dst[i].spansDenseScanned += w.stats[i].spansDenseScanned
		dst[i].spanObjsDenseScanned += w.stats[i].spanObjsDenseScanned
		dst[i].spansSparseScanned += w.stats[i].spansSparseScanned
		dst[i].spanObjsSparseScanned += w.stats[i].spanObjsSparseScanned
		dst[i].sparseObjsScanned += w.stats[i].sparseObjsScanned
	}
	clear(w.stats[:])
}

// scanObject scans the object starting at b, adding pointers to gcw.
// b must point to the beginning of a heap object or an oblet.
// scanObject consults the GC bitmap for the pointer mask and the
// spans for the size of the object.
//
// Used only for !gcUsesSpanInlineMarkBits spans, but supports all
// object sizes and is safe to be called on all heap objects.
//
//go:nowritebarrier
func scanObject(b uintptr, gcw *gcWork) {
	// Prefetch object before we scan it.
	//
	// This will overlap fetching the beginning of the object with initial
	// setup before we start scanning the object.
	sys.Prefetch(b)

	// Find the bits for b and the size of the object at b.
	//
	// b is either the beginning of an object, in which case this
	// is the size of the object to scan, or it points to an
	// oblet, in which case we compute the size to scan below.
	s := spanOfUnchecked(b)
	n := s.elemsize
	if n == 0 {
		throw("scanObject n == 0")
	}
	if s.spanclass.noscan() {
		// Correctness-wise this is ok, but it's inefficient
		// if noscan objects reach here.
		throw("scanObject of a noscan object")
	}

	var tp typePointers
	if n > maxObletBytes {
		// Large object. Break into oblets for better
		// parallelism and lower latency.
		if b == s.base() {
			// Enqueue the other oblets to scan later.
			// Some oblets may be in b's scalar tail, but
			// these will be marked as "no more pointers",
			// so we'll drop out immediately when we go to
			// scan those.
			for oblet := b + maxObletBytes; oblet < s.base()+s.elemsize; oblet += maxObletBytes {
				if !gcw.putObjFast(oblet) {
					gcw.putObj(oblet)
				}
			}
		}

		// Compute the size of the oblet. Since this object
		// must be a large object, s.base() is the beginning
		// of the object.
		n = s.base() + s.elemsize - b
		n = min(n, maxObletBytes)
		tp = s.typePointersOfUnchecked(s.base())
		tp = tp.fastForward(b-tp.addr, b+n)
	} else {
		tp = s.typePointersOfUnchecked(b)
	}

	var scanSize uintptr
	for {
		var addr uintptr
		if tp, addr = tp.nextFast(); addr == 0 {
			if tp, addr = tp.next(b + n); addr == 0 {
				break
			}
		}

		// Keep track of farthest pointer we found, so we can
		// update heapScanWork. TODO: is there a better metric,
		// now that we can skip scalar portions pretty efficiently?
		scanSize = addr - b + goarch.PtrSize

		// Work here is duplicated in scanblock and above.
		// If you make changes here, make changes there too.
		obj := *(*uintptr)(unsafe.Pointer(addr))

		// At this point we have extracted the next potential pointer.
		// Quickly filter out nil and pointers back to the current object.
		if obj != 0 && obj-b >= n {
			// Test if obj points into the Go heap and, if so,
			// mark the object.
			//
			// Note that it's possible for findObject to
			// fail if obj points to a just-allocated heap
			// object because of a race with growing the
			// heap. In this case, we know the object was
			// just allocated and hence will be marked by
			// allocation itself.
			if !tryDeferToSpanScan(obj, gcw) {
				if obj, span, objIndex := findObject(obj, b, addr-b); obj != 0 {
					greyobject(obj, b, addr-b, span, gcw, objIndex)
				}
			}
		}
	}
	gcw.bytesMarked += uint64(n)
	gcw.heapScanWork += int64(scanSize)
	if debug.gctrace > 1 {
		gcw.stats[s.spanclass.sizeclass()].sparseObjsScanned++
	}
}
