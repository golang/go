// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector: type and heap bitmaps.
//
// Stack, data, and bss bitmaps
//
// Stack frames and global variables in the data and bss sections are described
// by 1-bit bitmaps in which 0 means uninteresting and 1 means live pointer
// to be visited during GC. The bits in each byte are consumed starting with
// the low bit: 1<<0, 1<<1, and so on.
//
// Heap bitmap
//
// The allocated heap comes from a subset of the memory in the range [start, used),
// where start == mheap_.arena_start and used == mheap_.arena_used.
// The heap bitmap comprises 2 bits for each pointer-sized word in that range,
// stored in bytes indexed backward in memory from start.
// That is, the byte at address start-1 holds the 2-bit entries for the four words
// start through start+3*ptrSize, the byte at start-2 holds the entries for
// start+4*ptrSize through start+7*ptrSize, and so on.
//
// In each 2-bit entry, the lower bit holds the same information as in the 1-bit
// bitmaps: 0 means uninteresting and 1 means live pointer to be visited during GC.
// The meaning of the high bit depends on the position of the word being described
// in its allocated object. In the first word, the high bit is the GC ``marked'' bit.
// In the second word, the high bit is the GC ``checkmarked'' bit (see below).
// In the third and later words, the high bit indicates that the object is still
// being described. In these words, if a bit pair with a high bit 0 is encountered,
// the low bit can also be assumed to be 0, and the object description is over.
// This 00 is called the ``dead'' encoding: it signals that the rest of the words
// in the object are uninteresting to the garbage collector.
//
// The 2-bit entries are split when written into the byte, so that the top half
// of the byte contains 4 mark bits and the bottom half contains 4 pointer bits.
// This form allows a copy from the 1-bit to the 4-bit form to keep the
// pointer bits contiguous, instead of having to space them out.
//
// The code makes use of the fact that the zero value for a heap bitmap
// has no live pointer bit set and is (depending on position), not marked,
// not checkmarked, and is the dead encoding.
// These properties must be preserved when modifying the encoding.
//
// Checkmarks
//
// In a concurrent garbage collector, one worries about failing to mark
// a live object due to mutations without write barriers or bugs in the
// collector implementation. As a sanity check, the GC has a 'checkmark'
// mode that retraverses the object graph with the world stopped, to make
// sure that everything that should be marked is marked.
// In checkmark mode, in the heap bitmap, the high bit of the 2-bit entry
// for the second word of the object holds the checkmark bit.
// When not in checkmark mode, this bit is set to 1.
//
// The smallest possible allocation is 8 bytes. On a 32-bit machine, that
// means every allocated object has two words, so there is room for the
// checkmark bit. On a 64-bit machine, however, the 8-byte allocation is
// just one word, so the second bit pair is not available for encoding the
// checkmark. However, because non-pointer allocations are combined
// into larger 16-byte (maxTinySize) allocations, a plain 8-byte allocation
// must be a pointer, so the type bit in the first word is not actually needed.
// It is still used in general, except in checkmark the type bit is repurposed
// as the checkmark bit and then reinitialized (to 1) as the type bit when
// finished.

package runtime

import "unsafe"

const (
	bitPointer = 1 << 0
	bitMarked  = 1 << 4

	heapBitsShift   = 1                 // shift offset between successive bitPointer or bitMarked entries
	heapBitmapScale = ptrSize * (8 / 2) // number of data bytes described by one heap bitmap byte

	// all mark/pointer bits in a byte
	bitMarkedAll  = bitMarked | bitMarked<<heapBitsShift | bitMarked<<(2*heapBitsShift) | bitMarked<<(3*heapBitsShift)
	bitPointerAll = bitPointer | bitPointer<<heapBitsShift | bitPointer<<(2*heapBitsShift) | bitPointer<<(3*heapBitsShift)
)

// addb returns the byte pointer p+n.
//go:nowritebarrier
func addb(p *byte, n uintptr) *byte {
	return (*byte)(add(unsafe.Pointer(p), n))
}

// subtractb returns the byte pointer p-n.
//go:nowritebarrier
func subtractb(p *byte, n uintptr) *byte {
	return (*byte)(add(unsafe.Pointer(p), -n))
}

// mHeap_MapBits is called each time arena_used is extended.
// It maps any additional bitmap memory needed for the new arena memory.
//
//go:nowritebarrier
func mHeap_MapBits(h *mheap) {
	// Caller has added extra mappings to the arena.
	// Add extra mappings of bitmap words as needed.
	// We allocate extra bitmap pieces in chunks of bitmapChunk.
	const bitmapChunk = 8192

	n := (mheap_.arena_used - mheap_.arena_start) / heapBitmapScale
	n = round(n, bitmapChunk)
	n = round(n, _PhysPageSize)
	if h.bitmap_mapped >= n {
		return
	}

	sysMap(unsafe.Pointer(h.arena_start-n), n-h.bitmap_mapped, h.arena_reserved, &memstats.gc_sys)
	h.bitmap_mapped = n
}

// heapBits provides access to the bitmap bits for a single heap word.
// The methods on heapBits take value receivers so that the compiler
// can more easily inline calls to those methods and registerize the
// struct fields independently.
type heapBits struct {
	bitp  *uint8
	shift uint32
}

// heapBitsForAddr returns the heapBits for the address addr.
// The caller must have already checked that addr is in the range [mheap_.arena_start, mheap_.arena_used).
//
// nosplit because it is used during write barriers and must not be preempted.
//go:nosplit
func heapBitsForAddr(addr uintptr) heapBits {
	// 2 bits per work, 4 pairs per byte, and a mask is hard coded.
	off := (addr - mheap_.arena_start) / ptrSize
	return heapBits{(*uint8)(unsafe.Pointer(mheap_.arena_start - off/4 - 1)), uint32(off & 3)}
}

// heapBitsForSpan returns the heapBits for the span base address base.
func heapBitsForSpan(base uintptr) (hbits heapBits) {
	if base < mheap_.arena_start || base >= mheap_.arena_end {
		throw("heapBitsForSpan: base out of range")
	}
	hbits = heapBitsForAddr(base)
	if hbits.shift != 0 {
		throw("heapBitsForSpan: unaligned start")
	}
	return hbits
}

// heapBitsForObject returns the base address for the heap object
// containing the address p, along with the heapBits for base.
// If p does not point into a heap object,
// return base == 0
// otherwise return the base of the object.
func heapBitsForObject(p uintptr) (base uintptr, hbits heapBits, s *mspan) {
	arenaStart := mheap_.arena_start
	if p < arenaStart || p >= mheap_.arena_used {
		return
	}
	off := p - arenaStart
	idx := off >> _PageShift
	// p points into the heap, but possibly to the middle of an object.
	// Consult the span table to find the block beginning.
	k := p >> _PageShift
	s = h_spans[idx]
	if s == nil || pageID(k) < s.start || p >= s.limit || s.state != mSpanInUse {
		if s == nil || s.state == _MSpanStack {
			// If s is nil, the virtual address has never been part of the heap.
			// This pointer may be to some mmap'd region, so we allow it.
			// Pointers into stacks are also ok, the runtime manages these explicitly.
			return
		}

		// The following ensures that we are rigorous about what data
		// structures hold valid pointers.
		// TODO(rsc): Check if this still happens.
		if false {
			// Still happens sometimes. We don't know why.
			printlock()
			print("runtime:objectstart Span weird: p=", hex(p), " k=", hex(k))
			if s == nil {
				print(" s=nil\n")
			} else {
				print(" s.start=", hex(s.start<<_PageShift), " s.limit=", hex(s.limit), " s.state=", s.state, "\n")
			}
			printunlock()
			throw("objectstart: bad pointer in unexpected span")
		}
		return
	}
	// If this span holds object of a power of 2 size, just mask off the bits to
	// the interior of the object. Otherwise use the size to get the base.
	if s.baseMask != 0 {
		// optimize for power of 2 sized objects.
		base = s.base()
		base = base + (p-base)&s.baseMask
		// base = p & s.baseMask is faster for small spans,
		// but doesn't work for large spans.
		// Overall, it's faster to use the more general computation above.
	} else {
		base = s.base()
		if p-base >= s.elemsize {
			// n := (p - base) / s.elemsize, using division by multiplication
			n := uintptr(uint64(p-base) >> s.divShift * uint64(s.divMul) >> s.divShift2)
			base += n * s.elemsize
		}
	}
	// Now that we know the actual base, compute heapBits to return to caller.
	hbits = heapBitsForAddr(base)
	return
}

// prefetch the bits.
func (h heapBits) prefetch() {
	prefetchnta(uintptr(unsafe.Pointer((h.bitp))))
}

// next returns the heapBits describing the next pointer-sized word in memory.
// That is, if h describes address p, h.next() describes p+ptrSize.
// Note that next does not modify h. The caller must record the result.
func (h heapBits) next() heapBits {
	if h.shift < 3*heapBitsShift {
		return heapBits{h.bitp, h.shift + heapBitsShift}
	}
	return heapBits{subtractb(h.bitp, 1), 0}
}

// forward returns the heapBits describing n pointer-sized words ahead of h in memory.
// That is, if h describes address p, h.forward(n) describes p+n*ptrSize.
// h.forward(1) is equivalent to h.next(), just slower.
// Note that forward does not modify h. The caller must record the result.
// bits returns the heap bits for the current word.
func (h heapBits) forward(n uintptr) heapBits {
	n += uintptr(h.shift) / heapBitsShift
	return heapBits{subtractb(h.bitp, n/4), uint32(n%4) * heapBitsShift}
}

// The caller can test isMarked and isPointer by &-ing with bitMarked and bitPointer.
// The result includes in its higher bits the bits for subsequent words
// described by the same bitmap byte.
func (h heapBits) bits() uint32 {
	return uint32(*h.bitp) >> h.shift
}

// isMarked reports whether the heap bits have the marked bit set.
// h must describe the initial word of the object.
func (h heapBits) isMarked() bool {
	return *h.bitp&(bitMarked<<h.shift) != 0
}

// setMarked sets the marked bit in the heap bits, atomically.
// h must describe the initial word of the object.
func (h heapBits) setMarked() {
	// Each byte of GC bitmap holds info for four words.
	// Might be racing with other updates, so use atomic update always.
	// We used to be clever here and use a non-atomic update in certain
	// cases, but it's not worth the risk.
	atomicor8(h.bitp, bitMarked<<h.shift)
}

// setMarkedNonAtomic sets the marked bit in the heap bits, non-atomically.
// h must describe the initial word of the object.
func (h heapBits) setMarkedNonAtomic() {
	*h.bitp |= bitMarked << h.shift
}

// isPointer reports whether the heap bits describe a pointer word.
// h must describe the initial word of the object.
func (h heapBits) isPointer() bool {
	return (*h.bitp>>h.shift)&bitPointer != 0
}

// hasPointers reports whether the given object has any pointers.
// It must be told how large the object at h is, so that it does not read too
// far into the bitmap.
// h must describe the initial word of the object.
func (h heapBits) hasPointers(size uintptr) bool {
	if size == ptrSize { // 1-word objects are always pointers
		return true
	}
	// Otherwise, at least a 2-word object, and at least 2-word aligned,
	// so h.shift is either 0 or 4, so we know we can get the bits for the
	// first two words out of *h.bitp.
	// If either of the first two words is a pointer, not pointer free.
	b := uint32(*h.bitp >> h.shift)
	if b&(bitPointer|bitPointer<<heapBitsShift) != 0 {
		return true
	}
	if size == 2*ptrSize {
		return false
	}
	// At least a 4-word object. Check scan bit (aka marked bit) in third word.
	if h.shift == 0 {
		return b&(bitMarked<<(2*heapBitsShift)) != 0
	}
	return uint32(*subtractb(h.bitp, 1))&bitMarked != 0
}

// isCheckmarked reports whether the heap bits have the checkmarked bit set.
// It must be told how large the object at h is, because the encoding of the
// checkmark bit varies by size.
// h must describe the initial word of the object.
func (h heapBits) isCheckmarked(size uintptr) bool {
	if size == ptrSize {
		return (*h.bitp>>h.shift)&bitPointer != 0
	}
	// All multiword objects are 2-word aligned,
	// so we know that the initial word's 2-bit pair
	// and the second word's 2-bit pair are in the
	// same heap bitmap byte, *h.bitp.
	return (*h.bitp>>(heapBitsShift+h.shift))&bitMarked != 0
}

// setCheckmarked sets the checkmarked bit.
// It must be told how large the object at h is, because the encoding of the
// checkmark bit varies by size.
// h must describe the initial word of the object.
func (h heapBits) setCheckmarked(size uintptr) {
	if size == ptrSize {
		atomicor8(h.bitp, bitPointer<<h.shift)
		return
	}
	atomicor8(h.bitp, bitMarked<<(heapBitsShift+h.shift))
}

// heapBitsBulkBarrier executes writebarrierptr_nostore
// for every pointer slot in the memory range [p, p+size),
// using the heap bitmap to locate those pointer slots.
// This executes the write barriers necessary after a memmove.
// Both p and size must be pointer-aligned.
// The range [p, p+size) must lie within a single allocation.
//
// Callers should call heapBitsBulkBarrier immediately after
// calling memmove(p, src, size). This function is marked nosplit
// to avoid being preempted; the GC must not stop the goroutine
// betwen the memmove and the execution of the barriers.
//go:nosplit
func heapBitsBulkBarrier(p, size uintptr) {
	if (p|size)&(ptrSize-1) != 0 {
		throw("heapBitsBulkBarrier: unaligned arguments")
	}
	if !writeBarrierEnabled || !inheap(p) {
		return
	}

	for i := uintptr(0); i < size; i += ptrSize {
		if heapBitsForAddr(p + i).isPointer() {
			x := (*uintptr)(unsafe.Pointer(p + i))
			writebarrierptr_nostore(x, *x)
		}
	}
}

// The methods operating on spans all require that h has been returned
// by heapBitsForSpan and that size, n, total are the span layout description
// returned by the mspan's layout method.
// If total > size*n, it means that there is extra leftover memory in the span,
// usually due to rounding.
//
// TODO(rsc): Perhaps introduce a different heapBitsSpan type.

// initSpan initializes the heap bitmap for a span.
func (h heapBits) initSpan(size, n, total uintptr) {
	if total%heapBitmapScale != 0 {
		throw("initSpan: unaligned length")
	}
	nbyte := total / heapBitmapScale
	memclr(unsafe.Pointer(subtractb(h.bitp, nbyte-1)), nbyte)
}

// initCheckmarkSpan initializes a span for being checkmarked.
// It clears the checkmark bits, which are set to 1 in normal operation.
func (h heapBits) initCheckmarkSpan(size, n, total uintptr) {
	// The ptrSize == 8 is a compile-time constant false on 32-bit and eliminates this code entirely.
	if ptrSize == 8 && size == ptrSize {
		// Checkmark bit is type bit, bottom bit of every 2-bit entry.
		// Only possible on 64-bit system, since minimum size is 8.
		// Must clear type bit (checkmark bit) of every word.
		// The type bit is the lower of every two-bit pair.
		bitp := h.bitp
		for i := uintptr(0); i < n; i += 4 {
			*bitp &^= bitPointerAll
			bitp = subtractb(bitp, 1)
		}
		return
	}
	for i := uintptr(0); i < n; i++ {
		*h.bitp &^= bitMarked << (heapBitsShift + h.shift)
		h = h.forward(size / ptrSize)
	}
}

// clearCheckmarkSpan undoes all the checkmarking in a span.
// The actual checkmark bits are ignored, so the only work to do
// is to fix the pointer bits. (Pointer bits are ignored by scanobject
// but consulted by typedmemmove.)
func (h heapBits) clearCheckmarkSpan(size, n, total uintptr) {
	// The ptrSize == 8 is a compile-time constant false on 32-bit and eliminates this code entirely.
	if ptrSize == 8 && size == ptrSize {
		// Checkmark bit is type bit, bottom bit of every 2-bit entry.
		// Only possible on 64-bit system, since minimum size is 8.
		// Must clear type bit (checkmark bit) of every word.
		// The type bit is the lower of every two-bit pair.
		bitp := h.bitp
		for i := uintptr(0); i < n; i += 4 {
			*bitp |= bitPointerAll
			bitp = subtractb(bitp, 1)
		}
	}
}

// heapBitsSweepSpan coordinates the sweeping of a span by reading
// and updating the corresponding heap bitmap entries.
// For each free object in the span, heapBitsSweepSpan sets the type
// bits for the first two words (or one for single-word objects) to typeDead
// and then calls f(p), where p is the object's base address.
// f is expected to add the object to a free list.
// For non-free objects, heapBitsSweepSpan turns off the marked bit.
func heapBitsSweepSpan(base, size, n uintptr, f func(uintptr)) {
	h := heapBitsForSpan(base)
	switch {
	default:
		throw("heapBitsSweepSpan")
	case size == ptrSize:
		// Consider mark bits in all four 2-bit entries of each bitmap byte.
		bitp := h.bitp
		for i := uintptr(0); i < n; i += 4 {
			x := uint32(*bitp)
			if x&bitMarked != 0 {
				x &^= bitMarked
			} else {
				x &^= bitPointer
				f(base + i*ptrSize)
			}
			if x&(bitMarked<<heapBitsShift) != 0 {
				x &^= bitMarked << heapBitsShift
			} else {
				x &^= bitPointer << heapBitsShift
				f(base + (i+1)*ptrSize)
			}
			if x&(bitMarked<<(2*heapBitsShift)) != 0 {
				x &^= bitMarked << (2 * heapBitsShift)
			} else {
				x &^= bitPointer << (2 * heapBitsShift)
				f(base + (i+2)*ptrSize)
			}
			if x&(bitMarked<<(3*heapBitsShift)) != 0 {
				x &^= bitMarked << (3 * heapBitsShift)
			} else {
				x &^= bitPointer << (3 * heapBitsShift)
				f(base + (i+3)*ptrSize)
			}
			*bitp = uint8(x)
			bitp = subtractb(bitp, 1)
		}

	case size%(4*ptrSize) == 0:
		// Mark bit is in first word of each object.
		// Each object starts at bit 0 of a heap bitmap byte.
		bitp := h.bitp
		step := size / heapBitmapScale
		for i := uintptr(0); i < n; i++ {
			x := uint32(*bitp)
			if x&bitMarked != 0 {
				x &^= bitMarked
			} else {
				x = 0
				f(base + i*size)
			}
			*bitp = uint8(x)
			bitp = subtractb(bitp, step)
		}

	case size%(4*ptrSize) == 2*ptrSize:
		// Mark bit is in first word of each object,
		// but every other object starts halfway through a heap bitmap byte.
		// Unroll loop 2x to handle alternating shift count and step size.
		bitp := h.bitp
		step := size / heapBitmapScale
		var i uintptr
		for i = uintptr(0); i < n; i += 2 {
			x := uint32(*bitp)
			if x&bitMarked != 0 {
				x &^= bitMarked
			} else {
				x &^= bitMarked | bitPointer | (bitMarked|bitPointer)<<heapBitsShift
				f(base + i*size)
				if size > 2*ptrSize {
					x = 0
				}
			}
			*bitp = uint8(x)
			if i+1 >= n {
				break
			}
			bitp = subtractb(bitp, step)
			x = uint32(*bitp)
			if x&(bitMarked<<(2*heapBitsShift)) != 0 {
				x &^= bitMarked << (2 * heapBitsShift)
			} else {
				x &^= (bitMarked|bitPointer)<<(2*heapBitsShift) | (bitMarked|bitPointer)<<(3*heapBitsShift)
				f(base + (i+1)*size)
				if size > 2*ptrSize {
					*subtractb(bitp, 1) = 0
				}
			}
			*bitp = uint8(x)
			bitp = subtractb(bitp, step+1)
		}
	}
}

// TODO(rsc): Clean up the next two functions.

// heapBitsSetType records that the new allocation [x, x+size)
// holds in [x, x+dataSize) one or more values of type typ.
// (The number of values is given by dataSize / typ.size.)
// If dataSize < size, the fragment [x+dataSize, x+size) is
// recorded as non-pointer data.
// It is known that the type has pointers somewhere;
// malloc does not call heapBitsSetType when there are no pointers,
// because all free objects are marked as noscan during
// heapBitsSweepSpan.
// There can only be one allocation from a given span active at a time,
// so this code is not racing with other instances of itself,
// and we don't allocate from a span until it has been swept,
// so this code is not racing with heapBitsSweepSpan.
// It is, however, racing with the concurrent GC mark phase,
// which can be setting the mark bit in the leading 2-bit entry
// of an allocated block. The block we are modifying is not quite
// allocated yet, so the GC marker is not racing with updates to x's bits,
// but if the start or end of x shares a bitmap byte with an adjacent
// object, the GC marker is racing with updates to those object's mark bits.
func heapBitsSetType(x, size, dataSize uintptr, typ *_type) {
	const doubleCheck = false // slow but helpful; enable to test modifications to this function

	// From here till marked label marking the object as allocated
	// and storing type info in the GC bitmap.
	h := heapBitsForAddr(x)

	// dataSize is always size rounded up to the next malloc size class,
	// except in the case of allocating a defer block, in which case
	// size is sizeof(_defer{}) (at least 6 words) and dataSize may be
	// arbitrarily larger.
	//
	// The checks for size == ptrSize and size == 2*ptrSize can therefore
	// assume that dataSize == size without checking it explicitly.

	if size == ptrSize {
		// It's one word and it has pointers, it must be a pointer.
		// The bitmap byte is shared with the one-word object
		// next to it, and concurrent GC might be marking that
		// object, so we must use an atomic update.
		// TODO(rsc): It may make sense to set all the pointer bits
		// when initializing the span, and then the atomicor8 here
		// goes away - heapBitsSetType would be a no-op
		// in that case.
		atomicor8(h.bitp, bitPointer<<h.shift)
		return
	}

	ptrmask := (*uint8)(unsafe.Pointer(typ.gc[0])) // pointer to unrolled mask
	if typ.kind&kindGCProg != 0 {
		nptr := typ.ptrdata / ptrSize
		masksize := (nptr + 7) / 8
		masksize++ // unroll flag in the beginning
		if masksize > maxGCMask && typ.gc[1] != 0 {
			// write barriers have not been updated to deal with this case yet.
			throw("maxGCMask too small for now")
			// If the mask is too large, unroll the program directly
			// into the GC bitmap. It's 7 times slower than copying
			// from the pre-unrolled mask, but saves 1/16 of type size
			// memory for the mask.
			systemstack(func() {
				unrollgcproginplace_m(unsafe.Pointer(x), typ, size, dataSize)
			})
			return
		}
		// Check whether the program is already unrolled
		// by checking if the unroll flag byte is set
		maskword := uintptr(atomicloadp(unsafe.Pointer(ptrmask)))
		if *(*uint8)(unsafe.Pointer(&maskword)) == 0 {
			systemstack(func() {
				unrollgcprog_m(typ)
			})
		}
		ptrmask = addb(ptrmask, 1) // skip the unroll flag byte
	}

	// Heap bitmap bits for 2-word object are only 4 bits,
	// so also shared with objects next to it; use atomic updates.
	// This is called out as a special case primarily for 32-bit systems,
	// so that on 32-bit systems the code below can assume all objects
	// are 4-word aligned (because they're all 16-byte aligned).
	if size == 2*ptrSize {
		if typ.size == ptrSize {
			// 2-element slice of pointer.
			atomicor8(h.bitp, (bitPointer|bitPointer<<heapBitsShift)<<h.shift)
			return
		}
		// Otherwise typ.size must be 2*ptrSize, and typ.kind&kindGCProg == 0.
		b := uint32(*ptrmask)
		hb := b & 3
		atomicor8(h.bitp, uint8(hb<<h.shift))
		return
	}

	// Copy from 1-bit ptrmask into 2-bit bitmap.
	// The basic approach is to use a single uintptr as a bit buffer,
	// alternating between reloading the buffer and writing bitmap bytes.
	// In general, one load can supply two bitmap byte writes.
	// This is a lot of lines of code, but it compiles into relatively few
	// machine instructions.

	// Ptrmask buffer.
	var (
		p     *byte   // last ptrmask byte read
		b     uintptr // ptrmask bits already loaded
		nb    uintptr // number of bits in b at next read
		endp  *byte   // final ptrmask byte to read (then repeat)
		endnb uintptr // number of valid bits in *endp
		pbits uintptr // alternate source of bits
	)

	// Note about sizes:
	//
	// typ.size is the number of words in the object,
	// and typ.ptrdata is the number of words in the prefix
	// of the object that contains pointers. That is, the final
	// typ.size - typ.ptrdata words contain no pointers.
	// This allows optimization of a common pattern where
	// an object has a small header followed by a large scalar
	// buffer. If we know the pointers are over, we don't have
	// to scan the buffer's heap bitmap at all.
	// The 1-bit ptrmasks are sized to contain only bits for
	// the typ.ptrdata prefix, zero padded out to a full byte
	// of bitmap. This code sets nw (below) so that heap bitmap
	// bits are only written for the typ.ptrdata prefix; if there is
	// more room in the allocated object, the next heap bitmap
	// entry is a 00, indicating that there are no more pointers
	// to scan. So only the ptrmask for the ptrdata bytes is needed.
	//
	// Replicated copies are not as nice: if there is an array of
	// objects with scalar tails, all but the last tail does have to
	// be initialized, because there is no way to say "skip forward".
	// However, because of the possibility of a repeated type with
	// size not a multiple of 4 pointers (one heap bitmap byte),
	// the code already must handle the last ptrmask byte specially
	// by treating it as containing only the bits for endnb pointers,
	// where endnb <= 4. We represent large scalar tails that must
	// be expanded in the replication by setting endnb larger than 4.
	// This will have the effect of reading many bits out of b,
	// but once the real bits are shifted out, b will supply as many
	// zero bits as we try to read, which is exactly what we need.

	p = ptrmask
	if typ.size < dataSize {
		// Filling in bits for an array of typ.
		// Set up for repetition of ptrmask during main loop.
		// Note that ptrmask describes only a prefix of
		const maxBits = ptrSize*8 - 7
		if typ.ptrdata/ptrSize <= maxBits {
			// Entire ptrmask fits in uintptr with room for a byte fragment.
			// Load into pbits and never read from ptrmask again.
			// This is especially important when the ptrmask has
			// fewer than 8 bits in it; otherwise the reload in the middle
			// of the Phase 2 loop would itself need to loop to gather
			// at least 8 bits.

			// Accumulate ptrmask into b.
			// ptrmask is sized to describe only typ.ptrdata, but we record
			// it as describing typ.size bytes, since all the high bits are zero.
			nb = typ.ptrdata / ptrSize
			for i := uintptr(0); i < nb; i += 8 {
				b |= uintptr(*p) << i
				p = addb(p, 1)
			}
			nb = typ.size / ptrSize

			// Replicate ptrmask to fill entire pbits uintptr.
			// Doubling and truncating is fewer steps than
			// iterating by nb each time. (nb could be 1.)
			// Since we loaded typ.ptrdata/ptrSize bits
			// but are pretending to have typ.size/ptrSize,
			// there might be no replication necessary/possible.
			pbits = b
			endnb = nb
			if nb+nb <= maxBits {
				for endnb <= ptrSize*8 {
					pbits |= pbits << endnb
					endnb += endnb
				}
				// Truncate to a multiple of original ptrmask.
				endnb = maxBits / nb * nb
				pbits &= 1<<endnb - 1
				b = pbits
				nb = endnb
			}

			// Clear p and endp as sentinel for using pbits.
			// Checked during Phase 2 loop.
			p = nil
			endp = nil
		} else {
			// Ptrmask is larger. Read it multiple times.
			n := (typ.ptrdata/ptrSize+7)/8 - 1
			endp = addb(ptrmask, n)
			endnb = typ.size/ptrSize - n*8
		}
	}
	if p != nil {
		b = uintptr(*p)
		p = addb(p, 1)
		nb = 8
	}

	var w uintptr  // words processed
	var nw uintptr // number of words to process
	if typ.size == dataSize {
		// Single entry: can stop once we reach the non-pointer data.
		nw = typ.ptrdata / ptrSize
	} else {
		// Repeated instances of typ in an array.
		// Have to process first N-1 entries in full, but can stop
		// once we reach the non-pointer data in the final entry.
		nw = ((dataSize/typ.size-1)*typ.size + typ.ptrdata) / ptrSize
	}
	if nw == 0 {
		// No pointers! Caller was supposed to check.
		println("runtime: invalid type ", *typ._string)
		throw("heapBitsSetType: called with non-pointer type")
		return
	}
	if nw < 2 {
		// Must write at least 2 words, because the "no scan"
		// encoding doesn't take effect until the third word.
		nw = 2
	}

	hbitp := h.bitp // next heap bitmap byte to write
	var hb uintptr  // bits being preapred for *h.bitp

	// Phase 1: Special case for leading byte (shift==0) or half-byte (shift==4).
	// The leading byte is special because it contains the bits for words 0 and 1,
	// which do not have the marked bits set.
	// The leading half-byte is special because it's a half a byte and must be
	// manipulated atomically.
	switch {
	default:
		throw("heapBitsSetType: unexpected shift")

	case h.shift == 0:
		// Ptrmask and heap bitmap are aligned.
		// Handle first byte of bitmap specially.
		// The first byte we write out contains the first two words of the object.
		// In those words, the mark bits are mark and checkmark, respectively,
		// and must not be set. In all following words, we want to set the mark bit
		// as a signal that the object continues to the next 2-bit entry in the bitmap.
		hb = b & bitPointerAll
		hb |= bitMarked<<(2*heapBitsShift) | bitMarked<<(3*heapBitsShift)
		if w += 4; w >= nw {
			goto Phase3
		}
		*hbitp = uint8(hb)
		hbitp = subtractb(hbitp, 1)
		b >>= 4
		nb -= 4

	case ptrSize == 8 && h.shift == 2:
		// Ptrmask and heap bitmap are misaligned.
		// The bits for the first two words are in a byte shared with another object
		// and must be updated atomically.
		// NOTE(rsc): The atomic here may not be necessary.
		// We took care of 1-word and 2-word objects above,
		// so this is at least a 6-word object, so our start bits
		// are shared only with the type bits of another object,
		// not with its mark bit. Since there is only one allocation
		// from a given span at a time, we should be able to set
		// these bits non-atomically. Not worth the risk right now.
		hb = (b & 3) << (2 * heapBitsShift)
		b >>= 2
		nb -= 2
		// Note: no bitMarker in hb because the first two words don't get markers from us.
		atomicor8(hbitp, uint8(hb))
		hbitp = subtractb(hbitp, 1)
		if w += 2; w >= nw {
			// We know that there is more data, because we handled 2-word objects above.
			// This must be at least a 6-word object. If we're out of pointer words,
			// mark no scan in next bitmap byte and finish.
			hb = 0
			w += 4
			goto Phase3
		}
	}

	// Phase 2: Full bytes in bitmap, up to but not including write to last byte (full or partial) in bitmap.
	// The loop computes the bits for that last write but does not execute the write;
	// it leaves the bits in hb for processing by phase 3.
	// To avoid repeated adjustment of nb, we subtract out the 4 bits we're going to
	// use in the first half of the loop right now, and then we only adjust nb explicitly
	// if the 8 bits used by each iteration isn't balanced by 8 bits loaded mid-loop.
	nb -= 4
	for {
		// Emit bitmap byte.
		// b has at least nb+4 bits, with one exception:
		// if w+4 >= nw, then b has only nw-w bits,
		// but we'll stop at the break and then truncate
		// appropriately in Phase 3.
		hb = b & bitPointerAll
		hb |= bitMarkedAll
		if w += 4; w >= nw {
			break
		}
		*hbitp = uint8(hb)
		hbitp = subtractb(hbitp, 1)
		b >>= 4

		// Load more bits. b has nb right now.
		if p != endp {
			// Fast path: keep reading from ptrmask.
			// nb unmodified: we just loaded 8 bits,
			// and the next iteration will consume 8 bits,
			// leaving us with the same nb the next time we're here.
			b |= uintptr(*p) << nb
			p = addb(p, 1)
		} else if p == nil {
			// Almost as fast path: track bit count and refill from pbits.
			// For short repetitions.
			if nb < 8 {
				b |= pbits << nb
				nb += endnb
			}
			nb -= 8 // for next iteration
		} else {
			// Slow path: reached end of ptrmask.
			// Process final partial byte and rewind to start.
			b |= uintptr(*p) << nb
			nb += endnb
			if nb < 8 {
				b |= uintptr(*ptrmask) << nb
				p = addb(ptrmask, 1)
			} else {
				nb -= 8
				p = ptrmask
			}
		}

		// Emit bitmap byte.
		hb = b & bitPointerAll
		hb |= bitMarkedAll
		if w += 4; w >= nw {
			break
		}
		*hbitp = uint8(hb)
		hbitp = subtractb(hbitp, 1)
		b >>= 4
	}

Phase3:
	// Phase 3: Write last byte or partial byte and zero the rest of the bitmap entries.
	if w > nw {
		// Counting the 4 entries in hb not yet written to memory,
		// there are more entries than possible pointer slots.
		// Discard the excess entries (can't be more than 3).
		mask := uintptr(1)<<(4-(w-nw)) - 1
		hb &= mask | mask<<4 // apply mask to both pointer bits and mark bits
	}

	// Change nw from counting possibly-pointer words to total words in allocation.
	nw = size / ptrSize

	// Write whole bitmap bytes.
	// The first is hb, the rest are zero.
	if w <= nw {
		*hbitp = uint8(hb)
		hbitp = subtractb(hbitp, 1)
		hb = 0 // for possible final half-byte below
		for w += 4; w <= nw; w += 4 {
			*hbitp = 0
			hbitp = subtractb(hbitp, 1)
		}
	}

	// Write final partial bitmap byte if any.
	// We know w > nw, or else we'd still be in the loop above.
	// It can be bigger only due to the 4 entries in hb that it counts.
	// If w == nw+4 then there's nothing left to do: we wrote all nw entries
	// and can discard the 4 sitting in hb.
	// But if w == nw+2, we need to write first two in hb.
	// The byte is shared with the next object so we may need an atomic.
	if w == nw+2 {
		if gcphase == _GCoff {
			*hbitp = *hbitp&^(bitPointer|bitMarked|(bitPointer|bitMarked)<<heapBitsShift) | uint8(hb)
		} else {
			atomicand8(hbitp, ^uint8(bitPointer|bitMarked|(bitPointer|bitMarked)<<heapBitsShift))
			atomicor8(hbitp, uint8(hb))
		}
	}

	// Phase 4: all done, but perhaps double check.
	if doubleCheck {
		end := heapBitsForAddr(x + size)
		if hbitp != end.bitp || (w == nw+2) != (end.shift == 2) {
			println("ended at wrong bitmap byte for", *typ._string, "x", dataSize/typ.size)
			print("typ.size=", typ.size, " typ.ptrdata=", typ.ptrdata, " dataSize=", dataSize, " size=", size, "\n")
			print("w=", w, " nw=", nw, " b=", hex(b), " nb=", nb, " hb=", hex(hb), "\n")
			h0 := heapBitsForAddr(x)
			print("initial bits h0.bitp=", h0.bitp, " h0.shift=", h0.shift, "\n")
			print("ended at hbitp=", hbitp, " but next starts at bitp=", end.bitp, " shift=", end.shift, "\n")
			throw("bad heapBitsSetType")
		}

		// Double-check that bits to be written were written correctly.
		// Does not check that other bits were not written, unfortunately.
		h := heapBitsForAddr(x)
		nptr := typ.ptrdata / ptrSize
		ndata := typ.size / ptrSize
		count := dataSize / typ.size
		for i := uintptr(0); i <= dataSize/ptrSize; i++ {
			j := i % ndata
			var have, want uint8
			if i == dataSize/ptrSize && dataSize >= size {
				break
			}
			have = (*h.bitp >> h.shift) & (bitPointer | bitMarked)
			if i == dataSize/ptrSize || i/ndata == count-1 && j >= nptr {
				want = 0 // dead marker
			} else {
				if j < nptr && (*addb(ptrmask, j/8)>>(j%8))&1 != 0 {
					want |= bitPointer
				}
				if i >= 2 {
					want |= bitMarked
				} else {
					have &^= bitMarked
				}
			}
			if have != want {
				println("mismatch writing bits for", *typ._string, "x", dataSize/typ.size)
				print("typ.size=", typ.size, " typ.ptrdata=", typ.ptrdata, " dataSize=", dataSize, " size=", size, "\n")
				print("w=", w, " nw=", nw, " b=", hex(b), " nb=", nb, " hb=", hex(hb), "\n")
				h0 := heapBitsForAddr(x)
				print("initial bits h0.bitp=", h0.bitp, " h0.shift=", h0.shift, "\n")
				print("current bits h.bitp=", h.bitp, " h.shift=", h.shift, " *h.bitp=", hex(*h.bitp), "\n")
				print("ptrmask=", ptrmask, " p=", p, " endp=", endp, " endnb=", endnb, " pbits=", hex(pbits), " b=", hex(b), " nb=", nb, "\n")
				println("at word", i, "offset", i*ptrSize, "have", have, "want", want)
				throw("bad heapBitsSetType")
			}
			h = h.next()
		}
	}
}

// GC type info programs
//
// TODO(rsc): Clean up and enable.

const (
	// GC type info programs.
	// The programs allow to store type info required for GC in a compact form.
	// Most importantly arrays take O(1) space instead of O(n).
	// The program grammar is:
	//
	// Program = {Block} "insEnd"
	// Block = Data | Array
	// Data = "insData" DataSize DataBlock
	// DataSize = int // size of the DataBlock in bit pairs, 1 byte
	// DataBlock = binary // dense GC mask (2 bits per word) of size ]DataSize/4[ bytes
	// Array = "insArray" ArrayLen Block "insArrayEnd"
	// ArrayLen = int // length of the array, 8 bytes (4 bytes for 32-bit arch)
	//
	// Each instruction (insData, insArray, etc) is 1 byte.
	// For example, for type struct { x []byte; y [20]struct{ z int; w *byte }; }
	// the program looks as:
	//
	// insData 3 (typePointer typeScalar typeScalar)
	//	insArray 20 insData 2 (typeScalar typePointer) insArrayEnd insEnd
	//
	// Total size of the program is 17 bytes (13 bytes on 32-bits).
	// The corresponding GC mask would take 43 bytes (it would be repeated
	// because the type has odd number of words).
	insData = 1 + iota
	insArray
	insArrayEnd
	insEnd

	// 64 bytes cover objects of size 1024/512 on 64/32 bits, respectively.
	maxGCMask = 65536 // TODO(rsc): change back to 64
)

// Recursively unrolls GC program in prog.
// mask is where to store the result.
// If inplace is true, store the result not in mask but in the heap bitmap for mask.
// ppos is a pointer to position in mask, in bits.
// sparse says to generate 4-bits per word mask for heap (1-bit for data/bss otherwise).
//go:nowritebarrier
func unrollgcprog1(maskp *byte, prog *byte, ppos *uintptr, inplace bool) *byte {
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
				v := p[i/8] >> (uint(i) % 8) & 1
				if inplace {
					throw("gc inplace")
					const typeShift = 2
					// Store directly into GC bitmap.
					h := heapBitsForAddr(uintptr(unsafe.Pointer(&mask[pos])))
					if h.shift == 0 {
						*h.bitp = v << typeShift
					} else {
						*h.bitp |= v << (4 + typeShift)
					}
					pos += ptrSize
				} else {
					// 1 bit per word, for data/bss bitmap
					mask[pos/8] |= v << (pos % 8)
					pos++
				}
			}
			prog = addb(prog, (uintptr(siz)+7)/8)

		case insArray:
			prog = (*byte)(add(unsafe.Pointer(prog), 1))
			siz := uintptr(0)
			for i := uintptr(0); i < ptrSize; i++ {
				siz = (siz << 8) + uintptr(*(*byte)(add(unsafe.Pointer(prog), ptrSize-i-1)))
			}
			prog = (*byte)(add(unsafe.Pointer(prog), ptrSize))
			var prog1 *byte
			for i := uintptr(0); i < siz; i++ {
				prog1 = unrollgcprog1(&mask[0], prog, &pos, inplace)
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

// Unrolls GC program prog for data/bss, returns 1-bit GC mask.
func unrollglobgcprog(prog *byte, size uintptr) bitvector {
	masksize := round(round(size, ptrSize)/ptrSize, 8) / 8
	mask := (*[1 << 30]byte)(persistentalloc(masksize+1, 0, &memstats.gc_sys))
	mask[masksize] = 0xa1
	pos := uintptr(0)
	prog = unrollgcprog1(&mask[0], prog, &pos, false)
	if pos != size/ptrSize {
		print("unrollglobgcprog: bad program size, got ", pos, ", expect ", size/ptrSize, "\n")
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
	throw("unrollinplace")
	// TODO(rsc): Update for 1-bit bitmaps.
	// TODO(rsc): Explain why these non-atomic updates are okay.
	pos := uintptr(0)
	prog := (*byte)(unsafe.Pointer(uintptr(typ.gc[1])))
	for pos != size0 {
		unrollgcprog1((*byte)(v), prog, &pos, true)
	}

	// Mark first word as bitAllocated.
	// Mark word after last as typeDead.
	if size0 < size {
		h := heapBitsForAddr(uintptr(v) + size0)
		const typeMask = 0
		const typeShift = 0
		*h.bitp &^= typeMask << typeShift
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
		prog = unrollgcprog1(mask, prog, &pos, false)
		if *prog != insEnd {
			throw("unrollgcprog: program does not end with insEnd")
		}
		// atomic way to say mask[0] = 1
		atomicor8(mask, 1)
	}
	unlock(&unroll)
}

// Testing.

func getgcmaskcb(frame *stkframe, ctxt unsafe.Pointer) bool {
	target := (*stkframe)(ctxt)
	if frame.sp <= target.sp && target.sp < frame.varp {
		*target = *frame
		return false
	}
	return true
}

// Returns GC type info for object p for testing.
func getgcmask(ep interface{}) (mask []byte) {
	e := *(*eface)(unsafe.Pointer(&ep))
	p := e.data
	t := e._type
	// data or bss
	for datap := &firstmoduledata; datap != nil; datap = datap.next {
		// data
		if datap.data <= uintptr(p) && uintptr(p) < datap.edata {
			bitmap := datap.gcdatamask.bytedata
			n := (*ptrtype)(unsafe.Pointer(t)).elem.size
			mask = make([]byte, n/ptrSize)
			for i := uintptr(0); i < n; i += ptrSize {
				off := (uintptr(p) + i - datap.data) / ptrSize
				mask[i/ptrSize] = (*addb(bitmap, off/8) >> (off % 8)) & 1
			}
			return
		}

		// bss
		if datap.bss <= uintptr(p) && uintptr(p) < datap.ebss {
			bitmap := datap.gcbssmask.bytedata
			n := (*ptrtype)(unsafe.Pointer(t)).elem.size
			mask = make([]byte, n/ptrSize)
			for i := uintptr(0); i < n; i += ptrSize {
				off := (uintptr(p) + i - datap.bss) / ptrSize
				mask[i/ptrSize] = (*addb(bitmap, off/8) >> (off % 8)) & 1
			}
			return
		}
	}

	// heap
	var n uintptr
	var base uintptr
	if mlookup(uintptr(p), &base, &n, nil) != 0 {
		mask = make([]byte, n/ptrSize)
		for i := uintptr(0); i < n; i += ptrSize {
			hbits := heapBitsForAddr(base + i)
			if hbits.isPointer() {
				mask[i/ptrSize] = 1
			}
			if i >= 2*ptrSize && !hbits.isMarked() {
				mask = mask[:i/ptrSize]
				break
			}
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
		size := uintptr(bv.n) * ptrSize
		n := (*ptrtype)(unsafe.Pointer(t)).elem.size
		mask = make([]byte, n/ptrSize)
		for i := uintptr(0); i < n; i += ptrSize {
			bitmap := bv.bytedata
			off := (uintptr(p) + i - frame.varp + size) / ptrSize
			mask[i/ptrSize] = (*addb(bitmap, off/8) >> (off % 8)) & 1
		}
	}
	return
}
