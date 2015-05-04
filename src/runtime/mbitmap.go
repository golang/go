// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector: type and heap bitmaps.
//
// Stack, data, and bss bitmaps
//
// Stack frames and global variables in the data and bss sections are described
// by 1-bit bitmaps in which 0 means uninteresting and 1 means live pointer
// to be visited during GC.
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
// In each byte, the low 2 bits describe the first word, the next 2 bits describe
// the next word, and so on.
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
	bitPointer = 1
	bitMarked  = 2

	heapBitsWidth   = 2                             // heap bitmap bits to describe one pointer
	heapBitmapScale = ptrSize * (8 / heapBitsWidth) // number of data bytes described by one heap bitmap byte
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
func heapBitsForAddr(addr uintptr) heapBits {
	// 2 bits per work, 4 pairs per byte, and a mask is hard coded.
	off := (addr - mheap_.arena_start) / ptrSize
	return heapBits{(*uint8)(unsafe.Pointer(mheap_.arena_start - off/4 - 1)), uint32(2 * (off & 3))}
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
	if h.shift < 8-heapBitsWidth {
		return heapBits{h.bitp, h.shift + heapBitsWidth}
	}
	return heapBits{subtractb(h.bitp, 1), 0}
}

// forward returns the heapBits describing n pointer-sized words ahead of h in memory.
// That is, if h describes address p, h.forward(n) describes p+n*ptrSize.
// h.forward(1) is equivalent to h.next(), just slower.
// Note that forward does not modify h. The caller must record the result.
// bits returns the heap bits for the current word.
func (h heapBits) forward(n uintptr) heapBits {
	n += uintptr(h.shift) / heapBitsWidth
	return heapBits{subtractb(h.bitp, n/4), uint32(n%4) * heapBitsWidth}
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
	if b&(bitPointer|bitPointer<<heapBitsWidth) != 0 {
		return true
	}
	if size == 2*ptrSize {
		return false
	}
	// At least a 4-word object. Check scan bit (aka marked bit) in third word.
	if h.shift == 0 {
		return b&(bitMarked<<(2*heapBitsWidth)) != 0
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
	return (*h.bitp>>(heapBitsWidth+h.shift))&bitMarked != 0
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
	atomicor8(h.bitp, bitMarked<<(heapBitsWidth+h.shift))
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
			*bitp &^= bitPointer | bitPointer<<2 | bitPointer<<4 | bitPointer<<6
			bitp = subtractb(bitp, 1)
		}
		return
	}
	for i := uintptr(0); i < n; i++ {
		*h.bitp &^= bitMarked << (heapBitsWidth + h.shift)
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
			*bitp |= bitPointer | bitPointer<<2 | bitPointer<<4 | bitPointer<<6
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
			if x&(bitMarked<<2) != 0 {
				x &^= bitMarked << 2
			} else {
				x &^= bitPointer << 2
				f(base + (i+1)*ptrSize)
			}
			if x&(bitMarked<<4) != 0 {
				x &^= bitMarked << 4
			} else {
				x &^= bitPointer << 4
				f(base + (i+2)*ptrSize)
			}
			if x&(bitMarked<<6) != 0 {
				x &^= bitMarked << 6
			} else {
				x &^= bitPointer << 6
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
				x &^= 0x0f
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
			if x&(bitMarked<<4) != 0 {
				x &^= bitMarked << 4
			} else {
				x &^= 0xf0
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
func heapBitsSetType(x, size, dataSize uintptr, typ *_type) {
	// From here till marked label marking the object as allocated
	// and storing type info in the GC bitmap.
	h := heapBitsForAddr(x)

	var ptrmask *uint8
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
	if typ.kind&kindGCProg != 0 {
		nptr := (uintptr(typ.size) + ptrSize - 1) / ptrSize
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
		ptrmask = (*uint8)(unsafe.Pointer(uintptr(typ.gc[0])))
		// Check whether the program is already unrolled
		// by checking if the unroll flag byte is set
		maskword := uintptr(atomicloadp(unsafe.Pointer(ptrmask)))
		if *(*uint8)(unsafe.Pointer(&maskword)) == 0 {
			systemstack(func() {
				unrollgcprog_m(typ)
			})
		}
		ptrmask = (*uint8)(add(unsafe.Pointer(ptrmask), 1)) // skip the unroll flag byte
	} else {
		ptrmask = (*uint8)(unsafe.Pointer(typ.gc[0])) // pointer to unrolled mask
	}

	// Copy from 1-bit ptrmask into 2-bit bitmap.
	// If size is a multiple of 4 words, then the bitmap bytes for the object
	// are not shared with any other object and can be written directly.
	// On 64-bit systems, many sizes are only 16-byte aligned; half of
	// those are not multiples of 4 words (for example, 48/8 = 6 words);
	// those share either the leading byte or the trailing byte of their bitmaps
	// with another object.
	nptr := typ.size / ptrSize
	_ = nptr
	for i := uintptr(0); i < dataSize/ptrSize; i++ {
		atomicand8(h.bitp, ^((bitPointer | bitMarked) << h.shift))
		j := i % nptr
		if (*addb(ptrmask, j/8)>>(j%8))&1 != 0 {
			atomicor8(h.bitp, bitPointer<<h.shift)
		}
		if i >= 2 {
			atomicor8(h.bitp, bitMarked<<h.shift)
		}
		h = h.next()
	}
	if dataSize < size {
		atomicand8(h.bitp, ^((bitPointer | bitMarked) << h.shift))
	}
}

// ptrBitmapForType returns a bitmap indicating where pointers are
// in the memory representation of the type typ.
// The bit x[i/8]&(1<<(i%8)) is 1 if the i'th word in a value of type typ
// is a pointer.
func ptrBitmapForType(typ *_type) []uint8 {
	var ptrmask *uint8
	nptr := (uintptr(typ.size) + ptrSize - 1) / ptrSize
	if typ.kind&kindGCProg != 0 {
		masksize := (nptr + 7) / 8
		masksize++ // unroll flag in the beginning
		if masksize > maxGCMask && typ.gc[1] != 0 {
			// write barriers have not been updated to deal with this case yet.
			throw("maxGCMask too small for now")
		}
		ptrmask = (*uint8)(unsafe.Pointer(uintptr(typ.gc[0])))
		// Check whether the program is already unrolled
		// by checking if the unroll flag byte is set
		maskword := uintptr(atomicloadp(unsafe.Pointer(ptrmask)))
		if *(*uint8)(unsafe.Pointer(&maskword)) == 0 {
			systemstack(func() {
				unrollgcprog_m(typ)
			})
		}
		ptrmask = (*uint8)(add(unsafe.Pointer(ptrmask), 1)) // skip the unroll flag byte
	} else {
		ptrmask = (*uint8)(unsafe.Pointer(typ.gc[0])) // pointer to unrolled mask
	}
	return (*[1 << 30]byte)(unsafe.Pointer(ptrmask))[:(nptr+7)/8]
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
				mask[i/ptrSize] = 255
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
