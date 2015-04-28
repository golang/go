// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector: type and heap bitmaps.
//
// Stack, data, and bss bitmaps
//
// Not handled in this file, but worth mentioning: stack frames and global data
// in the data and bss sections are described by 1-bit bitmaps in which 0 means
// scalar or uninitialized or dead and 1 means pointer to visit during GC.
//
// Comparing this 1-bit form with the 2-bit form described below, 0 represents
// both the 2-bit 00 and 01, while 1 represents the 2-bit 10.
// Therefore conversions between the two (until the 2-bit form is gone)
// can be done by x>>1 for 2-bit to 1-bit and x+1 for 1-bit to 2-bit.
//
// Type bitmaps
//
// Types that aren't too large
// record information about the layout of their memory words using a type bitmap.
// The bitmap holds two bits for each pointer-sized word. The two-bit values are:
//
// 	00 - typeDead: not a pointer, and no pointers in the rest of the object
//	01 - typeScalar: not a pointer
//	10 - typePointer: a pointer that GC should trace
//	11 - unused
//
// typeDead only appears in type bitmaps in Go type descriptors
// and in type bitmaps embedded in the heap bitmap (see below).
//
// Heap bitmap
//
// The allocated heap comes from a subset of the memory in the range [start, used),
// where start == mheap_.arena_start and used == mheap_.arena_used.
// The heap bitmap comprises 4 bits for each pointer-sized word in that range,
// stored in bytes indexed backward in memory from start.
// That is, the byte at address start-1 holds the 4-bit entries for the two words
// start, start+ptrSize, the byte at start-2 holds the entries for start+2*ptrSize,
// start+3*ptrSize, and so on.
// In the byte holding the entries for addresses p and p+ptrSize, the low 4 bits
// describe p and the high 4 bits describe p+ptrSize.
//
// The 4 bits for each word are:
//	0001 - not used
//	0010 - bitMarked: this object has been marked by GC
//	tt00 - word type bits, as in a type bitmap.
//
// The code makes use of the fact that the zero value for a heap bitmap nibble
// has no boundary bit set, no marked bit set, and type bits == typeDead.
// These properties must be preserved when modifying the encoding.
//
// Checkmarks
//
// In a concurrent garbage collector, one worries about failing to mark
// a live object due to mutations without write barriers or bugs in the
// collector implementation. As a sanity check, the GC has a 'checkmark'
// mode that retraverses the object graph with the world stopped, to make
// sure that everything that should be marked is marked.
// In checkmark mode, in the heap bitmap, the type bits for the first word
// of an object are redefined:
//
//	00 - typeScalarCheckmarked // typeScalar, checkmarked
//	01 - typeScalar // typeScalar, not checkmarked
//	10 - typePointer // typePointer, not checkmarked
//	11 - typePointerCheckmarked // typePointer, checkmarked
//
// That is, typeDead is redefined to be typeScalar + a checkmark, and the
// previously unused 11 pattern is redefined to be typePointer + a checkmark.
// To prepare for this mode, we must move any typeDead in the first word of
// a multiword object to the second word.

package runtime

import "unsafe"

const (
	typeDead               = 0
	typeScalarCheckmarked  = 0
	typeScalar             = 1
	typePointer            = 2
	typePointerCheckmarked = 3

	typeBitsWidth = 2 // # of type bits per pointer-sized word
	typeMask      = 1<<typeBitsWidth - 1

	heapBitsWidth   = 4
	heapBitmapScale = ptrSize * (8 / heapBitsWidth) // number of data bytes per heap bitmap byte
	bitMarked       = 2
	typeShift       = 2
)

// Information from the compiler about the layout of stack frames.
type bitvector struct {
	n        int32 // # of bits
	bytedata *uint8
}

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
	off := (addr - mheap_.arena_start) / ptrSize
	return heapBits{(*uint8)(unsafe.Pointer(mheap_.arena_start - off/2 - 1)), uint32(4 * (off & 1))}
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
	if h.shift == 0 {
		return heapBits{h.bitp, 4}
	}
	return heapBits{subtractb(h.bitp, 1), 0}
}

// isMarked reports whether the heap bits have the marked bit set.
func (h heapBits) isMarked() bool {
	return *h.bitp&(bitMarked<<h.shift) != 0
}

// setMarked sets the marked bit in the heap bits, atomically.
func (h heapBits) setMarked() {
	// Each byte of GC bitmap holds info for two words.
	// Might be racing with other updates, so use atomic update always.
	// We used to be clever here and use a non-atomic update in certain
	// cases, but it's not worth the risk.
	atomicor8(h.bitp, bitMarked<<h.shift)
}

// setMarkedNonAtomic sets the marked bit in the heap bits, non-atomically.
func (h heapBits) setMarkedNonAtomic() {
	*h.bitp |= bitMarked << h.shift
}

// typeBits returns the heap bits' type bits.
func (h heapBits) typeBits() uint8 {
	return (*h.bitp >> (h.shift + typeShift)) & typeMask
}

// isCheckmarked reports whether the heap bits have the checkmarked bit set.
func (h heapBits) isCheckmarked() bool {
	typ := h.typeBits()
	return typ == typeScalarCheckmarked || typ == typePointerCheckmarked
}

// setCheckmarked sets the checkmarked bit.
func (h heapBits) setCheckmarked() {
	typ := h.typeBits()
	if typ == typeScalar {
		// Clear low type bit to turn 01 into 00.
		atomicand8(h.bitp, ^((1 << typeShift) << h.shift))
	} else if typ == typePointer {
		// Set low type bit to turn 10 into 11.
		atomicor8(h.bitp, (1<<typeShift)<<h.shift)
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
// This would be a no-op except that we need to rewrite any
// typeDead bits in the first word of the object into typeScalar
// followed by a typeDead in the second word of the object.
func (h heapBits) initCheckmarkSpan(size, n, total uintptr) {
	if size == ptrSize {
		// Only possible on 64-bit system, since minimum size is 8.
		// Must update both top and bottom nibble of each byte.
		// There is no second word in these objects, so all we have
		// to do is rewrite typeDead to typeScalar by adding the 1<<typeShift bit.
		bitp := h.bitp
		for i := uintptr(0); i < n; i += 2 {
			x := int(*bitp)

			if (x>>typeShift)&typeMask == typeDead {
				x += (typeScalar - typeDead) << typeShift
			}
			if (x>>(4+typeShift))&typeMask == typeDead {
				x += (typeScalar - typeDead) << (4 + typeShift)
			}
			*bitp = uint8(x)
			bitp = subtractb(bitp, 1)
		}
		return
	}

	// Update bottom nibble for first word of each object.
	// If the bottom nibble says typeDead, change to typeScalar
	// and clear top nibble to mark as typeDead.
	bitp := h.bitp
	step := size / heapBitmapScale
	for i := uintptr(0); i < n; i++ {
		x := *bitp
		if (x>>typeShift)&typeMask == typeDead {
			x += (typeScalar - typeDead) << typeShift
			x &= 0x0f // clear top nibble to typeDead
		}
		bitp = subtractb(bitp, step)
	}
}

// clearCheckmarkSpan removes all the checkmarks from a span.
// If it finds a multiword object starting with typeScalar typeDead,
// it rewrites the heap bits to the simpler typeDead typeDead.
func (h heapBits) clearCheckmarkSpan(size, n, total uintptr) {
	if size == ptrSize {
		// Only possible on 64-bit system, since minimum size is 8.
		// Must update both top and bottom nibble of each byte.
		// typeScalarCheckmarked can be left as typeDead,
		// but we want to change typeScalar back to typeDead.
		bitp := h.bitp
		for i := uintptr(0); i < n; i += 2 {
			x := int(*bitp)
			switch typ := (x >> typeShift) & typeMask; typ {
			case typeScalar:
				x += (typeDead - typeScalar) << typeShift
			case typePointerCheckmarked:
				x += (typePointer - typePointerCheckmarked) << typeShift
			}

			switch typ := (x >> (4 + typeShift)) & typeMask; typ {
			case typeScalar:
				x += (typeDead - typeScalar) << (4 + typeShift)
			case typePointerCheckmarked:
				x += (typePointer - typePointerCheckmarked) << (4 + typeShift)
			}

			*bitp = uint8(x)
			bitp = subtractb(bitp, 1)
		}
		return
	}

	// Update bottom nibble for first word of each object.
	// If the bottom nibble says typeScalarCheckmarked and the top is not typeDead,
	// change to typeScalar. Otherwise leave, since typeScalarCheckmarked == typeDead.
	// If the bottom nibble says typePointerCheckmarked, change to typePointer.
	bitp := h.bitp
	step := size / heapBitmapScale
	for i := uintptr(0); i < n; i++ {
		x := int(*bitp)
		switch typ := (x >> typeShift) & typeMask; {
		case typ == typeScalarCheckmarked && (x>>(4+typeShift))&typeMask != typeDead:
			x += (typeScalar - typeScalarCheckmarked) << typeShift
		case typ == typePointerCheckmarked:
			x += (typePointer - typePointerCheckmarked) << typeShift
		}

		*bitp = uint8(x)
		bitp = subtractb(bitp, step)
	}
}

// heapBitsSweepSpan coordinates the sweeping of a span by reading
// and updating the corresponding heap bitmap entries.
// For each free object in the span, heapBitsSweepSpan sets the type
// bits for the first two words (or one for single-word objects) to typeDead
// and then calls f(p), where p is the object's base address.
// f is expected to add the object to a free list.
func heapBitsSweepSpan(base, size, n uintptr, f func(uintptr)) {
	h := heapBitsForSpan(base)
	if size == ptrSize {
		// Only possible on 64-bit system, since minimum size is 8.
		// Must read and update both top and bottom nibble of each byte.
		bitp := h.bitp
		for i := uintptr(0); i < n; i += 2 {
			x := int(*bitp)
			if x&bitMarked != 0 {
				x &^= bitMarked
			} else {
				x &^= typeMask << typeShift
				f(base + i*ptrSize)
			}
			if x&(bitMarked<<4) != 0 {
				x &^= bitMarked << 4
			} else {
				x &^= typeMask << (4 + typeShift)
				f(base + (i+1)*ptrSize)
			}
			*bitp = uint8(x)
			bitp = subtractb(bitp, 1)
		}
		return
	}

	bitp := h.bitp
	step := size / heapBitmapScale
	for i := uintptr(0); i < n; i++ {
		x := int(*bitp)
		if x&bitMarked != 0 {
			x &^= bitMarked
		} else {
			x = 0
			f(base + i*size)
		}
		*bitp = uint8(x)
		bitp = subtractb(bitp, step)
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

	var ti, te uintptr
	var ptrmask *uint8
	if size == ptrSize {
		// It's one word and it has pointers, it must be a pointer.
		// The bitmap byte is shared with the one-word object
		// next to it, and concurrent GC might be marking that
		// object, so we must use an atomic update.
		atomicor8(h.bitp, typePointer<<(typeShift+h.shift))
		return
	}
	if typ.kind&kindGCProg != 0 {
		nptr := (uintptr(typ.size) + ptrSize - 1) / ptrSize
		masksize := nptr
		if masksize%2 != 0 {
			masksize *= 2 // repeated
		}
		const typeBitsPerByte = 8 / typeBitsWidth
		masksize = masksize * typeBitsPerByte / 8 // 4 bits per word
		masksize++                                // unroll flag in the beginning
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
	if size == 2*ptrSize {
		// h.shift is 0 for all sizes > ptrSize.
		*h.bitp = *ptrmask
		return
	}
	te = uintptr(typ.size) / ptrSize
	// If the type occupies odd number of words, its mask is repeated.
	if te%2 == 0 {
		te /= 2
	}
	// Copy pointer bitmask into the bitmap.
	// TODO(rlh): add comment addressing the following concerns:
	// If size > 2*ptrSize, is x guaranteed to be at least 2*ptrSize-aligned?
	// And if type occupies and odd number of words, why are we only going through half
	// of ptrmask and why don't we have to shift everything by 4 on odd iterations?

	for i := uintptr(0); i < dataSize; i += 2 * ptrSize {
		v := *(*uint8)(add(unsafe.Pointer(ptrmask), ti))
		ti++
		if ti == te {
			ti = 0
		}
		if i+ptrSize == dataSize {
			v &^= typeMask << (4 + typeShift)
		}

		*h.bitp = v
		h.bitp = subtractb(h.bitp, 1)
	}
	if dataSize%(2*ptrSize) == 0 && dataSize < size {
		// Mark the word after last object's word as typeDead.
		*h.bitp = 0
	}
}

// typeBitmapInHeapBitmapFormat returns a bitmap holding
// the type bits for the type typ, but expanded into heap bitmap format
// to make it easier to copy them into the heap bitmap.
// TODO(rsc): Change clients to use the type bitmap format instead,
// which can be stored more densely (especially if we drop to 1 bit per pointer).
//
// To make it easier to replicate the bits when filling out the heap
// bitmap for an array of typ, if typ holds an odd number of words
// (meaning the heap bitmap would stop halfway through a byte),
// typeBitmapInHeapBitmapFormat returns the bitmap for two instances
// of typ in a row.
// TODO(rsc): Remove doubling.
func typeBitmapInHeapBitmapFormat(typ *_type) []uint8 {
	var ptrmask *uint8
	nptr := (uintptr(typ.size) + ptrSize - 1) / ptrSize
	if typ.kind&kindGCProg != 0 {
		masksize := nptr
		if masksize%2 != 0 {
			masksize *= 2 // repeated
		}
		const typeBitsPerByte = 8 / typeBitsWidth
		masksize = masksize * typeBitsPerByte / 8 // 4 bits per word
		masksize++                                // unroll flag in the beginning
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
	return (*[1 << 30]byte)(unsafe.Pointer(ptrmask))[:(nptr+1)/2]
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
func unrollgcprog1(maskp *byte, prog *byte, ppos *uintptr, inplace, sparse bool) *byte {
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
				const typeBitsPerByte = 8 / typeBitsWidth
				v := p[i/typeBitsPerByte]
				v >>= (uint(i) % typeBitsPerByte) * typeBitsWidth
				v &= typeMask
				if inplace {
					// Store directly into GC bitmap.
					h := heapBitsForAddr(uintptr(unsafe.Pointer(&mask[pos])))
					if h.shift == 0 {
						*h.bitp = v << typeShift
					} else {
						*h.bitp |= v << (4 + typeShift)
					}
					pos += ptrSize
				} else if sparse {
					// 4-bits per word, type bits in high bits
					v <<= (pos % 8) + typeShift
					mask[pos/8] |= v
					pos += heapBitsWidth
				} else {
					// 1 bit per word, for data/bss bitmap
					v >>= 1 // convert typePointer to 1, others to 0
					mask[pos/8] |= v << (pos % 8)
					pos++
				}
			}
			prog = addb(prog, round(uintptr(siz)*typeBitsWidth, 8)/8)

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
	masksize := round(round(size, ptrSize)/ptrSize, 8) / 8
	mask := (*[1 << 30]byte)(persistentalloc(masksize+1, 0, &memstats.gc_sys))
	mask[masksize] = 0xa1
	pos := uintptr(0)
	prog = unrollgcprog1(&mask[0], prog, &pos, false, false)
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
	// TODO(rsc): Explain why these non-atomic updates are okay.
	pos := uintptr(0)
	prog := (*byte)(unsafe.Pointer(uintptr(typ.gc[1])))
	for pos != size0 {
		unrollgcprog1((*byte)(v), prog, &pos, true, true)
	}

	// Mark first word as bitAllocated.
	// Mark word after last as typeDead.
	if size0 < size {
		h := heapBitsForAddr(uintptr(v) + size0)
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
func getgcmask(p unsafe.Pointer, t *_type, mask **byte, len *uintptr) {
	*mask = nil
	*len = 0

	// data
	for datap := &firstmoduledata; datap != nil; datap = datap.next {
		if datap.data <= uintptr(p) && uintptr(p) < datap.edata {
			n := (*ptrtype)(unsafe.Pointer(t)).elem.size
			*len = n / ptrSize
			*mask = &make([]byte, *len)[0]
			for i := uintptr(0); i < n; i += ptrSize {
				off := (uintptr(p) + i - datap.data) / ptrSize
				bits := (*addb(datap.gcdatamask.bytedata, off/8) >> (off % 8)) & 1
				bits += 1 // convert 1-bit to 2-bit
				*addb(*mask, i/ptrSize) = bits
			}
			return
		}

		// bss
		if datap.bss <= uintptr(p) && uintptr(p) < datap.ebss {
			n := (*ptrtype)(unsafe.Pointer(t)).elem.size
			*len = n / ptrSize
			*mask = &make([]byte, *len)[0]
			for i := uintptr(0); i < n; i += ptrSize {
				off := (uintptr(p) + i - datap.bss) / ptrSize
				bits := (*addb(datap.gcbssmask.bytedata, off/8) >> (off % 8)) & 1
				bits += 1 // convert 1-bit to 2-bit
				*addb(*mask, i/ptrSize) = bits
			}
			return
		}
	}

	// heap
	var n uintptr
	var base uintptr
	if mlookup(uintptr(p), &base, &n, nil) != 0 {
		*len = n / ptrSize
		*mask = &make([]byte, *len)[0]
		for i := uintptr(0); i < n; i += ptrSize {
			bits := heapBitsForAddr(base + i).typeBits()
			*addb(*mask, i/ptrSize) = bits
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
		*len = n / ptrSize
		*mask = &make([]byte, *len)[0]
		for i := uintptr(0); i < n; i += ptrSize {
			off := (uintptr(p) + i - frame.varp + size) / ptrSize
			bits := (*addb(bv.bytedata, off/8) >> (off % 8)) & 1
			bits += 1 // convert 1-bit to 2-bit
			*addb(*mask, i/ptrSize) = bits
		}
	}
}
