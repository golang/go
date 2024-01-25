// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.allocheaders

// Garbage collector: type and heap bitmaps.
//
// Stack, data, and bss bitmaps
//
// Stack frames and global variables in the data and bss sections are
// described by bitmaps with 1 bit per pointer-sized word. A "1" bit
// means the word is a live pointer to be visited by the GC (referred to
// as "pointer"). A "0" bit means the word should be ignored by GC
// (referred to as "scalar", though it could be a dead pointer value).
//
// Heap bitmaps
//
// The heap bitmap comprises 1 bit for each pointer-sized word in the heap,
// recording whether a pointer is stored in that word or not. This bitmap
// is stored at the end of a span for small objects and is unrolled at
// runtime from type metadata for all larger objects. Objects without
// pointers have neither a bitmap nor associated type metadata.
//
// Bits in all cases correspond to words in little-endian order.
//
// For small objects, if s is the mspan for the span starting at "start",
// then s.heapBits() returns a slice containing the bitmap for the whole span.
// That is, s.heapBits()[0] holds the goarch.PtrSize*8 bits for the first
// goarch.PtrSize*8 words from "start" through "start+63*ptrSize" in the span.
// On a related note, small objects are always small enough that their bitmap
// fits in goarch.PtrSize*8 bits, so writing out bitmap data takes two bitmap
// writes at most (because object boundaries don't generally lie on
// s.heapBits()[i] boundaries).
//
// For larger objects, if t is the type for the object starting at "start",
// within some span whose mspan is s, then the bitmap at t.GCData is "tiled"
// from "start" through "start+s.elemsize".
// Specifically, the first bit of t.GCData corresponds to the word at "start",
// the second to the word after "start", and so on up to t.PtrBytes. At t.PtrBytes,
// we skip to "start+t.Size_" and begin again from there. This process is
// repeated until we hit "start+s.elemsize".
// This tiling algorithm supports array data, since the type always refers to
// the element type of the array. Single objects are considered the same as
// single-element arrays.
// The tiling algorithm may scan data past the end of the compiler-recognized
// object, but any unused data within the allocation slot (i.e. within s.elemsize)
// is zeroed, so the GC just observes nil pointers.
// Note that this "tiled" bitmap isn't stored anywhere; it is generated on-the-fly.
//
// For objects without their own span, the type metadata is stored in the first
// word before the object at the beginning of the allocation slot. For objects
// with their own span, the type metadata is stored in the mspan.
//
// The bitmap for small unallocated objects in scannable spans is not maintained
// (can be junk).

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"runtime/internal/sys"
	"unsafe"
)

const (
	// A malloc header is functionally a single type pointer, but
	// we need to use 8 here to ensure 8-byte alignment of allocations
	// on 32-bit platforms. It's wasteful, but a lot of code relies on
	// 8-byte alignment for 8-byte atomics.
	mallocHeaderSize = 8

	// The minimum object size that has a malloc header, exclusive.
	//
	// The size of this value controls overheads from the malloc header.
	// The minimum size is bound by writeHeapBitsSmall, which assumes that the
	// pointer bitmap for objects of a size smaller than this doesn't cross
	// more than one pointer-word boundary. This sets an upper-bound on this
	// value at the number of bits in a uintptr, multiplied by the pointer
	// size in bytes.
	//
	// We choose a value here that has a natural cutover point in terms of memory
	// overheads. This value just happens to be the maximum possible value this
	// can be.
	//
	// A span with heap bits in it will have 128 bytes of heap bits on 64-bit
	// platforms, and 256 bytes of heap bits on 32-bit platforms. The first size
	// class where malloc headers match this overhead for 64-bit platforms is
	// 512 bytes (8 KiB / 512 bytes * 8 bytes-per-header = 128 bytes of overhead).
	// On 32-bit platforms, this same point is the 256 byte size class
	// (8 KiB / 256 bytes * 8 bytes-per-header = 256 bytes of overhead).
	//
	// Guaranteed to be exactly at a size class boundary. The reason this value is
	// an exclusive minimum is subtle. Suppose we're allocating a 504-byte object
	// and its rounded up to 512 bytes for the size class. If minSizeForMallocHeader
	// is 512 and an inclusive minimum, then a comparison against minSizeForMallocHeader
	// by the two values would produce different results. In other words, the comparison
	// would not be invariant to size-class rounding. Eschewing this property means a
	// more complex check or possibly storing additional state to determine whether a
	// span has malloc headers.
	minSizeForMallocHeader = goarch.PtrSize * ptrBits
)

// heapBitsInSpan returns true if the size of an object implies its ptr/scalar
// data is stored at the end of the span, and is accessible via span.heapBits.
//
// Note: this works for both rounded-up sizes (span.elemsize) and unrounded
// type sizes because minSizeForMallocHeader is guaranteed to be at a size
// class boundary.
//
//go:nosplit
func heapBitsInSpan(userSize uintptr) bool {
	// N.B. minSizeForMallocHeader is an exclusive minimum so that this function is
	// invariant under size-class rounding on its input.
	return userSize <= minSizeForMallocHeader
}

// heapArenaPtrScalar contains the per-heapArena pointer/scalar metadata for the GC.
type heapArenaPtrScalar struct {
	// N.B. This is no longer necessary with allocation headers.
}

// typePointers is an iterator over the pointers in a heap object.
//
// Iteration through this type implements the tiling algorithm described at the
// top of this file.
type typePointers struct {
	// elem is the address of the current array element of type typ being iterated over.
	// Objects that are not arrays are treated as single-element arrays, in which case
	// this value does not change.
	elem uintptr

	// addr is the address the iterator is currently working from and describes
	// the address of the first word referenced by mask.
	addr uintptr

	// mask is a bitmask where each bit corresponds to pointer-words after addr.
	// Bit 0 is the pointer-word at addr, Bit 1 is the next word, and so on.
	// If a bit is 1, then there is a pointer at that word.
	// nextFast and next mask out bits in this mask as their pointers are processed.
	mask uintptr

	// typ is a pointer to the type information for the heap object's type.
	// This may be nil if the object is in a span where heapBitsInSpan(span.elemsize) is true.
	typ *_type
}

// typePointersOf returns an iterator over all heap pointers in the range [addr, addr+size).
//
// addr and addr+size must be in the range [span.base(), span.limit).
//
// Note: addr+size must be passed as the limit argument to the iterator's next method on
// each iteration. This slightly awkward API is to allow typePointers to be destructured
// by the compiler.
//
// nosplit because it is used during write barriers and must not be preempted.
//
//go:nosplit
func (span *mspan) typePointersOf(addr, size uintptr) typePointers {
	base := span.objBase(addr)
	tp := span.typePointersOfUnchecked(base)
	if base == addr && size == span.elemsize {
		return tp
	}
	return tp.fastForward(addr-tp.addr, addr+size)
}

// typePointersOfUnchecked is like typePointersOf, but assumes addr is the base
// of an allocation slot in a span (the start of the object if no header, the
// header otherwise). It returns an iterator that generates all pointers
// in the range [addr, addr+span.elemsize).
//
// nosplit because it is used during write barriers and must not be preempted.
//
//go:nosplit
func (span *mspan) typePointersOfUnchecked(addr uintptr) typePointers {
	const doubleCheck = false
	if doubleCheck && span.objBase(addr) != addr {
		print("runtime: addr=", addr, " base=", span.objBase(addr), "\n")
		throw("typePointersOfUnchecked consisting of non-base-address for object")
	}

	spc := span.spanclass
	if spc.noscan() {
		return typePointers{}
	}
	if heapBitsInSpan(span.elemsize) {
		// Handle header-less objects.
		return typePointers{elem: addr, addr: addr, mask: span.heapBitsSmallForAddr(addr)}
	}

	// All of these objects have a header.
	var typ *_type
	if spc.sizeclass() != 0 {
		// Pull the allocation header from the first word of the object.
		typ = *(**_type)(unsafe.Pointer(addr))
		addr += mallocHeaderSize
	} else {
		typ = span.largeType
	}
	gcdata := typ.GCData
	return typePointers{elem: addr, addr: addr, mask: readUintptr(gcdata), typ: typ}
}

// typePointersOfType is like typePointersOf, but assumes addr points to one or more
// contiguous instances of the provided type. The provided type must not be nil and
// it must not have its type metadata encoded as a gcprog.
//
// It returns an iterator that tiles typ.GCData starting from addr. It's the caller's
// responsibility to limit iteration.
//
// nosplit because its callers are nosplit and require all their callees to be nosplit.
//
//go:nosplit
func (span *mspan) typePointersOfType(typ *abi.Type, addr uintptr) typePointers {
	const doubleCheck = false
	if doubleCheck && (typ == nil || typ.Kind_&kindGCProg != 0) {
		throw("bad type passed to typePointersOfType")
	}
	if span.spanclass.noscan() {
		return typePointers{}
	}
	// Since we have the type, pretend we have a header.
	gcdata := typ.GCData
	return typePointers{elem: addr, addr: addr, mask: readUintptr(gcdata), typ: typ}
}

// nextFast is the fast path of next. nextFast is written to be inlineable and,
// as the name implies, fast.
//
// Callers that are performance-critical should iterate using the following
// pattern:
//
//	for {
//		var addr uintptr
//		if tp, addr = tp.nextFast(); addr == 0 {
//			if tp, addr = tp.next(limit); addr == 0 {
//				break
//			}
//		}
//		// Use addr.
//		...
//	}
//
// nosplit because it is used during write barriers and must not be preempted.
//
//go:nosplit
func (tp typePointers) nextFast() (typePointers, uintptr) {
	// TESTQ/JEQ
	if tp.mask == 0 {
		return tp, 0
	}
	// BSFQ
	var i int
	if goarch.PtrSize == 8 {
		i = sys.TrailingZeros64(uint64(tp.mask))
	} else {
		i = sys.TrailingZeros32(uint32(tp.mask))
	}
	// BTCQ
	tp.mask ^= uintptr(1) << (i & (ptrBits - 1))
	// LEAQ (XX)(XX*8)
	return tp, tp.addr + uintptr(i)*goarch.PtrSize
}

// next advances the pointers iterator, returning the updated iterator and
// the address of the next pointer.
//
// limit must be the same each time it is passed to next.
//
// nosplit because it is used during write barriers and must not be preempted.
//
//go:nosplit
func (tp typePointers) next(limit uintptr) (typePointers, uintptr) {
	for {
		if tp.mask != 0 {
			return tp.nextFast()
		}

		// Stop if we don't actually have type information.
		if tp.typ == nil {
			return typePointers{}, 0
		}

		// Advance to the next element if necessary.
		if tp.addr+goarch.PtrSize*ptrBits >= tp.elem+tp.typ.PtrBytes {
			tp.elem += tp.typ.Size_
			tp.addr = tp.elem
		} else {
			tp.addr += ptrBits * goarch.PtrSize
		}

		// Check if we've exceeded the limit with the last update.
		if tp.addr >= limit {
			return typePointers{}, 0
		}

		// Grab more bits and try again.
		tp.mask = readUintptr(addb(tp.typ.GCData, (tp.addr-tp.elem)/goarch.PtrSize/8))
		if tp.addr+goarch.PtrSize*ptrBits > limit {
			bits := (tp.addr + goarch.PtrSize*ptrBits - limit) / goarch.PtrSize
			tp.mask &^= ((1 << (bits)) - 1) << (ptrBits - bits)
		}
	}
}

// fastForward moves the iterator forward by n bytes. n must be a multiple
// of goarch.PtrSize. limit must be the same limit passed to next for this
// iterator.
//
// nosplit because it is used during write barriers and must not be preempted.
//
//go:nosplit
func (tp typePointers) fastForward(n, limit uintptr) typePointers {
	// Basic bounds check.
	target := tp.addr + n
	if target >= limit {
		return typePointers{}
	}
	if tp.typ == nil {
		// Handle small objects.
		// Clear any bits before the target address.
		tp.mask &^= (1 << ((target - tp.addr) / goarch.PtrSize)) - 1
		// Clear any bits past the limit.
		if tp.addr+goarch.PtrSize*ptrBits > limit {
			bits := (tp.addr + goarch.PtrSize*ptrBits - limit) / goarch.PtrSize
			tp.mask &^= ((1 << (bits)) - 1) << (ptrBits - bits)
		}
		return tp
	}

	// Move up elem and addr.
	// Offsets within an element are always at a ptrBits*goarch.PtrSize boundary.
	if n >= tp.typ.Size_ {
		// elem needs to be moved to the element containing
		// tp.addr + n.
		oldelem := tp.elem
		tp.elem += (tp.addr - tp.elem + n) / tp.typ.Size_ * tp.typ.Size_
		tp.addr = tp.elem + alignDown(n-(tp.elem-oldelem), ptrBits*goarch.PtrSize)
	} else {
		tp.addr += alignDown(n, ptrBits*goarch.PtrSize)
	}

	if tp.addr-tp.elem >= tp.typ.PtrBytes {
		// We're starting in the non-pointer area of an array.
		// Move up to the next element.
		tp.elem += tp.typ.Size_
		tp.addr = tp.elem
		tp.mask = readUintptr(tp.typ.GCData)

		// We may have exceeded the limit after this. Bail just like next does.
		if tp.addr >= limit {
			return typePointers{}
		}
	} else {
		// Grab the mask, but then clear any bits before the target address and any
		// bits over the limit.
		tp.mask = readUintptr(addb(tp.typ.GCData, (tp.addr-tp.elem)/goarch.PtrSize/8))
		tp.mask &^= (1 << ((target - tp.addr) / goarch.PtrSize)) - 1
	}
	if tp.addr+goarch.PtrSize*ptrBits > limit {
		bits := (tp.addr + goarch.PtrSize*ptrBits - limit) / goarch.PtrSize
		tp.mask &^= ((1 << (bits)) - 1) << (ptrBits - bits)
	}
	return tp
}

// objBase returns the base pointer for the object containing addr in span.
//
// Assumes that addr points into a valid part of span (span.base() <= addr < span.limit).
//
//go:nosplit
func (span *mspan) objBase(addr uintptr) uintptr {
	return span.base() + span.objIndex(addr)*span.elemsize
}

// bulkBarrierPreWrite executes a write barrier
// for every pointer slot in the memory range [src, src+size),
// using pointer/scalar information from [dst, dst+size).
// This executes the write barriers necessary before a memmove.
// src, dst, and size must be pointer-aligned.
// The range [dst, dst+size) must lie within a single object.
// It does not perform the actual writes.
//
// As a special case, src == 0 indicates that this is being used for a
// memclr. bulkBarrierPreWrite will pass 0 for the src of each write
// barrier.
//
// Callers should call bulkBarrierPreWrite immediately before
// calling memmove(dst, src, size). This function is marked nosplit
// to avoid being preempted; the GC must not stop the goroutine
// between the memmove and the execution of the barriers.
// The caller is also responsible for cgo pointer checks if this
// may be writing Go pointers into non-Go memory.
//
// Pointer data is not maintained for allocations containing
// no pointers at all; any caller of bulkBarrierPreWrite must first
// make sure the underlying allocation contains pointers, usually
// by checking typ.PtrBytes.
//
// The typ argument is the type of the space at src and dst (and the
// element type if src and dst refer to arrays) and it is optional.
// If typ is nil, the barrier will still behave as expected and typ
// is used purely as an optimization. However, it must be used with
// care.
//
// If typ is not nil, then src and dst must point to one or more values
// of type typ. The caller must ensure that the ranges [src, src+size)
// and [dst, dst+size) refer to one or more whole values of type src and
// dst (leaving off the pointerless tail of the space is OK). If this
// precondition is not followed, this function will fail to scan the
// right pointers.
//
// When in doubt, pass nil for typ. That is safe and will always work.
//
// Callers must perform cgo checks if goexperiment.CgoCheck2.
//
//go:nosplit
func bulkBarrierPreWrite(dst, src, size uintptr, typ *abi.Type) {
	if (dst|src|size)&(goarch.PtrSize-1) != 0 {
		throw("bulkBarrierPreWrite: unaligned arguments")
	}
	if !writeBarrier.enabled {
		return
	}
	s := spanOf(dst)
	if s == nil {
		// If dst is a global, use the data or BSS bitmaps to
		// execute write barriers.
		for _, datap := range activeModules() {
			if datap.data <= dst && dst < datap.edata {
				bulkBarrierBitmap(dst, src, size, dst-datap.data, datap.gcdatamask.bytedata)
				return
			}
		}
		for _, datap := range activeModules() {
			if datap.bss <= dst && dst < datap.ebss {
				bulkBarrierBitmap(dst, src, size, dst-datap.bss, datap.gcbssmask.bytedata)
				return
			}
		}
		return
	} else if s.state.get() != mSpanInUse || dst < s.base() || s.limit <= dst {
		// dst was heap memory at some point, but isn't now.
		// It can't be a global. It must be either our stack,
		// or in the case of direct channel sends, it could be
		// another stack. Either way, no need for barriers.
		// This will also catch if dst is in a freed span,
		// though that should never have.
		return
	}
	buf := &getg().m.p.ptr().wbBuf

	// Double-check that the bitmaps generated in the two possible paths match.
	const doubleCheck = false
	if doubleCheck {
		doubleCheckTypePointersOfType(s, typ, dst, size)
	}

	var tp typePointers
	if typ != nil && typ.Kind_&kindGCProg == 0 {
		tp = s.typePointersOfType(typ, dst)
	} else {
		tp = s.typePointersOf(dst, size)
	}
	if src == 0 {
		for {
			var addr uintptr
			if tp, addr = tp.next(dst + size); addr == 0 {
				break
			}
			dstx := (*uintptr)(unsafe.Pointer(addr))
			p := buf.get1()
			p[0] = *dstx
		}
	} else {
		for {
			var addr uintptr
			if tp, addr = tp.next(dst + size); addr == 0 {
				break
			}
			dstx := (*uintptr)(unsafe.Pointer(addr))
			srcx := (*uintptr)(unsafe.Pointer(src + (addr - dst)))
			p := buf.get2()
			p[0] = *dstx
			p[1] = *srcx
		}
	}
}

// bulkBarrierPreWriteSrcOnly is like bulkBarrierPreWrite but
// does not execute write barriers for [dst, dst+size).
//
// In addition to the requirements of bulkBarrierPreWrite
// callers need to ensure [dst, dst+size) is zeroed.
//
// This is used for special cases where e.g. dst was just
// created and zeroed with malloc.
//
// The type of the space can be provided purely as an optimization.
// See bulkBarrierPreWrite's comment for more details -- use this
// optimization with great care.
//
//go:nosplit
func bulkBarrierPreWriteSrcOnly(dst, src, size uintptr, typ *abi.Type) {
	if (dst|src|size)&(goarch.PtrSize-1) != 0 {
		throw("bulkBarrierPreWrite: unaligned arguments")
	}
	if !writeBarrier.enabled {
		return
	}
	buf := &getg().m.p.ptr().wbBuf
	s := spanOf(dst)

	// Double-check that the bitmaps generated in the two possible paths match.
	const doubleCheck = false
	if doubleCheck {
		doubleCheckTypePointersOfType(s, typ, dst, size)
	}

	var tp typePointers
	if typ != nil && typ.Kind_&kindGCProg == 0 {
		tp = s.typePointersOfType(typ, dst)
	} else {
		tp = s.typePointersOf(dst, size)
	}
	for {
		var addr uintptr
		if tp, addr = tp.next(dst + size); addr == 0 {
			break
		}
		srcx := (*uintptr)(unsafe.Pointer(addr - dst + src))
		p := buf.get1()
		p[0] = *srcx
	}
}

// initHeapBits initializes the heap bitmap for a span.
//
// TODO(mknyszek): This should set the heap bits for single pointer
// allocations eagerly to avoid calling heapSetType at allocation time,
// just to write one bit.
func (s *mspan) initHeapBits(forceClear bool) {
	if (!s.spanclass.noscan() && heapBitsInSpan(s.elemsize)) || s.isUserArenaChunk {
		b := s.heapBits()
		for i := range b {
			b[i] = 0
		}
	}
}

// bswapIfBigEndian swaps the byte order of the uintptr on goarch.BigEndian platforms,
// and leaves it alone elsewhere.
func bswapIfBigEndian(x uintptr) uintptr {
	if goarch.BigEndian {
		if goarch.PtrSize == 8 {
			return uintptr(sys.Bswap64(uint64(x)))
		}
		return uintptr(sys.Bswap32(uint32(x)))
	}
	return x
}

type writeUserArenaHeapBits struct {
	offset uintptr // offset in span that the low bit of mask represents the pointer state of.
	mask   uintptr // some pointer bits starting at the address addr.
	valid  uintptr // number of bits in buf that are valid (including low)
	low    uintptr // number of low-order bits to not overwrite
}

func (s *mspan) writeUserArenaHeapBits(addr uintptr) (h writeUserArenaHeapBits) {
	offset := addr - s.base()

	// We start writing bits maybe in the middle of a heap bitmap word.
	// Remember how many bits into the word we started, so we can be sure
	// not to overwrite the previous bits.
	h.low = offset / goarch.PtrSize % ptrBits

	// round down to heap word that starts the bitmap word.
	h.offset = offset - h.low*goarch.PtrSize

	// We don't have any bits yet.
	h.mask = 0
	h.valid = h.low

	return
}

// write appends the pointerness of the next valid pointer slots
// using the low valid bits of bits. 1=pointer, 0=scalar.
func (h writeUserArenaHeapBits) write(s *mspan, bits, valid uintptr) writeUserArenaHeapBits {
	if h.valid+valid <= ptrBits {
		// Fast path - just accumulate the bits.
		h.mask |= bits << h.valid
		h.valid += valid
		return h
	}
	// Too many bits to fit in this word. Write the current word
	// out and move on to the next word.

	data := h.mask | bits<<h.valid       // mask for this word
	h.mask = bits >> (ptrBits - h.valid) // leftover for next word
	h.valid += valid - ptrBits           // have h.valid+valid bits, writing ptrBits of them

	// Flush mask to the memory bitmap.
	idx := h.offset / (ptrBits * goarch.PtrSize)
	m := uintptr(1)<<h.low - 1
	bitmap := s.heapBits()
	bitmap[idx] = bswapIfBigEndian(bswapIfBigEndian(bitmap[idx])&m | data)
	// Note: no synchronization required for this write because
	// the allocator has exclusive access to the page, and the bitmap
	// entries are all for a single page. Also, visibility of these
	// writes is guaranteed by the publication barrier in mallocgc.

	// Move to next word of bitmap.
	h.offset += ptrBits * goarch.PtrSize
	h.low = 0
	return h
}

// Add padding of size bytes.
func (h writeUserArenaHeapBits) pad(s *mspan, size uintptr) writeUserArenaHeapBits {
	if size == 0 {
		return h
	}
	words := size / goarch.PtrSize
	for words > ptrBits {
		h = h.write(s, 0, ptrBits)
		words -= ptrBits
	}
	return h.write(s, 0, words)
}

// Flush the bits that have been written, and add zeros as needed
// to cover the full object [addr, addr+size).
func (h writeUserArenaHeapBits) flush(s *mspan, addr, size uintptr) {
	offset := addr - s.base()

	// zeros counts the number of bits needed to represent the object minus the
	// number of bits we've already written. This is the number of 0 bits
	// that need to be added.
	zeros := (offset+size-h.offset)/goarch.PtrSize - h.valid

	// Add zero bits up to the bitmap word boundary
	if zeros > 0 {
		z := ptrBits - h.valid
		if z > zeros {
			z = zeros
		}
		h.valid += z
		zeros -= z
	}

	// Find word in bitmap that we're going to write.
	bitmap := s.heapBits()
	idx := h.offset / (ptrBits * goarch.PtrSize)

	// Write remaining bits.
	if h.valid != h.low {
		m := uintptr(1)<<h.low - 1      // don't clear existing bits below "low"
		m |= ^(uintptr(1)<<h.valid - 1) // don't clear existing bits above "valid"
		bitmap[idx] = bswapIfBigEndian(bswapIfBigEndian(bitmap[idx])&m | h.mask)
	}
	if zeros == 0 {
		return
	}

	// Advance to next bitmap word.
	h.offset += ptrBits * goarch.PtrSize

	// Continue on writing zeros for the rest of the object.
	// For standard use of the ptr bits this is not required, as
	// the bits are read from the beginning of the object. Some uses,
	// like noscan spans, oblets, bulk write barriers, and cgocheck, might
	// start mid-object, so these writes are still required.
	for {
		// Write zero bits.
		idx := h.offset / (ptrBits * goarch.PtrSize)
		if zeros < ptrBits {
			bitmap[idx] = bswapIfBigEndian(bswapIfBigEndian(bitmap[idx]) &^ (uintptr(1)<<zeros - 1))
			break
		} else if zeros == ptrBits {
			bitmap[idx] = 0
			break
		} else {
			bitmap[idx] = 0
			zeros -= ptrBits
		}
		h.offset += ptrBits * goarch.PtrSize
	}
}

// heapBits returns the heap ptr/scalar bits stored at the end of the span for
// small object spans and heap arena spans.
//
// Note that the uintptr of each element means something different for small object
// spans and for heap arena spans. Small object spans are easy: they're never interpreted
// as anything but uintptr, so they're immune to differences in endianness. However, the
// heapBits for user arena spans is exposed through a dummy type descriptor, so the byte
// ordering needs to match the same byte ordering the compiler would emit. The compiler always
// emits the bitmap data in little endian byte ordering, so on big endian platforms these
// uintptrs will have their byte orders swapped from what they normally would be.
//
// heapBitsInSpan(span.elemsize) or span.isUserArenaChunk must be true.
//
//go:nosplit
func (span *mspan) heapBits() []uintptr {
	const doubleCheck = false

	if doubleCheck && !span.isUserArenaChunk {
		if span.spanclass.noscan() {
			throw("heapBits called for noscan")
		}
		if span.elemsize > minSizeForMallocHeader {
			throw("heapBits called for span class that should have a malloc header")
		}
	}
	// Find the bitmap at the end of the span.
	//
	// Nearly every span with heap bits is exactly one page in size. Arenas are the only exception.
	if span.npages == 1 {
		// This will be inlined and constant-folded down.
		return heapBitsSlice(span.base(), pageSize)
	}
	return heapBitsSlice(span.base(), span.npages*pageSize)
}

// Helper for constructing a slice for the span's heap bits.
//
//go:nosplit
func heapBitsSlice(spanBase, spanSize uintptr) []uintptr {
	bitmapSize := spanSize / goarch.PtrSize / 8
	elems := int(bitmapSize / goarch.PtrSize)
	var sl notInHeapSlice
	sl = notInHeapSlice{(*notInHeap)(unsafe.Pointer(spanBase + spanSize - bitmapSize)), elems, elems}
	return *(*[]uintptr)(unsafe.Pointer(&sl))
}

// heapBitsSmallForAddr loads the heap bits for the object stored at addr from span.heapBits.
//
// addr must be the base pointer of an object in the span. heapBitsInSpan(span.elemsize)
// must be true.
//
//go:nosplit
func (span *mspan) heapBitsSmallForAddr(addr uintptr) uintptr {
	spanSize := span.npages * pageSize
	bitmapSize := spanSize / goarch.PtrSize / 8
	hbits := (*byte)(unsafe.Pointer(span.base() + spanSize - bitmapSize))

	// These objects are always small enough that their bitmaps
	// fit in a single word, so just load the word or two we need.
	//
	// Mirrors mspan.writeHeapBitsSmall.
	//
	// We should be using heapBits(), but unfortunately it introduces
	// both bounds checks panics and throw which causes us to exceed
	// the nosplit limit in quite a few cases.
	i := (addr - span.base()) / goarch.PtrSize / ptrBits
	j := (addr - span.base()) / goarch.PtrSize % ptrBits
	bits := span.elemsize / goarch.PtrSize
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

// writeHeapBitsSmall writes the heap bits for small objects whose ptr/scalar data is
// stored as a bitmap at the end of the span.
//
// Assumes dataSize is <= ptrBits*goarch.PtrSize. x must be a pointer into the span.
// heapBitsInSpan(dataSize) must be true. dataSize must be >= typ.Size_.
//
//go:nosplit
func (span *mspan) writeHeapBitsSmall(x, dataSize uintptr, typ *_type) (scanSize uintptr) {
	// The objects here are always really small, so a single load is sufficient.
	src0 := readUintptr(typ.GCData)

	// Create repetitions of the bitmap if we have a small array.
	bits := span.elemsize / goarch.PtrSize
	scanSize = typ.PtrBytes
	src := src0
	switch typ.Size_ {
	case goarch.PtrSize:
		src = (1 << (dataSize / goarch.PtrSize)) - 1
	default:
		for i := typ.Size_; i < dataSize; i += typ.Size_ {
			src |= src0 << (i / goarch.PtrSize)
			scanSize += typ.Size_
		}
	}

	// Since we're never writing more than one uintptr's worth of bits, we're either going
	// to do one or two writes.
	dst := span.heapBits()
	o := (x - span.base()) / goarch.PtrSize
	i := o / ptrBits
	j := o % ptrBits
	if j+bits > ptrBits {
		// Two writes.
		bits0 := ptrBits - j
		bits1 := bits - bits0
		dst[i+0] = dst[i+0]&(^uintptr(0)>>bits0) | (src << j)
		dst[i+1] = dst[i+1]&^((1<<bits1)-1) | (src >> bits0)
	} else {
		// One write.
		dst[i] = (dst[i] &^ (((1 << bits) - 1) << j)) | (src << j)
	}

	const doubleCheck = false
	if doubleCheck {
		srcRead := span.heapBitsSmallForAddr(x)
		if srcRead != src {
			print("runtime: x=", hex(x), " i=", i, " j=", j, " bits=", bits, "\n")
			print("runtime: dataSize=", dataSize, " typ.Size_=", typ.Size_, " typ.PtrBytes=", typ.PtrBytes, "\n")
			print("runtime: src0=", hex(src0), " src=", hex(src), " srcRead=", hex(srcRead), "\n")
			throw("bad pointer bits written for small object")
		}
	}
	return
}

// For !goexperiment.AllocHeaders.
func heapBitsSetType(x, size, dataSize uintptr, typ *_type) {
}

// heapSetType records that the new allocation [x, x+size)
// holds in [x, x+dataSize) one or more values of type typ.
// (The number of values is given by dataSize / typ.Size.)
// If dataSize < size, the fragment [x+dataSize, x+size) is
// recorded as non-pointer data.
// It is known that the type has pointers somewhere;
// malloc does not call heapSetType when there are no pointers.
//
// There can be read-write races between heapSetType and things
// that read the heap metadata like scanobject. However, since
// heapSetType is only used for objects that have not yet been
// made reachable, readers will ignore bits being modified by this
// function. This does mean this function cannot transiently modify
// shared memory that belongs to neighboring objects. Also, on weakly-ordered
// machines, callers must execute a store/store (publication) barrier
// between calling this function and making the object reachable.
func heapSetType(x, dataSize uintptr, typ *_type, header **_type, span *mspan) (scanSize uintptr) {
	const doubleCheck = false

	gctyp := typ
	if header == nil {
		if doubleCheck && (!heapBitsInSpan(dataSize) || !heapBitsInSpan(span.elemsize)) {
			throw("tried to write heap bits, but no heap bits in span")
		}
		// Handle the case where we have no malloc header.
		scanSize = span.writeHeapBitsSmall(x, dataSize, typ)
	} else {
		if typ.Kind_&kindGCProg != 0 {
			// Allocate space to unroll the gcprog. This space will consist of
			// a dummy _type value and the unrolled gcprog. The dummy _type will
			// refer to the bitmap, and the mspan will refer to the dummy _type.
			if span.spanclass.sizeclass() != 0 {
				throw("GCProg for type that isn't large")
			}
			spaceNeeded := alignUp(unsafe.Sizeof(_type{}), goarch.PtrSize)
			heapBitsOff := spaceNeeded
			spaceNeeded += alignUp(typ.PtrBytes/goarch.PtrSize/8, goarch.PtrSize)
			npages := alignUp(spaceNeeded, pageSize) / pageSize
			var progSpan *mspan
			systemstack(func() {
				progSpan = mheap_.allocManual(npages, spanAllocPtrScalarBits)
				memclrNoHeapPointers(unsafe.Pointer(progSpan.base()), progSpan.npages*pageSize)
			})
			// Write a dummy _type in the new space.
			//
			// We only need to write size, PtrBytes, and GCData, since that's all
			// the GC cares about.
			gctyp = (*_type)(unsafe.Pointer(progSpan.base()))
			gctyp.Size_ = typ.Size_
			gctyp.PtrBytes = typ.PtrBytes
			gctyp.GCData = (*byte)(add(unsafe.Pointer(progSpan.base()), heapBitsOff))
			gctyp.TFlag = abi.TFlagUnrolledBitmap

			// Expand the GC program into space reserved at the end of the new span.
			runGCProg(addb(typ.GCData, 4), gctyp.GCData)
		}

		// Write out the header.
		*header = gctyp
		scanSize = span.elemsize
	}

	if doubleCheck {
		doubleCheckHeapPointers(x, dataSize, gctyp, header, span)

		// To exercise the less common path more often, generate
		// a random interior pointer and make sure iterating from
		// that point works correctly too.
		maxIterBytes := span.elemsize
		if header == nil {
			maxIterBytes = dataSize
		}
		off := alignUp(uintptr(cheaprand())%dataSize, goarch.PtrSize)
		size := dataSize - off
		if size == 0 {
			off -= goarch.PtrSize
			size += goarch.PtrSize
		}
		interior := x + off
		size -= alignDown(uintptr(cheaprand())%size, goarch.PtrSize)
		if size == 0 {
			size = goarch.PtrSize
		}
		// Round up the type to the size of the type.
		size = (size + gctyp.Size_ - 1) / gctyp.Size_ * gctyp.Size_
		if interior+size > x+maxIterBytes {
			size = x + maxIterBytes - interior
		}
		doubleCheckHeapPointersInterior(x, interior, size, dataSize, gctyp, header, span)
	}
	return
}

func doubleCheckHeapPointers(x, dataSize uintptr, typ *_type, header **_type, span *mspan) {
	// Check that scanning the full object works.
	tp := span.typePointersOfUnchecked(span.objBase(x))
	maxIterBytes := span.elemsize
	if header == nil {
		maxIterBytes = dataSize
	}
	bad := false
	for i := uintptr(0); i < maxIterBytes; i += goarch.PtrSize {
		// Compute the pointer bit we want at offset i.
		want := false
		if i < span.elemsize {
			off := i % typ.Size_
			if off < typ.PtrBytes {
				j := off / goarch.PtrSize
				want = *addb(typ.GCData, j/8)>>(j%8)&1 != 0
			}
		}
		if want {
			var addr uintptr
			tp, addr = tp.next(x + span.elemsize)
			if addr == 0 {
				println("runtime: found bad iterator")
			}
			if addr != x+i {
				print("runtime: addr=", hex(addr), " x+i=", hex(x+i), "\n")
				bad = true
			}
		}
	}
	if !bad {
		var addr uintptr
		tp, addr = tp.next(x + span.elemsize)
		if addr == 0 {
			return
		}
		println("runtime: extra pointer:", hex(addr))
	}
	print("runtime: hasHeader=", header != nil, " typ.Size_=", typ.Size_, " hasGCProg=", typ.Kind_&kindGCProg != 0, "\n")
	print("runtime: x=", hex(x), " dataSize=", dataSize, " elemsize=", span.elemsize, "\n")
	print("runtime: typ=", unsafe.Pointer(typ), " typ.PtrBytes=", typ.PtrBytes, "\n")
	print("runtime: limit=", hex(x+span.elemsize), "\n")
	tp = span.typePointersOfUnchecked(x)
	dumpTypePointers(tp)
	for {
		var addr uintptr
		if tp, addr = tp.next(x + span.elemsize); addr == 0 {
			println("runtime: would've stopped here")
			dumpTypePointers(tp)
			break
		}
		print("runtime: addr=", hex(addr), "\n")
		dumpTypePointers(tp)
	}
	throw("heapSetType: pointer entry not correct")
}

func doubleCheckHeapPointersInterior(x, interior, size, dataSize uintptr, typ *_type, header **_type, span *mspan) {
	bad := false
	if interior < x {
		print("runtime: interior=", hex(interior), " x=", hex(x), "\n")
		throw("found bad interior pointer")
	}
	off := interior - x
	tp := span.typePointersOf(interior, size)
	for i := off; i < off+size; i += goarch.PtrSize {
		// Compute the pointer bit we want at offset i.
		want := false
		if i < span.elemsize {
			off := i % typ.Size_
			if off < typ.PtrBytes {
				j := off / goarch.PtrSize
				want = *addb(typ.GCData, j/8)>>(j%8)&1 != 0
			}
		}
		if want {
			var addr uintptr
			tp, addr = tp.next(interior + size)
			if addr == 0 {
				println("runtime: found bad iterator")
				bad = true
			}
			if addr != x+i {
				print("runtime: addr=", hex(addr), " x+i=", hex(x+i), "\n")
				bad = true
			}
		}
	}
	if !bad {
		var addr uintptr
		tp, addr = tp.next(interior + size)
		if addr == 0 {
			return
		}
		println("runtime: extra pointer:", hex(addr))
	}
	print("runtime: hasHeader=", header != nil, " typ.Size_=", typ.Size_, "\n")
	print("runtime: x=", hex(x), " dataSize=", dataSize, " elemsize=", span.elemsize, " interior=", hex(interior), " size=", size, "\n")
	print("runtime: limit=", hex(interior+size), "\n")
	tp = span.typePointersOf(interior, size)
	dumpTypePointers(tp)
	for {
		var addr uintptr
		if tp, addr = tp.next(interior + size); addr == 0 {
			println("runtime: would've stopped here")
			dumpTypePointers(tp)
			break
		}
		print("runtime: addr=", hex(addr), "\n")
		dumpTypePointers(tp)
	}

	print("runtime: want: ")
	for i := off; i < off+size; i += goarch.PtrSize {
		// Compute the pointer bit we want at offset i.
		want := false
		if i < dataSize {
			off := i % typ.Size_
			if off < typ.PtrBytes {
				j := off / goarch.PtrSize
				want = *addb(typ.GCData, j/8)>>(j%8)&1 != 0
			}
		}
		if want {
			print("1")
		} else {
			print("0")
		}
	}
	println()

	throw("heapSetType: pointer entry not correct")
}

//go:nosplit
func doubleCheckTypePointersOfType(s *mspan, typ *_type, addr, size uintptr) {
	if typ == nil || typ.Kind_&kindGCProg != 0 {
		return
	}
	if typ.Kind_&kindMask == kindInterface {
		// Interfaces are unfortunately inconsistently handled
		// when it comes to the type pointer, so it's easy to
		// produce a lot of false positives here.
		return
	}
	tp0 := s.typePointersOfType(typ, addr)
	tp1 := s.typePointersOf(addr, size)
	failed := false
	for {
		var addr0, addr1 uintptr
		tp0, addr0 = tp0.next(addr + size)
		tp1, addr1 = tp1.next(addr + size)
		if addr0 != addr1 {
			failed = true
			break
		}
		if addr0 == 0 {
			break
		}
	}
	if failed {
		tp0 := s.typePointersOfType(typ, addr)
		tp1 := s.typePointersOf(addr, size)
		print("runtime: addr=", hex(addr), " size=", size, "\n")
		print("runtime: type=", toRType(typ).string(), "\n")
		dumpTypePointers(tp0)
		dumpTypePointers(tp1)
		for {
			var addr0, addr1 uintptr
			tp0, addr0 = tp0.next(addr + size)
			tp1, addr1 = tp1.next(addr + size)
			print("runtime: ", hex(addr0), " ", hex(addr1), "\n")
			if addr0 == 0 && addr1 == 0 {
				break
			}
		}
		throw("mismatch between typePointersOfType and typePointersOf")
	}
}

func dumpTypePointers(tp typePointers) {
	print("runtime: tp.elem=", hex(tp.elem), " tp.typ=", unsafe.Pointer(tp.typ), "\n")
	print("runtime: tp.addr=", hex(tp.addr), " tp.mask=")
	for i := uintptr(0); i < ptrBits; i++ {
		if tp.mask&(uintptr(1)<<i) != 0 {
			print("1")
		} else {
			print("0")
		}
	}
	println()
}

// Testing.

// Returns GC type info for the pointer stored in ep for testing.
// If ep points to the stack, only static live information will be returned
// (i.e. not for objects which are only dynamically live stack objects).
func getgcmask(ep any) (mask []byte) {
	e := *efaceOf(&ep)
	p := e.data
	t := e._type

	var et *_type
	if t.Kind_&kindMask != kindPtr {
		throw("bad argument to getgcmask: expected type to be a pointer to the value type whose mask is being queried")
	}
	et = (*ptrtype)(unsafe.Pointer(t)).Elem

	// data or bss
	for _, datap := range activeModules() {
		// data
		if datap.data <= uintptr(p) && uintptr(p) < datap.edata {
			bitmap := datap.gcdatamask.bytedata
			n := et.Size_
			mask = make([]byte, n/goarch.PtrSize)
			for i := uintptr(0); i < n; i += goarch.PtrSize {
				off := (uintptr(p) + i - datap.data) / goarch.PtrSize
				mask[i/goarch.PtrSize] = (*addb(bitmap, off/8) >> (off % 8)) & 1
			}
			return
		}

		// bss
		if datap.bss <= uintptr(p) && uintptr(p) < datap.ebss {
			bitmap := datap.gcbssmask.bytedata
			n := et.Size_
			mask = make([]byte, n/goarch.PtrSize)
			for i := uintptr(0); i < n; i += goarch.PtrSize {
				off := (uintptr(p) + i - datap.bss) / goarch.PtrSize
				mask[i/goarch.PtrSize] = (*addb(bitmap, off/8) >> (off % 8)) & 1
			}
			return
		}
	}

	// heap
	if base, s, _ := findObject(uintptr(p), 0, 0); base != 0 {
		if s.spanclass.noscan() {
			return nil
		}
		limit := base + s.elemsize

		// Move the base up to the iterator's start, because
		// we want to hide evidence of a malloc header from the
		// caller.
		tp := s.typePointersOfUnchecked(base)
		base = tp.addr

		// Unroll the full bitmap the GC would actually observe.
		maskFromHeap := make([]byte, (limit-base)/goarch.PtrSize)
		for {
			var addr uintptr
			if tp, addr = tp.next(limit); addr == 0 {
				break
			}
			maskFromHeap[(addr-base)/goarch.PtrSize] = 1
		}

		// Double-check that every part of the ptr/scalar we're not
		// showing the caller is zeroed. This keeps us honest that
		// that information is actually irrelevant.
		for i := limit; i < s.elemsize; i++ {
			if *(*byte)(unsafe.Pointer(i)) != 0 {
				throw("found non-zeroed tail of allocation")
			}
		}

		// Callers (and a check we're about to run) expects this mask
		// to end at the last pointer.
		for len(maskFromHeap) > 0 && maskFromHeap[len(maskFromHeap)-1] == 0 {
			maskFromHeap = maskFromHeap[:len(maskFromHeap)-1]
		}

		if et.Kind_&kindGCProg == 0 {
			// Unroll again, but this time from the type information.
			maskFromType := make([]byte, (limit-base)/goarch.PtrSize)
			tp = s.typePointersOfType(et, base)
			for {
				var addr uintptr
				if tp, addr = tp.next(limit); addr == 0 {
					break
				}
				maskFromType[(addr-base)/goarch.PtrSize] = 1
			}

			// Validate that the prefix of maskFromType is equal to
			// maskFromHeap. maskFromType may contain more pointers than
			// maskFromHeap produces because maskFromHeap may be able to
			// get exact type information for certain classes of objects.
			// With maskFromType, we're always just tiling the type bitmap
			// through to the elemsize.
			//
			// It's OK if maskFromType has pointers in elemsize that extend
			// past the actual populated space; we checked above that all
			// that space is zeroed, so just the GC will just see nil pointers.
			differs := false
			for i := range maskFromHeap {
				if maskFromHeap[i] != maskFromType[i] {
					differs = true
					break
				}
			}

			if differs {
				print("runtime: heap mask=")
				for _, b := range maskFromHeap {
					print(b)
				}
				println()
				print("runtime: type mask=")
				for _, b := range maskFromType {
					print(b)
				}
				println()
				print("runtime: type=", toRType(et).string(), "\n")
				throw("found two different masks from two different methods")
			}
		}

		// Select the heap mask to return. We may not have a type mask.
		mask = maskFromHeap

		// Make sure we keep ep alive. We may have stopped referencing
		// ep's data pointer sometime before this point and it's possible
		// for that memory to get freed.
		KeepAlive(ep)
		return
	}

	// stack
	if gp := getg(); gp.m.curg.stack.lo <= uintptr(p) && uintptr(p) < gp.m.curg.stack.hi {
		found := false
		var u unwinder
		for u.initAt(gp.m.curg.sched.pc, gp.m.curg.sched.sp, 0, gp.m.curg, 0); u.valid(); u.next() {
			if u.frame.sp <= uintptr(p) && uintptr(p) < u.frame.varp {
				found = true
				break
			}
		}
		if found {
			locals, _, _ := u.frame.getStackMap(false)
			if locals.n == 0 {
				return
			}
			size := uintptr(locals.n) * goarch.PtrSize
			n := (*ptrtype)(unsafe.Pointer(t)).Elem.Size_
			mask = make([]byte, n/goarch.PtrSize)
			for i := uintptr(0); i < n; i += goarch.PtrSize {
				off := (uintptr(p) + i - u.frame.varp + size) / goarch.PtrSize
				mask[i/goarch.PtrSize] = locals.ptrbit(off)
			}
		}
		return
	}

	// otherwise, not something the GC knows about.
	// possibly read-only data, like malloc(0).
	// must not have pointers
	return
}

// userArenaHeapBitsSetType is the equivalent of heapSetType but for
// non-slice-backing-store Go values allocated in a user arena chunk. It
// sets up the type metadata for the value with type typ allocated at address ptr.
// base is the base address of the arena chunk.
func userArenaHeapBitsSetType(typ *_type, ptr unsafe.Pointer, s *mspan) {
	base := s.base()
	h := s.writeUserArenaHeapBits(uintptr(ptr))

	p := typ.GCData // start of 1-bit pointer mask (or GC program)
	var gcProgBits uintptr
	if typ.Kind_&kindGCProg != 0 {
		// Expand gc program, using the object itself for storage.
		gcProgBits = runGCProg(addb(p, 4), (*byte)(ptr))
		p = (*byte)(ptr)
	}
	nb := typ.PtrBytes / goarch.PtrSize

	for i := uintptr(0); i < nb; i += ptrBits {
		k := nb - i
		if k > ptrBits {
			k = ptrBits
		}
		// N.B. On big endian platforms we byte swap the data that we
		// read from GCData, which is always stored in little-endian order
		// by the compiler. writeUserArenaHeapBits handles data in
		// a platform-ordered way for efficiency, but stores back the
		// data in little endian order, since we expose the bitmap through
		// a dummy type.
		h = h.write(s, readUintptr(addb(p, i/8)), k)
	}
	// Note: we call pad here to ensure we emit explicit 0 bits
	// for the pointerless tail of the object. This ensures that
	// there's only a single noMorePtrs mark for the next object
	// to clear. We don't need to do this to clear stale noMorePtrs
	// markers from previous uses because arena chunk pointer bitmaps
	// are always fully cleared when reused.
	h = h.pad(s, typ.Size_-typ.PtrBytes)
	h.flush(s, uintptr(ptr), typ.Size_)

	if typ.Kind_&kindGCProg != 0 {
		// Zero out temporary ptrmask buffer inside object.
		memclrNoHeapPointers(ptr, (gcProgBits+7)/8)
	}

	// Update the PtrBytes value in the type information. After this
	// point, the GC will observe the new bitmap.
	s.largeType.PtrBytes = uintptr(ptr) - base + typ.PtrBytes

	// Double-check that the bitmap was written out correctly.
	const doubleCheck = false
	if doubleCheck {
		doubleCheckHeapPointersInterior(uintptr(ptr), uintptr(ptr), typ.Size_, typ.Size_, typ, &s.largeType, s)
	}
}

// For !goexperiment.AllocHeaders, to pass TestIntendedInlining.
func writeHeapBitsForAddr() {
	panic("not implemented")
}

// For !goexperiment.AllocHeaders.
type heapBits struct {
}

// For !goexperiment.AllocHeaders.
//
//go:nosplit
func heapBitsForAddr(addr, size uintptr) heapBits {
	panic("not implemented")
}

// For !goexperiment.AllocHeaders.
//
//go:nosplit
func (h heapBits) next() (heapBits, uintptr) {
	panic("not implemented")
}

// For !goexperiment.AllocHeaders.
//
//go:nosplit
func (h heapBits) nextFast() (heapBits, uintptr) {
	panic("not implemented")
}
