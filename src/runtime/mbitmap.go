// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
	"internal/runtime/atomic"
	"internal/runtime/sys"
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
		// Synchronize with allocator, in case this came from the conservative scanner.
		// See heapSetTypeLarge for more details.
		typ = (*_type)(atomic.Loadp(unsafe.Pointer(&span.largeType)))
		if typ == nil {
			// Allow a nil type here for delayed zeroing. See mallocgc.
			return typePointers{}
		}
	}
	gcmask := getGCMask(typ)
	return typePointers{elem: addr, addr: addr, mask: readUintptr(gcmask), typ: typ}
}

// typePointersOfType is like typePointersOf, but assumes addr points to one or more
// contiguous instances of the provided type. The provided type must not be nil.
//
// It returns an iterator that tiles typ's gcmask starting from addr. It's the caller's
// responsibility to limit iteration.
//
// nosplit because its callers are nosplit and require all their callees to be nosplit.
//
//go:nosplit
func (span *mspan) typePointersOfType(typ *abi.Type, addr uintptr) typePointers {
	const doubleCheck = false
	if doubleCheck && typ == nil {
		throw("bad type passed to typePointersOfType")
	}
	if span.spanclass.noscan() {
		return typePointers{}
	}
	// Since we have the type, pretend we have a header.
	gcmask := getGCMask(typ)
	return typePointers{elem: addr, addr: addr, mask: readUintptr(gcmask), typ: typ}
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
		tp.mask = readUintptr(addb(getGCMask(tp.typ), (tp.addr-tp.elem)/goarch.PtrSize/8))
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
		tp.mask = readUintptr(getGCMask(tp.typ))

		// We may have exceeded the limit after this. Bail just like next does.
		if tp.addr >= limit {
			return typePointers{}
		}
	} else {
		// Grab the mask, but then clear any bits before the target address and any
		// bits over the limit.
		tp.mask = readUintptr(addb(getGCMask(tp.typ), (tp.addr-tp.elem)/goarch.PtrSize/8))
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
	if typ != nil {
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
	if typ != nil {
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
func (s *mspan) initHeapBits() {
	if goarch.PtrSize == 8 && !s.spanclass.noscan() && s.spanclass.sizeclass() == 1 {
		b := s.heapBits()
		for i := range b {
			b[i] = ^uintptr(0)
		}
	} else if (!s.spanclass.noscan() && heapBitsInSpan(s.elemsize)) || s.isUserArenaChunk {
		b := s.heapBits()
		clear(b)
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
	src0 := readUintptr(getGCMask(typ))

	// Create repetitions of the bitmap if we have a small slice backing store.
	scanSize = typ.PtrBytes
	src := src0
	if typ.Size_ == goarch.PtrSize {
		src = (1 << (dataSize / goarch.PtrSize)) - 1
	} else {
		// N.B. We rely on dataSize being an exact multiple of the type size.
		// The alternative is to be defensive and mask out src to the length
		// of dataSize. The purpose is to save on one additional masking operation.
		if doubleCheckHeapSetType && !asanenabled && dataSize%typ.Size_ != 0 {
			throw("runtime: (*mspan).writeHeapBitsSmall: dataSize is not a multiple of typ.Size_")
		}
		for i := typ.Size_; i < dataSize; i += typ.Size_ {
			src |= src0 << (i / goarch.PtrSize)
			scanSize += typ.Size_
		}
		if asanenabled {
			// Mask src down to dataSize. dataSize is going to be a strange size because of
			// the redzone required for allocations when asan is enabled.
			src &= (1 << (dataSize / goarch.PtrSize)) - 1
		}
	}

	// Since we're never writing more than one uintptr's worth of bits, we're either going
	// to do one or two writes.
	dst := unsafe.Pointer(span.base() + pageSize - pageSize/goarch.PtrSize/8)
	o := (x - span.base()) / goarch.PtrSize
	i := o / ptrBits
	j := o % ptrBits
	bits := span.elemsize / goarch.PtrSize
	if j+bits > ptrBits {
		// Two writes.
		bits0 := ptrBits - j
		bits1 := bits - bits0
		dst0 := (*uintptr)(add(dst, (i+0)*goarch.PtrSize))
		dst1 := (*uintptr)(add(dst, (i+1)*goarch.PtrSize))
		*dst0 = (*dst0)&(^uintptr(0)>>bits0) | (src << j)
		*dst1 = (*dst1)&^((1<<bits1)-1) | (src >> bits0)
	} else {
		// One write.
		dst := (*uintptr)(add(dst, i*goarch.PtrSize))
		*dst = (*dst)&^(((1<<bits)-1)<<j) | (src << j)
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

// heapSetType* functions record that the new allocation [x, x+size)
// holds in [x, x+dataSize) one or more values of type typ.
// (The number of values is given by dataSize / typ.Size.)
// If dataSize < size, the fragment [x+dataSize, x+size) is
// recorded as non-pointer data.
// It is known that the type has pointers somewhere;
// malloc does not call heapSetType* when there are no pointers.
//
// There can be read-write races between heapSetType* and things
// that read the heap metadata like scanobject. However, since
// heapSetType* is only used for objects that have not yet been
// made reachable, readers will ignore bits being modified by this
// function. This does mean this function cannot transiently modify
// shared memory that belongs to neighboring objects. Also, on weakly-ordered
// machines, callers must execute a store/store (publication) barrier
// between calling this function and making the object reachable.

const doubleCheckHeapSetType = doubleCheckMalloc

func heapSetTypeNoHeader(x, dataSize uintptr, typ *_type, span *mspan) uintptr {
	if doubleCheckHeapSetType && (!heapBitsInSpan(dataSize) || !heapBitsInSpan(span.elemsize)) {
		throw("tried to write heap bits, but no heap bits in span")
	}
	scanSize := span.writeHeapBitsSmall(x, dataSize, typ)
	if doubleCheckHeapSetType {
		doubleCheckHeapType(x, dataSize, typ, nil, span)
	}
	return scanSize
}

func heapSetTypeSmallHeader(x, dataSize uintptr, typ *_type, header **_type, span *mspan) uintptr {
	*header = typ
	if doubleCheckHeapSetType {
		doubleCheckHeapType(x, dataSize, typ, header, span)
	}
	return span.elemsize
}

func heapSetTypeLarge(x, dataSize uintptr, typ *_type, span *mspan) uintptr {
	gctyp := typ
	// Write out the header atomically to synchronize with the garbage collector.
	//
	// This atomic store is paired with an atomic load in typePointersOfUnchecked.
	// This store ensures that initializing x's memory cannot be reordered after
	// this store. Meanwhile the load in typePointersOfUnchecked ensures that
	// reading x's memory cannot be reordered before largeType is loaded. Together,
	// these two operations guarantee that the garbage collector can only see
	// initialized memory if largeType is non-nil.
	//
	// Gory details below...
	//
	// Ignoring conservative scanning for a moment, this store need not be atomic
	// if we have a publication barrier on our side. This is because the garbage
	// collector cannot observe x unless:
	//   1. It stops this goroutine and scans its stack, or
	//   2. We return from mallocgc and publish the pointer somewhere.
	// Either case requires a write on our side, followed by some synchronization
	// followed by a read by the garbage collector.
	//
	// In case (1), the garbage collector can only observe a nil largeType, since it
	// had to stop our goroutine when it was preemptible during zeroing. For the
	// duration of the zeroing, largeType is nil and the object has nothing interesting
	// for the garbage collector to look at, so the garbage collector will not access
	// the object at all.
	//
	// In case (2), the garbage collector can also observe a nil largeType. This
	// might happen if the object was newly allocated, and a new GC cycle didn't start
	// (that would require a global barrier, STW). In this case, the garbage collector
	// will once again ignore the object, and that's safe because objects are
	// allocate-black.
	//
	// However, the garbage collector can also observe a non-nil largeType in case (2).
	// This is still okay, since to access the object's memory, it must have first
	// loaded the object's pointer from somewhere. This makes the access of the object's
	// memory a data-dependent load, and our publication barrier in the allocator
	// guarantees that a data-dependent load must observe a version of the object's
	// data from after the publication barrier executed.
	//
	// Unfortunately conservative scanning is a problem. There's no guarantee of a
	// data dependency as in case (2) because conservative scanning can produce pointers
	// 'out of thin air' in that it need not have been written somewhere by the allocating
	// thread first. It might not even be a pointer, or it could be a pointer written to
	// some stack location long ago. This is the fundamental reason why we need
	// explicit synchronization somewhere in this whole mess. We choose to put that
	// synchronization on largeType.
	//
	// As described at the very top, the treating largeType as an atomic variable, on
	// both the reader and writer side, is sufficient to ensure that only initialized
	// memory at x will be observed if largeType is non-nil.
	atomic.StorepNoWB(unsafe.Pointer(&span.largeType), unsafe.Pointer(gctyp))
	if doubleCheckHeapSetType {
		doubleCheckHeapType(x, dataSize, typ, &span.largeType, span)
	}
	return span.elemsize
}

func doubleCheckHeapType(x, dataSize uintptr, gctyp *_type, header **_type, span *mspan) {
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
				want = *addb(getGCMask(typ), j/8)>>(j%8)&1 != 0
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
	print("runtime: hasHeader=", header != nil, " typ.Size_=", typ.Size_, " TFlagGCMaskOnDemaind=", typ.TFlag&abi.TFlagGCMaskOnDemand != 0, "\n")
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
				want = *addb(getGCMask(typ), j/8)>>(j%8)&1 != 0
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
				want = *addb(getGCMask(typ), j/8)>>(j%8)&1 != 0
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
	if typ == nil {
		return
	}
	if typ.Kind_&abi.KindMask == abi.Interface {
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

// addb returns the byte pointer p+n.
//
//go:nowritebarrier
//go:nosplit
func addb(p *byte, n uintptr) *byte {
	// Note: wrote out full expression instead of calling add(p, n)
	// to reduce the number of temporaries generated by the
	// compiler for this trivial expression during inlining.
	return (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + n))
}

// subtractb returns the byte pointer p-n.
//
//go:nowritebarrier
//go:nosplit
func subtractb(p *byte, n uintptr) *byte {
	// Note: wrote out full expression instead of calling add(p, -n)
	// to reduce the number of temporaries generated by the
	// compiler for this trivial expression during inlining.
	return (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) - n))
}

// add1 returns the byte pointer p+1.
//
//go:nowritebarrier
//go:nosplit
func add1(p *byte) *byte {
	// Note: wrote out full expression instead of calling addb(p, 1)
	// to reduce the number of temporaries generated by the
	// compiler for this trivial expression during inlining.
	return (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + 1))
}

// subtract1 returns the byte pointer p-1.
//
// nosplit because it is used during write barriers and must not be preempted.
//
//go:nowritebarrier
//go:nosplit
func subtract1(p *byte) *byte {
	// Note: wrote out full expression instead of calling subtractb(p, 1)
	// to reduce the number of temporaries generated by the
	// compiler for this trivial expression during inlining.
	return (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) - 1))
}

// markBits provides access to the mark bit for an object in the heap.
// bytep points to the byte holding the mark bit.
// mask is a byte with a single bit set that can be &ed with *bytep
// to see if the bit has been set.
// *m.byte&m.mask != 0 indicates the mark bit is set.
// index can be used along with span information to generate
// the address of the object in the heap.
// We maintain one set of mark bits for allocation and one for
// marking purposes.
type markBits struct {
	bytep *uint8
	mask  uint8
	index uintptr
}

//go:nosplit
func (s *mspan) allocBitsForIndex(allocBitIndex uintptr) markBits {
	bytep, mask := s.allocBits.bitp(allocBitIndex)
	return markBits{bytep, mask, allocBitIndex}
}

// refillAllocCache takes 8 bytes s.allocBits starting at whichByte
// and negates them so that ctz (count trailing zeros) instructions
// can be used. It then places these 8 bytes into the cached 64 bit
// s.allocCache.
func (s *mspan) refillAllocCache(whichByte uint16) {
	bytes := (*[8]uint8)(unsafe.Pointer(s.allocBits.bytep(uintptr(whichByte))))
	aCache := uint64(0)
	aCache |= uint64(bytes[0])
	aCache |= uint64(bytes[1]) << (1 * 8)
	aCache |= uint64(bytes[2]) << (2 * 8)
	aCache |= uint64(bytes[3]) << (3 * 8)
	aCache |= uint64(bytes[4]) << (4 * 8)
	aCache |= uint64(bytes[5]) << (5 * 8)
	aCache |= uint64(bytes[6]) << (6 * 8)
	aCache |= uint64(bytes[7]) << (7 * 8)
	s.allocCache = ^aCache
}

// nextFreeIndex returns the index of the next free object in s at
// or after s.freeindex.
// There are hardware instructions that can be used to make this
// faster if profiling warrants it.
func (s *mspan) nextFreeIndex() uint16 {
	sfreeindex := s.freeindex
	snelems := s.nelems
	if sfreeindex == snelems {
		return sfreeindex
	}
	if sfreeindex > snelems {
		throw("s.freeindex > s.nelems")
	}

	aCache := s.allocCache

	bitIndex := sys.TrailingZeros64(aCache)
	for bitIndex == 64 {
		// Move index to start of next cached bits.
		sfreeindex = (sfreeindex + 64) &^ (64 - 1)
		if sfreeindex >= snelems {
			s.freeindex = snelems
			return snelems
		}
		whichByte := sfreeindex / 8
		// Refill s.allocCache with the next 64 alloc bits.
		s.refillAllocCache(whichByte)
		aCache = s.allocCache
		bitIndex = sys.TrailingZeros64(aCache)
		// nothing available in cached bits
		// grab the next 8 bytes and try again.
	}
	result := sfreeindex + uint16(bitIndex)
	if result >= snelems {
		s.freeindex = snelems
		return snelems
	}

	s.allocCache >>= uint(bitIndex + 1)
	sfreeindex = result + 1

	if sfreeindex%64 == 0 && sfreeindex != snelems {
		// We just incremented s.freeindex so it isn't 0.
		// As each 1 in s.allocCache was encountered and used for allocation
		// it was shifted away. At this point s.allocCache contains all 0s.
		// Refill s.allocCache so that it corresponds
		// to the bits at s.allocBits starting at s.freeindex.
		whichByte := sfreeindex / 8
		s.refillAllocCache(whichByte)
	}
	s.freeindex = sfreeindex
	return result
}

// isFree reports whether the index'th object in s is unallocated.
//
// The caller must ensure s.state is mSpanInUse, and there must have
// been no preemption points since ensuring this (which could allow a
// GC transition, which would allow the state to change).
func (s *mspan) isFree(index uintptr) bool {
	if index < uintptr(s.freeIndexForScan) {
		return false
	}
	bytep, mask := s.allocBits.bitp(index)
	return *bytep&mask == 0
}

// divideByElemSize returns n/s.elemsize.
// n must be within [0, s.npages*_PageSize),
// or may be exactly s.npages*_PageSize
// if s.elemsize is from sizeclasses.go.
//
// nosplit, because it is called by objIndex, which is nosplit
//
//go:nosplit
func (s *mspan) divideByElemSize(n uintptr) uintptr {
	const doubleCheck = false

	// See explanation in mksizeclasses.go's computeDivMagic.
	q := uintptr((uint64(n) * uint64(s.divMul)) >> 32)

	if doubleCheck && q != n/s.elemsize {
		println(n, "/", s.elemsize, "should be", n/s.elemsize, "but got", q)
		throw("bad magic division")
	}
	return q
}

// nosplit, because it is called by other nosplit code like findObject
//
//go:nosplit
func (s *mspan) objIndex(p uintptr) uintptr {
	return s.divideByElemSize(p - s.base())
}

func markBitsForAddr(p uintptr) markBits {
	s := spanOf(p)
	objIndex := s.objIndex(p)
	return s.markBitsForIndex(objIndex)
}

func (s *mspan) markBitsForIndex(objIndex uintptr) markBits {
	bytep, mask := s.gcmarkBits.bitp(objIndex)
	return markBits{bytep, mask, objIndex}
}

func (s *mspan) markBitsForBase() markBits {
	return markBits{&s.gcmarkBits.x, uint8(1), 0}
}

// isMarked reports whether mark bit m is set.
func (m markBits) isMarked() bool {
	return *m.bytep&m.mask != 0
}

// setMarked sets the marked bit in the markbits, atomically.
func (m markBits) setMarked() {
	// Might be racing with other updates, so use atomic update always.
	// We used to be clever here and use a non-atomic update in certain
	// cases, but it's not worth the risk.
	atomic.Or8(m.bytep, m.mask)
}

// setMarkedNonAtomic sets the marked bit in the markbits, non-atomically.
func (m markBits) setMarkedNonAtomic() {
	*m.bytep |= m.mask
}

// clearMarked clears the marked bit in the markbits, atomically.
func (m markBits) clearMarked() {
	// Might be racing with other updates, so use atomic update always.
	// We used to be clever here and use a non-atomic update in certain
	// cases, but it's not worth the risk.
	atomic.And8(m.bytep, ^m.mask)
}

// markBitsForSpan returns the markBits for the span base address base.
func markBitsForSpan(base uintptr) (mbits markBits) {
	mbits = markBitsForAddr(base)
	if mbits.mask != 1 {
		throw("markBitsForSpan: unaligned start")
	}
	return mbits
}

// advance advances the markBits to the next object in the span.
func (m *markBits) advance() {
	if m.mask == 1<<7 {
		m.bytep = (*uint8)(unsafe.Pointer(uintptr(unsafe.Pointer(m.bytep)) + 1))
		m.mask = 1
	} else {
		m.mask = m.mask << 1
	}
	m.index++
}

// clobberdeadPtr is a special value that is used by the compiler to
// clobber dead stack slots, when -clobberdead flag is set.
const clobberdeadPtr = uintptr(0xdeaddead | 0xdeaddead<<((^uintptr(0)>>63)*32))

// badPointer throws bad pointer in heap panic.
func badPointer(s *mspan, p, refBase, refOff uintptr) {
	// Typically this indicates an incorrect use
	// of unsafe or cgo to store a bad pointer in
	// the Go heap. It may also indicate a runtime
	// bug.
	//
	// TODO(austin): We could be more aggressive
	// and detect pointers to unallocated objects
	// in allocated spans.
	printlock()
	print("runtime: pointer ", hex(p))
	if s != nil {
		state := s.state.get()
		if state != mSpanInUse {
			print(" to unallocated span")
		} else {
			print(" to unused region of span")
		}
		print(" span.base()=", hex(s.base()), " span.limit=", hex(s.limit), " span.state=", state)
	}
	print("\n")
	if refBase != 0 {
		print("runtime: found in object at *(", hex(refBase), "+", hex(refOff), ")\n")
		gcDumpObject("object", refBase, refOff)
	}
	getg().m.traceback = 2
	throw("found bad pointer in Go heap (incorrect use of unsafe or cgo?)")
}

// findObject returns the base address for the heap object containing
// the address p, the object's span, and the index of the object in s.
// If p does not point into a heap object, it returns base == 0.
//
// If p points is an invalid heap pointer and debug.invalidptr != 0,
// findObject panics.
//
// refBase and refOff optionally give the base address of the object
// in which the pointer p was found and the byte offset at which it
// was found. These are used for error reporting.
//
// It is nosplit so it is safe for p to be a pointer to the current goroutine's stack.
// Since p is a uintptr, it would not be adjusted if the stack were to move.
//
// findObject should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/bytedance/sonic
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname findObject
//go:nosplit
func findObject(p, refBase, refOff uintptr) (base uintptr, s *mspan, objIndex uintptr) {
	s = spanOf(p)
	// If s is nil, the virtual address has never been part of the heap.
	// This pointer may be to some mmap'd region, so we allow it.
	if s == nil {
		if (GOARCH == "amd64" || GOARCH == "arm64") && p == clobberdeadPtr && debug.invalidptr != 0 {
			// Crash if clobberdeadPtr is seen. Only on AMD64 and ARM64 for now,
			// as they are the only platform where compiler's clobberdead mode is
			// implemented. On these platforms clobberdeadPtr cannot be a valid address.
			badPointer(s, p, refBase, refOff)
		}
		return
	}
	// If p is a bad pointer, it may not be in s's bounds.
	//
	// Check s.state to synchronize with span initialization
	// before checking other fields. See also spanOfHeap.
	if state := s.state.get(); state != mSpanInUse || p < s.base() || p >= s.limit {
		// Pointers into stacks are also ok, the runtime manages these explicitly.
		if state == mSpanManual {
			return
		}
		// The following ensures that we are rigorous about what data
		// structures hold valid pointers.
		if debug.invalidptr != 0 {
			badPointer(s, p, refBase, refOff)
		}
		return
	}

	objIndex = s.objIndex(p)
	base = s.base() + objIndex*s.elemsize
	return
}

// reflect_verifyNotInHeapPtr reports whether converting the not-in-heap pointer into a unsafe.Pointer is ok.
//
//go:linkname reflect_verifyNotInHeapPtr reflect.verifyNotInHeapPtr
func reflect_verifyNotInHeapPtr(p uintptr) bool {
	// Conversion to a pointer is ok as long as findObject above does not call badPointer.
	// Since we're already promised that p doesn't point into the heap, just disallow heap
	// pointers and the special clobbered pointer.
	return spanOf(p) == nil && p != clobberdeadPtr
}

const ptrBits = 8 * goarch.PtrSize

// bulkBarrierBitmap executes write barriers for copying from [src,
// src+size) to [dst, dst+size) using a 1-bit pointer bitmap. src is
// assumed to start maskOffset bytes into the data covered by the
// bitmap in bits (which may not be a multiple of 8).
//
// This is used by bulkBarrierPreWrite for writes to data and BSS.
//
//go:nosplit
func bulkBarrierBitmap(dst, src, size, maskOffset uintptr, bits *uint8) {
	word := maskOffset / goarch.PtrSize
	bits = addb(bits, word/8)
	mask := uint8(1) << (word % 8)

	buf := &getg().m.p.ptr().wbBuf
	for i := uintptr(0); i < size; i += goarch.PtrSize {
		if mask == 0 {
			bits = addb(bits, 1)
			if *bits == 0 {
				// Skip 8 words.
				i += 7 * goarch.PtrSize
				continue
			}
			mask = 1
		}
		if *bits&mask != 0 {
			dstx := (*uintptr)(unsafe.Pointer(dst + i))
			if src == 0 {
				p := buf.get1()
				p[0] = *dstx
			} else {
				srcx := (*uintptr)(unsafe.Pointer(src + i))
				p := buf.get2()
				p[0] = *dstx
				p[1] = *srcx
			}
		}
		mask <<= 1
	}
}

// typeBitsBulkBarrier executes a write barrier for every
// pointer that would be copied from [src, src+size) to [dst,
// dst+size) by a memmove using the type bitmap to locate those
// pointer slots.
//
// The type typ must correspond exactly to [src, src+size) and [dst, dst+size).
// dst, src, and size must be pointer-aligned.
//
// Must not be preempted because it typically runs right before memmove,
// and the GC must observe them as an atomic action.
//
// Callers must perform cgo checks if goexperiment.CgoCheck2.
//
//go:nosplit
func typeBitsBulkBarrier(typ *_type, dst, src, size uintptr) {
	if typ == nil {
		throw("runtime: typeBitsBulkBarrier without type")
	}
	if typ.Size_ != size {
		println("runtime: typeBitsBulkBarrier with type ", toRType(typ).string(), " of size ", typ.Size_, " but memory size", size)
		throw("runtime: invalid typeBitsBulkBarrier")
	}
	if !writeBarrier.enabled {
		return
	}
	ptrmask := getGCMask(typ)
	buf := &getg().m.p.ptr().wbBuf
	var bits uint32
	for i := uintptr(0); i < typ.PtrBytes; i += goarch.PtrSize {
		if i&(goarch.PtrSize*8-1) == 0 {
			bits = uint32(*ptrmask)
			ptrmask = addb(ptrmask, 1)
		} else {
			bits = bits >> 1
		}
		if bits&1 != 0 {
			dstx := (*uintptr)(unsafe.Pointer(dst + i))
			srcx := (*uintptr)(unsafe.Pointer(src + i))
			p := buf.get2()
			p[0] = *dstx
			p[1] = *srcx
		}
	}
}

// countAlloc returns the number of objects allocated in span s by
// scanning the mark bitmap.
func (s *mspan) countAlloc() int {
	count := 0
	bytes := divRoundUp(uintptr(s.nelems), 8)
	// Iterate over each 8-byte chunk and count allocations
	// with an intrinsic. Note that newMarkBits guarantees that
	// gcmarkBits will be 8-byte aligned, so we don't have to
	// worry about edge cases, irrelevant bits will simply be zero.
	for i := uintptr(0); i < bytes; i += 8 {
		// Extract 64 bits from the byte pointer and get a OnesCount.
		// Note that the unsafe cast here doesn't preserve endianness,
		// but that's OK. We only care about how many bits are 1, not
		// about the order we discover them in.
		mrkBits := *(*uint64)(unsafe.Pointer(s.gcmarkBits.bytep(i)))
		count += sys.OnesCount64(mrkBits)
	}
	return count
}

// Read the bytes starting at the aligned pointer p into a uintptr.
// Read is little-endian.
func readUintptr(p *byte) uintptr {
	x := *(*uintptr)(unsafe.Pointer(p))
	if goarch.BigEndian {
		if goarch.PtrSize == 8 {
			return uintptr(sys.Bswap64(uint64(x)))
		}
		return uintptr(sys.Bswap32(uint32(x)))
	}
	return x
}

var debugPtrmask struct {
	lock mutex
	data *byte
}

// progToPointerMask returns the 1-bit pointer mask output by the GC program prog.
// size the size of the region described by prog, in bytes.
// The resulting bitvector will have no more than size/goarch.PtrSize bits.
func progToPointerMask(prog *byte, size uintptr) bitvector {
	n := (size/goarch.PtrSize + 7) / 8
	x := (*[1 << 30]byte)(persistentalloc(n+1, 1, &memstats.buckhash_sys))[:n+1]
	x[len(x)-1] = 0xa1 // overflow check sentinel
	n = runGCProg(prog, &x[0])
	if x[len(x)-1] != 0xa1 {
		throw("progToPointerMask: overflow")
	}
	return bitvector{int32(n), &x[0]}
}

// Packed GC pointer bitmaps, aka GC programs.
//
// For large types containing arrays, the type information has a
// natural repetition that can be encoded to save space in the
// binary and in the memory representation of the type information.
//
// The encoding is a simple Lempel-Ziv style bytecode machine
// with the following instructions:
//
//	00000000: stop
//	0nnnnnnn: emit n bits copied from the next (n+7)/8 bytes
//	10000000 n c: repeat the previous n bits c times; n, c are varints
//	1nnnnnnn c: repeat the previous n bits c times; c is a varint
//
// Currently, gc programs are only used for describing data and bss
// sections of the binary.

// runGCProg returns the number of 1-bit entries written to memory.
func runGCProg(prog, dst *byte) uintptr {
	dstStart := dst

	// Bits waiting to be written to memory.
	var bits uintptr
	var nbits uintptr

	p := prog
Run:
	for {
		// Flush accumulated full bytes.
		// The rest of the loop assumes that nbits <= 7.
		for ; nbits >= 8; nbits -= 8 {
			*dst = uint8(bits)
			dst = add1(dst)
			bits >>= 8
		}

		// Process one instruction.
		inst := uintptr(*p)
		p = add1(p)
		n := inst & 0x7F
		if inst&0x80 == 0 {
			// Literal bits; n == 0 means end of program.
			if n == 0 {
				// Program is over.
				break Run
			}
			nbyte := n / 8
			for i := uintptr(0); i < nbyte; i++ {
				bits |= uintptr(*p) << nbits
				p = add1(p)
				*dst = uint8(bits)
				dst = add1(dst)
				bits >>= 8
			}
			if n %= 8; n > 0 {
				bits |= uintptr(*p) << nbits
				p = add1(p)
				nbits += n
			}
			continue Run
		}

		// Repeat. If n == 0, it is encoded in a varint in the next bytes.
		if n == 0 {
			for off := uint(0); ; off += 7 {
				x := uintptr(*p)
				p = add1(p)
				n |= (x & 0x7F) << off
				if x&0x80 == 0 {
					break
				}
			}
		}

		// Count is encoded in a varint in the next bytes.
		c := uintptr(0)
		for off := uint(0); ; off += 7 {
			x := uintptr(*p)
			p = add1(p)
			c |= (x & 0x7F) << off
			if x&0x80 == 0 {
				break
			}
		}
		c *= n // now total number of bits to copy

		// If the number of bits being repeated is small, load them
		// into a register and use that register for the entire loop
		// instead of repeatedly reading from memory.
		// Handling fewer than 8 bits here makes the general loop simpler.
		// The cutoff is goarch.PtrSize*8 - 7 to guarantee that when we add
		// the pattern to a bit buffer holding at most 7 bits (a partial byte)
		// it will not overflow.
		src := dst
		const maxBits = goarch.PtrSize*8 - 7
		if n <= maxBits {
			// Start with bits in output buffer.
			pattern := bits
			npattern := nbits

			// If we need more bits, fetch them from memory.
			src = subtract1(src)
			for npattern < n {
				pattern <<= 8
				pattern |= uintptr(*src)
				src = subtract1(src)
				npattern += 8
			}

			// We started with the whole bit output buffer,
			// and then we loaded bits from whole bytes.
			// Either way, we might now have too many instead of too few.
			// Discard the extra.
			if npattern > n {
				pattern >>= npattern - n
				npattern = n
			}

			// Replicate pattern to at most maxBits.
			if npattern == 1 {
				// One bit being repeated.
				// If the bit is 1, make the pattern all 1s.
				// If the bit is 0, the pattern is already all 0s,
				// but we can claim that the number of bits
				// in the word is equal to the number we need (c),
				// because right shift of bits will zero fill.
				if pattern == 1 {
					pattern = 1<<maxBits - 1
					npattern = maxBits
				} else {
					npattern = c
				}
			} else {
				b := pattern
				nb := npattern
				if nb+nb <= maxBits {
					// Double pattern until the whole uintptr is filled.
					for nb <= goarch.PtrSize*8 {
						b |= b << nb
						nb += nb
					}
					// Trim away incomplete copy of original pattern in high bits.
					// TODO(rsc): Replace with table lookup or loop on systems without divide?
					nb = maxBits / npattern * npattern
					b &= 1<<nb - 1
					pattern = b
					npattern = nb
				}
			}

			// Add pattern to bit buffer and flush bit buffer, c/npattern times.
			// Since pattern contains >8 bits, there will be full bytes to flush
			// on each iteration.
			for ; c >= npattern; c -= npattern {
				bits |= pattern << nbits
				nbits += npattern
				for nbits >= 8 {
					*dst = uint8(bits)
					dst = add1(dst)
					bits >>= 8
					nbits -= 8
				}
			}

			// Add final fragment to bit buffer.
			if c > 0 {
				pattern &= 1<<c - 1
				bits |= pattern << nbits
				nbits += c
			}
			continue Run
		}

		// Repeat; n too large to fit in a register.
		// Since nbits <= 7, we know the first few bytes of repeated data
		// are already written to memory.
		off := n - nbits // n > nbits because n > maxBits and nbits <= 7
		// Leading src fragment.
		src = subtractb(src, (off+7)/8)
		if frag := off & 7; frag != 0 {
			bits |= uintptr(*src) >> (8 - frag) << nbits
			src = add1(src)
			nbits += frag
			c -= frag
		}
		// Main loop: load one byte, write another.
		// The bits are rotating through the bit buffer.
		for i := c / 8; i > 0; i-- {
			bits |= uintptr(*src) << nbits
			src = add1(src)
			*dst = uint8(bits)
			dst = add1(dst)
			bits >>= 8
		}
		// Final src fragment.
		if c %= 8; c > 0 {
			bits |= (uintptr(*src) & (1<<c - 1)) << nbits
			nbits += c
		}
	}

	// Write any final bits out, using full-byte writes, even for the final byte.
	totalBits := (uintptr(unsafe.Pointer(dst))-uintptr(unsafe.Pointer(dstStart)))*8 + nbits
	nbits += -nbits & 7
	for ; nbits > 0; nbits -= 8 {
		*dst = uint8(bits)
		dst = add1(dst)
		bits >>= 8
	}
	return totalBits
}

func dumpGCProg(p *byte) {
	nptr := 0
	for {
		x := *p
		p = add1(p)
		if x == 0 {
			print("\t", nptr, " end\n")
			break
		}
		if x&0x80 == 0 {
			print("\t", nptr, " lit ", x, ":")
			n := int(x+7) / 8
			for i := 0; i < n; i++ {
				print(" ", hex(*p))
				p = add1(p)
			}
			print("\n")
			nptr += int(x)
		} else {
			nbit := int(x &^ 0x80)
			if nbit == 0 {
				for nb := uint(0); ; nb += 7 {
					x := *p
					p = add1(p)
					nbit |= int(x&0x7f) << nb
					if x&0x80 == 0 {
						break
					}
				}
			}
			count := 0
			for nb := uint(0); ; nb += 7 {
				x := *p
				p = add1(p)
				count |= int(x&0x7f) << nb
				if x&0x80 == 0 {
					break
				}
			}
			print("\t", nptr, " repeat ", nbit, " × ", count, "\n")
			nptr += nbit * count
		}
	}
}

// Testing.

// reflect_gcbits returns the GC type info for x, for testing.
// The result is the bitmap entries (0 or 1), one entry per byte.
//
//go:linkname reflect_gcbits reflect.gcbits
func reflect_gcbits(x any) []byte {
	return pointerMask(x)
}

// Returns GC type info for the pointer stored in ep for testing.
// If ep points to the stack, only static live information will be returned
// (i.e. not for objects which are only dynamically live stack objects).
func pointerMask(ep any) (mask []byte) {
	e := *efaceOf(&ep)
	p := e.data
	t := e._type

	var et *_type
	if t.Kind_&abi.KindMask != abi.Pointer {
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
