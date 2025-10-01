// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains stub functions that are not meant to be called directly,
// but that will be assembled together using the inlining logic in runtime/_mkmalloc
// to produce a full mallocgc function that's specialized for a span class
// or specific size in the case of the tiny allocator.
//
// To assemble a mallocgc function, the mallocStub function is cloned, and the call to
// inlinedMalloc is replaced with the inlined body of smallScanNoHeaderStub,
// smallNoScanStub or tinyStub, depending on the parameters being specialized.
//
// The size_ (for the tiny case) and elemsize_, sizeclass_, and noscanint_ (for all three cases)
// identifiers are replaced with the value of the parameter in the specialized case.
// The nextFreeFastStub, nextFreeFastTiny, heapSetTypeNoHeaderStub, and writeHeapBitsSmallStub
// functions are also inlined by _mkmalloc.

package runtime

import (
	"internal/goarch"
	"internal/runtime/sys"
	"unsafe"
)

// These identifiers will all be replaced by the inliner. So their values don't
// really matter: they just need to be set so that the stub functions, which
// will never be used on their own, can compile. elemsize_ can't be  set to
// zero because we divide by it in nextFreeFastTiny, and the compiler would
// complain about a division by zero. Its replaced value will always be greater
// than zero.
const elemsize_ = 8
const sizeclass_ = 0
const noscanint_ = 0
const size_ = 0

func malloc0(size uintptr, typ *_type, needzero bool) unsafe.Pointer {
	if doubleCheckMalloc {
		if gcphase == _GCmarktermination {
			throw("mallocgc called with gcphase == _GCmarktermination")
		}
	}

	// Short-circuit zero-sized allocation requests.
	return unsafe.Pointer(&zerobase)
}

func mallocPanic(size uintptr, typ *_type, needzero bool) unsafe.Pointer {
	panic("not defined for sizeclass")
}

// WARNING: mallocStub does not do any work for sanitizers so callers need
// to steer out of this codepath early if sanitizers are enabled.
func mallocStub(size uintptr, typ *_type, needzero bool) unsafe.Pointer {
	if doubleCheckMalloc {
		if gcphase == _GCmarktermination {
			throw("mallocgc called with gcphase == _GCmarktermination")
		}
	}

	// It's possible for any malloc to trigger sweeping, which may in
	// turn queue finalizers. Record this dynamic lock edge.
	// N.B. Compiled away if lockrank experiment is not enabled.
	lockRankMayQueueFinalizer()

	// Pre-malloc debug hooks.
	if debug.malloc {
		if x := preMallocgcDebug(size, typ); x != nil {
			return x
		}
	}

	// Assist the GC if needed.
	if gcBlackenEnabled != 0 {
		deductAssistCredit(size)
	}

	// Actually do the allocation.
	x, elemsize := inlinedMalloc(size, typ, needzero)

	// Notify valgrind, if enabled.
	// To allow the compiler to not know about valgrind, we do valgrind instrumentation
	// unlike the other sanitizers.
	if valgrindenabled {
		valgrindMalloc(x, size)
	}

	// Adjust our GC assist debt to account for internal fragmentation.
	if gcBlackenEnabled != 0 && elemsize != 0 {
		if assistG := getg().m.curg; assistG != nil {
			assistG.gcAssistBytes -= int64(elemsize - size)
		}
	}

	// Post-malloc debug hooks.
	if debug.malloc {
		postMallocgcDebug(x, elemsize, typ)
	}
	return x
}

// inlinedMalloc will never be called. It is defined just so that the compiler can compile
// the mallocStub function, which will also never be called, but instead used as a template
// to generate a size-specialized malloc function. The call to inlinedMalloc in mallocStub
// will be replaced with the inlined body of smallScanNoHeaderStub, smallNoScanStub, or tinyStub
// when generating the size-specialized malloc function. See the comment at the top of this
// file for more information.
func inlinedMalloc(size uintptr, typ *_type, needzero bool) (unsafe.Pointer, uintptr) {
	return unsafe.Pointer(uintptr(0)), 0
}

func doubleCheckSmallScanNoHeader(size uintptr, typ *_type, mp *m) {
	if mp.mallocing != 0 {
		throw("malloc deadlock")
	}
	if mp.gsignal == getg() {
		throw("malloc during signal")
	}
	if typ == nil || !typ.Pointers() {
		throw("noscan allocated in scan-only path")
	}
	if !heapBitsInSpan(size) {
		throw("heap bits in not in span for non-header-only path")
	}
}

func smallScanNoHeaderStub(size uintptr, typ *_type, needzero bool) (unsafe.Pointer, uintptr) {
	const sizeclass = sizeclass_
	const elemsize = elemsize_

	// Set mp.mallocing to keep from being preempted by GC.
	mp := acquirem()
	if doubleCheckMalloc {
		doubleCheckSmallScanNoHeader(size, typ, mp)
	}
	mp.mallocing = 1

	checkGCTrigger := false
	c := getMCache(mp)
	const spc = spanClass(sizeclass<<1) | spanClass(noscanint_)
	span := c.alloc[spc]
	v := nextFreeFastStub(span)
	if v == 0 {
		v, span, checkGCTrigger = c.nextFree(spc)
	}
	x := unsafe.Pointer(v)
	if span.needzero != 0 {
		memclrNoHeapPointers(x, elemsize)
	}
	if goarch.PtrSize == 8 && sizeclass == 1 {
		// initHeapBits already set the pointer bits for the 8-byte sizeclass
		// on 64-bit platforms.
		c.scanAlloc += 8
	} else {
		dataSize := size // make the inliner happy
		x := uintptr(x)
		scanSize := heapSetTypeNoHeaderStub(x, dataSize, typ, span)
		c.scanAlloc += scanSize
	}

	// Ensure that the stores above that initialize x to
	// type-safe memory and set the heap bits occur before
	// the caller can make x observable to the garbage
	// collector. Otherwise, on weakly ordered machines,
	// the garbage collector could follow a pointer to x,
	// but see uninitialized memory or stale heap bits.
	publicationBarrier()

	if writeBarrier.enabled {
		// Allocate black during GC.
		// All slots hold nil so no scanning is needed.
		// This may be racing with GC so do it atomically if there can be
		// a race marking the bit.
		gcmarknewobject(span, uintptr(x))
	} else {
		// Track the last free index before the mark phase. This field
		// is only used by the garbage collector. During the mark phase
		// this is used by the conservative scanner to filter out objects
		// that are both free and recently-allocated. It's safe to do that
		// because we allocate-black if the GC is enabled. The conservative
		// scanner produces pointers out of thin air, so without additional
		// synchronization it might otherwise observe a partially-initialized
		// object, which could crash the program.
		span.freeIndexForScan = span.freeindex
	}

	// Note cache c only valid while m acquired; see #47302
	//
	// N.B. Use the full size because that matches how the GC
	// will update the mem profile on the "free" side.
	//
	// TODO(mknyszek): We should really count the header as part
	// of gc_sys or something. The code below just pretends it is
	// internal fragmentation and matches the GC's accounting by
	// using the whole allocation slot.
	c.nextSample -= int64(elemsize)
	if c.nextSample < 0 || MemProfileRate != c.memProfRate {
		profilealloc(mp, x, elemsize)
	}
	mp.mallocing = 0
	releasem(mp)

	if checkGCTrigger {
		if t := (gcTrigger{kind: gcTriggerHeap}); t.test() {
			gcStart(t)
		}
	}

	return x, elemsize
}

func doubleCheckSmallNoScan(typ *_type, mp *m) {
	if mp.mallocing != 0 {
		throw("malloc deadlock")
	}
	if mp.gsignal == getg() {
		throw("malloc during signal")
	}
	if typ != nil && typ.Pointers() {
		throw("expected noscan type for noscan alloc")
	}
}

func smallNoScanStub(size uintptr, typ *_type, needzero bool) (unsafe.Pointer, uintptr) {
	// TODO(matloob): Add functionality to mkmalloc to allow us to inline a non-constant
	// sizeclass_ and elemsize_ value (instead just set to the expressions to look up the size class
	// and elemsize. We'd also need to teach mkmalloc that values that are touched by these (specifically
	// spc below) should turn into vars. This would allow us to generate mallocgcSmallNoScan itself,
	// so that its code could not diverge from the generated functions.
	const sizeclass = sizeclass_
	const elemsize = elemsize_

	// Set mp.mallocing to keep from being preempted by GC.
	mp := acquirem()
	if doubleCheckMalloc {
		doubleCheckSmallNoScan(typ, mp)
	}
	mp.mallocing = 1

	checkGCTrigger := false
	c := getMCache(mp)
	const spc = spanClass(sizeclass<<1) | spanClass(noscanint_)
	span := c.alloc[spc]
	v := nextFreeFastStub(span)
	if v == 0 {
		v, span, checkGCTrigger = c.nextFree(spc)
	}
	x := unsafe.Pointer(v)
	if needzero && span.needzero != 0 {
		memclrNoHeapPointers(x, elemsize)
	}

	// Ensure that the stores above that initialize x to
	// type-safe memory and set the heap bits occur before
	// the caller can make x observable to the garbage
	// collector. Otherwise, on weakly ordered machines,
	// the garbage collector could follow a pointer to x,
	// but see uninitialized memory or stale heap bits.
	publicationBarrier()

	if writeBarrier.enabled {
		// Allocate black during GC.
		// All slots hold nil so no scanning is needed.
		// This may be racing with GC so do it atomically if there can be
		// a race marking the bit.
		gcmarknewobject(span, uintptr(x))
	} else {
		// Track the last free index before the mark phase. This field
		// is only used by the garbage collector. During the mark phase
		// this is used by the conservative scanner to filter out objects
		// that are both free and recently-allocated. It's safe to do that
		// because we allocate-black if the GC is enabled. The conservative
		// scanner produces pointers out of thin air, so without additional
		// synchronization it might otherwise observe a partially-initialized
		// object, which could crash the program.
		span.freeIndexForScan = span.freeindex
	}

	// Note cache c only valid while m acquired; see #47302
	//
	// N.B. Use the full size because that matches how the GC
	// will update the mem profile on the "free" side.
	//
	// TODO(mknyszek): We should really count the header as part
	// of gc_sys or something. The code below just pretends it is
	// internal fragmentation and matches the GC's accounting by
	// using the whole allocation slot.
	c.nextSample -= int64(elemsize)
	if c.nextSample < 0 || MemProfileRate != c.memProfRate {
		profilealloc(mp, x, elemsize)
	}
	mp.mallocing = 0
	releasem(mp)

	if checkGCTrigger {
		if t := (gcTrigger{kind: gcTriggerHeap}); t.test() {
			gcStart(t)
		}
	}
	return x, elemsize
}

func doubleCheckTiny(size uintptr, typ *_type, mp *m) {
	if mp.mallocing != 0 {
		throw("malloc deadlock")
	}
	if mp.gsignal == getg() {
		throw("malloc during signal")
	}
	if typ != nil && typ.Pointers() {
		throw("expected noscan for tiny alloc")
	}
}

func tinyStub(size uintptr, typ *_type, needzero bool) (unsafe.Pointer, uintptr) {
	const constsize = size_
	const elemsize = elemsize_

	// Set mp.mallocing to keep from being preempted by GC.
	mp := acquirem()
	if doubleCheckMalloc {
		doubleCheckTiny(constsize, typ, mp)
	}
	mp.mallocing = 1

	// Tiny allocator.
	//
	// Tiny allocator combines several tiny allocation requests
	// into a single memory block. The resulting memory block
	// is freed when all subobjects are unreachable. The subobjects
	// must be noscan (don't have pointers), this ensures that
	// the amount of potentially wasted memory is bounded.
	//
	// Size of the memory block used for combining (maxTinySize) is tunable.
	// Current setting is 16 bytes, which relates to 2x worst case memory
	// wastage (when all but one subobjects are unreachable).
	// 8 bytes would result in no wastage at all, but provides less
	// opportunities for combining.
	// 32 bytes provides more opportunities for combining,
	// but can lead to 4x worst case wastage.
	// The best case winning is 8x regardless of block size.
	//
	// Objects obtained from tiny allocator must not be freed explicitly.
	// So when an object will be freed explicitly, we ensure that
	// its size >= maxTinySize.
	//
	// SetFinalizer has a special case for objects potentially coming
	// from tiny allocator, it such case it allows to set finalizers
	// for an inner byte of a memory block.
	//
	// The main targets of tiny allocator are small strings and
	// standalone escaping variables. On a json benchmark
	// the allocator reduces number of allocations by ~12% and
	// reduces heap size by ~20%.
	c := getMCache(mp)
	off := c.tinyoffset
	// Align tiny pointer for required (conservative) alignment.
	if constsize&7 == 0 {
		off = alignUp(off, 8)
	} else if goarch.PtrSize == 4 && constsize == 12 {
		// Conservatively align 12-byte objects to 8 bytes on 32-bit
		// systems so that objects whose first field is a 64-bit
		// value is aligned to 8 bytes and does not cause a fault on
		// atomic access. See issue 37262.
		// TODO(mknyszek): Remove this workaround if/when issue 36606
		// is resolved.
		off = alignUp(off, 8)
	} else if constsize&3 == 0 {
		off = alignUp(off, 4)
	} else if constsize&1 == 0 {
		off = alignUp(off, 2)
	}
	if off+constsize <= maxTinySize && c.tiny != 0 {
		// The object fits into existing tiny block.
		x := unsafe.Pointer(c.tiny + off)
		c.tinyoffset = off + constsize
		c.tinyAllocs++
		mp.mallocing = 0
		releasem(mp)
		return x, 0
	}
	// Allocate a new maxTinySize block.
	checkGCTrigger := false
	span := c.alloc[tinySpanClass]
	v := nextFreeFastTiny(span)
	if v == 0 {
		v, span, checkGCTrigger = c.nextFree(tinySpanClass)
	}
	x := unsafe.Pointer(v)
	(*[2]uint64)(x)[0] = 0 // Always zero
	(*[2]uint64)(x)[1] = 0
	// See if we need to replace the existing tiny block with the new one
	// based on amount of remaining free space.
	if !raceenabled && (constsize < c.tinyoffset || c.tiny == 0) {
		// Note: disabled when race detector is on, see comment near end of this function.
		c.tiny = uintptr(x)
		c.tinyoffset = constsize
	}

	// Ensure that the stores above that initialize x to
	// type-safe memory and set the heap bits occur before
	// the caller can make x observable to the garbage
	// collector. Otherwise, on weakly ordered machines,
	// the garbage collector could follow a pointer to x,
	// but see uninitialized memory or stale heap bits.
	publicationBarrier()

	if writeBarrier.enabled {
		// Allocate black during GC.
		// All slots hold nil so no scanning is needed.
		// This may be racing with GC so do it atomically if there can be
		// a race marking the bit.
		gcmarknewobject(span, uintptr(x))
	} else {
		// Track the last free index before the mark phase. This field
		// is only used by the garbage collector. During the mark phase
		// this is used by the conservative scanner to filter out objects
		// that are both free and recently-allocated. It's safe to do that
		// because we allocate-black if the GC is enabled. The conservative
		// scanner produces pointers out of thin air, so without additional
		// synchronization it might otherwise observe a partially-initialized
		// object, which could crash the program.
		span.freeIndexForScan = span.freeindex
	}

	// Note cache c only valid while m acquired; see #47302
	//
	// N.B. Use the full size because that matches how the GC
	// will update the mem profile on the "free" side.
	//
	// TODO(mknyszek): We should really count the header as part
	// of gc_sys or something. The code below just pretends it is
	// internal fragmentation and matches the GC's accounting by
	// using the whole allocation slot.
	c.nextSample -= int64(elemsize)
	if c.nextSample < 0 || MemProfileRate != c.memProfRate {
		profilealloc(mp, x, elemsize)
	}
	mp.mallocing = 0
	releasem(mp)

	if checkGCTrigger {
		if t := (gcTrigger{kind: gcTriggerHeap}); t.test() {
			gcStart(t)
		}
	}

	if raceenabled {
		// Pad tinysize allocations so they are aligned with the end
		// of the tinyalloc region. This ensures that any arithmetic
		// that goes off the top end of the object will be detectable
		// by checkptr (issue 38872).
		// Note that we disable tinyalloc when raceenabled for this to work.
		// TODO: This padding is only performed when the race detector
		// is enabled. It would be nice to enable it if any package
		// was compiled with checkptr, but there's no easy way to
		// detect that (especially at compile time).
		// TODO: enable this padding for all allocations, not just
		// tinyalloc ones. It's tricky because of pointer maps.
		// Maybe just all noscan objects?
		x = add(x, elemsize-constsize)
	}
	return x, elemsize
}

// TODO(matloob): Should we let the go compiler inline this instead of using mkmalloc?
// We won't be able to use elemsize_ but that's probably ok.
func nextFreeFastTiny(span *mspan) gclinkptr {
	const nbytes = 8192
	const nelems = uint16((nbytes - unsafe.Sizeof(spanInlineMarkBits{})) / elemsize_)
	var nextFreeFastResult gclinkptr
	if span.allocCache != 0 {
		theBit := sys.TrailingZeros64(span.allocCache) // Is there a free object in the allocCache?
		result := span.freeindex + uint16(theBit)
		if result < nelems {
			freeidx := result + 1
			if !(freeidx%64 == 0 && freeidx != nelems) {
				span.allocCache >>= uint(theBit + 1)
				span.freeindex = freeidx
				span.allocCount++
				nextFreeFastResult = gclinkptr(uintptr(result)*elemsize_ + span.base())
			}
		}
	}
	return nextFreeFastResult
}

func nextFreeFastStub(span *mspan) gclinkptr {
	var nextFreeFastResult gclinkptr
	if span.allocCache != 0 {
		theBit := sys.TrailingZeros64(span.allocCache) // Is there a free object in the allocCache?
		result := span.freeindex + uint16(theBit)
		if result < span.nelems {
			freeidx := result + 1
			if !(freeidx%64 == 0 && freeidx != span.nelems) {
				span.allocCache >>= uint(theBit + 1)
				span.freeindex = freeidx
				span.allocCount++
				nextFreeFastResult = gclinkptr(uintptr(result)*elemsize_ + span.base())
			}
		}
	}
	return nextFreeFastResult
}

func heapSetTypeNoHeaderStub(x, dataSize uintptr, typ *_type, span *mspan) uintptr {
	if doubleCheckHeapSetType && (!heapBitsInSpan(dataSize) || !heapBitsInSpan(elemsize_)) {
		throw("tried to write heap bits, but no heap bits in span")
	}
	scanSize := writeHeapBitsSmallStub(span, x, dataSize, typ)
	if doubleCheckHeapSetType {
		doubleCheckHeapType(x, dataSize, typ, nil, span)
	}
	return scanSize
}

// writeHeapBitsSmallStub writes the heap bits for small objects whose ptr/scalar data is
// stored as a bitmap at the end of the span.
//
// Assumes dataSize is <= ptrBits*goarch.PtrSize. x must be a pointer into the span.
// heapBitsInSpan(dataSize) must be true. dataSize must be >= typ.Size_.
//
//go:nosplit
func writeHeapBitsSmallStub(span *mspan, x, dataSize uintptr, typ *_type) uintptr {
	// The objects here are always really small, so a single load is sufficient.
	src0 := readUintptr(getGCMask(typ))

	const elemsize = elemsize_

	// Create repetitions of the bitmap if we have a small slice backing store.
	scanSize := typ.PtrBytes
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
	}

	// Since we're never writing more than one uintptr's worth of bits, we're either going
	// to do one or two writes.
	dstBase, _ := spanHeapBitsRange(span.base(), pageSize, elemsize)
	dst := unsafe.Pointer(dstBase)
	o := (x - span.base()) / goarch.PtrSize
	i := o / ptrBits
	j := o % ptrBits
	const bits uintptr = elemsize / goarch.PtrSize
	// In the if statement below, we have to do two uintptr writes if the bits
	// we need to write straddle across two different memory locations. But if
	// the number of bits we're writing divides evenly into the number of bits
	// in the uintptr we're writing, this can never happen. Since bitsIsPowerOfTwo
	// is a compile-time constant in the generated code, in the case where the size is
	// a power of two less than or equal to ptrBits, the compiler can remove the
	// 'two writes' branch of the if statement and always do only one write without
	// the check.
	const bitsIsPowerOfTwo = bits&(bits-1) == 0
	if bits > ptrBits || (!bitsIsPowerOfTwo && j+bits > ptrBits) {
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
		*dst = (*dst)&^(((1<<(min(bits, ptrBits)))-1)<<j) | (src << j) // We're taking the min so this compiles on 32 bit platforms. But if bits > ptrbits we always take the other branch
	}

	const doubleCheck = false
	if doubleCheck {
		writeHeapBitsDoubleCheck(span, x, dataSize, src, src0, i, j, bits, typ)
	}
	return scanSize
}

func writeHeapBitsDoubleCheck(span *mspan, x, dataSize, src, src0, i, j, bits uintptr, typ *_type) {
	srcRead := span.heapBitsSmallForAddr(x)
	if srcRead != src {
		print("runtime: x=", hex(x), " i=", i, " j=", j, " bits=", bits, "\n")
		print("runtime: dataSize=", dataSize, " typ.Size_=", typ.Size_, " typ.PtrBytes=", typ.PtrBytes, "\n")
		print("runtime: src0=", hex(src0), " src=", hex(src), " srcRead=", hex(srcRead), "\n")
		throw("bad pointer bits written for small object")
	}
}
