// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.allocheaders

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
// Heap bitmap
//
// The heap bitmap comprises 1 bit for each pointer-sized word in the heap,
// recording whether a pointer is stored in that word or not. This bitmap
// is stored in the heapArena metadata backing each heap arena.
// That is, if ha is the heapArena for the arena starting at "start",
// then ha.bitmap[0] holds the 64 bits for the 64 words "start"
// through start+63*ptrSize, ha.bitmap[1] holds the entries for
// start+64*ptrSize through start+127*ptrSize, and so on.
// Bits correspond to words in little-endian order. ha.bitmap[0]&1 represents
// the word at "start", ha.bitmap[0]>>1&1 represents the word at start+8, etc.
// (For 32-bit platforms, s/64/32/.)
//
// We also keep a noMorePtrs bitmap which allows us to stop scanning
// the heap bitmap early in certain situations. If ha.noMorePtrs[i]>>j&1
// is 1, then the object containing the last word described by ha.bitmap[8*i+j]
// has no more pointers beyond those described by ha.bitmap[8*i+j].
// If ha.noMorePtrs[i]>>j&1 is set, the entries in ha.bitmap[8*i+j+1] and
// beyond must all be zero until the start of the next object.
//
// The bitmap for noscan spans is set to all zero at span allocation time.
//
// The bitmap for unallocated objects in scannable spans is not maintained
// (can be junk).

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"runtime/internal/sys"
	"unsafe"
)

const (
	// For compatibility with the allocheaders GOEXPERIMENT.
	mallocHeaderSize       = 0
	minSizeForMallocHeader = ^uintptr(0)
)

// For compatibility with the allocheaders GOEXPERIMENT.
//
//go:nosplit
func heapBitsInSpan(_ uintptr) bool {
	return false
}

// heapArenaPtrScalar contains the per-heapArena pointer/scalar metadata for the GC.
type heapArenaPtrScalar struct {
	// bitmap stores the pointer/scalar bitmap for the words in
	// this arena. See mbitmap.go for a description.
	// This array uses 1 bit per word of heap, or 1.6% of the heap size (for 64-bit).
	bitmap [heapArenaBitmapWords]uintptr

	// If the ith bit of noMorePtrs is true, then there are no more
	// pointers for the object containing the word described by the
	// high bit of bitmap[i].
	// In that case, bitmap[i+1], ... must be zero until the start
	// of the next object.
	// We never operate on these entries using bit-parallel techniques,
	// so it is ok if they are small. Also, they can't be bigger than
	// uint16 because at that size a single noMorePtrs entry
	// represents 8K of memory, the minimum size of a span. Any larger
	// and we'd have to worry about concurrent updates.
	// This array uses 1 bit per word of bitmap, or .024% of the heap size (for 64-bit).
	noMorePtrs [heapArenaBitmapWords / 8]uint8
}

// heapBits provides access to the bitmap bits for a single heap word.
// The methods on heapBits take value receivers so that the compiler
// can more easily inline calls to those methods and registerize the
// struct fields independently.
type heapBits struct {
	// heapBits will report on pointers in the range [addr,addr+size).
	// The low bit of mask contains the pointerness of the word at addr
	// (assuming valid>0).
	addr, size uintptr

	// The next few pointer bits representing words starting at addr.
	// Those bits already returned by next() are zeroed.
	mask uintptr
	// Number of bits in mask that are valid. mask is always less than 1<<valid.
	valid uintptr
}

// heapBitsForAddr returns the heapBits for the address addr.
// The caller must ensure [addr,addr+size) is in an allocated span.
// In particular, be careful not to point past the end of an object.
//
// nosplit because it is used during write barriers and must not be preempted.
//
//go:nosplit
func heapBitsForAddr(addr, size uintptr) heapBits {
	// Find arena
	ai := arenaIndex(addr)
	ha := mheap_.arenas[ai.l1()][ai.l2()]

	// Word index in arena.
	word := addr / goarch.PtrSize % heapArenaWords

	// Word index and bit offset in bitmap array.
	idx := word / ptrBits
	off := word % ptrBits

	// Grab relevant bits of bitmap.
	mask := ha.bitmap[idx] >> off
	valid := ptrBits - off

	// Process depending on where the object ends.
	nptr := size / goarch.PtrSize
	if nptr < valid {
		// Bits for this object end before the end of this bitmap word.
		// Squash bits for the following objects.
		mask &= 1<<(nptr&(ptrBits-1)) - 1
		valid = nptr
	} else if nptr == valid {
		// Bits for this object end at exactly the end of this bitmap word.
		// All good.
	} else {
		// Bits for this object extend into the next bitmap word. See if there
		// may be any pointers recorded there.
		if uintptr(ha.noMorePtrs[idx/8])>>(idx%8)&1 != 0 {
			// No more pointers in this object after this bitmap word.
			// Update size so we know not to look there.
			size = valid * goarch.PtrSize
		}
	}

	return heapBits{addr: addr, size: size, mask: mask, valid: valid}
}

// Returns the (absolute) address of the next known pointer and
// a heapBits iterator representing any remaining pointers.
// If there are no more pointers, returns address 0.
// Note that next does not modify h. The caller must record the result.
//
// nosplit because it is used during write barriers and must not be preempted.
//
//go:nosplit
func (h heapBits) next() (heapBits, uintptr) {
	for {
		if h.mask != 0 {
			var i int
			if goarch.PtrSize == 8 {
				i = sys.TrailingZeros64(uint64(h.mask))
			} else {
				i = sys.TrailingZeros32(uint32(h.mask))
			}
			h.mask ^= uintptr(1) << (i & (ptrBits - 1))
			return h, h.addr + uintptr(i)*goarch.PtrSize
		}

		// Skip words that we've already processed.
		h.addr += h.valid * goarch.PtrSize
		h.size -= h.valid * goarch.PtrSize
		if h.size == 0 {
			return h, 0 // no more pointers
		}

		// Grab more bits and try again.
		h = heapBitsForAddr(h.addr, h.size)
	}
}

// nextFast is like next, but can return 0 even when there are more pointers
// to be found. Callers should call next if nextFast returns 0 as its second
// return value.
//
//	if addr, h = h.nextFast(); addr == 0 {
//	    if addr, h = h.next(); addr == 0 {
//	        ... no more pointers ...
//	    }
//	}
//	... process pointer at addr ...
//
// nextFast is designed to be inlineable.
//
//go:nosplit
func (h heapBits) nextFast() (heapBits, uintptr) {
	// TESTQ/JEQ
	if h.mask == 0 {
		return h, 0
	}
	// BSFQ
	var i int
	if goarch.PtrSize == 8 {
		i = sys.TrailingZeros64(uint64(h.mask))
	} else {
		i = sys.TrailingZeros32(uint32(h.mask))
	}
	// BTCQ
	h.mask ^= uintptr(1) << (i & (ptrBits - 1))
	// LEAQ (XX)(XX*8)
	return h, h.addr + uintptr(i)*goarch.PtrSize
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
// The pointer bitmap is not maintained for allocations containing
// no pointers at all; any caller of bulkBarrierPreWrite must first
// make sure the underlying allocation contains pointers, usually
// by checking typ.PtrBytes.
//
// The type of the space can be provided purely as an optimization,
// however it is not used with GOEXPERIMENT=noallocheaders.
//
// Callers must perform cgo checks if goexperiment.CgoCheck2.
//
//go:nosplit
func bulkBarrierPreWrite(dst, src, size uintptr, _ *abi.Type) {
	if (dst|src|size)&(goarch.PtrSize-1) != 0 {
		throw("bulkBarrierPreWrite: unaligned arguments")
	}
	if !writeBarrier.enabled {
		return
	}
	if s := spanOf(dst); s == nil {
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
	h := heapBitsForAddr(dst, size)
	if src == 0 {
		for {
			var addr uintptr
			if h, addr = h.next(); addr == 0 {
				break
			}
			dstx := (*uintptr)(unsafe.Pointer(addr))
			p := buf.get1()
			p[0] = *dstx
		}
	} else {
		for {
			var addr uintptr
			if h, addr = h.next(); addr == 0 {
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
// The type of the space can be provided purely as an optimization,
// however it is not used with GOEXPERIMENT=noallocheaders.
//
//go:nosplit
func bulkBarrierPreWriteSrcOnly(dst, src, size uintptr, _ *abi.Type) {
	if (dst|src|size)&(goarch.PtrSize-1) != 0 {
		throw("bulkBarrierPreWrite: unaligned arguments")
	}
	if !writeBarrier.enabled {
		return
	}
	buf := &getg().m.p.ptr().wbBuf
	h := heapBitsForAddr(dst, size)
	for {
		var addr uintptr
		if h, addr = h.next(); addr == 0 {
			break
		}
		srcx := (*uintptr)(unsafe.Pointer(addr - dst + src))
		p := buf.get1()
		p[0] = *srcx
	}
}

// initHeapBits initializes the heap bitmap for a span.
// If this is a span of single pointer allocations, it initializes all
// words to pointer. If force is true, clears all bits.
func (s *mspan) initHeapBits(forceClear bool) {
	if forceClear || s.spanclass.noscan() {
		// Set all the pointer bits to zero. We do this once
		// when the span is allocated so we don't have to do it
		// for each object allocation.
		base := s.base()
		size := s.npages * pageSize
		h := writeHeapBitsForAddr(base)
		h.flush(base, size)
		return
	}
	isPtrs := goarch.PtrSize == 8 && s.elemsize == goarch.PtrSize
	if !isPtrs {
		return // nothing to do
	}
	h := writeHeapBitsForAddr(s.base())
	size := s.npages * pageSize
	nptrs := size / goarch.PtrSize
	for i := uintptr(0); i < nptrs; i += ptrBits {
		h = h.write(^uintptr(0), ptrBits)
	}
	h.flush(s.base(), size)
}

type writeHeapBits struct {
	addr  uintptr // address that the low bit of mask represents the pointer state of.
	mask  uintptr // some pointer bits starting at the address addr.
	valid uintptr // number of bits in buf that are valid (including low)
	low   uintptr // number of low-order bits to not overwrite
}

func writeHeapBitsForAddr(addr uintptr) (h writeHeapBits) {
	// We start writing bits maybe in the middle of a heap bitmap word.
	// Remember how many bits into the word we started, so we can be sure
	// not to overwrite the previous bits.
	h.low = addr / goarch.PtrSize % ptrBits

	// round down to heap word that starts the bitmap word.
	h.addr = addr - h.low*goarch.PtrSize

	// We don't have any bits yet.
	h.mask = 0
	h.valid = h.low

	return
}

// write appends the pointerness of the next valid pointer slots
// using the low valid bits of bits. 1=pointer, 0=scalar.
func (h writeHeapBits) write(bits, valid uintptr) writeHeapBits {
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
	// TODO: figure out how to cache arena lookup.
	ai := arenaIndex(h.addr)
	ha := mheap_.arenas[ai.l1()][ai.l2()]
	idx := h.addr / (ptrBits * goarch.PtrSize) % heapArenaBitmapWords
	m := uintptr(1)<<h.low - 1
	ha.bitmap[idx] = ha.bitmap[idx]&m | data
	// Note: no synchronization required for this write because
	// the allocator has exclusive access to the page, and the bitmap
	// entries are all for a single page. Also, visibility of these
	// writes is guaranteed by the publication barrier in mallocgc.

	// Clear noMorePtrs bit, since we're going to be writing bits
	// into the following word.
	ha.noMorePtrs[idx/8] &^= uint8(1) << (idx % 8)
	// Note: same as above

	// Move to next word of bitmap.
	h.addr += ptrBits * goarch.PtrSize
	h.low = 0
	return h
}

// Add padding of size bytes.
func (h writeHeapBits) pad(size uintptr) writeHeapBits {
	if size == 0 {
		return h
	}
	words := size / goarch.PtrSize
	for words > ptrBits {
		h = h.write(0, ptrBits)
		words -= ptrBits
	}
	return h.write(0, words)
}

// Flush the bits that have been written, and add zeros as needed
// to cover the full object [addr, addr+size).
func (h writeHeapBits) flush(addr, size uintptr) {
	// zeros counts the number of bits needed to represent the object minus the
	// number of bits we've already written. This is the number of 0 bits
	// that need to be added.
	zeros := (addr+size-h.addr)/goarch.PtrSize - h.valid

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
	ai := arenaIndex(h.addr)
	ha := mheap_.arenas[ai.l1()][ai.l2()]
	idx := h.addr / (ptrBits * goarch.PtrSize) % heapArenaBitmapWords

	// Write remaining bits.
	if h.valid != h.low {
		m := uintptr(1)<<h.low - 1      // don't clear existing bits below "low"
		m |= ^(uintptr(1)<<h.valid - 1) // don't clear existing bits above "valid"
		ha.bitmap[idx] = ha.bitmap[idx]&m | h.mask
	}
	if zeros == 0 {
		return
	}

	// Record in the noMorePtrs map that there won't be any more 1 bits,
	// so readers can stop early.
	ha.noMorePtrs[idx/8] |= uint8(1) << (idx % 8)

	// Advance to next bitmap word.
	h.addr += ptrBits * goarch.PtrSize

	// Continue on writing zeros for the rest of the object.
	// For standard use of the ptr bits this is not required, as
	// the bits are read from the beginning of the object. Some uses,
	// like noscan spans, oblets, bulk write barriers, and cgocheck, might
	// start mid-object, so these writes are still required.
	for {
		// Write zero bits.
		ai := arenaIndex(h.addr)
		ha := mheap_.arenas[ai.l1()][ai.l2()]
		idx := h.addr / (ptrBits * goarch.PtrSize) % heapArenaBitmapWords
		if zeros < ptrBits {
			ha.bitmap[idx] &^= uintptr(1)<<zeros - 1
			break
		} else if zeros == ptrBits {
			ha.bitmap[idx] = 0
			break
		} else {
			ha.bitmap[idx] = 0
			zeros -= ptrBits
		}
		ha.noMorePtrs[idx/8] |= uint8(1) << (idx % 8)
		h.addr += ptrBits * goarch.PtrSize
	}
}

// heapBitsSetType records that the new allocation [x, x+size)
// holds in [x, x+dataSize) one or more values of type typ.
// (The number of values is given by dataSize / typ.Size.)
// If dataSize < size, the fragment [x+dataSize, x+size) is
// recorded as non-pointer data.
// It is known that the type has pointers somewhere;
// malloc does not call heapBitsSetType when there are no pointers,
// because all free objects are marked as noscan during
// heapBitsSweepSpan.
//
// There can only be one allocation from a given span active at a time,
// and the bitmap for a span always falls on word boundaries,
// so there are no write-write races for access to the heap bitmap.
// Hence, heapBitsSetType can access the bitmap without atomics.
//
// There can be read-write races between heapBitsSetType and things
// that read the heap bitmap like scanobject. However, since
// heapBitsSetType is only used for objects that have not yet been
// made reachable, readers will ignore bits being modified by this
// function. This does mean this function cannot transiently modify
// bits that belong to neighboring objects. Also, on weakly-ordered
// machines, callers must execute a store/store (publication) barrier
// between calling this function and making the object reachable.
func heapBitsSetType(x, size, dataSize uintptr, typ *_type) {
	const doubleCheck = false // slow but helpful; enable to test modifications to this code

	if doubleCheck && dataSize%typ.Size_ != 0 {
		throw("heapBitsSetType: dataSize not a multiple of typ.Size")
	}

	if goarch.PtrSize == 8 && size == goarch.PtrSize {
		// It's one word and it has pointers, it must be a pointer.
		// Since all allocated one-word objects are pointers
		// (non-pointers are aggregated into tinySize allocations),
		// (*mspan).initHeapBits sets the pointer bits for us.
		// Nothing to do here.
		if doubleCheck {
			h, addr := heapBitsForAddr(x, size).next()
			if addr != x {
				throw("heapBitsSetType: pointer bit missing")
			}
			_, addr = h.next()
			if addr != 0 {
				throw("heapBitsSetType: second pointer bit found")
			}
		}
		return
	}

	h := writeHeapBitsForAddr(x)

	// Handle GC program.
	if typ.Kind_&kindGCProg != 0 {
		// Expand the gc program into the storage we're going to use for the actual object.
		obj := (*uint8)(unsafe.Pointer(x))
		n := runGCProg(addb(typ.GCData, 4), obj)
		// Use the expanded program to set the heap bits.
		for i := uintptr(0); true; i += typ.Size_ {
			// Copy expanded program to heap bitmap.
			p := obj
			j := n
			for j > 8 {
				h = h.write(uintptr(*p), 8)
				p = add1(p)
				j -= 8
			}
			h = h.write(uintptr(*p), j)

			if i+typ.Size_ == dataSize {
				break // no padding after last element
			}

			// Pad with zeros to the start of the next element.
			h = h.pad(typ.Size_ - n*goarch.PtrSize)
		}

		h.flush(x, size)

		// Erase the expanded GC program.
		memclrNoHeapPointers(unsafe.Pointer(obj), (n+7)/8)
		return
	}

	// Note about sizes:
	//
	// typ.Size is the number of words in the object,
	// and typ.PtrBytes is the number of words in the prefix
	// of the object that contains pointers. That is, the final
	// typ.Size - typ.PtrBytes words contain no pointers.
	// This allows optimization of a common pattern where
	// an object has a small header followed by a large scalar
	// buffer. If we know the pointers are over, we don't have
	// to scan the buffer's heap bitmap at all.
	// The 1-bit ptrmasks are sized to contain only bits for
	// the typ.PtrBytes prefix, zero padded out to a full byte
	// of bitmap. If there is more room in the allocated object,
	// that space is pointerless. The noMorePtrs bitmap will prevent
	// scanning large pointerless tails of an object.
	//
	// Replicated copies are not as nice: if there is an array of
	// objects with scalar tails, all but the last tail does have to
	// be initialized, because there is no way to say "skip forward".

	ptrs := typ.PtrBytes / goarch.PtrSize
	if typ.Size_ == dataSize { // Single element
		if ptrs <= ptrBits { // Single small element
			m := readUintptr(typ.GCData)
			h = h.write(m, ptrs)
		} else { // Single large element
			p := typ.GCData
			for {
				h = h.write(readUintptr(p), ptrBits)
				p = addb(p, ptrBits/8)
				ptrs -= ptrBits
				if ptrs <= ptrBits {
					break
				}
			}
			m := readUintptr(p)
			h = h.write(m, ptrs)
		}
	} else { // Repeated element
		words := typ.Size_ / goarch.PtrSize // total words, including scalar tail
		if words <= ptrBits {               // Repeated small element
			n := dataSize / typ.Size_
			m := readUintptr(typ.GCData)
			// Make larger unit to repeat
			for words <= ptrBits/2 {
				if n&1 != 0 {
					h = h.write(m, words)
				}
				n /= 2
				m |= m << words
				ptrs += words
				words *= 2
				if n == 1 {
					break
				}
			}
			for n > 1 {
				h = h.write(m, words)
				n--
			}
			h = h.write(m, ptrs)
		} else { // Repeated large element
			for i := uintptr(0); true; i += typ.Size_ {
				p := typ.GCData
				j := ptrs
				for j > ptrBits {
					h = h.write(readUintptr(p), ptrBits)
					p = addb(p, ptrBits/8)
					j -= ptrBits
				}
				m := readUintptr(p)
				h = h.write(m, j)
				if i+typ.Size_ == dataSize {
					break // don't need the trailing nonptr bits on the last element.
				}
				// Pad with zeros to the start of the next element.
				h = h.pad(typ.Size_ - typ.PtrBytes)
			}
		}
	}
	h.flush(x, size)

	if doubleCheck {
		h := heapBitsForAddr(x, size)
		for i := uintptr(0); i < size; i += goarch.PtrSize {
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
				var addr uintptr
				h, addr = h.next()
				if addr != x+i {
					throw("heapBitsSetType: pointer entry not correct")
				}
			}
		}
		if _, addr := h.next(); addr != 0 {
			throw("heapBitsSetType: extra pointer")
		}
	}
}

// For goexperiment.AllocHeaders
func heapSetType(x, dataSize uintptr, typ *_type, header **_type, span *mspan) (scanSize uintptr) {
	return 0
}

// Testing.

// Returns GC type info for the pointer stored in ep for testing.
// If ep points to the stack, only static live information will be returned
// (i.e. not for objects which are only dynamically live stack objects).
func getgcmask(ep any) (mask []byte) {
	e := *efaceOf(&ep)
	p := e.data
	t := e._type
	// data or bss
	for _, datap := range activeModules() {
		// data
		if datap.data <= uintptr(p) && uintptr(p) < datap.edata {
			bitmap := datap.gcdatamask.bytedata
			n := (*ptrtype)(unsafe.Pointer(t)).Elem.Size_
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
			n := (*ptrtype)(unsafe.Pointer(t)).Elem.Size_
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
		n := s.elemsize
		hbits := heapBitsForAddr(base, n)
		mask = make([]byte, n/goarch.PtrSize)
		for {
			var addr uintptr
			if hbits, addr = hbits.next(); addr == 0 {
				break
			}
			mask[(addr-base)/goarch.PtrSize] = 1
		}
		// Callers expect this mask to end at the last pointer.
		for len(mask) > 0 && mask[len(mask)-1] == 0 {
			mask = mask[:len(mask)-1]
		}

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

// userArenaHeapBitsSetType is the equivalent of heapBitsSetType but for
// non-slice-backing-store Go values allocated in a user arena chunk. It
// sets up the heap bitmap for the value with type typ allocated at address ptr.
// base is the base address of the arena chunk.
func userArenaHeapBitsSetType(typ *_type, ptr unsafe.Pointer, s *mspan) {
	base := s.base()
	h := writeHeapBitsForAddr(uintptr(ptr))

	// Our last allocation might have ended right at a noMorePtrs mark,
	// which we would not have erased. We need to erase that mark here,
	// because we're going to start adding new heap bitmap bits.
	// We only need to clear one mark, because below we make sure to
	// pad out the bits with zeroes and only write one noMorePtrs bit
	// for each new object.
	// (This is only necessary at noMorePtrs boundaries, as noMorePtrs
	// marks within an object allocated with newAt will be erased by
	// the normal writeHeapBitsForAddr mechanism.)
	//
	// Note that we skip this if this is the first allocation in the
	// arena because there's definitely no previous noMorePtrs mark
	// (in fact, we *must* do this, because we're going to try to back
	// up a pointer to fix this up).
	if uintptr(ptr)%(8*goarch.PtrSize*goarch.PtrSize) == 0 && uintptr(ptr) != base {
		// Back up one pointer and rewrite that pointer. That will
		// cause the writeHeapBits implementation to clear the
		// noMorePtrs bit we need to clear.
		r := heapBitsForAddr(uintptr(ptr)-goarch.PtrSize, goarch.PtrSize)
		_, p := r.next()
		b := uintptr(0)
		if p == uintptr(ptr)-goarch.PtrSize {
			b = 1
		}
		h = writeHeapBitsForAddr(uintptr(ptr) - goarch.PtrSize)
		h = h.write(b, 1)
	}

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
		h = h.write(readUintptr(addb(p, i/8)), k)
	}
	// Note: we call pad here to ensure we emit explicit 0 bits
	// for the pointerless tail of the object. This ensures that
	// there's only a single noMorePtrs mark for the next object
	// to clear. We don't need to do this to clear stale noMorePtrs
	// markers from previous uses because arena chunk pointer bitmaps
	// are always fully cleared when reused.
	h = h.pad(typ.Size_ - typ.PtrBytes)
	h.flush(uintptr(ptr), typ.Size_)

	if typ.Kind_&kindGCProg != 0 {
		// Zero out temporary ptrmask buffer inside object.
		memclrNoHeapPointers(ptr, (gcProgBits+7)/8)
	}

	// Double-check that the bitmap was written out correctly.
	//
	// Derived from heapBitsSetType.
	const doubleCheck = false
	if doubleCheck {
		size := typ.Size_
		x := uintptr(ptr)
		h := heapBitsForAddr(x, size)
		for i := uintptr(0); i < size; i += goarch.PtrSize {
			// Compute the pointer bit we want at offset i.
			want := false
			off := i % typ.Size_
			if off < typ.PtrBytes {
				j := off / goarch.PtrSize
				want = *addb(typ.GCData, j/8)>>(j%8)&1 != 0
			}
			if want {
				var addr uintptr
				h, addr = h.next()
				if addr != x+i {
					throw("userArenaHeapBitsSetType: pointer entry not correct")
				}
			}
		}
		if _, addr := h.next(); addr != 0 {
			throw("userArenaHeapBitsSetType: extra pointer")
		}
	}
}

// For goexperiment.AllocHeaders.
type typePointers struct {
	addr uintptr
}

// For goexperiment.AllocHeaders.
//
//go:nosplit
func (span *mspan) typePointersOf(addr, size uintptr) typePointers {
	panic("not implemented")
}

// For goexperiment.AllocHeaders.
//
//go:nosplit
func (span *mspan) typePointersOfUnchecked(addr uintptr) typePointers {
	panic("not implemented")
}

// For goexperiment.AllocHeaders.
//
//go:nosplit
func (tp typePointers) nextFast() (typePointers, uintptr) {
	panic("not implemented")
}

// For goexperiment.AllocHeaders.
//
//go:nosplit
func (tp typePointers) next(limit uintptr) (typePointers, uintptr) {
	panic("not implemented")
}

// For goexperiment.AllocHeaders.
//
//go:nosplit
func (tp typePointers) fastForward(n, limit uintptr) typePointers {
	panic("not implemented")
}

// For goexperiment.AllocHeaders, to pass TestIntendedInlining.
func (s *mspan) writeUserArenaHeapBits() {
	panic("not implemented")
}

// For goexperiment.AllocHeaders, to pass TestIntendedInlining.
func heapBitsSlice() {
	panic("not implemented")
}
