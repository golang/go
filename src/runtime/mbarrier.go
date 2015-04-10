// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector: write barriers.
//
// For the concurrent garbage collector, the Go compiler implements
// updates to pointer-valued fields that may be in heap objects by
// emitting calls to write barriers. This file contains the actual write barrier
// implementation, markwb, and the various wrappers called by the
// compiler to implement pointer assignment, slice assignment,
// typed memmove, and so on.
//
// To check for missed write barriers, the GODEBUG=wbshadow debugging
// mode allocates a second copy of the heap. Write barrier-based pointer
// updates make changes to both the real heap and the shadow, and both
// the pointer updates and the GC look for inconsistencies between the two,
// indicating pointer writes that bypassed the barrier.

package runtime

import "unsafe"

// markwb is the mark-phase write barrier, the only barrier we have.
// The rest of this file exists only to make calls to this function.
//
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
// Without an expensive memory barrier between the st and the ld, the final
// result on most HW (including 386/amd64) can be r1==r2==0. This is a classic
// example of what can happen when loads are allowed to be reordered with older
// stores (avoiding such reorderings lies at the heart of the classic
// Peterson/Dekker algorithms for mutual exclusion). Rather than require memory
// barriers, which will slow down both the mutator and the GC, we always grey
// the ptr object regardless of the slot's color.
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

// needwb reports whether a write barrier is needed now
// (otherwise the write can be made directly).
//go:nosplit
func needwb() bool {
	return gcphase == _GCmark || gcphase == _GCmarktermination || mheap_.shadow_enabled
}

// Write barrier calls must not happen during critical GC and scheduler
// related operations. In particular there are times when the GC assumes
// that the world is stopped but scheduler related code is still being
// executed, dealing with syscalls, dealing with putting gs on runnable
// queues and so forth. This code can not execute write barriers because
// the GC might drop them on the floor. Stopping the world involves removing
// the p associated with an m. We use the fact that m.p == nil to indicate
// that we are in one these critical section and throw if the write is of
// a pointer to a heap object.
// The p, m, and g pointers are the pointers that are used by the scheduler
// and need to be operated on without write barriers. We use
// the setPNoWriteBarrier, setMNoWriteBarrier and setGNowriteBarrier to
// avoid having to do the write barrier.
//go:nosplit
func writebarrierptr_nostore1(dst *uintptr, src uintptr) {
	mp := acquirem()
	if mp.inwb || mp.dying > 0 {
		releasem(mp)
		return
	}
	systemstack(func() {
		if mp.p == nil && memstats.enablegc && !mp.inwb && inheap(src) {
			throw("writebarrierptr_nostore1 called with mp.p == nil")
		}
		mp.inwb = true
		gcmarkwb_m(dst, src)
	})
	mp.inwb = false
	releasem(mp)
}

// NOTE: Really dst *unsafe.Pointer, src unsafe.Pointer,
// but if we do that, Go inserts a write barrier on *dst = src.
//go:nosplit
func writebarrierptr(dst *uintptr, src uintptr) {
	if !needwb() {
		*dst = src
		return
	}

	if src != 0 && (src < _PhysPageSize || src == poisonStack) {
		systemstack(func() { throw("bad pointer in write barrier") })
	}

	if mheap_.shadow_enabled {
		writebarrierptr_shadow(dst, src)
	}

	*dst = src
	writebarrierptr_nostore1(dst, src)
}

//go:nosplit
func writebarrierptr_shadow(dst *uintptr, src uintptr) {
	systemstack(func() {
		addr := uintptr(unsafe.Pointer(dst))
		shadow := shadowptr(addr)
		if shadow == nil {
			return
		}
		// There is a race here but only if the program is using
		// racy writes instead of sync/atomic. In that case we
		// don't mind crashing.
		if *shadow != *dst && *shadow != noShadow && istrackedptr(*dst) {
			mheap_.shadow_enabled = false
			print("runtime: write barrier dst=", dst, " old=", hex(*dst), " shadow=", shadow, " old=", hex(*shadow), " new=", hex(src), "\n")
			throw("missed write barrier")
		}
		*shadow = src
	})
}

// Like writebarrierptr, but the store has already been applied.
// Do not reapply.
//go:nosplit
func writebarrierptr_nostore(dst *uintptr, src uintptr) {
	if !needwb() {
		return
	}

	if src != 0 && (src < _PhysPageSize || src == poisonStack) {
		systemstack(func() { throw("bad pointer in write barrier") })
	}

	// Apply changes to shadow.
	// Since *dst has been overwritten already, we cannot check
	// whether there were any missed updates, but writebarrierptr_nostore
	// is only rarely used.
	if mheap_.shadow_enabled {
		systemstack(func() {
			addr := uintptr(unsafe.Pointer(dst))
			shadow := shadowptr(addr)
			if shadow == nil {
				return
			}
			*shadow = src
		})
	}

	writebarrierptr_nostore1(dst, src)
}

// writebarrierptr_noshadow records that the value in *dst
// has been written to using an atomic operation and the shadow
// has not been updated. (In general if dst must be manipulated
// atomically we cannot get the right bits for use in the shadow.)
//go:nosplit
func writebarrierptr_noshadow(dst *uintptr) {
	addr := uintptr(unsafe.Pointer(dst))
	shadow := shadowptr(addr)
	if shadow == nil {
		return
	}

	*shadow = noShadow
}

//go:nosplit
func writebarrierstring(dst *[2]uintptr, src [2]uintptr) {
	writebarrierptr(&dst[0], src[0])
	dst[1] = src[1]
}

//go:nosplit
func writebarrierslice(dst *[3]uintptr, src [3]uintptr) {
	writebarrierptr(&dst[0], src[0])
	dst[1] = src[1]
	dst[2] = src[2]
}

//go:nosplit
func writebarrieriface(dst *[2]uintptr, src [2]uintptr) {
	writebarrierptr(&dst[0], src[0])
	writebarrierptr(&dst[1], src[1])
}

//go:generate go run wbfat_gen.go -- wbfat.go
//
// The above line generates multiword write barriers for
// all the combinations of ptr+scalar up to four words.
// The implementations are written to wbfat.go.

// typedmemmove copies a value of type t to dst from src.
//go:nosplit
func typedmemmove(typ *_type, dst, src unsafe.Pointer) {
	if !needwb() || (typ.kind&kindNoPointers) != 0 {
		memmove(dst, src, typ.size)
		return
	}

	systemstack(func() {
		mask := typeBitmapInHeapBitmapFormat(typ)
		nptr := typ.size / ptrSize
		for i := uintptr(0); i < nptr; i += 2 {
			bits := mask[i/2]
			if (bits>>2)&typeMask == typePointer {
				writebarrierptr((*uintptr)(dst), *(*uintptr)(src))
			} else {
				*(*uintptr)(dst) = *(*uintptr)(src)
			}
			// TODO(rsc): The noescape calls should be unnecessary.
			dst = add(noescape(dst), ptrSize)
			src = add(noescape(src), ptrSize)
			if i+1 == nptr {
				break
			}
			bits >>= 4
			if (bits>>2)&typeMask == typePointer {
				writebarrierptr((*uintptr)(dst), *(*uintptr)(src))
			} else {
				*(*uintptr)(dst) = *(*uintptr)(src)
			}
			dst = add(noescape(dst), ptrSize)
			src = add(noescape(src), ptrSize)
		}
	})
}

//go:linkname reflect_typedmemmove reflect.typedmemmove
func reflect_typedmemmove(typ *_type, dst, src unsafe.Pointer) {
	typedmemmove(typ, dst, src)
}

// typedmemmovepartial is like typedmemmove but assumes that
// dst and src point off bytes into the value and only copies size bytes.
//go:linkname reflect_typedmemmovepartial reflect.typedmemmovepartial
func reflect_typedmemmovepartial(typ *_type, dst, src unsafe.Pointer, off, size uintptr) {
	if !needwb() || (typ.kind&kindNoPointers) != 0 || size < ptrSize {
		memmove(dst, src, size)
		return
	}

	if off&(ptrSize-1) != 0 {
		frag := -off & (ptrSize - 1)
		// frag < size, because size >= ptrSize, checked above.
		memmove(dst, src, frag)
		size -= frag
		dst = add(noescape(dst), frag)
		src = add(noescape(src), frag)
		off += frag
	}

	mask := typeBitmapInHeapBitmapFormat(typ)
	nptr := (off + size) / ptrSize
	for i := uintptr(off / ptrSize); i < nptr; i++ {
		bits := mask[i/2] >> ((i & 1) << 2)
		if (bits>>2)&typeMask == typePointer {
			writebarrierptr((*uintptr)(dst), *(*uintptr)(src))
		} else {
			*(*uintptr)(dst) = *(*uintptr)(src)
		}
		// TODO(rsc): The noescape calls should be unnecessary.
		dst = add(noescape(dst), ptrSize)
		src = add(noescape(src), ptrSize)
	}
	size &= ptrSize - 1
	if size > 0 {
		memmove(dst, src, size)
	}
}

// callwritebarrier is invoked at the end of reflectcall, to execute
// write barrier operations to record the fact that a call's return
// values have just been copied to frame, starting at retoffset
// and continuing to framesize. The entire frame (not just the return
// values) is described by typ. Because the copy has already
// happened, we call writebarrierptr_nostore, and we must be careful
// not to be preempted before the write barriers have been run.
//go:nosplit
func callwritebarrier(typ *_type, frame unsafe.Pointer, framesize, retoffset uintptr) {
	if !needwb() || typ == nil || (typ.kind&kindNoPointers) != 0 || framesize-retoffset < ptrSize {
		return
	}

	systemstack(func() {
		mask := typeBitmapInHeapBitmapFormat(typ)
		// retoffset is known to be pointer-aligned (at least).
		// TODO(rsc): The noescape call should be unnecessary.
		dst := add(noescape(frame), retoffset)
		nptr := framesize / ptrSize
		for i := uintptr(retoffset / ptrSize); i < nptr; i++ {
			bits := mask[i/2] >> ((i & 1) << 2)
			if (bits>>2)&typeMask == typePointer {
				writebarrierptr_nostore((*uintptr)(dst), *(*uintptr)(dst))
			}
			// TODO(rsc): The noescape call should be unnecessary.
			dst = add(noescape(dst), ptrSize)
		}
	})
}

//go:nosplit
func typedslicecopy(typ *_type, dst, src slice) int {
	n := dst.len
	if n > src.len {
		n = src.len
	}
	if n == 0 {
		return 0
	}
	dstp := unsafe.Pointer(dst.array)
	srcp := unsafe.Pointer(src.array)

	if raceenabled {
		callerpc := getcallerpc(unsafe.Pointer(&typ))
		pc := funcPC(slicecopy)
		racewriterangepc(dstp, uintptr(n)*typ.size, callerpc, pc)
		racereadrangepc(srcp, uintptr(n)*typ.size, callerpc, pc)
	}

	if !needwb() {
		memmove(dstp, srcp, uintptr(n)*typ.size)
		return n
	}

	systemstack(func() {
		if uintptr(srcp) < uintptr(dstp) && uintptr(srcp)+uintptr(n)*typ.size > uintptr(dstp) {
			// Overlap with src before dst.
			// Copy backward, being careful not to move dstp/srcp
			// out of the array they point into.
			dstp = add(dstp, uintptr(n-1)*typ.size)
			srcp = add(srcp, uintptr(n-1)*typ.size)
			i := 0
			for {
				typedmemmove(typ, dstp, srcp)
				if i++; i >= n {
					break
				}
				dstp = add(dstp, -typ.size)
				srcp = add(srcp, -typ.size)
			}
		} else {
			// Copy forward, being careful not to move dstp/srcp
			// out of the array they point into.
			i := 0
			for {
				typedmemmove(typ, dstp, srcp)
				if i++; i >= n {
					break
				}
				dstp = add(dstp, typ.size)
				srcp = add(srcp, typ.size)
			}
		}
	})
	return int(n)
}

//go:linkname reflect_typedslicecopy reflect.typedslicecopy
func reflect_typedslicecopy(elemType *_type, dst, src slice) int {
	return typedslicecopy(elemType, dst, src)
}

// Shadow heap for detecting missed write barriers.

// noShadow is stored in as the shadow pointer to mark that there is no
// shadow word recorded. It matches any actual pointer word.
// noShadow is used when it is impossible to know the right word
// to store in the shadow heap, such as when the real heap word
// is being manipulated atomically.
const noShadow uintptr = 1

func wbshadowinit() {
	// Initialize write barrier shadow heap if we were asked for it
	// and we have enough address space (not on 32-bit).
	if debug.wbshadow == 0 {
		return
	}
	if ptrSize != 8 {
		print("runtime: GODEBUG=wbshadow=1 disabled on 32-bit system\n")
		return
	}

	var reserved bool
	p1 := sysReserveHigh(mheap_.arena_end-mheap_.arena_start, &reserved)
	if p1 == nil {
		throw("cannot map shadow heap")
	}
	mheap_.shadow_heap = uintptr(p1) - mheap_.arena_start
	sysMap(p1, mheap_.arena_used-mheap_.arena_start, reserved, &memstats.other_sys)
	memmove(p1, unsafe.Pointer(mheap_.arena_start), mheap_.arena_used-mheap_.arena_start)

	mheap_.shadow_reserved = reserved

	for datap := &firstmoduledata; datap != nil; datap = datap.next {
		start := ^uintptr(0)
		end := uintptr(0)
		if start > datap.noptrdata {
			start = datap.noptrdata
		}
		if start > datap.data {
			start = datap.data
		}
		if start > datap.noptrbss {
			start = datap.noptrbss
		}
		if start > datap.bss {
			start = datap.bss
		}
		if end < datap.enoptrdata {
			end = datap.enoptrdata
		}
		if end < datap.edata {
			end = datap.edata
		}
		if end < datap.enoptrbss {
			end = datap.enoptrbss
		}
		if end < datap.ebss {
			end = datap.ebss
		}
		start &^= _PhysPageSize - 1
		end = round(end, _PhysPageSize)
		datap.data_start = start
		datap.data_end = end
		reserved = false
		p1 = sysReserveHigh(end-start, &reserved)
		if p1 == nil {
			throw("cannot map shadow data")
		}
		datap.shadow_data = uintptr(p1) - start
		sysMap(p1, end-start, reserved, &memstats.other_sys)
		memmove(p1, unsafe.Pointer(start), end-start)
	}

	mheap_.shadow_enabled = true
}

// shadowptr returns a pointer to the shadow value for addr.
//go:nosplit
func shadowptr(addr uintptr) *uintptr {
	for datap := &firstmoduledata; datap != nil; datap = datap.next {
		if datap.data_start <= addr && addr < datap.data_end {
			return (*uintptr)(unsafe.Pointer(addr + datap.shadow_data))
		}
	}
	if inheap(addr) {
		return (*uintptr)(unsafe.Pointer(addr + mheap_.shadow_heap))
	}
	return nil
}

// istrackedptr reports whether the pointer value p requires a write barrier
// when stored into the heap.
func istrackedptr(p uintptr) bool {
	return inheap(p)
}

// checkwbshadow checks that p matches its shadow word.
// The garbage collector calls checkwbshadow for each pointer during the checkmark phase.
// It is only called when mheap_.shadow_enabled is true.
func checkwbshadow(p *uintptr) {
	addr := uintptr(unsafe.Pointer(p))
	shadow := shadowptr(addr)
	if shadow == nil {
		return
	}
	// There is no race on the accesses here, because the world is stopped,
	// but there may be racy writes that lead to the shadow and the
	// heap being inconsistent. If so, we will detect that here as a
	// missed write barrier and crash. We don't mind.
	// Code should use sync/atomic instead of racy pointer writes.
	if *shadow != *p && *shadow != noShadow && istrackedptr(*p) {
		mheap_.shadow_enabled = false
		print("runtime: checkwritebarrier p=", p, " *p=", hex(*p), " shadow=", shadow, " *shadow=", hex(*shadow), "\n")
		throw("missed write barrier")
	}
}

// clearshadow clears the shadow copy associated with the n bytes of memory at addr.
func clearshadow(addr, n uintptr) {
	if !mheap_.shadow_enabled {
		return
	}
	p := shadowptr(addr)
	if p == nil || n <= ptrSize {
		return
	}
	memclr(unsafe.Pointer(p), n)
}
