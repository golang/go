// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

//go:linkname runtime_debug_freeOSMemory runtime/debug.freeOSMemory
func runtime_debug_freeOSMemory() {
	gogc(2) // force GC and do eager sweep
	systemstack(scavenge_m)
}

var poolcleanup func()

//go:linkname sync_runtime_registerPoolCleanup sync.runtime_registerPoolCleanup
func sync_runtime_registerPoolCleanup(f func()) {
	poolcleanup = f
}

func clearpools() {
	// clear sync.Pools
	if poolcleanup != nil {
		poolcleanup()
	}

	for _, p := range &allp {
		if p == nil {
			break
		}
		// clear tinyalloc pool
		if c := p.mcache; c != nil {
			c.tiny = nil
			c.tinyoffset = 0

			// disconnect cached list before dropping it on the floor,
			// so that a dangling ref to one entry does not pin all of them.
			var sg, sgnext *sudog
			for sg = c.sudogcache; sg != nil; sg = sgnext {
				sgnext = sg.next
				sg.next = nil
			}
			c.sudogcache = nil
		}

		// clear defer pools
		for i := range p.deferpool {
			// disconnect cached list before dropping it on the floor,
			// so that a dangling ref to one entry does not pin all of them.
			var d, dlink *_defer
			for d = p.deferpool[i]; d != nil; d = dlink {
				dlink = d.link
				d.link = nil
			}
			p.deferpool[i] = nil
		}
	}
}

// backgroundgc is running in a goroutine and does the concurrent GC work.
// bggc holds the state of the backgroundgc.
func backgroundgc() {
	bggc.g = getg()
	bggc.g.issystem = true
	for {
		gcwork(0)
		lock(&bggc.lock)
		bggc.working = 0
		goparkunlock(&bggc.lock, "Concurrent GC wait")
	}
}

func bgsweep() {
	sweep.g = getg()
	getg().issystem = true
	for {
		for gosweepone() != ^uintptr(0) {
			sweep.nbgsweep++
			Gosched()
		}
		lock(&gclock)
		if !gosweepdone() {
			// This can happen if a GC runs between
			// gosweepone returning ^0 above
			// and the lock being acquired.
			unlock(&gclock)
			continue
		}
		sweep.parked = true
		goparkunlock(&gclock, "GC sweep wait")
	}
}

const (
	_PoisonGC    = 0xf969696969696969 & (1<<(8*ptrSize) - 1)
	_PoisonStack = 0x6868686868686868 & (1<<(8*ptrSize) - 1)
)

//go:nosplit
func needwb() bool {
	return gcphase == _GCmark || gcphase == _GCmarktermination || mheap_.shadow_enabled
}

// shadowptr returns a pointer to the shadow value for addr.
//go:nosplit
func shadowptr(addr uintptr) *uintptr {
	var shadow *uintptr
	if mheap_.data_start <= addr && addr < mheap_.data_end {
		shadow = (*uintptr)(unsafe.Pointer(addr + mheap_.shadow_data))
	} else if inheap(addr) {
		shadow = (*uintptr)(unsafe.Pointer(addr + mheap_.shadow_heap))
	}
	return shadow
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

// NOTE: Really dst *unsafe.Pointer, src unsafe.Pointer,
// but if we do that, Go inserts a write barrier on *dst = src.
//go:nosplit
func writebarrierptr(dst *uintptr, src uintptr) {
	if !needwb() {
		*dst = src
		return
	}

	if src != 0 && (src < _PageSize || src == _PoisonGC || src == _PoisonStack) {
		systemstack(func() { throw("bad pointer in write barrier") })
	}

	if mheap_.shadow_enabled {
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

	*dst = src
	writebarrierptr_nostore1(dst, src)
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

// noShadow is stored in as the shadow pointer to mark that there is no
// shadow word recorded. It matches any actual pointer word.
// noShadow is used when it is impossible to know the right word
// to store in the shadow heap, such as when the real heap word
// is being manipulated atomically.
const noShadow uintptr = 1

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

// Like writebarrierptr, but the store has already been applied.
// Do not reapply.
//go:nosplit
func writebarrierptr_nostore(dst *uintptr, src uintptr) {
	if !needwb() {
		return
	}

	if src != 0 && (src < _PageSize || src == _PoisonGC || src == _PoisonStack) {
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

//go:nosplit
func writebarrierptr_nostore1(dst *uintptr, src uintptr) {
	mp := acquirem()
	if mp.inwb || mp.dying > 0 {
		releasem(mp)
		return
	}
	mp.inwb = true
	systemstack(func() {
		gcmarkwb_m(dst, src)
	})
	mp.inwb = false
	releasem(mp)
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

//go:linkname reflect_typedmemmove reflect.typedmemmove
func reflect_typedmemmove(typ *_type, dst, src unsafe.Pointer) {
	typedmemmove(typ, dst, src)
}

// typedmemmove copies a value of type t to dst from src.
//go:nosplit
func typedmemmove(typ *_type, dst, src unsafe.Pointer) {
	if !needwb() || (typ.kind&kindNoPointers) != 0 {
		memmove(dst, src, typ.size)
		return
	}

	systemstack(func() {
		mask := loadPtrMask(typ)
		nptr := typ.size / ptrSize
		for i := uintptr(0); i < nptr; i += 2 {
			bits := mask[i/2]
			if (bits>>2)&_BitsMask == _BitsPointer {
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
			if (bits>>2)&_BitsMask == _BitsPointer {
				writebarrierptr((*uintptr)(dst), *(*uintptr)(src))
			} else {
				*(*uintptr)(dst) = *(*uintptr)(src)
			}
			dst = add(noescape(dst), ptrSize)
			src = add(noescape(src), ptrSize)
		}
	})
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

	mask := loadPtrMask(typ)
	nptr := (off + size) / ptrSize
	for i := uintptr(off / ptrSize); i < nptr; i++ {
		bits := mask[i/2] >> ((i & 1) << 2)
		if (bits>>2)&_BitsMask == _BitsPointer {
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
		mask := loadPtrMask(typ)
		// retoffset is known to be pointer-aligned (at least).
		// TODO(rsc): The noescape call should be unnecessary.
		dst := add(noescape(frame), retoffset)
		nptr := framesize / ptrSize
		for i := uintptr(retoffset / ptrSize); i < nptr; i++ {
			bits := mask[i/2] >> ((i & 1) << 2)
			if (bits>>2)&_BitsMask == _BitsPointer {
				writebarrierptr_nostore((*uintptr)(dst), *(*uintptr)(dst))
			}
			// TODO(rsc): The noescape call should be unnecessary.
			dst = add(noescape(dst), ptrSize)
		}
	})
}

//go:linkname reflect_typedslicecopy reflect.typedslicecopy
func reflect_typedslicecopy(elemType *_type, dst, src slice) int {
	return typedslicecopy(elemType, dst, src)
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

	if !needwb() {
		memmove(dstp, srcp, uintptr(n)*typ.size)
		return int(n)
	}

	systemstack(func() {
		if uintptr(srcp) < uintptr(dstp) && uintptr(srcp)+uintptr(n)*typ.size > uintptr(dstp) {
			// Overlap with src before dst.
			// Copy backward, being careful not to move dstp/srcp
			// out of the array they point into.
			dstp = add(dstp, uintptr(n-1)*typ.size)
			srcp = add(srcp, uintptr(n-1)*typ.size)
			i := uint(0)
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
			i := uint(0)
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
