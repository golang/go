// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector: write barriers.
//
// For the concurrent garbage collector, the Go compiler implements
// updates to pointer-valued fields that may be in heap objects by
// emitting calls to write barriers. This file contains the actual write barrier
// implementation, gcmarkwb_m, and the various wrappers called by the
// compiler to implement pointer assignment, slice assignment,
// typed memmove, and so on.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

// gcmarkwb_m is the mark-phase write barrier, the only barrier we have.
// The rest of this file exists only to make calls to this function.
//
// This is a hybrid barrier that combines a Yuasa-style deletion
// barrier—which shades the object whose reference is being
// overwritten—with Dijkstra insertion barrier—which shades the object
// whose reference is being written. The insertion part of the barrier
// is necessary while the calling goroutine's stack is grey. In
// pseudocode, the barrier is:
//
//     writePointer(slot, ptr):
//         shade(*slot)
//         if current stack is grey:
//             shade(ptr)
//         *slot = ptr
//
// slot is the destination in Go code.
// ptr is the value that goes into the slot in Go code.
//
// Shade indicates that it has seen a white pointer by adding the referent
// to wbuf as well as marking it.
//
// The two shades and the condition work together to prevent a mutator
// from hiding an object from the garbage collector:
//
// 1. shade(*slot) prevents a mutator from hiding an object by moving
// the sole pointer to it from the heap to its stack. If it attempts
// to unlink an object from the heap, this will shade it.
//
// 2. shade(ptr) prevents a mutator from hiding an object by moving
// the sole pointer to it from its stack into a black object in the
// heap. If it attempts to install the pointer into a black object,
// this will shade it.
//
// 3. Once a goroutine's stack is black, the shade(ptr) becomes
// unnecessary. shade(ptr) prevents hiding an object by moving it from
// the stack to the heap, but this requires first having a pointer
// hidden on the stack. Immediately after a stack is scanned, it only
// points to shaded objects, so it's not hiding anything, and the
// shade(*slot) prevents it from hiding any other pointers on its
// stack.
//
// For a detailed description of this barrier and proof of
// correctness, see https://github.com/golang/proposal/blob/master/design/17503-eliminate-rescan.md
//
//
//
// Dealing with memory ordering:
//
// Both the Yuasa and Dijkstra barriers can be made conditional on the
// color of the object containing the slot. We chose not to make these
// conditional because the cost of ensuring that the object holding
// the slot doesn't concurrently change color without the mutator
// noticing seems prohibitive.
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
//
// Another place where we intentionally omit memory barriers is when
// accessing mheap_.arena_used to check if a pointer points into the
// heap. On relaxed memory machines, it's possible for a mutator to
// extend the size of the heap by updating arena_used, allocate an
// object from this new region, and publish a pointer to that object,
// but for tracing running on another processor to observe the pointer
// but use the old value of arena_used. In this case, tracing will not
// mark the object, even though it's reachable. However, the mutator
// is guaranteed to execute a write barrier when it publishes the
// pointer, so it will take care of marking the object. A general
// consequence of this is that the garbage collector may cache the
// value of mheap_.arena_used. (See issue #9984.)
//
//
// Stack writes:
//
// The compiler omits write barriers for writes to the current frame,
// but if a stack pointer has been passed down the call stack, the
// compiler will generate a write barrier for writes through that
// pointer (because it doesn't know it's not a heap pointer).
//
// One might be tempted to ignore the write barrier if slot points
// into to the stack. Don't do it! Mark termination only re-scans
// frames that have potentially been active since the concurrent scan,
// so it depends on write barriers to track changes to pointers in
// stack frames that have not been active.
//
//
// Global writes:
//
// The Go garbage collector requires write barriers when heap pointers
// are stored in globals. Many garbage collectors ignore writes to
// globals and instead pick up global -> heap pointers during
// termination. This increases pause time, so we instead rely on write
// barriers for writes to globals so that we don't have to rescan
// global during mark termination.
//
//
// Publication ordering:
//
// The write barrier is *pre-publication*, meaning that the write
// barrier happens prior to the *slot = ptr write that may make ptr
// reachable by some goroutine that currently cannot reach it.
//
//
//go:nowritebarrierrec
//go:systemstack
func gcmarkwb_m(slot *uintptr, ptr uintptr) {
	if writeBarrier.needed {
		// Note: This turns bad pointer writes into bad
		// pointer reads, which could be confusing. We avoid
		// reading from obviously bad pointers, which should
		// take care of the vast majority of these. We could
		// patch this up in the signal handler, or use XCHG to
		// combine the read and the write. Checking inheap is
		// insufficient since we need to track changes to
		// roots outside the heap.
		//
		// Note: profbuf.go omits a barrier during signal handler
		// profile logging; that's safe only because this deletion barrier exists.
		// If we remove the deletion barrier, we'll have to work out
		// a new way to handle the profile logging.
		if slot1 := uintptr(unsafe.Pointer(slot)); slot1 >= minPhysPageSize {
			if optr := *slot; optr != 0 {
				shade(optr)
			}
		}
		// TODO: Make this conditional on the caller's stack color.
		if ptr != 0 && inheap(ptr) {
			shade(ptr)
		}
	}
}

// writebarrierptr_prewrite1 invokes a write barrier for *dst = src
// prior to the write happening.
//
// Write barrier calls must not happen during critical GC and scheduler
// related operations. In particular there are times when the GC assumes
// that the world is stopped but scheduler related code is still being
// executed, dealing with syscalls, dealing with putting gs on runnable
// queues and so forth. This code cannot execute write barriers because
// the GC might drop them on the floor. Stopping the world involves removing
// the p associated with an m. We use the fact that m.p == nil to indicate
// that we are in one these critical section and throw if the write is of
// a pointer to a heap object.
//go:nosplit
func writebarrierptr_prewrite1(dst *uintptr, src uintptr) {
	mp := acquirem()
	if mp.inwb || mp.dying > 0 {
		releasem(mp)
		return
	}
	systemstack(func() {
		if mp.p == 0 && memstats.enablegc && !mp.inwb && inheap(src) {
			throw("writebarrierptr_prewrite1 called with mp.p == nil")
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
	if writeBarrier.cgo {
		cgoCheckWriteBarrier(dst, src)
	}
	if !writeBarrier.needed {
		*dst = src
		return
	}
	if src != 0 && src < minPhysPageSize {
		systemstack(func() {
			print("runtime: writebarrierptr *", dst, " = ", hex(src), "\n")
			throw("bad pointer in write barrier")
		})
	}
	writebarrierptr_prewrite1(dst, src)
	*dst = src
}

// writebarrierptr_prewrite is like writebarrierptr, but the store
// will be performed by the caller after this call. The caller must
// not allow preemption between this call and the write.
//
//go:nosplit
func writebarrierptr_prewrite(dst *uintptr, src uintptr) {
	if writeBarrier.cgo {
		cgoCheckWriteBarrier(dst, src)
	}
	if !writeBarrier.needed {
		return
	}
	if src != 0 && src < minPhysPageSize {
		systemstack(func() { throw("bad pointer in write barrier") })
	}
	writebarrierptr_prewrite1(dst, src)
}

// typedmemmove copies a value of type t to dst from src.
// Must be nosplit, see #16026.
//go:nosplit
func typedmemmove(typ *_type, dst, src unsafe.Pointer) {
	if typ.kind&kindNoPointers == 0 {
		bulkBarrierPreWrite(uintptr(dst), uintptr(src), typ.size)
	}
	// There's a race here: if some other goroutine can write to
	// src, it may change some pointer in src after we've
	// performed the write barrier but before we perform the
	// memory copy. This safe because the write performed by that
	// other goroutine must also be accompanied by a write
	// barrier, so at worst we've unnecessarily greyed the old
	// pointer that was in src.
	memmove(dst, src, typ.size)
	if writeBarrier.cgo {
		cgoCheckMemmove(typ, dst, src, 0, typ.size)
	}
}

//go:linkname reflect_typedmemmove reflect.typedmemmove
func reflect_typedmemmove(typ *_type, dst, src unsafe.Pointer) {
	if raceenabled {
		raceWriteObjectPC(typ, dst, getcallerpc(unsafe.Pointer(&typ)), funcPC(reflect_typedmemmove))
		raceReadObjectPC(typ, src, getcallerpc(unsafe.Pointer(&typ)), funcPC(reflect_typedmemmove))
	}
	if msanenabled {
		msanwrite(dst, typ.size)
		msanread(src, typ.size)
	}
	typedmemmove(typ, dst, src)
}

// typedmemmovepartial is like typedmemmove but assumes that
// dst and src point off bytes into the value and only copies size bytes.
//go:linkname reflect_typedmemmovepartial reflect.typedmemmovepartial
func reflect_typedmemmovepartial(typ *_type, dst, src unsafe.Pointer, off, size uintptr) {
	if writeBarrier.needed && typ.kind&kindNoPointers == 0 && size >= sys.PtrSize {
		// Pointer-align start address for bulk barrier.
		adst, asrc, asize := dst, src, size
		if frag := -off & (sys.PtrSize - 1); frag != 0 {
			adst = add(dst, frag)
			asrc = add(src, frag)
			asize -= frag
		}
		bulkBarrierPreWrite(uintptr(adst), uintptr(asrc), asize&^(sys.PtrSize-1))
	}

	memmove(dst, src, size)
	if writeBarrier.cgo {
		cgoCheckMemmove(typ, dst, src, off, size)
	}
}

// reflectcallmove is invoked by reflectcall to copy the return values
// out of the stack and into the heap, invoking the necessary write
// barriers. dst, src, and size describe the return value area to
// copy. typ describes the entire frame (not just the return values).
// typ may be nil, which indicates write barriers are not needed.
//
// It must be nosplit and must only call nosplit functions because the
// stack map of reflectcall is wrong.
//
//go:nosplit
func reflectcallmove(typ *_type, dst, src unsafe.Pointer, size uintptr) {
	if writeBarrier.needed && typ != nil && typ.kind&kindNoPointers == 0 && size >= sys.PtrSize {
		bulkBarrierPreWrite(uintptr(dst), uintptr(src), size)
	}
	memmove(dst, src, size)
}

//go:nosplit
func typedslicecopy(typ *_type, dst, src slice) int {
	// TODO(rsc): If typedslicecopy becomes faster than calling
	// typedmemmove repeatedly, consider using during func growslice.
	n := dst.len
	if n > src.len {
		n = src.len
	}
	if n == 0 {
		return 0
	}
	dstp := dst.array
	srcp := src.array

	if raceenabled {
		callerpc := getcallerpc(unsafe.Pointer(&typ))
		pc := funcPC(slicecopy)
		racewriterangepc(dstp, uintptr(n)*typ.size, callerpc, pc)
		racereadrangepc(srcp, uintptr(n)*typ.size, callerpc, pc)
	}
	if msanenabled {
		msanwrite(dstp, uintptr(n)*typ.size)
		msanread(srcp, uintptr(n)*typ.size)
	}

	if writeBarrier.cgo {
		cgoCheckSliceCopy(typ, dst, src, n)
	}

	// Note: No point in checking typ.kind&kindNoPointers here:
	// compiler only emits calls to typedslicecopy for types with pointers,
	// and growslice and reflect_typedslicecopy check for pointers
	// before calling typedslicecopy.
	if !writeBarrier.needed {
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
	return n
}

//go:linkname reflect_typedslicecopy reflect.typedslicecopy
func reflect_typedslicecopy(elemType *_type, dst, src slice) int {
	if elemType.kind&kindNoPointers != 0 {
		n := dst.len
		if n > src.len {
			n = src.len
		}
		if n == 0 {
			return 0
		}

		size := uintptr(n) * elemType.size
		if raceenabled {
			callerpc := getcallerpc(unsafe.Pointer(&elemType))
			pc := funcPC(reflect_typedslicecopy)
			racewriterangepc(dst.array, size, callerpc, pc)
			racereadrangepc(src.array, size, callerpc, pc)
		}
		if msanenabled {
			msanwrite(dst.array, size)
			msanread(src.array, size)
		}

		memmove(dst.array, src.array, size)
		return n
	}
	return typedslicecopy(elemType, dst, src)
}

// typedmemclr clears the typed memory at ptr with type typ. The
// memory at ptr must already be initialized (and hence in type-safe
// state). If the memory is being initialized for the first time, see
// memclrNoHeapPointers.
//
// If the caller knows that typ has pointers, it can alternatively
// call memclrHasPointers.
//
//go:nosplit
func typedmemclr(typ *_type, ptr unsafe.Pointer) {
	if typ.kind&kindNoPointers == 0 {
		bulkBarrierPreWrite(uintptr(ptr), 0, typ.size)
	}
	memclrNoHeapPointers(ptr, typ.size)
}

// memclrHasPointers clears n bytes of typed memory starting at ptr.
// The caller must ensure that the type of the object at ptr has
// pointers, usually by checking typ.kind&kindNoPointers. However, ptr
// does not have to point to the start of the allocation.
//
//go:nosplit
func memclrHasPointers(ptr unsafe.Pointer, n uintptr) {
	bulkBarrierPreWrite(uintptr(ptr), 0, n)
	memclrNoHeapPointers(ptr, n)
}
