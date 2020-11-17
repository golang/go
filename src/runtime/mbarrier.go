// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector: write barriers.
//
// For the concurrent garbage collector, the Go compiler implements
// updates to pointer-valued fields that may be in heap objects by
// emitting calls to write barriers. The main write barrier for
// individual pointer writes is gcWriteBarrier and is implemented in
// assembly. This file contains write barrier entry points for bulk
// operations. See also mwbbuf.go.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

// Go uses a hybrid barrier that combines a Yuasa-style deletion
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
// Signal handler pointer writes:
//
// In general, the signal handler cannot safely invoke the write
// barrier because it may run without a P or even during the write
// barrier.
//
// There is exactly one exception: profbuf.go omits a barrier during
// signal handler profile logging. That's safe only because of the
// deletion barrier. See profbuf.go for a detailed argument. If we
// remove the deletion barrier, we'll have to work out a new way to
// handle the profile logging.

// typedmemmove copies a value of type t to dst from src.
// Must be nosplit, see #16026.
//
// TODO: Perfect for go:nosplitrec since we can't have a safe point
// anywhere in the bulk barrier or memmove.
//
//go:nosplit
func typedmemmove(typ *_type, dst, src unsafe.Pointer) {
	if dst == src {
		return
	}
	if writeBarrier.needed && typ.ptrdata != 0 {
		bulkBarrierPreWrite(uintptr(dst), uintptr(src), typ.ptrdata)
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
		raceWriteObjectPC(typ, dst, getcallerpc(), funcPC(reflect_typedmemmove))
		raceReadObjectPC(typ, src, getcallerpc(), funcPC(reflect_typedmemmove))
	}
	if msanenabled {
		msanwrite(dst, typ.size)
		msanread(src, typ.size)
	}
	typedmemmove(typ, dst, src)
}

//go:linkname reflectlite_typedmemmove internal/reflectlite.typedmemmove
func reflectlite_typedmemmove(typ *_type, dst, src unsafe.Pointer) {
	reflect_typedmemmove(typ, dst, src)
}

// typedmemmovepartial is like typedmemmove but assumes that
// dst and src point off bytes into the value and only copies size bytes.
// off must be a multiple of sys.PtrSize.
//go:linkname reflect_typedmemmovepartial reflect.typedmemmovepartial
func reflect_typedmemmovepartial(typ *_type, dst, src unsafe.Pointer, off, size uintptr) {
	if writeBarrier.needed && typ.ptrdata > off && size >= sys.PtrSize {
		if off&(sys.PtrSize-1) != 0 {
			panic("reflect: internal error: misaligned offset")
		}
		pwsize := alignDown(size, sys.PtrSize)
		if poff := typ.ptrdata - off; pwsize > poff {
			pwsize = poff
		}
		bulkBarrierPreWrite(uintptr(dst), uintptr(src), pwsize)
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
	if writeBarrier.needed && typ != nil && typ.ptrdata != 0 && size >= sys.PtrSize {
		bulkBarrierPreWrite(uintptr(dst), uintptr(src), size)
	}
	memmove(dst, src, size)
}

//go:nosplit
func typedslicecopy(typ *_type, dstPtr unsafe.Pointer, dstLen int, srcPtr unsafe.Pointer, srcLen int) int {
	n := dstLen
	if n > srcLen {
		n = srcLen
	}
	if n == 0 {
		return 0
	}

	// The compiler emits calls to typedslicecopy before
	// instrumentation runs, so unlike the other copying and
	// assignment operations, it's not instrumented in the calling
	// code and needs its own instrumentation.
	if raceenabled {
		callerpc := getcallerpc()
		pc := funcPC(slicecopy)
		racewriterangepc(dstPtr, uintptr(n)*typ.size, callerpc, pc)
		racereadrangepc(srcPtr, uintptr(n)*typ.size, callerpc, pc)
	}
	if msanenabled {
		msanwrite(dstPtr, uintptr(n)*typ.size)
		msanread(srcPtr, uintptr(n)*typ.size)
	}

	if writeBarrier.cgo {
		cgoCheckSliceCopy(typ, dstPtr, srcPtr, n)
	}

	if dstPtr == srcPtr {
		return n
	}

	// Note: No point in checking typ.ptrdata here:
	// compiler only emits calls to typedslicecopy for types with pointers,
	// and growslice and reflect_typedslicecopy check for pointers
	// before calling typedslicecopy.
	size := uintptr(n) * typ.size
	if writeBarrier.needed {
		pwsize := size - typ.size + typ.ptrdata
		bulkBarrierPreWrite(uintptr(dstPtr), uintptr(srcPtr), pwsize)
	}
	// See typedmemmove for a discussion of the race between the
	// barrier and memmove.
	memmove(dstPtr, srcPtr, size)
	return n
}

//go:linkname reflect_typedslicecopy reflect.typedslicecopy
func reflect_typedslicecopy(elemType *_type, dst, src slice) int {
	if elemType.ptrdata == 0 {
		return slicecopy(dst.array, dst.len, src.array, src.len, elemType.size)
	}
	return typedslicecopy(elemType, dst.array, dst.len, src.array, src.len)
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
	if writeBarrier.needed && typ.ptrdata != 0 {
		bulkBarrierPreWrite(uintptr(ptr), 0, typ.ptrdata)
	}
	memclrNoHeapPointers(ptr, typ.size)
}

//go:linkname reflect_typedmemclr reflect.typedmemclr
func reflect_typedmemclr(typ *_type, ptr unsafe.Pointer) {
	typedmemclr(typ, ptr)
}

//go:linkname reflect_typedmemclrpartial reflect.typedmemclrpartial
func reflect_typedmemclrpartial(typ *_type, ptr unsafe.Pointer, off, size uintptr) {
	if writeBarrier.needed && typ.ptrdata != 0 {
		bulkBarrierPreWrite(uintptr(ptr), 0, size)
	}
	memclrNoHeapPointers(ptr, size)
}

// memclrHasPointers clears n bytes of typed memory starting at ptr.
// The caller must ensure that the type of the object at ptr has
// pointers, usually by checking typ.ptrdata. However, ptr
// does not have to point to the start of the allocation.
//
//go:nosplit
func memclrHasPointers(ptr unsafe.Pointer, n uintptr) {
	bulkBarrierPreWrite(uintptr(ptr), 0, n)
	memclrNoHeapPointers(ptr, n)
}
