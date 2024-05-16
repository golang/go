// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Runtime -> tracer API for memory events.

package runtime

import (
	"internal/abi"
	"runtime/internal/sys"
)

// traceSnapshotMemory takes a snapshot of all runtime memory that there are events for
// (heap spans, heap objects, goroutine stacks, etc.) and writes out events for them.
//
// The world must be stopped and tracing must be enabled when this function is called.
func traceSnapshotMemory() {
	assertWorldStopped()

	// Start tracing.
	trace := traceAcquire()
	if !trace.ok() {
		throw("traceSnapshotMemory: tracing is not enabled")
	}

	// Write out all the goroutine stacks.
	forEachGRace(func(gp *g) {
		trace.GoroutineStackExists(gp.stack.lo, gp.stack.hi-gp.stack.lo)
	})

	// Write out all the heap spans and heap objects.
	for _, s := range mheap_.allspans {
		if s.state.get() == mSpanDead {
			continue
		}
		// It's some kind of span, so trace that it exists.
		trace.SpanExists(s)

		// Write out allocated objects if it's a heap span.
		if s.state.get() != mSpanInUse {
			continue
		}

		// Find all allocated objects.
		abits := s.allocBitsForIndex(0)
		for i := uintptr(0); i < uintptr(s.nelems); i++ {
			if !abits.isMarked() {
				continue
			}
			x := s.base() + i*s.elemsize
			trace.HeapObjectExists(x, s.typePointersOfUnchecked(x).typ)
			abits.advance()
		}
	}
	traceRelease(trace)
}

func traceSpanTypeAndClass(s *mspan) traceArg {
	if s.state.get() == mSpanInUse {
		return traceArg(s.spanclass) << 1
	}
	return traceArg(1)
}

// SpanExists records an event indicating that the span exists.
func (tl traceLocker) SpanExists(s *mspan) {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvSpan, traceSpanID(s), traceArg(s.npages), traceSpanTypeAndClass(s))
}

// SpanAlloc records an event indicating that the span has just been allocated.
func (tl traceLocker) SpanAlloc(s *mspan) {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvSpanAlloc, traceSpanID(s), traceArg(s.npages), traceSpanTypeAndClass(s))
}

// SpanFree records an event indicating that the span is about to be freed.
func (tl traceLocker) SpanFree(s *mspan) {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvSpanFree, traceSpanID(s))
}

// traceSpanID creates a trace ID for the span s for the trace.
func traceSpanID(s *mspan) traceArg {
	return traceArg(uint64(s.base())-trace.minPageHeapAddr) / pageSize
}

// HeapObjectExists records that an object already exists at addr with the provided type.
// The type is optional, and the size of the slot occupied the object is inferred from the
// span containing it.
func (tl traceLocker) HeapObjectExists(addr uintptr, typ *abi.Type) {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvHeapObject, traceHeapObjectID(addr), tl.rtype(typ))
}

// HeapObjectAlloc records that an object was newly allocated at addr with the provided type.
// The type is optional, and the size of the slot occupied the object is inferred from the
// span containing it.
func (tl traceLocker) HeapObjectAlloc(addr uintptr, typ *abi.Type) {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvHeapObjectAlloc, traceHeapObjectID(addr), tl.rtype(typ))
}

// HeapObjectFree records that an object at addr is about to be freed.
func (tl traceLocker) HeapObjectFree(addr uintptr) {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvHeapObjectFree, traceHeapObjectID(addr))
}

// traceHeapObjectID creates a trace ID for a heap object at address addr.
func traceHeapObjectID(addr uintptr) traceArg {
	return traceArg(uint64(addr)-trace.minPageHeapAddr) / 8
}

// GoroutineStackExists records that a goroutine stack already exists at address base with the provided size.
func (tl traceLocker) GoroutineStackExists(base, size uintptr) {
	order := traceCompressStackSize(size)
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGoroutineStack, traceGoroutineStackID(base), order)
}

// GoroutineStackAlloc records that a goroutine stack was newly allocated at address base with the provided size..
func (tl traceLocker) GoroutineStackAlloc(base, size uintptr) {
	order := traceCompressStackSize(size)
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGoroutineStackAlloc, traceGoroutineStackID(base), order)
}

// GoroutineStackFree records that a goroutine stack at address base is about to be freed.
func (tl traceLocker) GoroutineStackFree(base uintptr) {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGoroutineStackFree, traceGoroutineStackID(base))
}

// traceGoroutineStackID creates a trace ID for the goroutine stack from its base address.
func traceGoroutineStackID(base uintptr) traceArg {
	return traceArg(uint64(base)-trace.minPageHeapAddr) / fixedStack
}

// traceCompressStackSize assumes size is a power of 2 and returns log2(size).
func traceCompressStackSize(size uintptr) traceArg {
	if size&(size-1) != 0 {
		throw("goroutine stack size is not a power of 2")
	}
	return traceArg(sys.Len64(uint64(size)))
}
