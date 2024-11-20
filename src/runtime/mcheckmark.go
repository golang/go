// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// GC checkmarks
//
// In a concurrent garbage collector, one worries about failing to mark
// a live object due to mutations without write barriers or bugs in the
// collector implementation. As a sanity check, the GC has a 'checkmark'
// mode that retraverses the object graph with the world stopped, to make
// sure that everything that should be marked is marked.

package runtime

import (
	"internal/goarch"
	"internal/runtime/atomic"
	"internal/runtime/sys"
	"unsafe"
)

// A checkmarksMap stores the GC marks in "checkmarks" mode. It is a
// per-arena bitmap with a bit for every word in the arena. The mark
// is stored on the bit corresponding to the first word of the marked
// allocation.
type checkmarksMap struct {
	_ sys.NotInHeap
	b [heapArenaBytes / goarch.PtrSize / 8]uint8
}

// If useCheckmark is true, marking of an object uses the checkmark
// bits instead of the standard mark bits.
var useCheckmark = false

// startCheckmarks prepares for the checkmarks phase.
//
// The world must be stopped.
func startCheckmarks() {
	assertWorldStopped()

	// Clear all checkmarks.
	for _, ai := range mheap_.allArenas {
		arena := mheap_.arenas[ai.l1()][ai.l2()]
		bitmap := arena.checkmarks

		if bitmap == nil {
			// Allocate bitmap on first use.
			bitmap = (*checkmarksMap)(persistentalloc(unsafe.Sizeof(*bitmap), 0, &memstats.gcMiscSys))
			if bitmap == nil {
				throw("out of memory allocating checkmarks bitmap")
			}
			arena.checkmarks = bitmap
		} else {
			// Otherwise clear the existing bitmap.
			clear(bitmap.b[:])
		}
	}
	// Enable checkmarking.
	useCheckmark = true
}

// endCheckmarks ends the checkmarks phase.
func endCheckmarks() {
	if gcMarkWorkAvailable(nil) {
		throw("GC work not flushed")
	}
	useCheckmark = false
}

// setCheckmark throws if marking object is a checkmarks violation,
// and otherwise sets obj's checkmark. It returns true if obj was
// already checkmarked.
func setCheckmark(obj, base, off uintptr, mbits markBits) bool {
	if !mbits.isMarked() {
		printlock()
		print("runtime: checkmarks found unexpected unmarked object obj=", hex(obj), "\n")
		print("runtime: found obj at *(", hex(base), "+", hex(off), ")\n")

		// Dump the source (base) object
		gcDumpObject("base", base, off)

		// Dump the object
		gcDumpObject("obj", obj, ^uintptr(0))

		getg().m.traceback = 2
		throw("checkmark found unmarked object")
	}

	ai := arenaIndex(obj)
	arena := mheap_.arenas[ai.l1()][ai.l2()]
	arenaWord := (obj / heapArenaBytes / 8) % uintptr(len(arena.checkmarks.b))
	mask := byte(1 << ((obj / heapArenaBytes) % 8))
	bytep := &arena.checkmarks.b[arenaWord]

	if atomic.Load8(bytep)&mask != 0 {
		// Already checkmarked.
		return true
	}

	atomic.Or8(bytep, mask)
	return false
}
