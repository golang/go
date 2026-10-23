// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm.stackswitch

// Continuation table for WebAssembly stack switching (GOWASM=stackswitch).
//
// With stack switching, a g's WebAssembly execution stack is a continuation,
// a reference of type (ref null $cont). Reference values are opaque and
// cannot be stored in linear memory, so they cannot live in a g. Instead the
// linker declares a continuation table, wasm table wasmContTable, and every
// g owns one slot of it, recorded in g.wasmCont.index. A slot holds the g's
// suspended continuation, or null if the g has not started yet.
//
// m0.g0 is allocated statically rather than by malg, so its index is the zero
// value: slot wasmContG0Slot belongs to it. malg gives every other g a slot
// of its own, starting at 1, which the g and its continuation keep for their
// existence, including while the g is on a free list. Like the g's
// themselves, slots are never freed, so the table grows to the number of g's
// ever allocated.
//
// wasm runs a single thread, so the allocation state below needs no locking.

package runtime

const (
	// wasmContTable is the index of the continuation table among the
	// module's wasm tables. It must match cmd/link/internal/wasm's
	// writeTableSec, which declares it with an initial size of
	// contTableInitialSize. Assembly refers to it as $const_wasmContTable.
	wasmContTable = 1

	// wasmContG0Slot is m0.g0's slot in the continuation table.
	// Assembly refers to it as $const_wasmContG0Slot.
	wasmContG0Slot = 0
)

// wasmContPerG is the per-g state of WebAssembly stack switching.
type wasmContPerG struct {
	// index is the g's slot in the continuation table: wasmContG0Slot for
	// m0.g0, and a slot of its own, assigned by malg, for every other g.
	index int32
}

var (
	wasmContLast int32 // the highest slot assigned by wasmContAlloc so far
	wasmContSize int32 // the continuation table's size, as last read or grown
)

// wasmContAlloc gives gp, which malg just allocated, the next unused slot of
// the continuation table, growing the table if the slot does not fit.
func wasmContAlloc(gp *g) {
	slot := wasmContLast + 1
	if slot >= wasmContSize {
		wasmContGrow(slot)
	}
	wasmContLast = slot
	gp.wasmCont.index = slot
}

// wasmContGrow grows the continuation table so that it has slot n. It at
// least doubles the table and does not depend on the table's initial
// size, which the linker sets.
func wasmContGrow(n int32) {
	size := wasmContTableSize()
	if n >= size {
		delta := max(size, n+1-size)
		if wasmContTableGrow(delta) < 0 {
			throw("runtime: cannot grow WebAssembly continuation table")
		}
		size += delta
	}
	wasmContSize = size
}

// Implemented in stackswitch_wasm.s.

// wasmContTableSize returns the size of the continuation table (table.size).
func wasmContTableSize() int32

// wasmContTableGrow appends delta null slots to the continuation table and
// returns its previous size, or -1 if it cannot grow (table.grow).
func wasmContTableGrow(delta int32) int32
