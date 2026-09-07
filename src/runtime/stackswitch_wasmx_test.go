// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (wasm && wasm.stackswitch) || (wasm32 && wasm32.stackswitch)

package runtime_test

import (
	"runtime"
	"testing"
)

func TestWasmContinuationTable(t *testing.T) {
	size := runtime.WasmContTableSize()
	if size <= 0 {
		t.Fatalf("continuation table size = %d, want > 0", size)
	}
	if old := runtime.WasmContTableGrow(1); old != size {
		t.Fatalf("table.grow returned %d, want previous size %d", old, size)
	}
	if got := runtime.WasmContTableSize(); got != size+1 {
		t.Fatalf("continuation table size = %d, want %d", got, size+1)
	}

	// Keep the goroutines alive together so their slots must be distinct.
	const count = 32
	slots := make(chan int32, count)
	done := make(chan struct{})
	defer close(done)
	for range count {
		go func() {
			slots <- runtime.WasmContSlot()
			<-done
		}()
	}
	seen := make(map[int32]bool)
	for range count {
		slot := <-slots
		if slot <= 0 || slot >= runtime.WasmContTableSize() {
			t.Errorf("goroutine continuation slot %d is outside the table", slot)
		}
		if seen[slot] {
			t.Errorf("goroutines share continuation slot %d", slot)
		}
		seen[slot] = true
	}
}
