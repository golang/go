// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"testing"

	"cmd/compile/internal/ir"
)

func TestForeignLeaksAllResultLevelsRoundTrip(t *testing.T) {
	var want leaks
	for i := range numEscResults {
		want.AddResult(i, i)
	}
	want.AddForeign(5)

	const encoded = "escf:\x00\x00\x00\x01\x02\x03\x04\x05\x06"
	if got := want.Encode(); got != encoded {
		t.Fatalf("Encode() = %q, want %q", got, encoded)
	}
	got := ParseLeaks(encoded)
	assertForeignLeakLevels(t, got, [9]int{-1, -1, -1, 0, 1, 2, 3, 4, 5})
}

func TestForeignLeaksLegacyEsc8DoesNotInventForeign(t *testing.T) {
	const encoded = "esc:\x02\x03\x04\x01\x02\x03\x04\x05"
	got := ParseLeaks(encoded)
	assertForeignLeakLevels(t, got, [9]int{1, 2, 3, 0, 1, 2, 3, 4, -1})
	if roundTrip := got.Encode(); roundTrip != encoded {
		t.Fatalf("legacy Encode() = %q, want %q", roundTrip, encoded)
	}
}

func TestForeignLeaksEscfPreservesForeignLevels(t *testing.T) {
	const encoded = "escf:\x02\x03\x04\x02\x03\x04\x05\x06\x07"
	got := ParseLeaks(encoded)
	assertForeignLeakLevels(t, got, [9]int{1, 2, 3, 1, 2, 3, 4, 5, 6})
	if roundTrip := got.Encode(); roundTrip != encoded {
		t.Fatalf("escf Encode() = %q, want %q", roundTrip, encoded)
	}
}

func TestForeignLeaksHeapDominanceClearsWeakerFlows(t *testing.T) {
	var got leaks
	got.AddHeap(2)
	got.AddMutator(1)
	got.AddCallee(2)
	got.AddResult(0, 1)
	got.AddResult(1, 2)
	got.AddResult(2, 3)
	got.AddResult(3, 1)
	got.AddResult(4, 4)
	got.AddForeign(3)

	got.Optimize()
	assertForeignLeakLevels(t, got, [9]int{2, 1, -1, 1, -1, -1, 1, -1, -1})
}

func TestForeignLeaksUnknownPrefixFailsHeap(t *testing.T) {
	got := ParseLeaks("unknown:\x02\x03\x04")
	assertForeignLeakLevels(t, got, [9]int{0, -1, -1, -1, -1, -1, -1, -1, -1})
}

func TestForeignExactMakesPlacementAndNonEscape(t *testing.T) {
	enableLogosArena(t)
	control := logosArenaDynamicMake(ir.EscNone)
	if !foreignBacking(&location{n: control}) {
		t.Fatal("dynamic make is not foreignBacking-shaped")
	}
	if logosArenaSolvedHeap(control) {
		t.Fatal("EscNone control reported as heap-forced")
	}
	if !LogosArenaMake(control, &ir.Func{}) {
		t.Fatal("variable-capacity make without a foreign heap decision was vetoed")
	}
	forced := logosArenaDynamicMake(ir.EscHeap)
	if !foreignBacking(&location{n: forced}) {
		t.Fatal("forced make lost its foreignBacking shape")
	}
	if !logosArenaSolvedHeap(forced) {
		t.Fatal("EscHeap make not reported as heap-forced")
	}
	if LogosArenaMake(forced, &ir.Func{}) {
		t.Error("foreign-forced make was arena-placed")
	}
}

func assertForeignLeakLevels(t *testing.T, got leaks, want [9]int) {
	t.Helper()
	levels := [9]int{
		got.Heap(), got.Mutator(), got.Callee(),
		got.Result(0), got.Result(1), got.Result(2), got.Result(3), got.Result(4),
		got.Foreign(),
	}
	if levels != want {
		t.Errorf("leak levels = %v, want %v", levels, want)
	}
}
