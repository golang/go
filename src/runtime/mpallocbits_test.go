// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"testing"
)

// Ensures that got and want are the same, and if not, reports
// detailed diff information.
func checkPallocBits(t *testing.T, got, want *PallocBits) {
	d := DiffPallocBits(got, want)
	if len(d) != 0 {
		t.Errorf("%d range(s) different", len(d))
		for _, bits := range d {
			t.Logf("\t@ bit index %d", bits.I)
			t.Logf("\t|  got: %s", StringifyPallocBits(got, bits))
			t.Logf("\t| want: %s", StringifyPallocBits(want, bits))
		}
	}
}

// makePallocBits produces an initialized PallocBits by setting
// the ranges in s to 1 and the rest to zero.
func makePallocBits(s []BitRange) *PallocBits {
	b := new(PallocBits)
	for _, v := range s {
		b.AllocRange(v.I, v.N)
	}
	return b
}

// Ensures that PallocBits.AllocRange works, which is a fundamental
// method used for testing and initialization since it's used by
// makePallocBits.
func TestPallocBitsAllocRange(t *testing.T) {
	test := func(t *testing.T, i, n uint, want *PallocBits) {
		checkPallocBits(t, makePallocBits([]BitRange{{i, n}}), want)
	}
	t.Run("OneLow", func(t *testing.T) {
		want := new(PallocBits)
		want[0] = 0x1
		test(t, 0, 1, want)
	})
	t.Run("OneHigh", func(t *testing.T) {
		want := new(PallocBits)
		want[PallocChunkPages/64-1] = 1 << 63
		test(t, PallocChunkPages-1, 1, want)
	})
	t.Run("Inner", func(t *testing.T) {
		want := new(PallocBits)
		want[2] = 0x3e
		test(t, 129, 5, want)
	})
	t.Run("Aligned", func(t *testing.T) {
		want := new(PallocBits)
		want[2] = ^uint64(0)
		want[3] = ^uint64(0)
		test(t, 128, 128, want)
	})
	t.Run("Begin", func(t *testing.T) {
		want := new(PallocBits)
		want[0] = ^uint64(0)
		want[1] = ^uint64(0)
		want[2] = ^uint64(0)
		want[3] = ^uint64(0)
		want[4] = ^uint64(0)
		want[5] = 0x1
		test(t, 0, 321, want)
	})
	t.Run("End", func(t *testing.T) {
		want := new(PallocBits)
		want[PallocChunkPages/64-1] = ^uint64(0)
		want[PallocChunkPages/64-2] = ^uint64(0)
		want[PallocChunkPages/64-3] = ^uint64(0)
		want[PallocChunkPages/64-4] = 1 << 63
		test(t, PallocChunkPages-(64*3+1), 64*3+1, want)
	})
	t.Run("All", func(t *testing.T) {
		want := new(PallocBits)
		for i := range want {
			want[i] = ^uint64(0)
		}
		test(t, 0, PallocChunkPages, want)
	})
}

// Inverts every bit in the PallocBits.
func invertPallocBits(b *PallocBits) {
	for i := range b {
		b[i] = ^b[i]
	}
}

// Ensures page allocation works.
func TestPallocBitsAlloc(t *testing.T) {
	tests := map[string]struct {
		before []BitRange
		after  []BitRange
		npages uintptr
		hits   []uint
	}{
		"AllFree1": {
			npages: 1,
			hits:   []uint{0, 1, 2, 3, 4, 5},
			after:  []BitRange{{0, 6}},
		},
		"AllFree2": {
			npages: 2,
			hits:   []uint{0, 2, 4, 6, 8, 10},
			after:  []BitRange{{0, 12}},
		},
		"AllFree5": {
			npages: 5,
			hits:   []uint{0, 5, 10, 15, 20},
			after:  []BitRange{{0, 25}},
		},
		"AllFree64": {
			npages: 64,
			hits:   []uint{0, 64, 128},
			after:  []BitRange{{0, 192}},
		},
		"AllFree65": {
			npages: 65,
			hits:   []uint{0, 65, 130},
			after:  []BitRange{{0, 195}},
		},
		"SomeFree64": {
			before: []BitRange{{0, 32}, {64, 32}, {100, PallocChunkPages - 100}},
			npages: 64,
			hits:   []uint{^uint(0)},
			after:  []BitRange{{0, 32}, {64, 32}, {100, PallocChunkPages - 100}},
		},
		"NoneFree1": {
			before: []BitRange{{0, PallocChunkPages}},
			npages: 1,
			hits:   []uint{^uint(0), ^uint(0)},
			after:  []BitRange{{0, PallocChunkPages}},
		},
		"NoneFree2": {
			before: []BitRange{{0, PallocChunkPages}},
			npages: 2,
			hits:   []uint{^uint(0), ^uint(0)},
			after:  []BitRange{{0, PallocChunkPages}},
		},
		"NoneFree5": {
			before: []BitRange{{0, PallocChunkPages}},
			npages: 5,
			hits:   []uint{^uint(0), ^uint(0)},
			after:  []BitRange{{0, PallocChunkPages}},
		},
		"NoneFree65": {
			before: []BitRange{{0, PallocChunkPages}},
			npages: 65,
			hits:   []uint{^uint(0), ^uint(0)},
			after:  []BitRange{{0, PallocChunkPages}},
		},
		"ExactFit1": {
			before: []BitRange{{0, PallocChunkPages/2 - 3}, {PallocChunkPages/2 - 2, PallocChunkPages/2 + 2}},
			npages: 1,
			hits:   []uint{PallocChunkPages/2 - 3, ^uint(0)},
			after:  []BitRange{{0, PallocChunkPages}},
		},
		"ExactFit2": {
			before: []BitRange{{0, PallocChunkPages/2 - 3}, {PallocChunkPages/2 - 1, PallocChunkPages/2 + 1}},
			npages: 2,
			hits:   []uint{PallocChunkPages/2 - 3, ^uint(0)},
			after:  []BitRange{{0, PallocChunkPages}},
		},
		"ExactFit5": {
			before: []BitRange{{0, PallocChunkPages/2 - 3}, {PallocChunkPages/2 + 2, PallocChunkPages/2 - 2}},
			npages: 5,
			hits:   []uint{PallocChunkPages/2 - 3, ^uint(0)},
			after:  []BitRange{{0, PallocChunkPages}},
		},
		"ExactFit65": {
			before: []BitRange{{0, PallocChunkPages/2 - 31}, {PallocChunkPages/2 + 34, PallocChunkPages/2 - 34}},
			npages: 65,
			hits:   []uint{PallocChunkPages/2 - 31, ^uint(0)},
			after:  []BitRange{{0, PallocChunkPages}},
		},
		"SomeFree161": {
			before: []BitRange{{0, 185}, {331, 1}},
			npages: 161,
			hits:   []uint{332},
			after:  []BitRange{{0, 185}, {331, 162}},
		},
	}
	for name, v := range tests {
		v := v
		t.Run(name, func(t *testing.T) {
			b := makePallocBits(v.before)
			for iter, i := range v.hits {
				a, _ := b.Find(v.npages, 0)
				if i != a {
					t.Fatalf("find #%d picked wrong index: want %d, got %d", iter+1, i, a)
				}
				if i != ^uint(0) {
					b.AllocRange(a, uint(v.npages))
				}
			}
			want := makePallocBits(v.after)
			checkPallocBits(t, b, want)
		})
	}
}

// Ensures page freeing works.
func TestPallocBitsFree(t *testing.T) {
	tests := map[string]struct {
		beforeInv []BitRange
		afterInv  []BitRange
		frees     []uint
		npages    uintptr
	}{
		"SomeFree": {
			npages:    1,
			beforeInv: []BitRange{{0, 32}, {64, 32}, {100, 1}},
			frees:     []uint{32},
			afterInv:  []BitRange{{0, 33}, {64, 32}, {100, 1}},
		},
		"NoneFree1": {
			npages:   1,
			frees:    []uint{0, 1, 2, 3, 4, 5},
			afterInv: []BitRange{{0, 6}},
		},
		"NoneFree2": {
			npages:   2,
			frees:    []uint{0, 2, 4, 6, 8, 10},
			afterInv: []BitRange{{0, 12}},
		},
		"NoneFree5": {
			npages:   5,
			frees:    []uint{0, 5, 10, 15, 20},
			afterInv: []BitRange{{0, 25}},
		},
		"NoneFree64": {
			npages:   64,
			frees:    []uint{0, 64, 128},
			afterInv: []BitRange{{0, 192}},
		},
		"NoneFree65": {
			npages:   65,
			frees:    []uint{0, 65, 130},
			afterInv: []BitRange{{0, 195}},
		},
	}
	for name, v := range tests {
		v := v
		t.Run(name, func(t *testing.T) {
			b := makePallocBits(v.beforeInv)
			invertPallocBits(b)
			for _, i := range v.frees {
				b.Free(i, uint(v.npages))
			}
			want := makePallocBits(v.afterInv)
			invertPallocBits(want)
			checkPallocBits(t, b, want)
		})
	}
}

func TestFindBitRange64(t *testing.T) {
	check := func(x uint64, n uint, result uint) {
		i := FindBitRange64(x, n)
		if result == ^uint(0) && i < 64 {
			t.Errorf("case (%016x, %d): got %d, want failure", x, n, i)
		} else if result != ^uint(0) && i != result {
			t.Errorf("case (%016x, %d): got %d, want %d", x, n, i, result)
		}
	}
	for i := uint(0); i <= 64; i++ {
		check(^uint64(0), i, 0)
	}
	check(0, 0, 0)
	for i := uint(1); i <= 64; i++ {
		check(0, i, ^uint(0))
	}
	check(0x8000000000000000, 1, 63)
	check(0xc000010001010000, 2, 62)
	check(0xc000010001030000, 2, 16)
	check(0xe000030001030000, 3, 61)
	check(0xe000030001070000, 3, 16)
	check(0xffff03ff01070000, 16, 48)
	check(0xffff03ff0107ffff, 16, 0)
	check(0x0fff03ff01079fff, 16, ^uint(0))
}
