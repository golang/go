// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"math/rand"
	. "runtime"
	"testing"
)

// Ensures that got and want are the same, and if not, reports
// detailed diff information.
func checkPallocBits(t *testing.T, got, want *PallocBits) bool {
	d := DiffPallocBits(got, want)
	if len(d) != 0 {
		t.Errorf("%d range(s) different", len(d))
		for _, bits := range d {
			t.Logf("\t@ bit index %d", bits.I)
			t.Logf("\t|  got: %s", StringifyPallocBits(got, bits))
			t.Logf("\t| want: %s", StringifyPallocBits(want, bits))
		}
		return false
	}
	return true
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
	if PallocChunkPages >= 512 {
		t.Run("Inner", func(t *testing.T) {
			want := new(PallocBits)
			want[:][2] = 0x3e
			test(t, 129, 5, want)
		})
		t.Run("Aligned", func(t *testing.T) {
			want := new(PallocBits)
			want[:][2] = ^uint64(0)
			want[:][3] = ^uint64(0)
			test(t, 128, 128, want)
		})
		t.Run("Begin", func(t *testing.T) {
			want := new(PallocBits)
			want[:][0] = ^uint64(0)
			want[:][1] = ^uint64(0)
			want[:][2] = ^uint64(0)
			want[:][3] = ^uint64(0)
			want[:][4] = ^uint64(0)
			want[:][5] = 0x1
			test(t, 0, 321, want)
		})
		t.Run("End", func(t *testing.T) {
			// avoid constant overflow when PallocChunkPages is small
			var PallocChunkPages uint = PallocChunkPages
			want := new(PallocBits)
			want[PallocChunkPages/64-1] = ^uint64(0)
			want[PallocChunkPages/64-2] = ^uint64(0)
			want[PallocChunkPages/64-3] = ^uint64(0)
			want[PallocChunkPages/64-4] = 1 << 63
			test(t, PallocChunkPages-(64*3+1), 64*3+1, want)
		})
	}
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

// Ensures two packed summaries are identical, and reports a detailed description
// of the difference if they're not.
func checkPallocSum(t testing.TB, got, want PallocSum) {
	if got.Start() != want.Start() {
		t.Errorf("inconsistent start: got %d, want %d", got.Start(), want.Start())
	}
	if got.Max() != want.Max() {
		t.Errorf("inconsistent max: got %d, want %d", got.Max(), want.Max())
	}
	if got.End() != want.End() {
		t.Errorf("inconsistent end: got %d, want %d", got.End(), want.End())
	}
}

func TestMallocBitsPopcntRange(t *testing.T) {
	type test struct {
		i, n uint // bit range to popcnt over.
		want uint // expected popcnt result on that range.
	}
	type testCase struct {
		init  []BitRange // bit ranges to set to 1 in the bitmap.
		tests []test     // a set of popcnt tests to run over the bitmap.
	}
	tests := map[string]testCase{
		"None": {
			tests: []test{
				{0, 1, 0},
				{5, 3, 0},
				{2, 11, 0},
				{PallocChunkPages/4 + 1, PallocChunkPages / 2, 0},
				{0, PallocChunkPages, 0},
			},
		},
		"All": {
			init: []BitRange{{0, PallocChunkPages}},
			tests: []test{
				{0, 1, 1},
				{5, 3, 3},
				{2, 11, 11},
				{PallocChunkPages/4 + 1, PallocChunkPages / 2, PallocChunkPages / 2},
				{0, PallocChunkPages, PallocChunkPages},
			},
		},
		"Half": {
			init: []BitRange{{PallocChunkPages / 2, PallocChunkPages / 2}},
			tests: []test{
				{0, 1, 0},
				{5, 3, 0},
				{2, 11, 0},
				{PallocChunkPages/2 - 1, 1, 0},
				{PallocChunkPages / 2, 1, 1},
				{PallocChunkPages/2 + 10, 1, 1},
				{PallocChunkPages/2 - 1, 2, 1},
				{PallocChunkPages / 4, PallocChunkPages / 4, 0},
				{PallocChunkPages / 4, PallocChunkPages/4 + 1, 1},
				{PallocChunkPages/4 + 1, PallocChunkPages / 2, PallocChunkPages/4 + 1},
				{0, PallocChunkPages, PallocChunkPages / 2},
			},
		},
	}
	if PallocChunkPages >= 512 {
		tests["OddBound"] = testCase{
			init: []BitRange{{0, 111}},
			tests: []test{
				{0, 1, 1},
				{5, 3, 3},
				{2, 11, 11},
				{110, 2, 1},
				{99, 50, 12},
				{110, 1, 1},
				{111, 1, 0},
				{99, 1, 1},
				{120, 1, 0},
				{PallocChunkPages / 2, PallocChunkPages / 2, 0},
				{0, PallocChunkPages, 111},
			},
		}
		tests["Scattered"] = testCase{
			init: []BitRange{
				{1, 3}, {5, 1}, {7, 1}, {10, 2}, {13, 1}, {15, 4},
				{21, 1}, {23, 1}, {26, 2}, {30, 5}, {36, 2}, {40, 3},
				{44, 6}, {51, 1}, {53, 2}, {58, 3}, {63, 1}, {67, 2},
				{71, 10}, {84, 1}, {89, 7}, {99, 2}, {103, 1}, {107, 2},
				{111, 1}, {113, 1}, {115, 1}, {118, 1}, {120, 2}, {125, 5},
			},
			tests: []test{
				{0, 11, 6},
				{0, 64, 39},
				{13, 64, 40},
				{64, 64, 34},
				{0, 128, 73},
				{1, 128, 74},
				{0, PallocChunkPages, 75},
			},
		}
	}
	for name, v := range tests {
		v := v
		t.Run(name, func(t *testing.T) {
			b := makePallocBits(v.init)
			for _, h := range v.tests {
				if got := b.PopcntRange(h.i, h.n); got != h.want {
					t.Errorf("bad popcnt (i=%d, n=%d): got %d, want %d", h.i, h.n, got, h.want)
				}
			}
		})
	}
}

// Ensures computing bit summaries works as expected by generating random
// bitmaps and checking against a reference implementation.
func TestPallocBitsSummarizeRandom(t *testing.T) {
	b := new(PallocBits)
	for i := 0; i < 1000; i++ {
		// Randomize bitmap.
		for i := range b {
			b[i] = rand.Uint64()
		}
		// Check summary against reference implementation.
		checkPallocSum(t, b.Summarize(), SummarizeSlow(b))
	}
}

// Ensures computing bit summaries works as expected.
func TestPallocBitsSummarize(t *testing.T) {
	var emptySum = PackPallocSum(PallocChunkPages, PallocChunkPages, PallocChunkPages)
	type test struct {
		free []BitRange // Ranges of free (zero) bits.
		hits []PallocSum
	}
	tests := make(map[string]test)
	tests["NoneFree"] = test{
		free: []BitRange{},
		hits: []PallocSum{
			PackPallocSum(0, 0, 0),
		},
	}
	tests["OnlyStart"] = test{
		free: []BitRange{{0, 10}},
		hits: []PallocSum{
			PackPallocSum(10, 10, 0),
		},
	}
	tests["OnlyEnd"] = test{
		free: []BitRange{{PallocChunkPages - 40, 40}},
		hits: []PallocSum{
			PackPallocSum(0, 40, 40),
		},
	}
	tests["StartAndEnd"] = test{
		free: []BitRange{{0, 11}, {PallocChunkPages - 23, 23}},
		hits: []PallocSum{
			PackPallocSum(11, 23, 23),
		},
	}
	if PallocChunkPages >= 512 {
		tests["StartMaxEnd"] = test{
			free: []BitRange{{0, 4}, {50, 100}, {PallocChunkPages - 4, 4}},
			hits: []PallocSum{
				PackPallocSum(4, 100, 4),
			},
		}
		tests["OnlyMax"] = test{
			free: []BitRange{{1, 20}, {35, 241}, {PallocChunkPages - 50, 30}},
			hits: []PallocSum{
				PackPallocSum(0, 241, 0),
			},
		}
		tests["MultiMax"] = test{
			free: []BitRange{{35, 2}, {40, 5}, {100, 5}},
			hits: []PallocSum{
				PackPallocSum(0, 5, 0),
			},
		}
	}
	tests["One"] = test{
		free: []BitRange{{2, 1}},
		hits: []PallocSum{
			PackPallocSum(0, 1, 0),
		},
	}
	tests["AllFree"] = test{
		free: []BitRange{{0, PallocChunkPages}},
		hits: []PallocSum{
			emptySum,
		},
	}
	for name, v := range tests {
		v := v
		t.Run(name, func(t *testing.T) {
			b := makePallocBits(v.free)
			// In the PallocBits we create 1's represent free spots, but in our actual
			// PallocBits 1 means not free, so invert.
			invertPallocBits(b)
			for _, h := range v.hits {
				checkPallocSum(t, b.Summarize(), h)
			}
		})
	}
}

// Benchmarks how quickly we can summarize a PallocBits.
func BenchmarkPallocBitsSummarize(b *testing.B) {
	patterns := []uint64{
		0,
		^uint64(0),
		0xaa,
		0xaaaaaaaaaaaaaaaa,
		0x80000000aaaaaaaa,
		0xaaaaaaaa00000001,
		0xbbbbbbbbbbbbbbbb,
		0x80000000bbbbbbbb,
		0xbbbbbbbb00000001,
		0xcccccccccccccccc,
		0x4444444444444444,
		0x4040404040404040,
		0x4000400040004000,
		0x1000404044ccaaff,
	}
	for _, p := range patterns {
		buf := new(PallocBits)
		for i := 0; i < len(buf); i++ {
			buf[i] = p
		}
		b.Run(fmt.Sprintf("Unpacked%02X", p), func(b *testing.B) {
			checkPallocSum(b, buf.Summarize(), SummarizeSlow(buf))
			for i := 0; i < b.N; i++ {
				buf.Summarize()
			}
		})
	}
}

// Ensures page allocation works.
func TestPallocBitsAlloc(t *testing.T) {
	type test struct {
		before []BitRange
		after  []BitRange
		npages uintptr
		hits   []uint
	}
	tests := map[string]test{
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
	}
	if PallocChunkPages >= 512 {
		// avoid constant overflow when PallocChunkPages is small
		var PallocChunkPages uint = PallocChunkPages
		tests["AllFree64"] = test{
			npages: 64,
			hits:   []uint{0, 64, 128},
			after:  []BitRange{{0, 192}},
		}
		tests["AllFree65"] = test{
			npages: 65,
			hits:   []uint{0, 65, 130},
			after:  []BitRange{{0, 195}},
		}
		tests["SomeFree64"] = test{
			before: []BitRange{{0, 32}, {64, 32}, {100, PallocChunkPages - 100}},
			npages: 64,
			hits:   []uint{^uint(0)},
			after:  []BitRange{{0, 32}, {64, 32}, {100, PallocChunkPages - 100}},
		}
		tests["ExactFit65"] = test{
			before: []BitRange{{0, PallocChunkPages/2 - 31}, {PallocChunkPages/2 + 34, PallocChunkPages/2 - 34}},
			npages: 65,
			hits:   []uint{PallocChunkPages/2 - 31, ^uint(0)},
			after:  []BitRange{{0, PallocChunkPages}},
		}
		tests["SomeFree161"] = test{
			before: []BitRange{{0, 185}, {331, 1}},
			npages: 161,
			hits:   []uint{332},
			after:  []BitRange{{0, 185}, {331, 162}},
		}
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
	type test struct {
		beforeInv []BitRange
		afterInv  []BitRange
		frees     []uint
		npages    uintptr
	}
	tests := map[string]test{
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
	}
	if PallocChunkPages >= 512 {
		tests["SomeFree"] = test{
			npages:    1,
			beforeInv: []BitRange{{0, 32}, {64, 32}, {100, 1}},
			frees:     []uint{32},
			afterInv:  []BitRange{{0, 33}, {64, 32}, {100, 1}},
		}
		tests["NoneFree64"] = test{
			npages:   64,
			frees:    []uint{0, 64, 128},
			afterInv: []BitRange{{0, 192}},
		}
		tests["NoneFree65"] = test{
			npages:   65,
			frees:    []uint{0, 65, 130},
			afterInv: []BitRange{{0, 195}},
		}
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
	for i := uint(1); i <= 64; i++ {
		check(^uint64(0), i, 0)
	}
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

func BenchmarkFindBitRange64(b *testing.B) {
	patterns := []uint64{
		0,
		^uint64(0),
		0xaa,
		0xaaaaaaaaaaaaaaaa,
		0x80000000aaaaaaaa,
		0xaaaaaaaa00000001,
		0xbbbbbbbbbbbbbbbb,
		0x80000000bbbbbbbb,
		0xbbbbbbbb00000001,
		0xcccccccccccccccc,
		0x4444444444444444,
		0x4040404040404040,
		0x4000400040004000,
	}
	sizes := []uint{
		2, 8, 32,
	}
	for _, pattern := range patterns {
		for _, size := range sizes {
			b.Run(fmt.Sprintf("Pattern%02XSize%d", pattern, size), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					FindBitRange64(pattern, size)
				}
			})
		}
	}
}
