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

// makePallocData produces an initialized PallocData by setting
// the ranges of described in alloc and scavenge.
func makePallocData(alloc, scavenged []BitRange) *PallocData {
	b := new(PallocData)
	for _, v := range alloc {
		if v.N == 0 {
			// Skip N==0. It's harmless and allocRange doesn't
			// handle this case.
			continue
		}
		b.AllocRange(v.I, v.N)
	}
	for _, v := range scavenged {
		if v.N == 0 {
			// See the previous loop.
			continue
		}
		b.ScavengedSetRange(v.I, v.N)
	}
	return b
}

func TestFillAligned(t *testing.T) {
	fillAlignedSlow := func(x uint64, m uint) uint64 {
		if m == 1 {
			return x
		}
		out := uint64(0)
		for i := uint(0); i < 64; i += m {
			for j := uint(0); j < m; j++ {
				if x&(uint64(1)<<(i+j)) != 0 {
					out |= ((uint64(1) << m) - 1) << i
					break
				}
			}
		}
		return out
	}
	check := func(x uint64, m uint) {
		want := fillAlignedSlow(x, m)
		if got := FillAligned(x, m); got != want {
			t.Logf("got:  %064b", got)
			t.Logf("want: %064b", want)
			t.Errorf("bad fillAligned(%016x, %d)", x, m)
		}
	}
	for m := uint(1); m <= 64; m *= 2 {
		tests := []uint64{
			0x0000000000000000,
			0x00000000ffffffff,
			0xffffffff00000000,
			0x8000000000000001,
			0xf00000000000000f,
			0xf00000010050000f,
			0xffffffffffffffff,
			0x0000000000000001,
			0x0000000000000002,
			0x0000000000000008,
			uint64(1) << (m - 1),
			uint64(1) << m,
			// Try a few fixed arbitrary examples.
			0xb02b9effcf137016,
			0x3975a076a9fbff18,
			0x0f8c88ec3b81506e,
			0x60f14d80ef2fa0e6,
		}
		for _, test := range tests {
			check(test, m)
		}
		for i := 0; i < 1000; i++ {
			// Try a pseudo-random numbers.
			check(rand.Uint64(), m)

			if m > 1 {
				// For m != 1, let's construct a slightly more interesting
				// random test. Generate a bitmap which is either 0 or
				// randomly set bits for each m-aligned group of m bits.
				val := uint64(0)
				for n := uint(0); n < 64; n += m {
					// For each group of m bits, flip a coin:
					// * Leave them as zero.
					// * Set them randomly.
					if rand.Uint64()%2 == 0 {
						val |= (rand.Uint64() & ((1 << m) - 1)) << n
					}
				}
				check(val, m)
			}
		}
	}
}

func TestPallocDataFindScavengeCandidate(t *testing.T) {
	type test struct {
		alloc, scavenged []BitRange
		min, max         uintptr
		want             BitRange
	}
	tests := map[string]test{
		"MixedMin1": {
			alloc:     []BitRange{{0, 40}, {42, PallocChunkPages - 42}},
			scavenged: []BitRange{{0, 41}, {42, PallocChunkPages - 42}},
			min:       1,
			max:       PallocChunkPages,
			want:      BitRange{41, 1},
		},
		"MultiMin1": {
			alloc:     []BitRange{{0, 63}, {65, 20}, {87, PallocChunkPages - 87}},
			scavenged: []BitRange{{86, 1}},
			min:       1,
			max:       PallocChunkPages,
			want:      BitRange{85, 1},
		},
	}
	// Try out different page minimums.
	for m := uintptr(1); m <= 64; m *= 2 {
		suffix := fmt.Sprintf("Min%d", m)
		tests["AllFree"+suffix] = test{
			min:  m,
			max:  PallocChunkPages,
			want: BitRange{0, PallocChunkPages},
		}
		tests["AllScavenged"+suffix] = test{
			scavenged: []BitRange{{0, PallocChunkPages}},
			min:       m,
			max:       PallocChunkPages,
			want:      BitRange{0, 0},
		}
		tests["NoneFree"+suffix] = test{
			alloc:     []BitRange{{0, PallocChunkPages}},
			scavenged: []BitRange{{PallocChunkPages / 2, PallocChunkPages / 2}},
			min:       m,
			max:       PallocChunkPages,
			want:      BitRange{0, 0},
		}
		tests["StartFree"+suffix] = test{
			alloc: []BitRange{{uint(m), PallocChunkPages - uint(m)}},
			min:   m,
			max:   PallocChunkPages,
			want:  BitRange{0, uint(m)},
		}
		tests["StartFree"+suffix] = test{
			alloc: []BitRange{{uint(m), PallocChunkPages - uint(m)}},
			min:   m,
			max:   PallocChunkPages,
			want:  BitRange{0, uint(m)},
		}
		tests["EndFree"+suffix] = test{
			alloc: []BitRange{{0, PallocChunkPages - uint(m)}},
			min:   m,
			max:   PallocChunkPages,
			want:  BitRange{PallocChunkPages - uint(m), uint(m)},
		}
		tests["Straddle64"+suffix] = test{
			alloc: []BitRange{{0, 64 - uint(m)}, {64 + uint(m), PallocChunkPages - (64 + uint(m))}},
			min:   m,
			max:   2 * m,
			want:  BitRange{64 - uint(m), 2 * uint(m)},
		}
		tests["BottomEdge64WithFull"+suffix] = test{
			alloc:     []BitRange{{64, 64}, {128 + 3*uint(m), PallocChunkPages - (128 + 3*uint(m))}},
			scavenged: []BitRange{{1, 10}},
			min:       m,
			max:       3 * m,
			want:      BitRange{128, 3 * uint(m)},
		}
		tests["BottomEdge64WithPocket"+suffix] = test{
			alloc:     []BitRange{{64, 62}, {127, 1}, {128 + 3*uint(m), PallocChunkPages - (128 + 3*uint(m))}},
			scavenged: []BitRange{{1, 10}},
			min:       m,
			max:       3 * m,
			want:      BitRange{128, 3 * uint(m)},
		}
		tests["Max0"+suffix] = test{
			scavenged: []BitRange{{0, PallocChunkPages - uint(m)}},
			min:       m,
			max:       0,
			want:      BitRange{PallocChunkPages - uint(m), uint(m)},
		}
		if m <= 8 {
			tests["OneFree"] = test{
				alloc: []BitRange{{0, 40}, {40 + uint(m), PallocChunkPages - (40 + uint(m))}},
				min:   m,
				max:   PallocChunkPages,
				want:  BitRange{40, uint(m)},
			}
			tests["OneScavenged"] = test{
				alloc:     []BitRange{{0, 40}, {40 + uint(m), PallocChunkPages - (40 + uint(m))}},
				scavenged: []BitRange{{40, 1}},
				min:       m,
				max:       PallocChunkPages,
				want:      BitRange{0, 0},
			}
		}
		if m > 1 {
			tests["MaxUnaligned"+suffix] = test{
				scavenged: []BitRange{{0, PallocChunkPages - uint(m*2-1)}},
				min:       m,
				max:       m - 2,
				want:      BitRange{PallocChunkPages - uint(m), uint(m)},
			}
			tests["SkipSmall"+suffix] = test{
				alloc: []BitRange{{0, 64 - uint(m)}, {64, 5}, {70, 11}, {82, PallocChunkPages - 82}},
				min:   m,
				max:   m,
				want:  BitRange{64 - uint(m), uint(m)},
			}
			tests["SkipMisaligned"+suffix] = test{
				alloc: []BitRange{{0, 64 - uint(m)}, {64, 63}, {127 + uint(m), PallocChunkPages - (127 + uint(m))}},
				min:   m,
				max:   m,
				want:  BitRange{64 - uint(m), uint(m)},
			}
			tests["MaxLessThan"+suffix] = test{
				scavenged: []BitRange{{0, PallocChunkPages - uint(m)}},
				min:       m,
				max:       1,
				want:      BitRange{PallocChunkPages - uint(m), uint(m)},
			}
		}
	}
	if PhysHugePageSize > uintptr(PageSize) {
		// Check hugepage preserving behavior.
		bits := uint(PhysHugePageSize / uintptr(PageSize))
		tests["PreserveHugePageBottom"] = test{
			alloc: []BitRange{{bits + 2, PallocChunkPages - (bits + 2)}},
			min:   1,
			max:   3, // Make it so that max would have us try to break the huge page.
			want:  BitRange{0, bits + 2},
		}
		if 3*bits < PallocChunkPages {
			// We need at least 3 huge pages in a chunk for this test to make sense.
			tests["PreserveHugePageMiddle"] = test{
				alloc: []BitRange{{0, bits - 10}, {2*bits + 10, PallocChunkPages - (2*bits + 10)}},
				min:   1,
				max:   12, // Make it so that max would have us try to break the huge page.
				want:  BitRange{bits, bits + 10},
			}
		}
		tests["PreserveHugePageTop"] = test{
			alloc: []BitRange{{0, PallocChunkPages - bits}},
			min:   1,
			max:   1, // Even one page would break a huge page in this case.
			want:  BitRange{PallocChunkPages - bits, bits},
		}
	}
	for name, v := range tests {
		v := v
		t.Run(name, func(t *testing.T) {
			b := makePallocData(v.alloc, v.scavenged)
			start, size := b.FindScavengeCandidate(PallocChunkPages-1, v.min, v.max)
			got := BitRange{start, size}
			if !(got.N == 0 && v.want.N == 0) && got != v.want {
				t.Fatalf("candidate mismatch: got %v, want %v", got, v.want)
			}
		})
	}
}

// Tests end-to-end scavenging on a pageAlloc.
func TestPageAllocScavenge(t *testing.T) {
	if GOOS == "openbsd" && testing.Short() {
		t.Skip("skipping because virtual memory is limited; see #36210")
	}
	type test struct {
		request, expect uintptr
	}
	minPages := PhysPageSize / PageSize
	if minPages < 1 {
		minPages = 1
	}
	type setup struct {
		beforeAlloc map[ChunkIdx][]BitRange
		beforeScav  map[ChunkIdx][]BitRange
		expect      []test
		afterScav   map[ChunkIdx][]BitRange
	}
	tests := map[string]setup{
		"AllFreeUnscavExhaust": {
			beforeAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx:     {},
				BaseChunkIdx + 1: {},
				BaseChunkIdx + 2: {},
			},
			beforeScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx:     {},
				BaseChunkIdx + 1: {},
				BaseChunkIdx + 2: {},
			},
			expect: []test{
				{^uintptr(0), 3 * PallocChunkPages * PageSize},
			},
			afterScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx:     {{0, PallocChunkPages}},
				BaseChunkIdx + 1: {{0, PallocChunkPages}},
				BaseChunkIdx + 2: {{0, PallocChunkPages}},
			},
		},
		"NoneFreeUnscavExhaust": {
			beforeAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx:     {{0, PallocChunkPages}},
				BaseChunkIdx + 1: {},
				BaseChunkIdx + 2: {{0, PallocChunkPages}},
			},
			beforeScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx:     {},
				BaseChunkIdx + 1: {{0, PallocChunkPages}},
				BaseChunkIdx + 2: {},
			},
			expect: []test{
				{^uintptr(0), 0},
			},
			afterScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx:     {},
				BaseChunkIdx + 1: {{0, PallocChunkPages}},
				BaseChunkIdx + 2: {},
			},
		},
		"ScavHighestPageFirst": {
			beforeAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {},
			},
			beforeScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{uint(minPages), PallocChunkPages - uint(2*minPages)}},
			},
			expect: []test{
				{1, minPages * PageSize},
			},
			afterScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{uint(minPages), PallocChunkPages - uint(minPages)}},
			},
		},
		"ScavMultiple": {
			beforeAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {},
			},
			beforeScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{uint(minPages), PallocChunkPages - uint(2*minPages)}},
			},
			expect: []test{
				{minPages * PageSize, minPages * PageSize},
				{minPages * PageSize, minPages * PageSize},
			},
			afterScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{0, PallocChunkPages}},
			},
		},
		"ScavMultiple2": {
			beforeAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx:     {},
				BaseChunkIdx + 1: {},
			},
			beforeScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx:     {{uint(minPages), PallocChunkPages - uint(2*minPages)}},
				BaseChunkIdx + 1: {{0, PallocChunkPages - uint(2*minPages)}},
			},
			expect: []test{
				{2 * minPages * PageSize, 2 * minPages * PageSize},
				{minPages * PageSize, minPages * PageSize},
				{minPages * PageSize, minPages * PageSize},
			},
			afterScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx:     {{0, PallocChunkPages}},
				BaseChunkIdx + 1: {{0, PallocChunkPages}},
			},
		},
		"ScavDiscontiguous": {
			beforeAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx:       {},
				BaseChunkIdx + 0xe: {},
			},
			beforeScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx:       {{uint(minPages), PallocChunkPages - uint(2*minPages)}},
				BaseChunkIdx + 0xe: {{uint(2 * minPages), PallocChunkPages - uint(2*minPages)}},
			},
			expect: []test{
				{2 * minPages * PageSize, 2 * minPages * PageSize},
				{^uintptr(0), 2 * minPages * PageSize},
				{^uintptr(0), 0},
			},
			afterScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx:       {{0, PallocChunkPages}},
				BaseChunkIdx + 0xe: {{0, PallocChunkPages}},
			},
		},
	}
	if PageAlloc64Bit != 0 {
		tests["ScavAllVeryDiscontiguous"] = setup{
			beforeAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx:          {},
				BaseChunkIdx + 0x1000: {},
			},
			beforeScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx:          {},
				BaseChunkIdx + 0x1000: {},
			},
			expect: []test{
				{^uintptr(0), 2 * PallocChunkPages * PageSize},
				{^uintptr(0), 0},
			},
			afterScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx:          {{0, PallocChunkPages}},
				BaseChunkIdx + 0x1000: {{0, PallocChunkPages}},
			},
		}
	}
	for name, v := range tests {
		v := v
		runTest := func(t *testing.T, locked bool) {
			b := NewPageAlloc(v.beforeAlloc, v.beforeScav)
			defer FreePageAlloc(b)

			for iter, h := range v.expect {
				if got := b.Scavenge(h.request, locked); got != h.expect {
					t.Fatalf("bad scavenge #%d: want %d, got %d", iter+1, h.expect, got)
				}
			}
			want := NewPageAlloc(v.beforeAlloc, v.afterScav)
			defer FreePageAlloc(want)

			checkPageAlloc(t, want, b)
		}
		t.Run(name, func(t *testing.T) {
			runTest(t, false)
		})
		t.Run(name+"Locked", func(t *testing.T) {
			runTest(t, true)
		})
	}
}
