// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"internal/goos"
	"math"
	"math/rand"
	. "runtime"
	"runtime/internal/atomic"
	"testing"
	"time"
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
		if bits < PallocChunkPages {
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
		} else if bits == PallocChunkPages {
			tests["PreserveHugePageAll"] = test{
				min:  1,
				max:  1, // Even one page would break a huge page in this case.
				want: BitRange{0, PallocChunkPages},
			}
		} else {
			// The huge page size is greater than pallocChunkPages, so it should
			// be effectively disabled. There's no way we can possible scavenge
			// a huge page out of this bitmap chunk.
			tests["PreserveHugePageNone"] = test{
				min:  1,
				max:  1,
				want: BitRange{PallocChunkPages - 1, 1},
			}
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
	// Disable these tests on iOS since we have a small address space.
	// See #46860.
	if PageAlloc64Bit != 0 && goos.IsIos == 0 {
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
		t.Run(name, func(t *testing.T) {
			b := NewPageAlloc(v.beforeAlloc, v.beforeScav)
			defer FreePageAlloc(b)

			for iter, h := range v.expect {
				if got := b.Scavenge(h.request); got != h.expect {
					t.Fatalf("bad scavenge #%d: want %d, got %d", iter+1, h.expect, got)
				}
			}
			want := NewPageAlloc(v.beforeAlloc, v.afterScav)
			defer FreePageAlloc(want)

			checkPageAlloc(t, want, b)
		})
	}
}

func TestScavenger(t *testing.T) {
	// workedTime is a standard conversion of bytes of scavenge
	// work to time elapsed.
	workedTime := func(bytes uintptr) int64 {
		return int64((bytes+4095)/4096) * int64(10*time.Microsecond)
	}

	// Set up a bunch of state that we're going to track and verify
	// throughout the test.
	totalWork := uint64(64<<20 - 3*PhysPageSize)
	var totalSlept, totalWorked atomic.Int64
	var availableWork atomic.Uint64
	var stopAt atomic.Uint64 // How much available work to stop at.

	// Set up the scavenger.
	var s Scavenger
	s.Sleep = func(ns int64) int64 {
		totalSlept.Add(ns)
		return ns
	}
	s.Scavenge = func(bytes uintptr) (uintptr, int64) {
		avail := availableWork.Load()
		if uint64(bytes) > avail {
			bytes = uintptr(avail)
		}
		t := workedTime(bytes)
		if bytes != 0 {
			availableWork.Add(-int64(bytes))
			totalWorked.Add(t)
		}
		return bytes, t
	}
	s.ShouldStop = func() bool {
		if availableWork.Load() <= stopAt.Load() {
			return true
		}
		return false
	}
	s.GoMaxProcs = func() int32 {
		return 1
	}

	// Define a helper for verifying that various properties hold.
	verifyScavengerState := func(t *testing.T, expWork uint64) {
		t.Helper()

		// Check to make sure it did the amount of work we expected.
		if workDone := uint64(s.Released()); workDone != expWork {
			t.Errorf("want %d bytes of work done, got %d", expWork, workDone)
		}
		// Check to make sure the scavenger is meeting its CPU target.
		idealFraction := float64(ScavengePercent) / 100.0
		cpuFraction := float64(totalWorked.Load()) / float64(totalWorked.Load()+totalSlept.Load())
		if cpuFraction < idealFraction-0.005 || cpuFraction > idealFraction+0.005 {
			t.Errorf("want %f CPU fraction, got %f", idealFraction, cpuFraction)
		}
	}

	// Start the scavenger.
	s.Start()

	// Set up some work and let the scavenger run to completion.
	availableWork.Store(totalWork)
	s.Wake()
	if !s.BlockUntilParked(2e9 /* 2 seconds */) {
		t.Fatal("timed out waiting for scavenger to run to completion")
	}
	// Run a check.
	verifyScavengerState(t, totalWork)

	// Now let's do it again and see what happens when we have no work to do.
	// It should've gone right back to sleep.
	s.Wake()
	if !s.BlockUntilParked(2e9 /* 2 seconds */) {
		t.Fatal("timed out waiting for scavenger to run to completion")
	}
	// Run another check.
	verifyScavengerState(t, totalWork)

	// One more time, this time doing the same amount of work as the first time.
	// Let's see if we can get the scavenger to continue.
	availableWork.Store(totalWork)
	s.Wake()
	if !s.BlockUntilParked(2e9 /* 2 seconds */) {
		t.Fatal("timed out waiting for scavenger to run to completion")
	}
	// Run another check.
	verifyScavengerState(t, 2*totalWork)

	// This time, let's stop after a certain amount of work.
	//
	// Pick a stopping point such that when subtracted from totalWork
	// we get a multiple of a relatively large power of 2. verifyScavengerState
	// always makes an exact check, but the scavenger might go a little over,
	// which is OK. If this breaks often or gets annoying to maintain, modify
	// verifyScavengerState.
	availableWork.Store(totalWork)
	stoppingPoint := uint64(1<<20 - 3*PhysPageSize)
	stopAt.Store(stoppingPoint)
	s.Wake()
	if !s.BlockUntilParked(2e9 /* 2 seconds */) {
		t.Fatal("timed out waiting for scavenger to run to completion")
	}
	// Run another check.
	verifyScavengerState(t, 2*totalWork+(totalWork-stoppingPoint))

	// Clean up.
	s.Stop()
}

func TestScavengeIndex(t *testing.T) {
	// This test suite tests the scavengeIndex data structure.

	// markFunc is a function that makes the address range [base, limit)
	// available for scavenging in a test index.
	type markFunc func(base, limit uintptr)

	// findFunc is a function that searches for the next available page
	// to scavenge in the index. It asserts that the page is found in
	// chunk "ci" at page "offset."
	type findFunc func(ci ChunkIdx, offset uint)

	// The structure of the tests below is as follows:
	//
	// setup creates a fake scavengeIndex that can be mutated and queried by
	// the functions it returns. Those functions capture the testing.T that
	// setup is called with, so they're bound to the subtest they're created in.
	//
	// Tests are then organized into test cases which mark some pages as
	// scavenge-able then try to find them. Tests expect that the initial
	// state of the scavengeIndex has all of the chunks as dense in the last
	// generation and empty to the scavenger.
	//
	// There are a few additional tests that interleave mark and find operations,
	// so they're defined separately, but use the same infrastructure.
	setup := func(t *testing.T, force bool) (mark markFunc, find findFunc, nextGen func()) {
		t.Helper()

		// Pick some reasonable bounds. We don't need a huge range just to test.
		si := NewScavengeIndex(BaseChunkIdx, BaseChunkIdx+64)

		// Initialize all the chunks as dense and empty.
		//
		// Also, reset search addresses so that we can get page offsets.
		si.AllocRange(PageBase(BaseChunkIdx, 0), PageBase(BaseChunkIdx+64, 0))
		si.NextGen()
		si.FreeRange(PageBase(BaseChunkIdx, 0), PageBase(BaseChunkIdx+64, 0))
		for ci := BaseChunkIdx; ci < BaseChunkIdx+64; ci++ {
			si.SetEmpty(ci)
		}
		si.ResetSearchAddrs()

		// Create and return test functions.
		mark = func(base, limit uintptr) {
			t.Helper()

			si.AllocRange(base, limit)
			si.FreeRange(base, limit)
		}
		find = func(want ChunkIdx, wantOffset uint) {
			t.Helper()

			got, gotOffset := si.Find(force)
			if want != got {
				t.Errorf("find: wanted chunk index %d, got %d", want, got)
			}
			if wantOffset != gotOffset {
				t.Errorf("find: wanted page offset %d, got %d", wantOffset, gotOffset)
			}
			if t.Failed() {
				t.FailNow()
			}
			si.SetEmpty(got)
		}
		nextGen = func() {
			t.Helper()

			si.NextGen()
		}
		return
	}

	// Each of these test cases calls mark and then find once.
	type testCase struct {
		name string
		mark func(markFunc)
		find func(findFunc)
	}
	for _, test := range []testCase{
		{
			name: "Uninitialized",
			mark: func(_ markFunc) {},
			find: func(_ findFunc) {},
		},
		{
			name: "OnePage",
			mark: func(mark markFunc) {
				mark(PageBase(BaseChunkIdx, 3), PageBase(BaseChunkIdx, 4))
			},
			find: func(find findFunc) {
				find(BaseChunkIdx, 3)
			},
		},
		{
			name: "FirstPage",
			mark: func(mark markFunc) {
				mark(PageBase(BaseChunkIdx, 0), PageBase(BaseChunkIdx, 1))
			},
			find: func(find findFunc) {
				find(BaseChunkIdx, 0)
			},
		},
		{
			name: "SeveralPages",
			mark: func(mark markFunc) {
				mark(PageBase(BaseChunkIdx, 9), PageBase(BaseChunkIdx, 14))
			},
			find: func(find findFunc) {
				find(BaseChunkIdx, 13)
			},
		},
		{
			name: "WholeChunk",
			mark: func(mark markFunc) {
				mark(PageBase(BaseChunkIdx, 0), PageBase(BaseChunkIdx+1, 0))
			},
			find: func(find findFunc) {
				find(BaseChunkIdx, PallocChunkPages-1)
			},
		},
		{
			name: "LastPage",
			mark: func(mark markFunc) {
				mark(PageBase(BaseChunkIdx, PallocChunkPages-1), PageBase(BaseChunkIdx+1, 0))
			},
			find: func(find findFunc) {
				find(BaseChunkIdx, PallocChunkPages-1)
			},
		},
		{
			name: "TwoChunks",
			mark: func(mark markFunc) {
				mark(PageBase(BaseChunkIdx, 128), PageBase(BaseChunkIdx+1, 128))
			},
			find: func(find findFunc) {
				find(BaseChunkIdx+1, 127)
				find(BaseChunkIdx, PallocChunkPages-1)
			},
		},
		{
			name: "TwoChunksOffset",
			mark: func(mark markFunc) {
				mark(PageBase(BaseChunkIdx+7, 128), PageBase(BaseChunkIdx+8, 129))
			},
			find: func(find findFunc) {
				find(BaseChunkIdx+8, 128)
				find(BaseChunkIdx+7, PallocChunkPages-1)
			},
		},
		{
			name: "SevenChunksOffset",
			mark: func(mark markFunc) {
				mark(PageBase(BaseChunkIdx+6, 11), PageBase(BaseChunkIdx+13, 15))
			},
			find: func(find findFunc) {
				find(BaseChunkIdx+13, 14)
				for i := BaseChunkIdx + 12; i >= BaseChunkIdx+6; i-- {
					find(i, PallocChunkPages-1)
				}
			},
		},
		{
			name: "ThirtyTwoChunks",
			mark: func(mark markFunc) {
				mark(PageBase(BaseChunkIdx, 0), PageBase(BaseChunkIdx+32, 0))
			},
			find: func(find findFunc) {
				for i := BaseChunkIdx + 31; i >= BaseChunkIdx; i-- {
					find(i, PallocChunkPages-1)
				}
			},
		},
		{
			name: "ThirtyTwoChunksOffset",
			mark: func(mark markFunc) {
				mark(PageBase(BaseChunkIdx+3, 0), PageBase(BaseChunkIdx+35, 0))
			},
			find: func(find findFunc) {
				for i := BaseChunkIdx + 34; i >= BaseChunkIdx+3; i-- {
					find(i, PallocChunkPages-1)
				}
			},
		},
		{
			name: "Mark",
			mark: func(mark markFunc) {
				for i := BaseChunkIdx; i < BaseChunkIdx+32; i++ {
					mark(PageBase(i, 0), PageBase(i+1, 0))
				}
			},
			find: func(find findFunc) {
				for i := BaseChunkIdx + 31; i >= BaseChunkIdx; i-- {
					find(i, PallocChunkPages-1)
				}
			},
		},
		{
			name: "MarkIdempotentOneChunk",
			mark: func(mark markFunc) {
				mark(PageBase(BaseChunkIdx, 0), PageBase(BaseChunkIdx+1, 0))
				mark(PageBase(BaseChunkIdx, 0), PageBase(BaseChunkIdx+1, 0))
			},
			find: func(find findFunc) {
				find(BaseChunkIdx, PallocChunkPages-1)
			},
		},
		{
			name: "MarkIdempotentThirtyTwoChunks",
			mark: func(mark markFunc) {
				mark(PageBase(BaseChunkIdx, 0), PageBase(BaseChunkIdx+32, 0))
				mark(PageBase(BaseChunkIdx, 0), PageBase(BaseChunkIdx+32, 0))
			},
			find: func(find findFunc) {
				for i := BaseChunkIdx + 31; i >= BaseChunkIdx; i-- {
					find(i, PallocChunkPages-1)
				}
			},
		},
		{
			name: "MarkIdempotentThirtyTwoChunksOffset",
			mark: func(mark markFunc) {
				mark(PageBase(BaseChunkIdx+4, 0), PageBase(BaseChunkIdx+31, 0))
				mark(PageBase(BaseChunkIdx+5, 0), PageBase(BaseChunkIdx+36, 0))
			},
			find: func(find findFunc) {
				for i := BaseChunkIdx + 35; i >= BaseChunkIdx+4; i-- {
					find(i, PallocChunkPages-1)
				}
			},
		},
	} {
		test := test
		t.Run("Bg/"+test.name, func(t *testing.T) {
			mark, find, nextGen := setup(t, false)
			test.mark(mark)
			find(0, 0)      // Make sure we find nothing at this point.
			nextGen()       // Move to the next generation.
			test.find(find) // Now we should be able to find things.
			find(0, 0)      // The test should always fully exhaust the index.
		})
		t.Run("Force/"+test.name, func(t *testing.T) {
			mark, find, _ := setup(t, true)
			test.mark(mark)
			test.find(find) // Finding should always work when forced.
			find(0, 0)      // The test should always fully exhaust the index.
		})
	}
	t.Run("Bg/MarkInterleaved", func(t *testing.T) {
		mark, find, nextGen := setup(t, false)
		for i := BaseChunkIdx; i < BaseChunkIdx+32; i++ {
			mark(PageBase(i, 0), PageBase(i+1, 0))
			nextGen()
			find(i, PallocChunkPages-1)
		}
		find(0, 0)
	})
	t.Run("Force/MarkInterleaved", func(t *testing.T) {
		mark, find, _ := setup(t, true)
		for i := BaseChunkIdx; i < BaseChunkIdx+32; i++ {
			mark(PageBase(i, 0), PageBase(i+1, 0))
			find(i, PallocChunkPages-1)
		}
		find(0, 0)
	})
}

func TestScavChunkDataPack(t *testing.T) {
	if !CheckPackScavChunkData(1918237402, 512, 512, 0b11) {
		t.Error("failed pack/unpack check for scavChunkData 1")
	}
	if !CheckPackScavChunkData(^uint32(0), 12, 0, 0b00) {
		t.Error("failed pack/unpack check for scavChunkData 2")
	}
}

func FuzzPIController(f *testing.F) {
	isNormal := func(x float64) bool {
		return !math.IsInf(x, 0) && !math.IsNaN(x)
	}
	isPositive := func(x float64) bool {
		return isNormal(x) && x > 0
	}
	// Seed with constants from controllers in the runtime.
	// It's not critical that we keep these in sync, they're just
	// reasonable seed inputs.
	f.Add(0.3375, 3.2e6, 1e9, 0.001, 1000.0, 0.01)
	f.Add(0.9, 4.0, 1000.0, -1000.0, 1000.0, 0.84)
	f.Fuzz(func(t *testing.T, kp, ti, tt, min, max, setPoint float64) {
		// Ignore uninteresting invalid parameters. These parameters
		// are constant, so in practice surprising values will be documented
		// or will be other otherwise immediately visible.
		//
		// We just want to make sure that given a non-Inf, non-NaN input,
		// we always get a non-Inf, non-NaN output.
		if !isPositive(kp) || !isPositive(ti) || !isPositive(tt) {
			return
		}
		if !isNormal(min) || !isNormal(max) || min > max {
			return
		}
		// Use a random source, but make it deterministic.
		rs := rand.New(rand.NewSource(800))
		randFloat64 := func() float64 {
			return math.Float64frombits(rs.Uint64())
		}
		p := NewPIController(kp, ti, tt, min, max)
		state := float64(0)
		for i := 0; i < 100; i++ {
			input := randFloat64()
			// Ignore the "ok" parameter. We're just trying to break it.
			// state is intentionally completely uncorrelated with the input.
			var ok bool
			state, ok = p.Next(input, setPoint, 1.0)
			if !isNormal(state) {
				t.Fatalf("got NaN or Inf result from controller: %f %v", state, ok)
			}
		}
	})
}
