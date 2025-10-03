// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/goos"
	"math/rand"
	. "runtime"
	"testing"
)

func checkPageCache(t *testing.T, got, want PageCache) {
	if got.Base() != want.Base() {
		t.Errorf("bad pageCache base: got 0x%x, want 0x%x", got.Base(), want.Base())
	}
	if got.Cache() != want.Cache() {
		t.Errorf("bad pageCache bits: got %016x, want %016x", got.Base(), want.Base())
	}
	if got.Scav() != want.Scav() {
		t.Errorf("bad pageCache scav: got %016x, want %016x", got.Scav(), want.Scav())
	}
}

func TestPageCacheAlloc(t *testing.T) {
	base := PageBase(BaseChunkIdx, 0)
	type hit struct {
		npages uintptr
		base   uintptr
		scav   uintptr
	}
	tests := map[string]struct {
		cache PageCache
		hits  []hit
	}{
		"Empty": {
			cache: NewPageCache(base, 0, 0),
			hits: []hit{
				{1, 0, 0},
				{2, 0, 0},
				{3, 0, 0},
				{4, 0, 0},
				{5, 0, 0},
				{11, 0, 0},
				{12, 0, 0},
				{16, 0, 0},
				{27, 0, 0},
				{32, 0, 0},
				{43, 0, 0},
				{57, 0, 0},
				{64, 0, 0},
				{121, 0, 0},
			},
		},
		"Lo1": {
			cache: NewPageCache(base, 0x1, 0x1),
			hits: []hit{
				{1, base, PageSize},
				{1, 0, 0},
				{10, 0, 0},
			},
		},
		"Hi1": {
			cache: NewPageCache(base, 0x1<<63, 0x1),
			hits: []hit{
				{1, base + 63*PageSize, 0},
				{1, 0, 0},
				{10, 0, 0},
			},
		},
		"Swiss1": {
			cache: NewPageCache(base, 0x20005555, 0x5505),
			hits: []hit{
				{2, 0, 0},
				{1, base, PageSize},
				{1, base + 2*PageSize, PageSize},
				{1, base + 4*PageSize, 0},
				{1, base + 6*PageSize, 0},
				{1, base + 8*PageSize, PageSize},
				{1, base + 10*PageSize, PageSize},
				{1, base + 12*PageSize, PageSize},
				{1, base + 14*PageSize, PageSize},
				{1, base + 29*PageSize, 0},
				{1, 0, 0},
				{10, 0, 0},
			},
		},
		"Lo2": {
			cache: NewPageCache(base, 0x3, 0x2<<62),
			hits: []hit{
				{2, base, 0},
				{2, 0, 0},
				{1, 0, 0},
			},
		},
		"Hi2": {
			cache: NewPageCache(base, 0x3<<62, 0x3<<62),
			hits: []hit{
				{2, base + 62*PageSize, 2 * PageSize},
				{2, 0, 0},
				{1, 0, 0},
			},
		},
		"Swiss2": {
			cache: NewPageCache(base, 0x3333<<31, 0x3030<<31),
			hits: []hit{
				{2, base + 31*PageSize, 0},
				{2, base + 35*PageSize, 2 * PageSize},
				{2, base + 39*PageSize, 0},
				{2, base + 43*PageSize, 2 * PageSize},
				{2, 0, 0},
			},
		},
		"Hi53": {
			cache: NewPageCache(base, ((uint64(1)<<53)-1)<<10, ((uint64(1)<<16)-1)<<10),
			hits: []hit{
				{53, base + 10*PageSize, 16 * PageSize},
				{53, 0, 0},
				{1, 0, 0},
			},
		},
		"Full53": {
			cache: NewPageCache(base, ^uint64(0), ((uint64(1)<<16)-1)<<10),
			hits: []hit{
				{53, base, 16 * PageSize},
				{53, 0, 0},
				{1, base + 53*PageSize, 0},
			},
		},
		"Full64": {
			cache: NewPageCache(base, ^uint64(0), ^uint64(0)),
			hits: []hit{
				{64, base, 64 * PageSize},
				{64, 0, 0},
				{1, 0, 0},
			},
		},
		"FullMixed": {
			cache: NewPageCache(base, ^uint64(0), ^uint64(0)),
			hits: []hit{
				{5, base, 5 * PageSize},
				{7, base + 5*PageSize, 7 * PageSize},
				{1, base + 12*PageSize, 1 * PageSize},
				{23, base + 13*PageSize, 23 * PageSize},
				{63, 0, 0},
				{3, base + 36*PageSize, 3 * PageSize},
				{3, base + 39*PageSize, 3 * PageSize},
				{3, base + 42*PageSize, 3 * PageSize},
				{12, base + 45*PageSize, 12 * PageSize},
				{11, 0, 0},
				{4, base + 57*PageSize, 4 * PageSize},
				{4, 0, 0},
				{6, 0, 0},
				{36, 0, 0},
				{2, base + 61*PageSize, 2 * PageSize},
				{3, 0, 0},
				{1, base + 63*PageSize, 1 * PageSize},
				{4, 0, 0},
				{2, 0, 0},
				{62, 0, 0},
				{1, 0, 0},
			},
		},
	}
	for name, test := range tests {
		test := test
		t.Run(name, func(t *testing.T) {
			c := test.cache
			for i, h := range test.hits {
				b, s := c.Alloc(h.npages)
				if b != h.base {
					t.Fatalf("bad alloc base #%d: got 0x%x, want 0x%x", i, b, h.base)
				}
				if s != h.scav {
					t.Fatalf("bad alloc scav #%d: got %d, want %d", i, s, h.scav)
				}
			}
		})
	}
}

func TestPageCacheFlush(t *testing.T) {
	if GOOS == "openbsd" && testing.Short() {
		t.Skip("skipping because virtual memory is limited; see #36210")
	}
	bits64ToBitRanges := func(bits uint64, base uint) []BitRange {
		var ranges []BitRange
		start, size := uint(0), uint(0)
		for i := 0; i < 64; i++ {
			if bits&(1<<i) != 0 {
				if size == 0 {
					start = uint(i) + base
				}
				size++
			} else {
				if size != 0 {
					ranges = append(ranges, BitRange{start, size})
					size = 0
				}
			}
		}
		if size != 0 {
			ranges = append(ranges, BitRange{start, size})
		}
		return ranges
	}
	runTest := func(t *testing.T, base uint, cache, scav uint64) {
		// Set up the before state.
		beforeAlloc := map[ChunkIdx][]BitRange{
			BaseChunkIdx: {{base, 64}},
		}
		beforeScav := map[ChunkIdx][]BitRange{
			BaseChunkIdx: {},
		}
		b := NewPageAlloc(beforeAlloc, beforeScav)
		defer FreePageAlloc(b)

		// Create and flush the cache.
		c := NewPageCache(PageBase(BaseChunkIdx, base), cache, scav)
		c.Flush(b)
		if !c.Empty() {
			t.Errorf("pageCache flush did not clear cache")
		}

		// Set up the expected after state.
		afterAlloc := map[ChunkIdx][]BitRange{
			BaseChunkIdx: bits64ToBitRanges(^cache, base),
		}
		afterScav := map[ChunkIdx][]BitRange{
			BaseChunkIdx: bits64ToBitRanges(scav, base),
		}
		want := NewPageAlloc(afterAlloc, afterScav)
		defer FreePageAlloc(want)

		// Check to see if it worked.
		checkPageAlloc(t, want, b)
	}

	// Empty.
	runTest(t, 0, 0, 0)

	// Full.
	runTest(t, 0, ^uint64(0), ^uint64(0))

	// Random.
	for i := 0; i < 100; i++ {
		// Generate random valid base within a chunk.
		base := uint(rand.Intn(PallocChunkPages/64)) * 64

		// Generate random cache.
		cache := rand.Uint64()
		scav := rand.Uint64() & cache

		// Run the test.
		runTest(t, base, cache, scav)
	}
}

func TestPageAllocAllocToCache(t *testing.T) {
	if GOOS == "openbsd" && testing.Short() {
		t.Skip("skipping because virtual memory is limited; see #36210")
	}
	type test struct {
		beforeAlloc map[ChunkIdx][]BitRange
		beforeScav  map[ChunkIdx][]BitRange
		hits        []PageCache // expected base addresses and patterns
		afterAlloc  map[ChunkIdx][]BitRange
		afterScav   map[ChunkIdx][]BitRange
	}
	tests := map[string]test{
		"ManyArena": {
			beforeAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx:     {{0, PallocChunkPages}},
				BaseChunkIdx + 1: {{0, PallocChunkPages}},
				BaseChunkIdx + 2: {{0, PallocChunkPages - 64}},
			},
			beforeScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx:     {{0, PallocChunkPages}},
				BaseChunkIdx + 1: {{0, PallocChunkPages}},
				BaseChunkIdx + 2: {},
			},
			hits: []PageCache{
				NewPageCache(PageBase(BaseChunkIdx+2, PallocChunkPages-64), ^uint64(0), 0),
			},
			afterAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx:     {{0, PallocChunkPages}},
				BaseChunkIdx + 1: {{0, PallocChunkPages}},
				BaseChunkIdx + 2: {{0, PallocChunkPages}},
			},
		},

		"Fail": {
			beforeAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{0, PallocChunkPages}},
			},
			hits: []PageCache{
				NewPageCache(0, 0, 0),
				NewPageCache(0, 0, 0),
				NewPageCache(0, 0, 0),
			},
			afterAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{0, PallocChunkPages}},
			},
		},
		"RetainScavBits": {
			beforeAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{0, 1}, {10, 2}},
			},
			beforeScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{0, 4}, {11, 1}},
			},
			hits: []PageCache{
				NewPageCache(PageBase(BaseChunkIdx, 0), ^uint64(0x1|(0x3<<10)), 0x7<<1),
			},
			afterAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{0, 64}},
			},
			afterScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{0, 1}, {11, 1}},
			},
		},
	}
	if PallocChunkPages >= 512 {
		tests["AllFree"] = test{
			beforeAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {},
			},
			beforeScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{1, 1}, {64, 64}},
			},
			hits: []PageCache{
				NewPageCache(PageBase(BaseChunkIdx, 0), ^uint64(0), 0x2),
				NewPageCache(PageBase(BaseChunkIdx, 64), ^uint64(0), ^uint64(0)),
				NewPageCache(PageBase(BaseChunkIdx, 128), ^uint64(0), 0),
				NewPageCache(PageBase(BaseChunkIdx, 192), ^uint64(0), 0),
			},
			afterAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{0, 256}},
			},
		}
		tests["NotContiguous"] = test{
			beforeAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx:        {{0, PallocChunkPages}},
				BaseChunkIdx + 0xff: {{0, 0}},
			},
			beforeScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx:        {{0, PallocChunkPages}},
				BaseChunkIdx + 0xff: {{31, 67}},
			},
			hits: []PageCache{
				NewPageCache(PageBase(BaseChunkIdx+0xff, 0), ^uint64(0), ((uint64(1)<<33)-1)<<31),
			},
			afterAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx:        {{0, PallocChunkPages}},
				BaseChunkIdx + 0xff: {{0, 64}},
			},
			afterScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx:        {{0, PallocChunkPages}},
				BaseChunkIdx + 0xff: {{64, 34}},
			},
		}
		tests["First"] = test{
			beforeAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{0, 32}, {33, 31}, {96, 32}},
			},
			beforeScav: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{1, 4}, {31, 5}, {66, 2}},
			},
			hits: []PageCache{
				NewPageCache(PageBase(BaseChunkIdx, 0), 1<<32, 1<<32),
				NewPageCache(PageBase(BaseChunkIdx, 64), (uint64(1)<<32)-1, 0x3<<2),
			},
			afterAlloc: map[ChunkIdx][]BitRange{
				BaseChunkIdx: {{0, 128}},
			},
		}
	}
	// Disable these tests on iOS since we have a small address space.
	// See #46860.
	if PageAlloc64Bit != 0 && goos.IsIos == 0 {
		const chunkIdxBigJump = 0x100000 // chunk index offset which translates to O(TiB)

		// This test is similar to the one with the same name for
		// pageAlloc.alloc and serves the same purpose.
		// See mpagealloc_test.go for details.
		sumsPerPhysPage := ChunkIdx(PhysPageSize / PallocSumBytes)
		baseChunkIdx := BaseChunkIdx &^ (sumsPerPhysPage - 1)
		tests["DiscontiguousMappedSumBoundary"] = test{
			beforeAlloc: map[ChunkIdx][]BitRange{
				baseChunkIdx + sumsPerPhysPage - 1: {{0, PallocChunkPages - 1}},
				baseChunkIdx + chunkIdxBigJump:     {{1, PallocChunkPages - 1}},
			},
			beforeScav: map[ChunkIdx][]BitRange{
				baseChunkIdx + sumsPerPhysPage - 1: {},
				baseChunkIdx + chunkIdxBigJump:     {},
			},
			hits: []PageCache{
				NewPageCache(PageBase(baseChunkIdx+sumsPerPhysPage-1, PallocChunkPages-64), 1<<63, 0),
				NewPageCache(PageBase(baseChunkIdx+chunkIdxBigJump, 0), 1, 0),
				NewPageCache(0, 0, 0),
			},
			afterAlloc: map[ChunkIdx][]BitRange{
				baseChunkIdx + sumsPerPhysPage - 1: {{0, PallocChunkPages}},
				baseChunkIdx + chunkIdxBigJump:     {{0, PallocChunkPages}},
			},
		}
	}
	for name, v := range tests {
		v := v
		t.Run(name, func(t *testing.T) {
			b := NewPageAlloc(v.beforeAlloc, v.beforeScav)
			defer FreePageAlloc(b)

			for _, expect := range v.hits {
				checkPageCache(t, b.AllocToCache(), expect)
				if t.Failed() {
					return
				}
			}
			want := NewPageAlloc(v.afterAlloc, v.afterScav)
			defer FreePageAlloc(want)

			checkPageAlloc(t, want, b)
		})
	}
}
