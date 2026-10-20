// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm

package runtime_test

import (
	"runtime"
	"testing"
	"unsafe"
)

// On wasm the heap splits into a low permanent region and a high scratch
// region (SPEC Sec 19, T-301.2). memory.grow extends the high region
// only, scratch reclamation lowers the scratch break and never reaches the
// low region, and permanent requests stay low until the region is full.
// Each region keeps its own free list, so a heap arena reservation is never
// served from low memory. The split exists only for binaries linked with
// -gc=wasm.

const wasmPage = 64 << 10

func lowRegion(t *testing.T) (base, end uintptr) {
	t.Helper()
	base, end, _, _ = runtime.LogosWasmLowRegion()
	if base == 0 {
		t.Skip("heap split needs a binary linked with -gc=wasm")
	}
	return base, end
}

// The low region has the fixed size and page alignment, and the Go heap
// lives above it.
func TestLogosWasmHeapLowRegionShape(t *testing.T) {
	base, end := lowRegion(t)
	if end-base != runtime.LogosWasmLowBytes {
		t.Fatalf("low region is %d bytes, want %d", end-base, runtime.LogosWasmLowBytes)
	}
	if base%wasmPage != 0 {
		t.Fatalf("low region base %#x is not page aligned", base)
	}
	b := make([]byte, 1<<20)
	p := uintptr(unsafe.Pointer(&b[0]))
	runtime.KeepAlive(b)
	if p < end {
		t.Fatalf("heap object at %#x lies inside or below the low region [%#x, %#x)", p, base, end)
	}
}

// Growing the scratch region 100 times extends the linear memory, hands
// out only addresses above the low region and contiguous with the break,
// and never changes the low region.
func TestLogosWasmHeapScratchGrowthLeavesLowRegionFixed(t *testing.T) {
	base, end := lowRegion(t)
	_, _, used0, spilled0 := runtime.LogosWasmLowRegion()
	pagesBefore := runtime.LogosWasmMemoryPages()
	const steps = 100
	const stepBytes = 4 * wasmPage
	out := make([]runtime.LogosWasmStep, steps)
	got := runtime.LogosWasmScratchSteps(steps, stepBytes, out)
	if got != steps {
		t.Fatalf("scratch grew %d times, want %d", got, steps)
	}
	for i, s := range out {
		if s.Base != base || s.End != end || s.Used != used0 {
			t.Fatalf("step %d changed the low region: [%#x, %#x) used %d, was [%#x, %#x) used %d",
				i, s.Base, s.End, s.Used, base, end, used0)
		}
		if s.Addr < end+wasmPage {
			t.Fatalf("step %d returned %#x inside the guard page after the low region end %#x", i, s.Addr, end)
		}
		if s.Break != s.Addr+stepBytes {
			t.Fatalf("step %d: break %#x, want %#x", i, s.Break, s.Addr+stepBytes)
		}
		if i > 0 && s.Addr != out[i-1].Break {
			t.Fatalf("step %d: address %#x is not contiguous with break %#x", i, s.Addr, out[i-1].Break)
		}
	}
	top := out[steps-1].Break
	pagesAfter := runtime.LogosWasmMemoryPages()
	if pagesAfter*wasmPage < top {
		t.Fatalf("memory of %d pages does not cover the scratch break %#x", pagesAfter, top)
	}
	if top > pagesBefore*wasmPage && pagesAfter <= pagesBefore {
		t.Fatalf("break %#x passed the %d-page memory but memory did not grow (%d pages)", top, pagesBefore, pagesAfter)
	}
	if _, _, used, spilled := runtime.LogosWasmLowRegion(); used != used0 || spilled != spilled0 {
		t.Fatalf("scratch growth moved the low bump pointer or spilled: used %d->%d spilled %d->%d", used0, used, spilled0, spilled)
	}
	for i := steps - 1; i >= 0; i-- {
		runtime.LogosWasmScratchFree(out[i].Addr, stepBytes)
	}
	if brk := runtime.LogosWasmBreak(); brk != out[0].Addr {
		t.Fatalf("freeing the scratch top left the break at %#x, want %#x", brk, out[0].Addr)
	}
	if brk := runtime.LogosWasmBreak(); brk < runtime.LogosWasmScratchStart() {
		t.Fatalf("scratch reclamation lowered the break to %#x, below the scratch start %#x", brk, runtime.LogosWasmScratchStart())
	}
	if b2, e2, used, _ := runtime.LogosWasmLowRegion(); b2 != base || e2 != end || used != used0 {
		t.Fatalf("scratch reclamation moved the low region: [%#x, %#x) used %d", b2, e2, used)
	}
}

// A reservation without an address hint is permanent memory too: it goes
// through the permanent allocator whatever state the low region is in. The
// block comes from the low bump pointer or the low free list and stays in
// the low region, or the region is full, the request is counted as spilled
// and the block comes from scratch.
func TestLogosWasmHeapUnhintedReserveIsPermanent(t *testing.T) {
	base, end := lowRegion(t)
	_, _, used0, spilled0 := runtime.LogosWasmLowRegion()
	free0 := runtime.LogosWasmLowFreeBytes()
	p := runtime.LogosWasmPermReserve(wasmPage)
	if p == 0 {
		t.Fatal("unhinted reservation failed")
	}
	defer runtime.LogosWasmPermFree(p, wasmPage)
	_, _, used, spilled := runtime.LogosWasmLowRegion()
	free := runtime.LogosWasmLowFreeBytes()
	if p >= base && p < end {
		bumped := used == used0+wasmPage && free == free0
		reused := used == used0 && free == free0-wasmPage
		if spilled != spilled0 || bumped == reused {
			t.Fatalf("low block %#x: used %d->%d, free %d->%d, spilled %d->%d; want exactly one of bump and free-list reuse",
				p, used0, used, free0, free, spilled0, spilled)
		}
		return
	}
	if p < runtime.LogosWasmScratchStart() || spilled != spilled0+wasmPage || used != used0 || free != free0 {
		t.Fatalf("scratch block %#x: used %d->%d, free %d->%d, spilled %d->%d; want one spilled page above the scratch start %#x",
			p, used0, used, free0, free, spilled0, spilled, runtime.LogosWasmScratchStart())
	}
}

// Memory freed in the low region stays on the free list of the low region:
// the scratch free list never holds a low block, and a heap arena
// reservation, which takes any block of the scratch list that is large
// enough, is never served from low memory even when freed low blocks add up
// to more than an arena.
func TestLogosWasmHeapFreedLowBlocksNeverServeScratch(t *testing.T) {
	base, end := lowRegion(t)
	const pages = 24
	held := make([]uintptr, 0, pages)
	for i := range pages {
		p, _ := runtime.LogosWasmPermAlloc(wasmPage)
		if p == 0 {
			t.Fatalf("permanent allocation %d failed", i)
		}
		if p < base || p >= end {
			t.Fatalf("permanent allocation %d at %#x left the low region [%#x, %#x) while it had room", i, p, base, end)
		}
		held = append(held, p)
	}
	for _, p := range held {
		runtime.LogosWasmPermFree(p, wasmPage)
	}
	if free := runtime.LogosWasmLowFreeBytes(); free < pages*wasmPage {
		t.Fatalf("the low free list holds %d bytes after freeing %d low bytes", free, pages*wasmPage)
	}
	if runtime.LogosWasmFreeOverlaps(base, end) {
		t.Fatalf("the scratch free list holds a block of the low region [%#x, %#x)", base, end)
	}
	const arena = 512 << 10
	p, got := runtime.LogosWasmReserveAligned(arena, arena)
	if p == 0 {
		t.Fatal("arena reservation failed")
	}
	defer runtime.LogosWasmScratchFree(p, got)
	if p < runtime.LogosWasmScratchStart() {
		t.Fatalf("arena reserved at %#x, below the scratch start %#x", p, runtime.LogosWasmScratchStart())
	}
}

// Permanent requests come from the low region, zeroed and above the
// region base, until the region is full; the region bounds never move,
// and the first request that no longer fits is counted as spilled and
// served from scratch instead of growing the region.
func TestLogosWasmHeapPermanentStaysLowUntilFull(t *testing.T) {
	base, end := lowRegion(t)
	held := make([]uintptr, 0, runtime.LogosWasmLowBytes/wasmPage+2)
	defer func() {
		for _, p := range held {
			runtime.LogosWasmPermFree(p, wasmPage)
		}
	}()
	_, _, _, spilledBefore := runtime.LogosWasmLowRegion()
	spilled := false
	for i := 0; i <= int(runtime.LogosWasmLowBytes/wasmPage)+1 && !spilled; i++ {
		p, zeroed := runtime.LogosWasmPermAlloc(wasmPage)
		if p == 0 {
			t.Fatalf("permanent allocation %d failed", i)
		}
		held = append(held, p)
		if !zeroed {
			t.Fatalf("permanent allocation %d at %#x was not zeroed", i, p)
		}
		if p < base {
			t.Fatalf("permanent allocation %d at %#x lies below the low region base %#x", i, p, base)
		}
		inLow := p < end
		if inLow != runtime.LogosWasmInLow(p) {
			t.Fatalf("LogosWasmInLow(%#x) = %v, region says %v", p, !inLow, inLow)
		}
		if b2, e2, used, _ := runtime.LogosWasmLowRegion(); b2 != base || e2 != end || used > end-base {
			t.Fatalf("allocation %d moved the low region to [%#x, %#x) used %d", i, b2, e2, used)
		}
		if !inLow {
			spilled = true
			if _, _, used, _ := runtime.LogosWasmLowRegion(); used != end-base {
				t.Fatalf("allocation %d spilled with only %d of %d low bytes used", i, used, end-base)
			}
			if p < runtime.LogosWasmScratchStart() {
				t.Fatalf("allocation %d spilled to %#x, below the scratch start %#x", i, p, runtime.LogosWasmScratchStart())
			}
		}
	}
	if !spilled {
		t.Fatal("no permanent allocation spilled although the region was over-asked")
	}
	if _, _, _, spilledAfter := runtime.LogosWasmLowRegion(); spilledAfter != spilledBefore+wasmPage {
		t.Fatalf("spilled bytes went from %d to %d, want +%d", spilledBefore, spilledAfter, wasmPage)
	}
}

// One guard page separates the regions and belongs to neither: scratch
// starts a page after the low region ends, so freeing the whole scratch
// region never lowers the break into the low region or its guard page, and
// the guard never appears on a free list.
func TestLogosWasmHeapGuardPageSeparatesRegions(t *testing.T) {
	base, end := lowRegion(t)
	if got := runtime.LogosWasmScratchStart(); got != end+wasmPage {
		t.Fatalf("scratch starts at %#x, want one guard page after the low region end %#x", got, end)
	}
	out := make([]runtime.LogosWasmStep, 2)
	if got := runtime.LogosWasmScratchSteps(2, wasmPage, out); got != 2 {
		t.Fatalf("scratch grew %d times, want 2", got)
	}
	runtime.LogosWasmScratchFree(out[0].Addr, wasmPage)
	runtime.LogosWasmScratchFree(out[1].Addr, wasmPage)
	if runtime.LogosWasmFreeOverlaps(base, end+wasmPage) {
		t.Fatalf("the scratch free list covers [%#x, %#x), the low region and its guard page", base, end+wasmPage)
	}
	if brk := runtime.LogosWasmBreak(); brk < end+wasmPage {
		t.Fatalf("the scratch break fell to %#x, into the low region or its guard page", brk)
	}
}

// The region test is exact at both edges.
func TestLogosWasmHeapInLowBounds(t *testing.T) {
	base, end := lowRegion(t)
	for _, c := range []struct {
		p    uintptr
		want bool
	}{
		{base - 1, false},
		{base, true},
		{end - 1, true},
		{end, false},
	} {
		if got := runtime.LogosWasmInLow(c.p); got != c.want {
			t.Errorf("LogosWasmInLow(%#x) = %v, want %v", c.p, got, c.want)
		}
	}
}
