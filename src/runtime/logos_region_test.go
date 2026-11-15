// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime_test

import (
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"time"
	"unsafe"
)

//go:noinline
func regionOpenInFrame() (unsafe.Pointer, uintptr) {
	return runtime.LogosRegionOpen()
}

//go:noinline
func regionAllocInFrame(h unsafe.Pointer, gen uintptr, n uintptr) unsafe.Pointer {
	return runtime.LogosRegionAllocNoscan(h, gen, n, 8)
}

func regionFillPattern(p unsafe.Pointer, size uintptr, v byte) {
	fillBytes(p, size, v)
}

func TestLogosRegionChunkLinkExactFitOverflow(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		capacity := uintptr(runtime.LogosArenaCapacity)
		maxBytes := uintptr(runtime.LogosRegionMaxBytes)
		first, second := maxBytes, capacity-maxBytes
		pa := runtime.LogosRegionAllocNoscan(h, gen, first, 8)
		fillBytes(pa, first, 0xAB)
		pb := runtime.LogosRegionAllocNoscan(h, gen, second, 8)
		fillBytes(pb, second, 0xCD)
		pc := runtime.LogosRegionAllocNoscan(h, gen, 1, 8)
		fillBytes(pc, 1, 0x11)
		for i, p := range []unsafe.Pointer{pa, pb, pc} {
			if !runtime.LogosArenaInChunk(p) {
				t.Fatalf("object %d not placed in a region chunk", i)
			}
		}
		if got := runtime.LogosRegionBytes(); got != int(first+second+1) {
			t.Fatalf("RegionBytes = %d, want %d", got, first+second+1)
		}
		if n := len(runtime.LogosRegionChunks()); n != 2 {
			t.Fatalf("region chunks = %d, want 2 after the 1-byte overflow", n)
		}
		if !checkBytes(pa, first, 0xAB) || !checkBytes(pb, second, 0xCD) || !checkBytes(pc, 1, 0x11) {
			t.Fatal("region objects lost their values across the chunk link")
		}
		inner, innerGen := runtime.LogosRegionOpen()
		pi := runtime.LogosRegionAllocNoscan(inner, innerGen, 32, 8)
		fillBytes(pi, 32, 0x77)
		runtime.LogosRegionClose(inner, innerGen)
		if !checkBytes(pa, first, 0xAB) || !checkBytes(pb, second, 0xCD) || !checkBytes(pc, 1, 0x11) {
			t.Fatal("inner Close damaged the outer region")
		}
		again := runtime.LogosRegionAllocNoscan(h, gen, 8, 8)
		if !runtime.LogosArenaInChunk(again) {
			t.Fatal("outer region no longer places after inner Close")
		}
		runtime.LogosRegionClose(h, gen)
		if got := runtime.LogosRegionBytes(); got != 0 {
			t.Fatalf("RegionBytes = %d after Close, want 0", got)
		}
	})
}

//go:noinline
func regionOwnIntInScan(h unsafe.Pointer, gen uintptr, done chan struct{}) unsafe.Pointer {
	obj := new(int)
	runtime.AddCleanup(obj, func(c chan struct{}) { close(c) }, done)
	p := runtime.LogosRegionAllocScan(h, gen, (*int)(nil), 64)
	for i := range 64 {
		*(*byte)(unsafe.Pointer(uintptr(p) + uintptr(i))) = 0xAA
	}
	*(**int)(p) = obj
	return p
}

func TestLogosRegionScanLivenessZeroAndCleanup(t *testing.T) {
	inFreshGoroutine(t, func() {
		defer debug.SetGCPercent(debug.SetGCPercent(-1))
		runtime.GC()
		h, gen := runtime.LogosRegionOpen()
		done := make(chan struct{})
		p := regionOwnIntInScan(h, gen, done)
		runtime.GC()
		runtime.GC()
		select {
		case <-done:
			t.Fatal("cleanup ran while the region still held the only reference")
		case <-time.After(200 * time.Millisecond):
		}
		runtime.LogosRegionClose(h, gen)
		if !checkBytes(p, 64, 0) {
			t.Fatal("scan memory did not read zero after Close")
		}
		runtime.GC()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("cleanup did not run within 5s of a GC after Close")
		}
	})
}

func TestLogosRegionParkReleasesEmptyChainOnly(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		a := runtime.LogosRegionAllocNoscan(h, gen, 64, 8)
		fillBytes(a, 64, 0xA5)
		capacity := uintptr(runtime.LogosArenaCapacity)
		maxBytes := uintptr(runtime.LogosRegionMaxBytes)
		runtime.LogosRegionAllocNoscan(h, gen, maxBytes, 8)
		runtime.LogosRegionAllocNoscan(h, gen, capacity-64-maxBytes, 8)
		runtime.LogosRegionAllocNoscan(h, gen, 1, 8)
		chunks := runtime.LogosRegionChunks()
		if len(chunks) != 2 {
			t.Fatalf("region chunks = %d, want 2", len(chunks))
		}
		if !runtime.LogosRegionEmptyChunk(chunks[0]) {
			t.Fatal("could not empty the newest region chunk")
		}
		wantBytes := int(64 + maxBytes + (capacity - 64 - maxBytes))
		if got := runtime.LogosRegionBytes(); got != wantBytes {
			t.Fatalf("RegionBytes = %d, want %d before park", got, wantBytes)
		}
		release := make(chan struct{})
		go func() {
			time.Sleep(50 * time.Millisecond)
			close(release)
		}()
		<-release
		chunks = runtime.LogosRegionChunks()
		if len(chunks) != 1 {
			t.Fatalf("region chunks after park = %d, want 1 (the live chunk)", len(chunks))
		}
		if !checkBytes(a, 64, 0xA5) {
			t.Fatal("live region object moved or changed over park")
		}
		if got := runtime.LogosRegionBytes(); got != wantBytes {
			t.Fatalf("RegionBytes = %d after park, want %d", got, wantBytes)
		}
		runtime.LogosRegionClose(h, gen)
	})
}

func TestLogosRegionParkEmptyRegionsHoldNothing(t *testing.T) {
	inFreshGoroutine(t, func() {
		total, leased, _ := runtime.LogosArenaCounts()
		release := make(chan struct{})
		go func() {
			time.Sleep(50 * time.Millisecond)
			close(release)
		}()
		h1, g1 := runtime.LogosRegionOpen()
		h2, g2 := runtime.LogosRegionOpen()
		<-release
		if totalNow, leasedNow, _ := runtime.LogosArenaCounts(); totalNow != total || leasedNow != leased {
			t.Fatalf("counts after park = (%d,%d), want (%d,%d)", totalNow, leasedNow, total, leased)
		}
		if n := len(runtime.LogosRegionChunks()); n != 0 {
			t.Fatalf("empty regions hold %d chunks, want 0", n)
		}
		if got := runtime.LogosRegionBytes(); got != 0 {
			t.Fatalf("RegionBytes = %d, want 0", got)
		}
		runtime.LogosRegionClose(h2, g2)
		runtime.LogosRegionClose(h1, g1)
	})
}

func TestLogosRegionFinalizerCleanupPanicTexts(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		noscan := runtime.LogosRegionAllocNoscan(h, gen, 32, 8)
		scan := runtime.LogosRegionAllocScan(h, gen, (*int)(nil), 32)
		for _, obj := range []unsafe.Pointer{noscan, scan} {
			msg, panicked := recoverMsg(func() { runtime.SetFinalizer((*byte)(obj), func(*byte) {}) })
			if !panicked || msg != "runtime: SetFinalizer on loGo/s arena object" {
				t.Fatalf("SetFinalizer on region object: panicked=%v msg=%q", panicked, msg)
			}
			msg, panicked = recoverMsg(func() { runtime.AddCleanup((*byte)(obj), func(struct{}) {}, struct{}{}) })
			if !panicked || msg != "runtime: AddCleanup on loGo/s arena object" {
				t.Fatalf("AddCleanup on region object: panicked=%v msg=%q", panicked, msg)
			}
		}
		runtime.LogosRegionClose(h, gen)
	})
}

func TestLogosRegionCgoChecksAcceptRegionAddresses(t *testing.T) {
	type holder struct{ p *int }
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		leaf := (*int)(runtime.LogosRegionAllocNoscan(h, gen, 8, 8))
		hd := (*holder)(runtime.LogosRegionAllocScan(h, gen, (*int)(nil), unsafe.Sizeof(holder{})))
		hd.p = leaf
		bytes := unsafe.Slice((*byte)(runtime.LogosRegionAllocNoscan(h, gen, 32, 8)), 32)
		heapLeaf := new(int)
		if !runtime.LogosArenaIsGoPointer(unsafe.Pointer(leaf)) || !runtime.LogosArenaIsGoPointer(unsafe.Pointer(hd)) {
			t.Fatal("a region address is not a Go pointer for the cgo checks")
		}
		cases := []struct {
			name string
			fn   func()
		}{
			{"pointer to region noscan object", func() { runtime.LogosArenaCgoCheckPointer(leaf, true) }},
			{"pointer to region scan object", func() { runtime.LogosArenaCgoCheckPointer(hd, true) }},
			{"unsafe.Pointer to region object", func() { runtime.LogosArenaCgoCheckPointer(unsafe.Pointer(hd), true) }},
			{"slice over region bytes", func() { runtime.LogosArenaCgoCheckPointer(&bytes[0], bytes) }},
			{"region struct holding a region pointer", func() { runtime.LogosArenaCgoCheckPointer(*hd, nil) }},
			{"result holding a region pointer", func() { runtime.LogosArenaCgoCheckResult(hd) }},
			{"heap pointer stored into region memory", func() {
				runtime.LogosArenaCgoCheckPtrWrite((*unsafe.Pointer)(unsafe.Pointer(&hd.p)), unsafe.Pointer(heapLeaf))
			}},
			{"region pointer stored into heap memory", func() {
				var slot *int
				runtime.LogosArenaCgoCheckPtrWrite((*unsafe.Pointer)(unsafe.Pointer(&slot)), unsafe.Pointer(leaf))
			}},
			{"cgocheck memmove over region objects", func() {
				runtime.LogosArenaCgoCheckMemmove(holder{}, unsafe.Pointer(hd), unsafe.Pointer(hd))
			}},
		}
		for _, c := range cases {
			if msg, panicked := recoverMsg(c.fn); panicked {
				t.Errorf("%s: cgo check panicked: %s", c.name, msg)
			}
		}
		runtime.LogosRegionClose(h, gen)
	})
}

func TestLogosRegionInvalidHandlesGoToHeap(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		runtime.LogosRegionAllocNoscan(h, gen, 64, 8)
		runtime.LogosRegionClose(h, gen)
		h2, gen2 := runtime.LogosRegionOpen()
		runtime.LogosRegionAllocNoscan(h2, gen2, 64, 8)
		stale := runtime.LogosRegionAllocNoscan(h, gen, 64, 8)
		if runtime.LogosArenaInChunk(stale) {
			t.Fatal("a stale-gen handle placed in a chunk")
		}
		if got := runtime.LogosRegionBytes(); got != 64 {
			t.Fatalf("RegionBytes = %d, want 64 (heap fallback must not touch the region)", got)
		}
		done := make(chan struct{})
		go func() {
			defer close(done)
			foreign := runtime.LogosRegionAllocNoscan(h2, gen2, 64, 8)
			if runtime.LogosArenaInChunk(foreign) {
				t.Error("a foreign-g handle placed in a chunk")
			}
		}()
		<-done
		runtime.LogosRegionClose(h2, gen2)
	})
}

func TestLogosRegionCloseOrderThrows(t *testing.T) {
	if os.Getenv("GOLOGOS_REGION_THROW") == "1" {
		h1, g1 := runtime.LogosRegionOpen()
		h2, g2 := runtime.LogosRegionOpen()
		runtime.LogosRegionClose(h1, g1)
		runtime.LogosRegionClose(h2, g2)
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestLogosRegionCloseOrderThrows$", "-test.timeout=60s")
	cmd.Env = append(os.Environ(), "GOLOGOS_REGION_THROW=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("out-of-order Close did not throw: %s", out)
	}
	if !strings.Contains(string(out), "logosRegionClose") {
		t.Fatalf("out-of-order Close threw an unrelated error: %s", out)
	}
}

//go:noinline
func regionUnwindLeaf() {
	panic("region unwind leaf")
}

//go:noinline
func regionUnwindMid() {
	h, gen := regionOpenInFrame()
	regionAllocInFrame(h, gen, 32)
	regionUnwindLeaf()
}

//go:noinline
func regionUnwindTwoUp() (recovered bool) {
	h1, gen1 := regionOpenInFrame()
	regionAllocInFrame(h1, gen1, 32)
	defer func() {
		if r := recover(); r != nil {
			recovered = true
		}
	}()
	regionUnwindMid()
	return
}

func TestLogosRegionPanicUnwindTwoFramesUp(t *testing.T) {
	inFreshGoroutine(t, func() {
		if !regionUnwindTwoUp() {
			t.Fatal("panic was not recovered two frames up")
		}
		if got := runtime.LogosRegionBytes(); got != 0 {
			t.Fatalf("RegionBytes = %d after recovery, want 0", got)
		}
		if n := len(runtime.LogosRegionChunks()); n != 0 {
			t.Fatalf("region chunks after recovery = %d, want 0", n)
		}
	})
}

//go:noinline
func regionUnwindMidRecovers() {
	h, gen := regionOpenInFrame()
	regionAllocInFrame(h, gen, 32)
	defer func() {
		recover()
	}()
	regionUnwindLeaf()
}

func TestLogosRegionPanicUnwindOneFrameUp(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		survivor := regionAllocInFrame(h, gen, 64)
		fillBytes(survivor, 64, 0x5A)
		regionUnwindMidRecovers()
		if got := runtime.LogosRegionBytes(); got != 64 {
			t.Fatalf("RegionBytes = %d after one-frame recovery, want 64 (outer region must survive)", got)
		}
		if !checkBytes(survivor, 64, 0x5A) {
			t.Fatal("surviving region object damaged by recovery")
		}
		again := runtime.LogosRegionAllocNoscan(h, gen, 16, 8)
		if !runtime.LogosArenaInChunk(again) {
			t.Fatal("surviving region no longer places after a deeper recovery")
		}
		runtime.LogosRegionClose(h, gen)
		if got := runtime.LogosRegionBytes(); got != 0 {
			t.Fatalf("RegionBytes = %d after final Close, want 0", got)
		}
	})
}

func TestLogosRegionExitReturnsChunks(t *testing.T) {
	total := runtime.LogosRegionChunkTotal()
	done := make(chan struct{})
	go func() {
		defer close(done)
		h1, g1 := runtime.LogosRegionOpen()
		capacity := uintptr(runtime.LogosArenaCapacity)
		maxBytes := uintptr(runtime.LogosRegionMaxBytes)
		runtime.LogosRegionAllocNoscan(h1, g1, maxBytes, 8)
		runtime.LogosRegionAllocNoscan(h1, g1, capacity-maxBytes, 8)
		runtime.LogosRegionAllocNoscan(h1, g1, 1, 8)
		h2, g2 := runtime.LogosRegionOpen()
		runtime.LogosRegionAllocScan(h2, g2, (*int)(nil), 64)
		if n := len(runtime.LogosRegionChunks()); n != 3 {
			t.Errorf("region chunks held at exit = %d, want 3", n)
		}
	}()
	<-done
	deadline := time.Now().Add(20 * time.Second)
	for {
		if totalNow := runtime.LogosRegionChunkTotal(); totalNow == total {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("region chunks after goroutine exit = %d, want %d", runtime.LogosRegionChunkTotal(), total)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestLogosRegionMakeSlicePanicsAndPlaces(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		msg, panicked := recoverMsg(func() { runtime.LogosRegionMakeSlice(h, gen, int(0), -1, -1) })
		if !panicked || msg != "runtime error: makeslice: len out of range" {
			t.Fatalf("MakeSlice len=-1: panicked=%v msg=%q", panicked, msg)
		}
		msg, panicked = recoverMsg(func() { runtime.LogosRegionMakeSlice(h, gen, int(0), 2, 1) })
		if !panicked || msg != "runtime error: makeslice: cap out of range" {
			t.Fatalf("MakeSlice len>cap: panicked=%v msg=%q", panicked, msg)
		}
		p := runtime.LogosRegionMakeSlice(h, gen, int(0), 10, 10)
		if !runtime.LogosArenaInChunk(p) {
			t.Fatal("noscan MakeSlice did not place in a chunk")
		}
		sp := runtime.LogosRegionMakeSlice(h, gen, (*int)(nil), 4, 4)
		if !runtime.LogosArenaInChunk(sp) {
			t.Fatal("scan MakeSlice did not place in a chunk")
		}
		runtime.LogosRegionClose(h, gen)
	})
}

func TestLogosRegionGrowSliceCapacityPolicy(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		for _, oldCap := range []int{0, 1, 255, 256, 257, 1024} {
			var oldPtr unsafe.Pointer
			var old []int
			if oldCap > 0 {
				old = make([]int, oldCap)
				for i := range old {
					old[i] = i*7 + 3
				}
				oldPtr = unsafe.Pointer(&old[0])
			}
			newLen, num := oldCap+1, 1
			wantArr, wantCap := runtime.LogosHeapGrowSlice(oldPtr, newLen, oldCap, num, int(0))
			gotArr, gotCap := runtime.LogosRegionGrowSlice(h, gen, oldPtr, newLen, oldCap, num, int(0))
			if gotCap != wantCap {
				t.Fatalf("oldCap %d: region cap = %d, growslice cap = %d", oldCap, gotCap, wantCap)
			}
			if !runtime.LogosArenaInChunk(gotArr) {
				t.Fatalf("oldCap %d: region grow did not place in a chunk", oldCap)
			}
			if oldCap > 0 {
				got := unsafe.Slice((*int)(gotArr), newLen)
				for i := range old {
					if got[i] != old[i] {
						t.Fatalf("oldCap %d: element %d = %d, want %d", oldCap, i, got[i], old[i])
					}
				}
			}
			_ = wantArr
		}
		runtime.LogosRegionClose(h, gen)
	})
}

func TestLogosRegionSizeLimits(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		maxBytes := uintptr(runtime.LogosRegionMaxBytes)
		p1 := runtime.LogosRegionAllocNoscan(h, gen, maxBytes-1, 8)
		if !runtime.LogosArenaInChunk(p1) {
			t.Fatal("MaxBytes-1 not placed in a chunk")
		}
		p2 := runtime.LogosRegionAllocNoscan(h, gen, maxBytes, 8)
		if !runtime.LogosArenaInChunk(p2) {
			t.Fatal("MaxBytes not placed in a chunk")
		}
		p3 := runtime.LogosRegionAllocNoscan(h, gen, maxBytes+1, 8)
		if runtime.LogosArenaInChunk(p3) {
			t.Fatal("MaxBytes+1 placed in a chunk")
		}
		if !checkBytes(p3, maxBytes+1, 0) {
			t.Fatal("heap fallback memory is not zero")
		}
		runtime.LogosRegionClose(h, gen)
	})
}

func TestLogosRegionChunkCountLimit(t *testing.T) {
	total, _, pooled := runtime.LogosArenaCounts()
	old := runtime.LogosArenaSetLimit(int64(total) + 1)
	defer runtime.LogosArenaSetLimit(old)
	inFreshGoroutine(t, func() {
		const size = 60000
		h, gen := runtime.LogosRegionOpen()
		var placed []unsafe.Pointer
		var heapPtr unsafe.Pointer
		for i := 0; i <= pooled+2; i++ {
			p := runtime.LogosRegionAllocNoscan(h, gen, size, 8)
			if !runtime.LogosArenaInChunk(p) {
				heapPtr = p
				break
			}
			fillBytes(p, size, byte(i+1))
			placed = append(placed, p)
		}
		if len(placed) == 0 || heapPtr == nil {
			t.Fatalf("placed=%d fallback=%v with the limit one above the current chunk count", len(placed), heapPtr != nil)
		}
		if !checkBytes(heapPtr, size, 0) {
			t.Fatal("fallback memory is not zero")
		}
		fillBytes(heapPtr, size, 0xEE)
		for i, p := range placed {
			if !checkBytes(p, size, byte(i+1)) {
				t.Errorf("region object %d was damaged by the fallback", i)
			}
		}
		runtime.LogosRegionClose(h, gen)
	})
}

func TestLogosRegionPointersTable(t *testing.T) {
	table := []struct {
		val  any
		want bool
	}{
		{struct{}{}, false},
		{[0]*int{}, false},
		{0, false},
		{byte(0), false},
		{complex128(0), false},
		{[4]int64{}, false},
		{[1]*int{}, true},
		{"s", true},
		{(*int)(nil), true},
		{[]int(nil), true},
		{map[string]int(nil), true},
		{chan int(nil), true},
		{(func())(nil), true},
		{struct {
			p *int
			x [8]byte
		}{}, true},
		{struct {
			x [8]byte
			p *int
		}{}, true},
	}
	for _, e := range table {
		if got := runtime.LogosTypePointers(e.val); got != e.want {
			t.Fatalf("typ.Pointers(%T) = %v, want %v", e.val, got, e.want)
		}
	}
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		for _, e := range table {
			p := runtime.LogosRegionMakeSlice(h, gen, e.val, 2, 2)
			if !e.want {
				continue
			}
			if !runtime.LogosArenaInChunk(p) {
				t.Fatalf("scan dispatch for %T did not place in a chunk", e.val)
			}
		}
		runtime.LogosRegionClose(h, gen)
	})
}

func TestLogosRegionBumpSlowPath(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		if p := runtime.LogosRegionBump(nil, gen, 8); p != nil {
			t.Fatal("bump on a nil handle did not return nil")
		}
		runtime.LogosRegionAllocNoscan(h, gen, 8, 8)
		used := uintptr(8)
		if !runtime.Raceenabled {
			p := runtime.LogosRegionBump(h, gen, 8)
			if p == nil || !runtime.LogosArenaInChunk(p) {
				t.Fatal("bump did not place in a chunk")
			}
			fillBytes(p, 8, 0x42)
			q := runtime.LogosRegionBump(h, gen, 16)
			if q == nil || uintptr(q) != uintptr(p)+8 {
				t.Fatal("bump did not advance the cursor contiguously")
			}
			used += 24
		}
		capacity := uintptr(runtime.LogosArenaCapacity)
		for left := capacity - used; left > 0; {
			n := min(left, runtime.LogosRegionMaxBytes)
			runtime.LogosRegionAllocNoscan(h, gen, n, 8)
			left -= n
		}
		if full := runtime.LogosRegionBump(h, gen, 8); full != nil {
			t.Fatal("bump crossed the chunk without returning nil")
		}
		over := runtime.LogosRegionAllocNoscan(h, gen, 8, 8)
		if !runtime.LogosArenaInChunk(over) {
			t.Fatal("the slow path did not link a chunk after a failed bump")
		}
		runtime.LogosRegionClose(h, gen)
		if stale := runtime.LogosRegionBump(h, gen, 8); stale != nil {
			t.Fatal("bump on a stale handle did not return nil")
		}
	})
}

func TestLogosRegionReuseRaceClean(t *testing.T) {
	inFreshGoroutine(t, func() {
		for i := range 64 {
			h, gen := runtime.LogosRegionOpen()
			p := runtime.LogosRegionAllocNoscan(h, gen, 8192, 8)
			regionFillPattern(p, 8192, byte(i))
			runtime.LogosRegionClose(h, gen)
			h2, gen2 := runtime.LogosRegionOpen()
			q := runtime.LogosRegionAllocNoscan(h2, gen2, 8192, 8)
			if !checkBytes(q, 8192, 0) {
				t.Fatal("reused region memory was not cleared")
			}
			regionFillPattern(q, 8192, byte(i+1))
			runtime.LogosRegionClose(h2, gen2)
		}
	})
}

func TestLogosRegionChunkGranularity64K(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		for i := range 63 * 1024 / 8 {
			p := runtime.LogosRegionAllocNoscan(h, gen, 8, 8)
			if p == nil || !runtime.LogosArenaInChunk(p) {
				t.Fatalf("63 KiB fill stalled at object %d", i)
			}
			*(*byte)(p) = byte(i)
		}
		if n := len(runtime.LogosRegionChunks()); n != 1 {
			t.Fatalf("63 KiB of region noscan = %d chunks, want 1", n)
		}
		runtime.LogosRegionClose(h, gen)
	})
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		for i := range 65 * 1024 / 8 {
			p := runtime.LogosRegionAllocNoscan(h, gen, 8, 8)
			if p == nil || !runtime.LogosArenaInChunk(p) {
				t.Fatalf("65 KiB fill stalled at object %d", i)
			}
			*(*byte)(p) = byte(i)
		}
		if n := len(runtime.LogosRegionChunks()); n != 2 {
			t.Fatalf("65 KiB of region noscan = %d chunks, want 2", n)
		}
		runtime.LogosRegionClose(h, gen)
	})
}

func TestLogosRegionChainsArePrivate(t *testing.T) {
	inFreshGoroutine(t, func() {
		chunkSet := func() map[uintptr]bool {
			s := map[uintptr]bool{}
			for _, c := range runtime.LogosRegionChunks() {
				s[c] = true
			}
			return s
		}
		hA, gA := runtime.LogosRegionOpen()
		pa := runtime.LogosRegionAllocNoscan(hA, gA, 32, 8)
		fillBytes(pa, 32, 0xA5)
		outer := chunkSet()
		if len(outer) != 1 {
			t.Fatalf("outer region chunks = %d, want 1", len(outer))
		}
		hB, gB := runtime.LogosRegionOpen()
		capacity := uintptr(runtime.LogosArenaCapacity)
		maxBytes := uintptr(runtime.LogosRegionMaxBytes)
		pb := runtime.LogosRegionAllocNoscan(hB, gB, maxBytes, 8)
		fillBytes(pb, maxBytes, 0x5A)
		pc := runtime.LogosRegionAllocNoscan(hB, gB, capacity-maxBytes, 8)
		fillBytes(pc, capacity-maxBytes, 0x33)
		pd := runtime.LogosRegionAllocNoscan(hB, gB, 1024, 8)
		fillBytes(pd, 1024, 0x77)
		during := chunkSet()
		for c := range outer {
			if !during[c] {
				t.Fatalf("outer chunk %x left its chain while the inner region allocated", c)
			}
		}
		if len(during) != 3 {
			t.Fatalf("chunks with both regions open = %d, want 3 (outer plus the inner chain)", len(during))
		}
		runtime.LogosRegionClose(hB, gB)
		after := chunkSet()
		if len(after) != len(outer) {
			t.Fatalf("chunks after inner Close = %d, want %d (the outer chain only)", len(after), len(outer))
		}
		for c := range outer {
			if !after[c] {
				t.Fatalf("outer chunk %x missing after inner Close", c)
			}
		}
		if got := runtime.LogosRegionBytes(); got != 32 {
			t.Fatalf("RegionBytes after inner Close = %d, want 32", got)
		}
		if !checkBytes(pa, 32, 0xA5) {
			t.Fatal("outer object damaged by inner Close")
		}
		again := runtime.LogosRegionAllocNoscan(hA, gA, 64, 8)
		if !runtime.LogosArenaInChunk(again) {
			t.Fatal("outer region no longer places after inner Close")
		}
		runtime.LogosRegionClose(hA, gA)
		if n := len(chunkSet()); n != 0 {
			t.Fatalf("chunks after both Closed = %d, want 0", n)
		}
	})
}

func TestLogosRegionParkRetainsStackAndMetadata(t *testing.T) {
	inFreshGoroutine(t, func() {
		canary := 0x5A5A5A5A
		before := runtime.LogosArenaGoid()
		h1, g1 := runtime.LogosRegionOpen()
		h2, g2 := runtime.LogosRegionOpen()
		release := make(chan struct{})
		go func() {
			time.Sleep(50 * time.Millisecond)
			close(release)
		}()
		canary += 1
		<-release
		if canary != 0x5A5A5A5B {
			t.Fatalf("stack canary = %#x, want %#x after park", canary, 0x5A5A5A5B)
		}
		if got := runtime.LogosArenaGoid(); got != before {
			t.Fatalf("goid = %d, want %d after park", got, before)
		}
		if n := len(runtime.LogosRegionChunks()); n != 0 {
			t.Fatalf("empty regions hold %d chunks, want 0 after park", n)
		}
		if got := runtime.LogosRegionBytes(); got != 0 {
			t.Fatalf("RegionBytes = %d, want 0 after park", got)
		}
		runtime.LogosRegionClose(h2, g2)
		runtime.LogosRegionClose(h1, g1)
		h3, g3 := runtime.LogosRegionOpen()
		p := runtime.LogosRegionAllocNoscan(h3, g3, 32, 8)
		fillBytes(p, 32, 0xA5)
		if !checkBytes(p, 32, 0xA5) {
			t.Fatal("region object damaged after park and reopen")
		}
		if !runtime.LogosArenaInChunk(p) {
			t.Fatal("reopened region no longer places after park")
		}
		runtime.LogosRegionClose(h3, g3)
		if n := len(runtime.LogosRegionChunks()); n != 0 {
			t.Fatalf("chunks after Close = %d, want 0", n)
		}
	})
}

func TestLogosRegionOwnerEnforcedCrossGoroutine(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		p := runtime.LogosRegionAllocNoscan(h, gen, 32, 8)
		if p == nil || !runtime.LogosArenaInChunk(p) {
			t.Fatal("owner region did not place")
		}
		q := runtime.LogosRegionBump(h, gen, 8)
		if runtime.Raceenabled {
			if q != nil {
				t.Fatal("race-enabled owner bump bypassed allocation tracking")
			}
		} else if q == nil || !runtime.LogosArenaInChunk(q) {
			t.Fatal("owner bump did not place")
		}
		bumpCh := make(chan unsafe.Pointer, 1)
		allocCh := make(chan unsafe.Pointer, 1)
		go func() {
			bumpCh <- runtime.LogosRegionBump(h, gen, 8)
			allocCh <- runtime.LogosRegionAllocNoscan(h, gen, 32, 8)
		}()
		if r := <-bumpCh; r != nil {
			t.Fatalf("foreign bump = %v, want nil", r)
		}
		if r := <-allocCh; r == nil || runtime.LogosArenaInChunk(r) {
			t.Fatalf("foreign alloc = %v, want heap fallback", r)
		}
		runtime.LogosRegionClose(h, gen)
	})
}
