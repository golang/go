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
	"sync"
	"testing"
	"time"
	"unsafe"
)

func TestLogosRegionAdvGroupChildSharing(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		const kids = 8
		vals := make([]unsafe.Pointer, kids)
		total := 0
		for i := range kids {
			p := runtime.LogosRegionAllocScan(h, gen, (*int)(nil), 64)
			*(*int)(p) = i*7 + 3
			vals[i] = p
			total += i*7 + 3
		}
		blob := runtime.LogosRegionAllocNoscan(h, gen, 128, 8)
		fillBytes(blob, 128, 0xE6)
		sums := make(chan int, kids)
		var wg sync.WaitGroup
		for i := range kids {
			wg.Go(func() {
				if q := runtime.LogosRegionAllocNoscan(h, gen, 32, 8); runtime.LogosArenaInChunk(q) {
					t.Error("child allocate with a parent handle placed in a chunk")
					return
				}
				if b := runtime.LogosRegionBump(h, gen, 8); b != nil {
					t.Error("child bump with a parent handle returned region memory")
					return
				}
				sums <- *(*int)(vals[i])
			})
		}
		wg.Wait()
		close(sums)
		got := 0
		for v := range sums {
			got += v
		}
	})
}

func TestLogosRegionAdvNilZeroHandlesAndRaces(t *testing.T) {
	inFreshGoroutine(t, func() {
		if p := runtime.LogosRegionAllocNoscan(nil, 0, 64, 8); runtime.LogosArenaInChunk(p) {
			t.Fatal("nil handle placed in a chunk")
		}
		if p := runtime.LogosRegionAllocScan(nil, 0, (*int)(nil), 64); runtime.LogosArenaInChunk(p) {
			t.Fatal("nil scan handle placed in a chunk")
		}
		if p := runtime.LogosRegionBump(nil, 0, 8); p != nil {
			t.Fatal("bump on a nil handle did not return nil")
		}
		if p := runtime.LogosRegionMakeSlice(nil, 0, int(0), 10, 10); runtime.LogosArenaInChunk(p) {
			t.Fatal("nil-handle MakeSlice placed in a chunk")
		}
		old := []int{11, 22}
		grown, _ := runtime.LogosRegionGrowSlice(nil, 0, unsafe.Pointer(&old[0]), 3, 2, 1, int(0))
		if runtime.LogosArenaInChunk(grown) {
			t.Fatal("nil-handle GrowSlice placed in a chunk")
		}
		got := unsafe.Slice((*int)(grown), 3)
		if got[0] != 11 || got[1] != 22 || got[2] != 0 {
			t.Fatalf("nil-handle GrowSlice = %v, want [11 22 0]", got)
		}
		if z := runtime.LogosRegionAllocNoscan(nil, 0, 0, 8); z == nil {
			t.Fatal("zero-size allocation returned nil")
		}
		h, gen := runtime.LogosRegionOpen()
		var wg sync.WaitGroup
		for range 8 {
			wg.Go(func() {
				for range 100 {
					if q := runtime.LogosRegionAllocNoscan(h, gen, 8, 8); runtime.LogosArenaInChunk(q) {
						t.Error("foreign handle placed in a chunk")
						return
					}
					if b := runtime.LogosRegionBump(h, gen, 8); b != nil {
						t.Error("foreign bump returned region memory")
						return
					}
					runtime.Gosched()
				}
			})
		}
		for range 100 {
			p := runtime.LogosRegionAllocNoscan(h, gen, 8, 8)
			if !runtime.LogosArenaInChunk(p) {
				t.Fatal("owner allocation fell back to the heap")
			}
			fillBytes(p, 8, 0x9C)
			runtime.Gosched()
		}
		wg.Wait()
		runtime.LogosRegionClose(h, gen)
		if got := runtime.LogosRegionBytes(); got != 0 {
			t.Fatalf("RegionBytes = %d after Close, want 0", got)
		}
	})
}
func TestLogosRegionAdvChunkLifetimeSizes(t *testing.T) {
	inFreshGoroutine(t, func() {
		maxBytes := uintptr(runtime.LogosRegionMaxBytes)
		capacity := uintptr(runtime.LogosArenaCapacity)
		for _, size := range []uintptr{maxBytes - 1, maxBytes} {
			h, gen := runtime.LogosRegionOpen()
			p := runtime.LogosRegionAllocNoscan(h, gen, size, 8)
			if !runtime.LogosArenaInChunk(p) {
				t.Fatalf("size %d did not place in a chunk", size)
			}
			fillBytes(p, size, 0xA1)
			if !checkBytes(p, size, 0xA1) {
				t.Fatalf("size %d lost its value", size)
			}
			runtime.LogosRegionClose(h, gen)
		}
		h, gen := runtime.LogosRegionOpen()
		p := runtime.LogosRegionAllocNoscan(h, gen, maxBytes+1, 8)
		if runtime.LogosArenaInChunk(p) {
			t.Fatal("MaxBytes+1 placed in a chunk")
		}
		if !checkBytes(p, maxBytes+1, 0) {
			t.Fatal("heap fallback memory is not zero")
		}
		z := runtime.LogosRegionAllocNoscan(h, gen, 0, 8)
		if z == nil {
			t.Fatal("zero-size allocation returned nil")
		}
		if got := runtime.LogosRegionBytes(); got != 0 {
			t.Fatalf("RegionBytes = %d during idle region, want 0", got)
		}
		runtime.LogosRegionClose(h, gen)
		h2, gen2 := runtime.LogosRegionOpen()
		first := maxBytes
		second := capacity - maxBytes
		pa := runtime.LogosRegionAllocNoscan(h2, gen2, first, 8)
		fillBytes(pa, first, 0xB2)
		pb := runtime.LogosRegionAllocNoscan(h2, gen2, second, 8)
		fillBytes(pb, second, 0xC3)
		pc := runtime.LogosRegionAllocNoscan(h2, gen2, 1, 8)
		fillBytes(pc, 1, 0xD4)
		if n := len(runtime.LogosRegionChunks()); n != 2 {
			t.Fatalf("region chunks = %d, want 2 after the 1-byte overflow", n)
		}
		if !checkBytes(pa, first, 0xB2) || !checkBytes(pb, second, 0xC3) || !checkBytes(pc, 1, 0xD4) {
			t.Fatal("objects lost their values across the chunk link")
		}
		runtime.LogosRegionClose(h2, gen2)
	})
}

func TestLogosRegionAdvCloseOnceThrows(t *testing.T) {
	if os.Getenv("GOLOGOS_REGION_ADV_THROW") == "double" {
		h, gen := runtime.LogosRegionOpen()
		runtime.LogosRegionClose(h, gen)
		runtime.LogosRegionClose(h, gen)
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestLogosRegionAdvCloseOnceThrows$", "-test.timeout=60s")
	cmd.Env = append(os.Environ(), "GOLOGOS_REGION_ADV_THROW=double")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("double Close did not throw: %s", out)
	}
	if !strings.Contains(string(out), "logosRegionClose") {
		t.Fatalf("double Close threw an unrelated error: %s", out)
	}
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		runtime.LogosRegionAllocNoscan(h, gen, 64, 8)
		runtime.LogosRegionClose(h, gen)
		stale := runtime.LogosRegionAllocNoscan(h, gen, 64, 8)
		if runtime.LogosArenaInChunk(stale) {
			t.Fatal("stale-gen handle placed in a chunk after Close")
		}
		if got := runtime.LogosRegionBytes(); got != 0 {
			t.Fatalf("RegionBytes = %d after Close, want 0", got)
		}
		runtime.LogosRegionClose(nil, 0)
	})
}

func TestLogosRegionAdvRestoreOrderThrows(t *testing.T) {
	if os.Getenv("GOLOGOS_REGION_ADV_THROW") == "order" {
		h1, g1 := runtime.LogosRegionOpen()
		h2, g2 := runtime.LogosRegionOpen()
		runtime.LogosRegionClose(h1, g1)
		runtime.LogosRegionClose(h2, g2)
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestLogosRegionAdvRestoreOrderThrows$", "-test.timeout=60s")
	cmd.Env = append(os.Environ(), "GOLOGOS_REGION_ADV_THROW=order")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("out-of-order Close did not throw: %s", out)
	}
	if !strings.Contains(string(out), "logosRegionClose") {
		t.Fatalf("out-of-order Close threw an unrelated error: %s", out)
	}
	inFreshGoroutine(t, func() {
		outer, og := runtime.LogosRegionOpen()
		survivor := runtime.LogosRegionAllocNoscan(outer, og, 64, 8)
		fillBytes(survivor, 64, 0x5A)
		inner, ig := runtime.LogosRegionOpen()
		runtime.LogosRegionAllocNoscan(inner, ig, 32, 8)
		runtime.LogosRegionClose(inner, ig)
		if !checkBytes(survivor, 64, 0x5A) {
			t.Fatal("inner Close damaged the outer region")
		}
		again := runtime.LogosRegionAllocNoscan(outer, og, 16, 8)
		if !runtime.LogosArenaInChunk(again) {
			t.Fatal("outer region no longer places after inner Close")
		}
		runtime.LogosRegionClose(outer, og)
		if got := runtime.LogosRegionBytes(); got != 0 {
			t.Fatalf("RegionBytes = %d after Close, want 0", got)
		}
	})
}

func TestLogosRegionAdvScanKeepsReferents(t *testing.T) {
	inFreshGoroutine(t, func() {
		defer debug.SetGCPercent(debug.SetGCPercent(-1))
		h2, g2 := runtime.LogosRegionOpen()
		maxBytes := uintptr(runtime.LogosRegionMaxBytes)
		for _, size := range []uintptr{maxBytes - 1, maxBytes} {
			p := runtime.LogosRegionAllocScan(h2, g2, (*int)(nil), size)
			if !runtime.LogosArenaInChunk(p) {
				t.Fatalf("scan size %d did not place in a chunk", size)
			}
		}
		p := runtime.LogosRegionAllocScan(h2, g2, (*int)(nil), maxBytes+1)
		if runtime.LogosArenaInChunk(p) {
			t.Fatal("scan MaxBytes+1 placed in a chunk")
		}
		runtime.LogosRegionClose(h2, g2)
		runtime.GC()
		h, gen := runtime.LogosRegionOpen()
		done := make(chan struct{})
		obj := new(int)
		*obj = 12345
		runtime.AddCleanup(obj, func(c chan struct{}) { close(c) }, done)
		sp := runtime.LogosRegionAllocScan(h, gen, (*int)(nil), 64)
		for i := range 64 {
			*(*byte)(unsafe.Pointer(uintptr(sp) + uintptr(i))) = 0xAA
		}
		*(**int)(sp) = obj
		runtime.GC()
		runtime.GC()
		select {
		case <-done:
			t.Fatal("cleanup ran while the region still held the only reference")
		case <-time.After(200 * time.Millisecond):
		}
		if *(**int)(sp) != obj {
			t.Fatal("scan object lost its heap referent across GC")
		}
		runtime.LogosRegionClose(h, gen)
		if !checkBytes(sp, 64, 0) {
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

func TestLogosRegionAdvParkKeepsLive(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		live := runtime.LogosRegionAllocNoscan(h, gen, 64, 8)
		fillBytes(live, 64, 0xA5)
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
		oldMax := runtime.GOMAXPROCS(1)
		defer runtime.GOMAXPROCS(oldMax)
		ready := make(chan struct{})
		release := make(chan struct{})
		go func() {
			select {
			case <-ready:
			case <-time.After(20 * time.Second):
				return
			}
			close(release)
		}()
		close(ready)
		select {
		case <-release:
		case <-time.After(20 * time.Second):
			t.Fatal("park release did not arrive")
		}
		deadline := time.Now().Add(20 * time.Second)
		for {
			chunks = runtime.LogosRegionChunks()
			if len(chunks) == 1 {
				break
			}
			if time.Now().After(deadline) {
				t.Fatalf("region chunks after park = %d, want 1 (the live chunk)", len(chunks))
			}
			time.Sleep(time.Millisecond)
		}
		if !checkBytes(live, 64, 0xA5) {
			t.Fatal("live region object moved or changed over park")
		}
		if got := runtime.LogosRegionBytes(); got != wantBytes {
			t.Fatalf("RegionBytes = %d after park, want %d", got, wantBytes)
		}
		for range 100 {
			tmp := runtime.LogosRegionAllocNoscan(h, gen, 16, 8)
			fillBytes(tmp, 16, 0x1B)
			runtime.Gosched()
		}
		if !checkBytes(live, 64, 0xA5) {
			t.Fatal("live region object changed under preemption")
		}
		runtime.LogosRegionClose(h, gen)
	})
}

func TestLogosRegionAdvExitReleasesChunks(t *testing.T) {
	if os.Getenv("GOLOGOS_REGION_ADV_THROW") == "exitclose" {
		type handle struct {
			h   unsafe.Pointer
			gen uintptr
		}
		ch := make(chan handle, 1)
		exited := make(chan struct{})
		go func() {
			defer close(exited)
			h, gen := runtime.LogosRegionOpen()
			runtime.LogosRegionAllocNoscan(h, gen, 64, 8)
			ch <- handle{h, gen}
		}()
		m := <-ch
		<-exited
		runtime.LogosRegionClose(m.h, m.gen)
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestLogosRegionAdvExitReleasesChunks$", "-test.timeout=60s")
	cmd.Env = append(os.Environ(), "GOLOGOS_REGION_ADV_THROW=exitclose")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("Close after goroutine exit did not throw: %s", out)
	}
	if !strings.Contains(string(out), "logosRegionClose") {
		t.Fatalf("Close after exit threw an unrelated error: %s", out)
	}
	oldMax := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(oldMax)
	total := runtime.LogosRegionChunkTotal()
	type handle struct {
		h   unsafe.Pointer
		gen uintptr
	}
	ch := make(chan handle, 1)
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		h, gen := runtime.LogosRegionOpen()
		runtime.LogosRegionAllocNoscan(h, gen, 4096, 8)
		ch <- handle{h, gen}
	}()
	stale := <-ch
	<-exited
	deadline := time.Now().Add(20 * time.Second)
	for runtime.LogosRegionChunkTotal() != total {
		if time.Now().After(deadline) {
			t.Fatalf("region chunks after goroutine exit = %d, want %d", runtime.LogosRegionChunkTotal(), total)
		}
		time.Sleep(time.Millisecond)
	}
	var held []handle
	reused := false
	for range 128 {
		h2, gen2 := runtime.LogosRegionOpen()
		if h2 == stale.h {
			reused = true
			if gen2 == stale.gen {
				t.Fatal("exited region record reused with an unchanged generation")
			}
			runtime.LogosRegionClose(h2, gen2)
			break
		}
		held = append(held, handle{h2, gen2})
	}
	for _, m := range held {
		runtime.LogosRegionClose(m.h, m.gen)
	}
	if !reused {
		t.Fatal("exited region record was never recycled")
	}
	foreign := runtime.LogosRegionAllocNoscan(stale.h, stale.gen, 64, 8)
	if runtime.LogosArenaInChunk(foreign) {
		t.Fatal("handle from an exited goroutine placed in a chunk")
	}
	if got := runtime.LogosRegionBytes(); got != 0 {
		t.Fatalf("RegionBytes = %d, want 0", got)
	}
}
