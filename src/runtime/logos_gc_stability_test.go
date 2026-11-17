// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime_test

import (
	"runtime"
	"runtime/debug"
	"testing"
	"unsafe"
	"weak"
)

func TestLogosRegionReleaseOwnership(t *testing.T) {
	oldProcs := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(oldProcs)

	oldGC := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(oldGC)
	oldLimit := debug.SetMemoryLimit(1<<63 - 1)
	defer debug.SetMemoryLimit(oldLimit)
	runtime.GC()

	const (
		rounds          = 32
		chunks          = 128
		samplesPerClose = 16
	)

	type preparedRegion struct {
		goid   uint64
		placed bool
	}
	prepared := make(chan preparedRegion)
	closeNow := make(chan struct{})
	stop := make(chan struct{})
	done := make(chan struct{})
	completed := 0

	go func() {
		defer close(done)
		goid := runtime.LogosArenaGoid()
		for range rounds {
			select {
			case <-stop:
				return
			default:
			}

			h, gen := runtime.LogosRegionOpen()
			placed := h != nil
			for i := 0; placed && i < chunks; i++ {
				p := runtime.LogosRegionAllocScan(
					h, gen, (*byte)(nil),
					uintptr(runtime.LogosRegionMaxBytes),
				)
				placed = runtime.LogosArenaInChunk(p)
			}

			select {
			case prepared <- preparedRegion{goid: goid, placed: placed}:
			case <-stop:
				runtime.LogosRegionClose(h, gen)
				return
			}
			select {
			case <-closeNow:
			case <-stop:
				runtime.LogosRegionClose(h, gen)
				return
			}

			runtime.LogosRegionClose(h, gen)
			completed++
		}
	}()

	joined := false
	defer func() {
		if !joined {
			close(stop)
			<-done
		}
	}()

	var pooled uintptr
	var missingControl, failedPlacement bool
roundLoop:
	for round := range rounds {
		setup := <-prepared
		if !setup.placed {
			failedPlacement = true
			break
		}
		if round == 0 {
			var roots int
			pooled, roots = runtime.LogosRegionPooledRoot(setup.goid)
			if pooled != 0 {
				break
			}
			if roots == 0 {
				missingControl = true
				break
			}
		}

		closeNow <- struct{}{}
		for range samplesPerClose {
			pooled, _ = runtime.LogosRegionPooledRoot(setup.goid)
			if pooled != 0 {
				break roundLoop
			}
			runtime.Gosched()
		}
	}

	// Join before reporting failures or restoring global runtime settings.
	// The worker closes any prepared region when stop is closed.
	close(stop)
	<-done
	joined = true

	if failedPlacement {
		t.Fatal("ownership fixture could not place its scan objects in region chunks")
	}
	if missingControl {
		t.Fatal("ownership fixture did not expose its prepared region root")
	}
	if pooled != 0 {
		t.Fatalf("published region root %#x is also in a reusable chunk pool", pooled)
	}
	if completed != rounds {
		t.Fatalf("completed %d ordinary region closes, want %d", completed, rounds)
	}
}

type logosGCStabilityLeaf struct {
	value int
	pad   [8]uintptr // Keep leaves out of the tiny allocator.
}

type logosGCStabilitySmall struct {
	child *logosGCStabilityLeaf
	pad   [16]byte // Reach the inline-mark size range on every pointer width.
}

type logosGCStabilityLarge struct {
	child *logosGCStabilityLeaf
	pad   [1024]byte
}

type logosGCStabilityRoots struct {
	self  *logosGCStabilityRoots
	small *logosGCStabilitySmall
	large *logosGCStabilityLarge
}

type logosGCStabilityRefs struct {
	small     weak.Pointer[logosGCStabilitySmall]
	large     weak.Pointer[logosGCStabilityLarge]
	smallLeaf weak.Pointer[logosGCStabilityLeaf]
	largeLeaf weak.Pointer[logosGCStabilityLeaf]
}

//go:noinline
func makeLogosGCStabilityGraph(alloc func(any, uintptr) unsafe.Pointer) (
	roots [3]unsafe.Pointer,
	refs [3]logosGCStabilityRefs,
	valid bool,
) {
	const objectSize = unsafe.Sizeof(logosGCStabilityRoots{})
	// Each allocation occupies more than half a chunk, so the three
	// allocations require distinct chunks. Unused records remain nil.
	const payload = (runtime.LogosArenaChunkBytes/2/objectSize + 1) * objectSize

	for i := range roots {
		p := alloc(logosGCStabilityRoots{}, payload)
		roots[i] = p
		if !runtime.LogosArenaInChunk(p) {
			return roots, refs, false
		}

		smallLeaf := &logosGCStabilityLeaf{value: i + 1}
		largeLeaf := &logosGCStabilityLeaf{value: i + 101}
		small := &logosGCStabilitySmall{child: smallLeaf}
		large := &logosGCStabilityLarge{child: largeLeaf}
		smallHeap, smallInline := runtime.LogosGCHeapPath(unsafe.Pointer(small))
		largeHeap, largeInline := runtime.LogosGCHeapPath(unsafe.Pointer(large))
		smallLeafHeap, _ := runtime.LogosGCHeapPath(unsafe.Pointer(smallLeaf))
		largeLeafHeap, _ := runtime.LogosGCHeapPath(unsafe.Pointer(largeLeaf))
		if !smallHeap || !largeHeap || !smallLeafHeap || !largeLeafHeap ||
			smallInline != runtime.LogosGCGreenTea || largeInline {
			return roots, refs, false
		}
		root := (*logosGCStabilityRoots)(p)
		root.self = root
		root.small = small
		root.large = large

		refs[i] = logosGCStabilityRefs{
			small:     weak.Make(small),
			large:     weak.Make(large),
			smallLeaf: weak.Make(smallLeaf),
			largeLeaf: weak.Make(largeLeaf),
		}
	}
	return roots, refs, true
}

// Keep strong pointers obtained from weak references out of the caller's
// frame before the next collection.
//
//go:noinline
func checkLogosGCStabilityGraph(t *testing.T, refs [3]logosGCStabilityRefs, live bool) {
	t.Helper()
	for i, ref := range refs {
		small := ref.small.Value()
		large := ref.large.Value()
		smallLeaf := ref.smallLeaf.Value()
		largeLeaf := ref.largeLeaf.Value()
		if (small != nil) != live || (large != nil) != live ||
			(smallLeaf != nil) != live || (largeLeaf != nil) != live {
			t.Errorf(
				"graph %d liveness: small=%t large=%t smallLeaf=%t largeLeaf=%t, want all %t",
				i, small != nil, large != nil,
				smallLeaf != nil, largeLeaf != nil, live,
			)
		}
		if live && smallLeaf != nil && smallLeaf.value != i+1 {
			t.Errorf("graph %d small leaf value = %d, want %d", i, smallLeaf.value, i+1)
		}
		if live && largeLeaf != nil && largeLeaf.value != i+101 {
			t.Errorf("graph %d large leaf value = %d, want %d", i, largeLeaf.value, i+101)
		}
	}
}

func TestLogosGCHeapGraphLiveness(t *testing.T) {
	oldGC := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(oldGC)

	for _, mode := range []string{"arena", "region"} {
		t.Run(mode, func(t *testing.T) {
			inFreshGoroutine(t, func() {
				runtime.GC()
				mark := runtime.LogosArenaSave()
				alloc := runtime.LogosArenaAllocScan
				release := func() { runtime.LogosArenaRestore(mark) }
				if mode == "region" {
					h, gen := runtime.LogosRegionOpen()
					alloc = func(elem any, size uintptr) unsafe.Pointer {
						return runtime.LogosRegionAllocScan(h, gen, elem, size)
					}
					release = func() { runtime.LogosRegionClose(h, gen) }
				}

				released := false
				defer func() {
					if !released {
						release()
					}
				}()

				roots, refs, valid := makeLogosGCStabilityGraph(alloc)
				if !valid {
					t.Fatal("heap graph fixture did not reach the required manual and heap scan paths")
				}

				runtime.GC()
				runtime.GC()
				checkLogosGCStabilityGraph(t, refs, true)

				runtime.KeepAlive(roots)
				release()
				released = true

				runtime.GC()
				runtime.GC()
				checkLogosGCStabilityGraph(t, refs, false)
			})
		})
	}
}
