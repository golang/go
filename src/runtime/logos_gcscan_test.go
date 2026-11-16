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

type logosGCScanReferent struct {
	value int
	pad   [8]uintptr // Keep these objects out of the tiny allocator.
}

type logosGCScanNode struct {
	self     *logosGCScanNode
	scalar   [70]uintptr // Put later pointers in a second bitmap word.
	interior unsafe.Pointer
	referent *logosGCScanReferent
	tail     [5]uintptr
}

type logosGCScanRefs [3]weak.Pointer[logosGCScanReferent]

//go:noinline
func makeLogosGCScanNodes(alloc func(any, uintptr) unsafe.Pointer) (p unsafe.Pointer, live, scalar logosGCScanRefs) {
	const count = len(logosGCScanRefs{})
	size := uintptr(count) * unsafe.Sizeof(logosGCScanNode{})
	p = alloc(logosGCScanNode{}, size)
	nodes := unsafe.Slice((*logosGCScanNode)(p), count)
	for i := range nodes {
		nodes[i].self = &nodes[i]
		nodes[i].interior = unsafe.Add(p, size-1)
		obj := &logosGCScanReferent{value: i + 1}
		nodes[i].referent = obj
		live[i] = weak.Make(obj)
		dead := &logosGCScanReferent{value: -i - 1}
		nodes[i].scalar[0] = uintptr(unsafe.Pointer(dead))
		scalar[i] = weak.Make(dead)
	}
	return
}

//go:noinline
func checkLogosGCScanLive(t *testing.T, refs logosGCScanRefs) {
	t.Helper()
	for i, ref := range refs {
		obj := ref.Value()
		if obj == nil {
			t.Fatalf("referent %d was collected while its scan object was live", i)
		}
		if obj.value != i+1 {
			t.Fatalf("referent %d value = %d, want %d", i, obj.value, i+1)
		}
	}
}

func TestLogosGCScanExternalPointers(t *testing.T) {
	for _, mode := range []string{"arena", "region"} {
		t.Run(mode, func(t *testing.T) {
			inFreshGoroutine(t, func() {
				defer debug.SetGCPercent(debug.SetGCPercent(-1))
				runtime.GC()
				mark := runtime.LogosArenaSave()
				alloc := runtime.LogosArenaAllocScan
				closeRegion := func() { runtime.LogosArenaRestore(mark) }
				if mode == "region" {
					h, gen := runtime.LogosRegionOpen()
					alloc = func(elem any, size uintptr) unsafe.Pointer {
						return runtime.LogosRegionAllocScan(h, gen, elem, size)
					}
					closeRegion = func() { runtime.LogosRegionClose(h, gen) }
				}
				p, live, scalar := makeLogosGCScanNodes(alloc)
				if !runtime.LogosArenaInChunk(p) {
					closeRegion()
					t.Fatal("scan fixture fell back to the heap")
				}
				runtime.GC()
				runtime.GC()
				checkLogosGCScanLive(t, live)
				for i, ref := range scalar {
					if ref.Value() != nil {
						t.Errorf("scalar word %d kept an unreachable heap object alive", i)
					}
				}
				runtime.KeepAlive(p)
				closeRegion()
				runtime.GC()
				runtime.GC()
				for i, ref := range live {
					if ref.Value() != nil {
						t.Errorf("referent %d survived after its scan object was released", i)
					}
				}
			})
		})
	}
}

func BenchmarkLogosGCScan(b *testing.B) {
	for _, mode := range []string{"Self", "Nil", "External"} {
		b.Run(mode, func(b *testing.B) {
			defer debug.SetGCPercent(debug.SetGCPercent(-1))
			runtime.GC()
			mark := runtime.LogosArenaSave()
			defer runtime.LogosArenaRestore(mark)
			const words = 512
			const size = words * unsafe.Sizeof(unsafe.Pointer(nil))
			p := runtime.LogosArenaAllocScan(unsafe.Pointer(nil), size)
			if !runtime.LogosArenaInChunk(p) {
				b.Fatal("scan benchmark fell back to the heap")
			}
			ptrs := unsafe.Slice((*unsafe.Pointer)(p), words)
			switch mode {
			case "Self":
				for i := range ptrs {
					ptrs[i] = unsafe.Add(p, uintptr(i)*unsafe.Sizeof(unsafe.Pointer(nil)))
				}
			case "External":
				external := runtime.LogosArenaAlloc(byte(0), 64)
				if !runtime.LogosArenaInChunk(external) {
					b.Fatal("external benchmark target fell back to the heap")
				}
				for i := range ptrs {
					ptrs[i] = external
				}
			}
			b.SetBytes(int64(size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				runtime.LogosArenaScanPointerBlock(p, size)
			}
			b.StopTimer()
			runtime.KeepAlive(p)
		})
	}
}
