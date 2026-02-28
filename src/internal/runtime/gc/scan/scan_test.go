// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scan_test

import (
	"fmt"
	"internal/cpu"
	"internal/goarch"
	"internal/runtime/gc"
	"internal/runtime/gc/scan"
	"math/bits"
	"math/rand/v2"
	"slices"
	"sync"
	"testing"
	"unsafe"
)

type scanFunc func(mem unsafe.Pointer, bufp *uintptr, objMarks *gc.ObjMask, sizeClass uintptr, ptrMask *gc.PtrMask) (count int32)

func testScanSpanPacked(t *testing.T, scanF scanFunc) {
	scanR := scan.ScanSpanPackedReference

	// Construct a fake memory
	mem, free := makeMem(t, 1)
	defer free()
	for i := range mem {
		// Use values > heap.PageSize because a scan function can discard
		// pointers smaller than this.
		mem[i] = uintptr(int(gc.PageSize) + i + 1)
	}

	// Construct a random pointer mask
	rnd := rand.New(rand.NewPCG(42, 42))
	var ptrs gc.PtrMask
	for i := range ptrs {
		ptrs[i] = uintptr(rnd.Uint64())
	}

	bufF := make([]uintptr, gc.PageWords)
	bufR := make([]uintptr, gc.PageWords)
	testObjs(t, func(t *testing.T, sizeClass int, objs *gc.ObjMask) {
		nF := scanF(unsafe.Pointer(&mem[0]), &bufF[0], objs, uintptr(sizeClass), &ptrs)
		nR := scanR(unsafe.Pointer(&mem[0]), &bufR[0], objs, uintptr(sizeClass), &ptrs)

		if nR != nF {
			t.Errorf("want %d count, got %d", nR, nF)
		} else if !slices.Equal(bufF[:nF], bufR[:nR]) {
			t.Errorf("want scanned pointers %d, got %d", bufR[:nR], bufF[:nF])
		}
	})
}

func testObjs(t *testing.T, f func(t *testing.T, sizeClass int, objMask *gc.ObjMask)) {
	for sizeClass := range gc.NumSizeClasses {
		if sizeClass == 0 {
			continue
		}
		size := uintptr(gc.SizeClassToSize[sizeClass])
		if size > gc.MinSizeForMallocHeader {
			break // Pointer/scalar metadata is not packed for larger sizes.
		}
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			// Scan a few objects near i to test boundary conditions.
			const objMask = 0x101
			nObj := uintptr(gc.SizeClassToNPages[sizeClass]) * gc.PageSize / size
			for i := range nObj - uintptr(bits.Len(objMask)-1) {
				t.Run(fmt.Sprintf("objs=0x%x<<%d", objMask, i), func(t *testing.T) {
					var objs gc.ObjMask
					objs[i/goarch.PtrBits] = objMask << (i % goarch.PtrBits)
					f(t, sizeClass, &objs)
				})
			}
		})
	}
}

var dataCacheSizes = sync.OnceValue(func() []uintptr {
	cs := cpu.DataCacheSizes()
	for i, c := range cs {
		fmt.Printf("# L%d cache: %d (%d Go pages)\n", i+1, c, c/gc.PageSize)
	}
	return cs
})

func BenchmarkScanSpanPacked(b *testing.B) {
	benchmarkCacheSizes(b, benchmarkScanSpanPackedAllSizeClasses)
}

func benchmarkCacheSizes(b *testing.B, fn func(b *testing.B, heapPages int)) {
	cacheSizes := dataCacheSizes()
	b.Run("cache=tiny/pages=1", func(b *testing.B) {
		fn(b, 1)
	})
	for i, cacheBytes := range cacheSizes {
		pages := int(cacheBytes*3/4) / gc.PageSize
		b.Run(fmt.Sprintf("cache=L%d/pages=%d", i+1, pages), func(b *testing.B) {
			fn(b, pages)
		})
	}
	if len(cacheSizes) == 0 {
		return
	}
	ramPages := int(cacheSizes[len(cacheSizes)-1]*3/2) / gc.PageSize
	b.Run(fmt.Sprintf("cache=ram/pages=%d", ramPages), func(b *testing.B) {
		fn(b, ramPages)
	})
}

func benchmarkScanSpanPackedAllSizeClasses(b *testing.B, nPages int) {
	for sc := range gc.NumSizeClasses {
		if sc == 0 {
			continue
		}
		size := gc.SizeClassToSize[sc]
		if size >= gc.MinSizeForMallocHeader {
			break
		}
		b.Run(fmt.Sprintf("sizeclass=%d", sc), func(b *testing.B) {
			benchmarkScanSpanPacked(b, nPages, sc)
		})
	}
}

func benchmarkScanSpanPacked(b *testing.B, nPages int, sizeClass int) {
	rnd := rand.New(rand.NewPCG(42, 42))

	// Construct a fake memory
	mem, free := makeMem(b, nPages)
	defer free()
	for i := range mem {
		// Use values > heap.PageSize because a scan function can discard
		// pointers smaller than this.
		mem[i] = uintptr(int(gc.PageSize) + i + 1)
	}

	// Construct a random pointer mask
	ptrs := make([]gc.PtrMask, nPages)
	for i := range ptrs {
		for j := range ptrs[i] {
			ptrs[i][j] = uintptr(rnd.Uint64())
		}
	}

	// Visit the pages in a random order
	pageOrder := rnd.Perm(nPages)

	// Create the scan buffer.
	buf := make([]uintptr, gc.PageWords)

	// Sweep from 0 marks to all marks. We'll use the same marks for each page
	// because I don't think that predictability matters.
	objBytes := uintptr(gc.SizeClassToSize[sizeClass])
	nObj := gc.PageSize / objBytes
	markOrder := rnd.Perm(int(nObj))
	const steps = 11
	for i := 0; i < steps; i++ {
		frac := float64(i) / float64(steps-1)
		// Set frac marks.
		nMarks := int(float64(len(markOrder))*frac + 0.5)
		var objMarks gc.ObjMask
		for _, mark := range markOrder[:nMarks] {
			objMarks[mark/goarch.PtrBits] |= 1 << (mark % goarch.PtrBits)
		}
		greyClusters := 0
		for page := range ptrs {
			greyClusters += countGreyClusters(sizeClass, &objMarks, &ptrs[page])
		}

		// Report MB/s of how much memory they're actually hitting. This assumes
		// 64 byte cache lines (TODO: Should it assume 128 byte cache lines?)
		// and expands each access to the whole cache line. This is useful for
		// comparing against memory bandwidth.
		//
		// TODO: Add a benchmark that just measures single core memory bandwidth
		// for comparison. (See runtime memcpy benchmarks.)
		//
		// TODO: Should there be a separate measure where we don't expand to
		// cache lines?
		avgBytes := int64(greyClusters) * int64(cpu.CacheLineSize) / int64(len(ptrs))

		b.Run(fmt.Sprintf("pct=%d", int(100*frac)), func(b *testing.B) {
			b.Run("impl=Reference", func(b *testing.B) {
				b.SetBytes(avgBytes)
				for i := range b.N {
					page := pageOrder[i%len(pageOrder)]
					scan.ScanSpanPackedReference(unsafe.Pointer(&mem[gc.PageWords*page]), &buf[0], &objMarks, uintptr(sizeClass), &ptrs[page])
				}
			})
			b.Run("impl=Go", func(b *testing.B) {
				b.SetBytes(avgBytes)
				for i := range b.N {
					page := pageOrder[i%len(pageOrder)]
					scan.ScanSpanPackedGo(unsafe.Pointer(&mem[gc.PageWords*page]), &buf[0], &objMarks, uintptr(sizeClass), &ptrs[page])
				}
			})
			if scan.HasFastScanSpanPacked() {
				b.Run("impl=Platform", func(b *testing.B) {
					b.SetBytes(avgBytes)
					for i := range b.N {
						page := pageOrder[i%len(pageOrder)]
						scan.ScanSpanPacked(unsafe.Pointer(&mem[gc.PageWords*page]), &buf[0], &objMarks, uintptr(sizeClass), &ptrs[page])
					}
				})
			}
		})
	}
}

func countGreyClusters(sizeClass int, objMarks *gc.ObjMask, ptrMask *gc.PtrMask) int {
	clusters := 0
	lastCluster := -1

	expandBy := uintptr(gc.SizeClassToSize[sizeClass]) / goarch.PtrSize
	for word := range gc.PageWords {
		objI := uintptr(word) / expandBy
		if objMarks[objI/goarch.PtrBits]&(1<<(objI%goarch.PtrBits)) == 0 {
			continue
		}
		if ptrMask[word/goarch.PtrBits]&(1<<(word%goarch.PtrBits)) == 0 {
			continue
		}
		c := word * 8 / goarch.PtrBits
		if c != lastCluster {
			lastCluster = c
			clusters++
		}
	}
	return clusters
}

func BenchmarkScanMaxBandwidth(b *testing.B) {
	// Measure the theoretical "maximum" bandwidth of scanning by reproducing
	// the memory access pattern of a full page scan, but using memcpy as the
	// kernel instead of scanning.
	benchmarkCacheSizes(b, func(b *testing.B, heapPages int) {
		mem, free := makeMem(b, heapPages)
		defer free()
		for i := range mem {
			mem[i] = uintptr(int(gc.PageSize) + i + 1)
		}
		buf := make([]uintptr, gc.PageWords)

		// Visit the pages in a random order
		rnd := rand.New(rand.NewPCG(42, 42))
		pageOrder := rnd.Perm(heapPages)

		b.SetBytes(int64(gc.PageSize))

		b.ResetTimer()
		for i := range b.N {
			page := pageOrder[i%len(pageOrder)]
			copy(buf, mem[gc.PageWords*page:])
		}
	})
}
