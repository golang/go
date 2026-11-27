// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"strconv"
	"testing"
	"weak"
)

type batchObject [4]uint64

//go:noinline
func fillBatchObjects(dst []*batchObject) {
	for i := range dst {
		dst[i] = new(batchObject)
	}
}

func TestBatchAllocSlice(t *testing.T) {
	fillBatchObjects(nil)
	for n := range 24 {
		var outside batchObject
		dst := make([]*batchObject, n+2)
		dst[0], dst[n+1] = &outside, &outside
		fillBatchObjects(dst[1 : n+1])
		for i, p := range dst[1 : n+1] {
			if p == nil || *p != (batchObject{}) {
				t.Fatalf("length %d, index %d: allocation is absent or nonzero", n, i)
			}
			p[0] = uint64(i + 1)
		}
		for i, p := range dst[1 : n+1] {
			if p[0] != uint64(i+1) {
				t.Fatalf("length %d, index %d: allocation aliases another slot", n, i)
			}
		}
		if dst[0] != &outside || dst[n+1] != &outside {
			t.Fatalf("length %d: wrote outside slice bounds", n)
		}
	}
}

//go:noinline
func batchAllocWeakSiblings() (*batchObject, [7]weak.Pointer[batchObject]) {
	var dst [8]*batchObject
	fillBatchObjects(dst[:])
	var weakRefs [7]weak.Pointer[batchObject]
	for i, obj := range dst[1:] {
		weakRefs[i] = weak.Make(obj)
	}
	return dst[0], weakRefs
}

func TestBatchAllocSliceGC(t *testing.T) {
	survivors := make([]*batchObject, 128)
	weakRefs := make([][7]weak.Pointer[batchObject], len(survivors))
	for i := range survivors {
		survivors[i], weakRefs[i] = batchAllocWeakSiblings()
		survivors[i][0] = uint64(i + 1)
	}
	runtime.GC()
	for i, refs := range weakRefs {
		for _, ref := range refs {
			if ref.Value() != nil {
				t.Fatalf("batch %d: retaining one object kept a sibling alive", i)
			}
		}
	}
	for i, p := range survivors {
		if p[0] != uint64(i+1) {
			t.Fatalf("batch %d: retaining the first object did not preserve its data", i)
		}
	}
}

func BenchmarkFillBatchObjects(b *testing.B) {
	for _, n := range []int{1, 8, 64, 256, 4096} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			dst := make([]*batchObject, n)
			b.ReportAllocs()
			for b.Loop() {
				fillBatchObjects(dst)
			}
		})
	}
}
