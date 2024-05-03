// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.swissmap

package runtime_test

import (
	"internal/abi"
	"internal/goarch"
	"internal/runtime/maps"
	"slices"
	"testing"
	"unsafe"
)

func TestHmapSize(t *testing.T) {
	// The structure of Map is defined in internal/runtime/maps/map.go
	// and in cmd/compile/internal/reflectdata/map_swiss.go and must be in sync.
	// The size of Map should be 72 bytes on 64 bit and 56 bytes on 32 bit platforms.
	wantSize := uintptr(4*goarch.PtrSize + 5*8)
	gotSize := unsafe.Sizeof(maps.Map{})
	if gotSize != wantSize {
		t.Errorf("sizeof(maps.Map{})==%d, want %d", gotSize, wantSize)
	}
}

// See also reflect_test.TestGroupSizeZero.
func TestGroupSizeZero(t *testing.T) {
	var m map[struct{}]struct{}
	mTyp := abi.TypeOf(m)
	mt := (*abi.SwissMapType)(unsafe.Pointer(mTyp))

	// internal/runtime/maps when create pointers to slots, even if slots
	// are size 0. The compiler should have reserved an extra word to
	// ensure that pointers to the zero-size type at the end of group are
	// valid.
	if mt.Group.Size() <= 8 {
		t.Errorf("Group size got %d want >8", mt.Group.Size())
	}
}

func TestMapIterOrder(t *testing.T) {
	sizes := []int{3, 7, 9, 15}
	for _, n := range sizes {
		for i := 0; i < 1000; i++ {
			// Make m be {0: true, 1: true, ..., n-1: true}.
			m := make(map[int]bool)
			for i := 0; i < n; i++ {
				m[i] = true
			}
			// Check that iterating over the map produces at least two different orderings.
			ord := func() []int {
				var s []int
				for key := range m {
					s = append(s, key)
				}
				return s
			}
			first := ord()
			ok := false
			for try := 0; try < 100; try++ {
				if !slices.Equal(first, ord()) {
					ok = true
					break
				}
			}
			if !ok {
				t.Errorf("Map with n=%d elements had consistent iteration order: %v", n, first)
				break
			}
		}
	}
}

func TestMapBuckets(t *testing.T) {
	t.Skipf("todo")
}
