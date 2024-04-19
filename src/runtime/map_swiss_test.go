// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.swissmap

package runtime_test

import (
	"internal/abi"
	"runtime"
	"slices"
	"testing"
)

func TestMapIterOrder(t *testing.T) {
	sizes := []int{3, 7, 9, 15}
	if abi.SwissMapBucketCountBits >= 5 {
		// it gets flaky (often only one iteration order) at size 3 when abi.MapBucketCountBits >=5.
		t.Fatalf("This test becomes flaky if abi.MapBucketCountBits(=%d) is 5 or larger", abi.SwissMapBucketCountBits)
	}
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

const bs = abi.SwissMapBucketCount

// belowOverflow should be a pretty-full pair of buckets;
// atOverflow is 1/8 bs larger = 13/8 buckets or two buckets
// that are 13/16 full each, which is the overflow boundary.
// Adding one to that should ensure overflow to the next higher size.
const (
	belowOverflow = bs * 3 / 2           // 1.5 bs = 2 buckets @ 75%
	atOverflow    = belowOverflow + bs/8 // 2 buckets at 13/16 fill.
)

var mapBucketTests = [...]struct {
	n        int // n is the number of map elements
	noescape int // number of expected buckets for non-escaping map
	escape   int // number of expected buckets for escaping map
}{
	{-(1 << 30), 1, 1},
	{-1, 1, 1},
	{0, 1, 1},
	{1, 1, 1},
	{bs, 1, 1},
	{bs + 1, 2, 2},
	{belowOverflow, 2, 2},  // 1.5 bs = 2 buckets @ 75%
	{atOverflow + 1, 4, 4}, // 13/8 bs + 1 == overflow to 4

	{2 * belowOverflow, 4, 4}, // 3 bs = 4 buckets @75%
	{2*atOverflow + 1, 8, 8},  // 13/4 bs + 1 = overflow to 8

	{4 * belowOverflow, 8, 8},  // 6 bs = 8 buckets @ 75%
	{4*atOverflow + 1, 16, 16}, // 13/2 bs + 1 = overflow to 16
}

func TestMapBuckets(t *testing.T) {
	// Test that maps of different sizes have the right number of buckets.
	// Non-escaping maps with small buckets (like map[int]int) never
	// have a nil bucket pointer due to starting with preallocated buckets
	// on the stack. Escaping maps start with a non-nil bucket pointer if
	// hint size is above bucketCnt and thereby have more than one bucket.
	// These tests depend on bucketCnt and loadFactor* in map.go.
	t.Run("mapliteral", func(t *testing.T) {
		for _, tt := range mapBucketTests {
			localMap := map[int]int{}
			if runtime.MapBucketsPointerIsNil(localMap) {
				t.Errorf("no escape: buckets pointer is nil for non-escaping map")
			}
			for i := 0; i < tt.n; i++ {
				localMap[i] = i
			}
			if got := runtime.MapBucketsCount(localMap); got != tt.noescape {
				t.Errorf("no escape: n=%d want %d buckets, got %d", tt.n, tt.noescape, got)
			}
			escapingMap := runtime.Escape(map[int]int{})
			if count := runtime.MapBucketsCount(escapingMap); count > 1 && runtime.MapBucketsPointerIsNil(escapingMap) {
				t.Errorf("escape: buckets pointer is nil for n=%d buckets", count)
			}
			for i := 0; i < tt.n; i++ {
				escapingMap[i] = i
			}
			if got := runtime.MapBucketsCount(escapingMap); got != tt.escape {
				t.Errorf("escape n=%d want %d buckets, got %d", tt.n, tt.escape, got)
			}
		}
	})
	t.Run("nohint", func(t *testing.T) {
		for _, tt := range mapBucketTests {
			localMap := make(map[int]int)
			if runtime.MapBucketsPointerIsNil(localMap) {
				t.Errorf("no escape: buckets pointer is nil for non-escaping map")
			}
			for i := 0; i < tt.n; i++ {
				localMap[i] = i
			}
			if got := runtime.MapBucketsCount(localMap); got != tt.noescape {
				t.Errorf("no escape: n=%d want %d buckets, got %d", tt.n, tt.noescape, got)
			}
			escapingMap := runtime.Escape(make(map[int]int))
			if count := runtime.MapBucketsCount(escapingMap); count > 1 && runtime.MapBucketsPointerIsNil(escapingMap) {
				t.Errorf("escape: buckets pointer is nil for n=%d buckets", count)
			}
			for i := 0; i < tt.n; i++ {
				escapingMap[i] = i
			}
			if got := runtime.MapBucketsCount(escapingMap); got != tt.escape {
				t.Errorf("escape: n=%d want %d buckets, got %d", tt.n, tt.escape, got)
			}
		}
	})
	t.Run("makemap", func(t *testing.T) {
		for _, tt := range mapBucketTests {
			localMap := make(map[int]int, tt.n)
			if runtime.MapBucketsPointerIsNil(localMap) {
				t.Errorf("no escape: buckets pointer is nil for non-escaping map")
			}
			for i := 0; i < tt.n; i++ {
				localMap[i] = i
			}
			if got := runtime.MapBucketsCount(localMap); got != tt.noescape {
				t.Errorf("no escape: n=%d want %d buckets, got %d", tt.n, tt.noescape, got)
			}
			escapingMap := runtime.Escape(make(map[int]int, tt.n))
			if count := runtime.MapBucketsCount(escapingMap); count > 1 && runtime.MapBucketsPointerIsNil(escapingMap) {
				t.Errorf("escape: buckets pointer is nil for n=%d buckets", count)
			}
			for i := 0; i < tt.n; i++ {
				escapingMap[i] = i
			}
			if got := runtime.MapBucketsCount(escapingMap); got != tt.escape {
				t.Errorf("escape: n=%d want %d buckets, got %d", tt.n, tt.escape, got)
			}
		}
	})
	t.Run("makemap64", func(t *testing.T) {
		for _, tt := range mapBucketTests {
			localMap := make(map[int]int, int64(tt.n))
			if runtime.MapBucketsPointerIsNil(localMap) {
				t.Errorf("no escape: buckets pointer is nil for non-escaping map")
			}
			for i := 0; i < tt.n; i++ {
				localMap[i] = i
			}
			if got := runtime.MapBucketsCount(localMap); got != tt.noescape {
				t.Errorf("no escape: n=%d want %d buckets, got %d", tt.n, tt.noescape, got)
			}
			escapingMap := runtime.Escape(make(map[int]int, tt.n))
			if count := runtime.MapBucketsCount(escapingMap); count > 1 && runtime.MapBucketsPointerIsNil(escapingMap) {
				t.Errorf("escape: buckets pointer is nil for n=%d buckets", count)
			}
			for i := 0; i < tt.n; i++ {
				escapingMap[i] = i
			}
			if got := runtime.MapBucketsCount(escapingMap); got != tt.escape {
				t.Errorf("escape: n=%d want %d buckets, got %d", tt.n, tt.escape, got)
			}
		}
	})
}
