// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests of map internals that need to use the builtin map type, and thus must
// be built with GOEXPERIMENT=swissmap.

//go:build goexperiment.swissmap

package maps_test

import (
	"fmt"
	"internal/abi"
	"internal/runtime/maps"
	"testing"
	"unsafe"
)

var alwaysFalse bool
var escapeSink any

func escape[T any](x T) T {
	if alwaysFalse {
		escapeSink = x
	}
	return x
}

const (
	belowMax = abi.SwissMapGroupSlots * 3 / 2                                               // 1.5 * group max = 2 groups @ 75%
	atMax    = (2 * abi.SwissMapGroupSlots * maps.MaxAvgGroupLoad) / abi.SwissMapGroupSlots // 2 groups at 7/8 full.
)

func TestTableGroupCount(t *testing.T) {
	// Test that maps of different sizes have the right number of
	// tables/groups.

	type mapCount struct {
		tables int
		groups uint64
	}

	type mapCase struct {
		initialLit  mapCount
		initialHint mapCount
		after       mapCount
	}

	var testCases = []struct {
		n      int     // n is the number of map elements
		escape mapCase // expected values for escaping map
	}{
		{
			n: -(1 << 30),
			escape: mapCase{
				initialLit:  mapCount{0, 0},
				initialHint: mapCount{0, 0},
				after:       mapCount{0, 0},
			},
		},
		{
			n: -1,
			escape: mapCase{
				initialLit:  mapCount{0, 0},
				initialHint: mapCount{0, 0},
				after:       mapCount{0, 0},
			},
		},
		{
			n: 0,
			escape: mapCase{
				initialLit:  mapCount{0, 0},
				initialHint: mapCount{0, 0},
				after:       mapCount{0, 0},
			},
		},
		{
			n: 1,
			escape: mapCase{
				initialLit:  mapCount{0, 0},
				initialHint: mapCount{0, 0},
				after:       mapCount{0, 1},
			},
		},
		{
			n: abi.SwissMapGroupSlots,
			escape: mapCase{
				initialLit:  mapCount{0, 0},
				initialHint: mapCount{0, 0},
				after:       mapCount{0, 1},
			},
		},
		{
			n: abi.SwissMapGroupSlots + 1,
			escape: mapCase{
				initialLit:  mapCount{0, 0},
				initialHint: mapCount{1, 2},
				after:       mapCount{1, 2},
			},
		},
		{
			n: belowMax, // 1.5 group max = 2 groups @ 75%
			escape: mapCase{
				initialLit:  mapCount{0, 0},
				initialHint: mapCount{1, 2},
				after:       mapCount{1, 2},
			},
		},
		{
			n: atMax, // 2 groups at max
			escape: mapCase{
				initialLit:  mapCount{0, 0},
				initialHint: mapCount{1, 2},
				after:       mapCount{1, 2},
			},
		},
		{
			n: atMax + 1, // 2 groups at max + 1 -> grow to 4 groups
			escape: mapCase{
				initialLit:  mapCount{0, 0},
				initialHint: mapCount{1, 4},
				after:       mapCount{1, 4},
			},
		},
		{
			n: 2 * belowMax, // 3 * group max = 4 groups @75%
			escape: mapCase{
				initialLit:  mapCount{0, 0},
				initialHint: mapCount{1, 4},
				after:       mapCount{1, 4},
			},
		},
		{
			n: 2*atMax + 1, // 4 groups at max + 1 -> grow to 8 groups
			escape: mapCase{
				initialLit:  mapCount{0, 0},
				initialHint: mapCount{1, 8},
				after:       mapCount{1, 8},
			},
		},
	}

	testMap := func(t *testing.T, m map[int]int, n int, initial, after mapCount) {
		mm := *(**maps.Map)(unsafe.Pointer(&m))

		gotTab := mm.TableCount()
		if gotTab != initial.tables {
			t.Errorf("initial TableCount got %d want %d", gotTab, initial.tables)
		}

		gotGroup := mm.GroupCount()
		if gotGroup != initial.groups {
			t.Errorf("initial GroupCount got %d want %d", gotGroup, initial.groups)
		}

		for i := 0; i < n; i++ {
			m[i] = i
		}

		gotTab = mm.TableCount()
		if gotTab != after.tables {
			t.Errorf("after TableCount got %d want %d", gotTab, after.tables)
		}

		gotGroup = mm.GroupCount()
		if gotGroup != after.groups {
			t.Errorf("after GroupCount got %d want %d", gotGroup, after.groups)
		}
	}

	t.Run("mapliteral", func(t *testing.T) {
		for _, tc := range testCases {
			t.Run(fmt.Sprintf("n=%d", tc.n), func(t *testing.T) {
				t.Run("escape", func(t *testing.T) {
					m := escape(map[int]int{})
					testMap(t, m, tc.n, tc.escape.initialLit, tc.escape.after)
				})
			})
		}
	})
	t.Run("nohint", func(t *testing.T) {
		for _, tc := range testCases {
			t.Run(fmt.Sprintf("n=%d", tc.n), func(t *testing.T) {
				t.Run("escape", func(t *testing.T) {
					m := escape(make(map[int]int))
					testMap(t, m, tc.n, tc.escape.initialLit, tc.escape.after)
				})
			})
		}
	})
	t.Run("makemap", func(t *testing.T) {
		for _, tc := range testCases {
			t.Run(fmt.Sprintf("n=%d", tc.n), func(t *testing.T) {
				t.Run("escape", func(t *testing.T) {
					m := escape(make(map[int]int, tc.n))
					testMap(t, m, tc.n, tc.escape.initialHint, tc.escape.after)
				})
			})
		}
	})
	t.Run("makemap64", func(t *testing.T) {
		for _, tc := range testCases {
			t.Run(fmt.Sprintf("n=%d", tc.n), func(t *testing.T) {
				t.Run("escape", func(t *testing.T) {
					m := escape(make(map[int]int, int64(tc.n)))
					testMap(t, m, tc.n, tc.escape.initialHint, tc.escape.after)
				})
			})
		}
	})
}
