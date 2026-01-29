// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"internal/asan"
	"internal/goexperiment"
	"internal/msan"
	"internal/race"
	"testing"
)

func TestFreeAppendAllocations(t *testing.T) {
	t.Run("slice-no-alias", func(t *testing.T) {
		if !goexperiment.RuntimeFreegc {
			t.Skip("skipping allocation test when runtime.freegc is disabled")
		}
		if race.Enabled || msan.Enabled || asan.Enabled {
			// TODO(thepudds): we get 8 allocs for slice-no-alias instead of 1 with -race. This
			// might be expected given some allocation optimizations are already disabled
			// under race, but if not, we might need to update walk.
			t.Skip("skipping allocation test under race detector and other sanitizers")
		}

		allocs := testing.AllocsPerRun(100, func() {
			var s []int64
			for i := range 100 {
				s = append(s, int64(i))
			}
			_ = s
		})
		t.Logf("allocs: %v", allocs)
		if allocs != 1 {
			t.Errorf("allocs: %v, want 1", allocs)
		}
	})

	t.Run("slice-aliased", func(t *testing.T) {
		allocs := testing.AllocsPerRun(100, func() {
			var s []int64
			var alias []int64
			for i := range 100 {
				s = append(s, int64(i))
				alias = s
			}
			_ = alias
		})
		t.Logf("allocs: %v", allocs)
		if allocs < 2 {
			t.Errorf("allocs: %v, want >= 2", allocs)
		}
	})
}
