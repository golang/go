// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"internal/race"
	"internal/testenv"
	"testing"
)

func wantAllocs(t *testing.T, want int, f func()) {
	if race.Enabled {
		t.Skip("skipping test in race mode")
	}
	testenv.SkipIfOptimizationOff(t)
	t.Helper()
	got := int(testing.AllocsPerRun(5, f))
	if got != want {
		t.Errorf("got %d allocs, want %d", got, want)
	}
}
