// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"testing"
)

func TestDeepCompletionIsHighScore(t *testing.T) {
	// Test that deepCompletionState.isHighScore properly tracks the top
	// N=MaxDeepCompletions scores.

	var s deepCompletionState

	if !s.isHighScore(1) {
		// No other scores yet, anything is a winner.
		t.Error("1 should be high score")
	}

	// Fill up with higher scores.
	for i := 0; i < MaxDeepCompletions; i++ {
		if !s.isHighScore(10) {
			t.Error("10 should be high score")
		}
	}

	// High scores should be filled with 10s so 2 is not a high score.
	if s.isHighScore(2) {
		t.Error("2 shouldn't be high score")
	}
}
