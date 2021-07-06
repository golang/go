// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"testing"
)

// Check that the partial order in lockPartialOrder fits within the total order
// determined by the order of the lockRank constants.
func TestLockRankPartialOrder(t *testing.T) {
	for r, list := range LockPartialOrder {
		rank := LockRank(r)
		for _, e := range list {
			entry := LockRank(e)
			if entry > rank {
				t.Errorf("lockPartialOrder row %v entry %v is inconsistent with total lock ranking order", rank, entry)
			}
		}
	}
}

// Verify that partial order lists are kept sorted. This is a purely cosemetic
// check to make manual reviews simpler. It does not affect correctness, unlike
// the above test.
func TestLockRankPartialOrderSortedEntries(t *testing.T) {
	for r, list := range LockPartialOrder {
		rank := LockRank(r)
		var prev LockRank
		for _, e := range list {
			entry := LockRank(e)
			if entry <= prev {
				t.Errorf("Partial order for rank %v out of order: %v <= %v in %v", rank, entry, prev, list)
			}
			prev = entry
		}
	}
}
