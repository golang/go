// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"testing"

	"cmd/compile/internal/logos/facts"
)

func TestMergeAllocFactsDedups(t *testing.T) {
	prev := facts.Current()
	t.Cleanup(func() { facts.SetCurrent(prev) })
	facts.SetCurrent(&facts.Facts{})
	logosMergeAllocFacts([]string{"F", "G", "F", "G", "F"})
	counts := make(map[string]int)
	for _, a := range facts.Current().AllocFrees {
		counts[a.Object]++
	}
	for _, obj := range []string{"F", "G"} {
		if counts[obj] != 1 {
			t.Errorf("allocfree records for %q = %d, want 1", obj, counts[obj])
		}
	}
}
