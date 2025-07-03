// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/synctest"
	"runtime"
	"testing"
)

func TestSynctest(t *testing.T) {
	output := runTestProg(t, "testsynctest", "")
	want := "success\n"
	if output != want {
		t.Fatalf("output:\n%s\n\nwanted:\n%s", output, want)
	}
}

// TestSynctestAssocConsts verifies that constants defined
// in both runtime and internal/synctest match.
func TestSynctestAssocConsts(t *testing.T) {
	if runtime.BubbleAssocUnbubbled != synctest.Unbubbled ||
		runtime.BubbleAssocCurrentBubble != synctest.CurrentBubble ||
		runtime.BubbleAssocOtherBubble != synctest.OtherBubble {
		t.Fatal("mismatch: runtime.BubbleAssoc? != synctest.*")
	}
}
