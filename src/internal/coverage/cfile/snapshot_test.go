// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build SELECT_USING_THIS_TAG

package cfile

import "testing"

var funcInvoked bool

//go:noinline
func thisFunctionOnlyCalledFromSnapshotTest(n int) int {
	if funcInvoked {
		panic("bad")
	}
	funcInvoked = true

	// Contents here not especially important, just so long as we
	// have some statements.
	t := 0
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			t += i ^ j
		}
	}
	return t
}

// Tests runtime/coverage.snapshot() directly. Note that if
// coverage is not enabled, the hook is designed to just return
// zero.
func TestCoverageSnapshotImpl(t *testing.T) {
	C1 := Snapshot()
	thisFunctionOnlyCalledFromSnapshotTest(15)
	C2 := Snapshot()
	cond := "C1 > C2"
	val := C1 > C2
	if testing.CoverMode() != "" {
		cond = "C1 >= C2"
		val = C1 >= C2
	}
	t.Logf("%f %f\n", C1, C2)
	if val {
		t.Errorf("erroneous snapshots, %s = true C1=%f C2=%f",
			cond, C1, C2)
	}
}
