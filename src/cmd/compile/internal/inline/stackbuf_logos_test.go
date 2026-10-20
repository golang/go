// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

import (
	"testing"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
)

// isolateRecursiveState gives the test its own mark set and restores the
// debug flags it changes.
func isolateRecursiveState(t *testing.T) {
	t.Helper()
	oldMarks := logosRecursiveMarks
	oldThreshold := base.Debug.VariableMakeThreshold
	oldLowerL := base.Flag.LowerL
	logosRecursiveMarks = make(map[*ir.Func]bool)
	t.Cleanup(func() {
		logosRecursiveMarks = oldMarks
		base.Debug.VariableMakeThreshold = oldThreshold
		base.Flag.LowerL = oldLowerL
	})
}

// TestLogosVariableMakeThreshold pins row #134: a function in a recursive
// call-graph cycle keeps the upstream variable-make threshold whatever the
// -O tier asks for, every other function gets the tier threshold, and a
// threshold below the upstream value stays as set.
func TestLogosVariableMakeThreshold(t *testing.T) {
	isolateRecursiveState(t)
	rec, plain := new(ir.Func), new(ir.Func)
	logosNoteSCC([]*ir.Func{rec})

	for _, tc := range []struct {
		tier            int
		wantRec, wantNo int64
	}{
		{tier: 4096, wantRec: 32, wantNo: 4096}, // -O=3
		{tier: 256, wantRec: 32, wantNo: 256},   // -O=2
		{tier: 33, wantRec: 32, wantNo: 33},
		{tier: 32, wantRec: 32, wantNo: 32}, // upstream default
		{tier: 8, wantRec: 8, wantNo: 8},    // explicit -d value below upstream
		{tier: 0, wantRec: 0, wantNo: 0},
	} {
		base.Debug.VariableMakeThreshold = tc.tier
		if got := LogosVariableMakeThreshold(rec); got != tc.wantRec {
			t.Errorf("tier %d: recursive function threshold = %d, want %d", tc.tier, got, tc.wantRec)
		}
		if got := LogosVariableMakeThreshold(plain); got != tc.wantNo {
			t.Errorf("tier %d: non-recursive function threshold = %d, want %d", tc.tier, got, tc.wantNo)
		}
		if got := LogosVariableMakeThreshold(nil); got != tc.wantNo {
			t.Errorf("tier %d: nil function threshold = %d, want %d", tc.tier, got, tc.wantNo)
		}
	}
}

// TestLogosRecursiveMarksSurviveLaterPass pins why the marks are
// cumulative: noder/reader.go calls CanInlineFuncs again for generic
// functions instantiated while the package is being inlined, and that pass
// must not drop the marks of the package's own recursive functions, which
// the backend reads long after.
func TestLogosRecursiveMarksSurviveLaterPass(t *testing.T) {
	isolateRecursiveState(t)
	base.Flag.LowerL = 1
	base.Debug.VariableMakeThreshold = 4096
	rec := new(ir.Func)
	logosNoteSCC([]*ir.Func{rec})

	CanInlineFuncs(nil, nil) // a later pass over a different batch

	if got := LogosVariableMakeThreshold(rec); got != 32 {
		t.Fatalf("threshold after a later CanInlineFuncs pass = %d, want 32 (marks were dropped)", got)
	}
}
