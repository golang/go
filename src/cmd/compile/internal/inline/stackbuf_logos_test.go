// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

import (
	"testing"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
)

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

func TestLogosVariableMakeThreshold(t *testing.T) {
	isolateRecursiveState(t)
	rec, plain := new(ir.Func), new(ir.Func)
	logosNoteSCC([]*ir.Func{rec})

	for _, tc := range []struct {
		tier            int
		wantRec, wantNo int64
	}{
		{tier: 4096, wantRec: 32, wantNo: 4096},
		{tier: 256, wantRec: 32, wantNo: 256},
		{tier: 33, wantRec: 32, wantNo: 33},
		{tier: 32, wantRec: 32, wantNo: 32},
		{tier: 8, wantRec: 8, wantNo: 8},
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

func TestLogosRecursiveMarksSurviveLaterPass(t *testing.T) {
	isolateRecursiveState(t)
	base.Flag.LowerL = 1
	base.Debug.VariableMakeThreshold = 4096
	rec := new(ir.Func)
	logosNoteSCC([]*ir.Func{rec})

	CanInlineFuncs(nil, nil)

	if got := LogosVariableMakeThreshold(rec); got != 32 {
		t.Fatalf("threshold after a later CanInlineFuncs pass = %d, want 32 (marks were dropped)", got)
	}
}
