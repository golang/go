// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"internal/testenv"
	"testing"
)

func TestLogosScalarOptimizationRuntime(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()
	for _, mode := range []string{"0", "1"} {
		t.Run("logosopt="+mode, func(t *testing.T) {
			cmd := testenv.Command(t, testenv.GoToolPath(t), "test", "-count=1",
				"-gcflags=cmd/compile/internal/test/testdata/scalaropt=-d=logosopt="+mode+",ssa/check/on",
				"./testdata/scalaropt")
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("scalar runtime parity at logosopt=%s: %v\n%s", mode, err, out)
			}
		})
	}
}
