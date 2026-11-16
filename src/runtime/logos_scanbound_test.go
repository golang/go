// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime_test

import (
	"runtime"
	"testing"
)

func TestLogosArenaScanBound(t *testing.T) {
	bounds, reused, rejected, ok := runtime.LogosArenaLeaseScanBounds()
	if !ok {
		t.Fatal("fixture could not lease an arena chunk")
	}
	if !reused {
		t.Error("second lease did not reuse the released chunk")
	}
	if !rejected {
		t.Error("lease succeeded with an empty pool at the chunk limit")
	}
	for i, name := range []string{"baseline", "new_lease", "release", "pooled_lease", "second_release", "free_pool", "failed_lease"} {
		var want int64
		if i == 1 || i == 3 {
			want = runtime.LogosArenaChunkBytes
		}
		if delta := bounds[i] - bounds[0]; delta != want {
			t.Errorf("%s: root scan bound delta = %d, want %d", name, delta, want)
		}
	}
}
