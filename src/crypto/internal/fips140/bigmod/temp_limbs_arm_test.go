// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm

package bigmod

import "testing"

func TestMakeWideLimbsAllocatesOnArm(t *testing.T) {
	allocs := testing.AllocsPerRun(100, func() {
		_ = makeWideLimbs(128)
	})
	if allocs < 1 {
		t.Fatalf("makeWideLimbs(128) allocations = %v, want at least 1", allocs)
	}
}
