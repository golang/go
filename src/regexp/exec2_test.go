// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !race
// +build !race

package regexp

import (
	"testing"
)

// This test is excluded when running under the race detector because
// it is a very expensive test and takes too long.
func TestRE2Exhaustive(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping TestRE2Exhaustive during short test")
	}
	testRE2(t, "testdata/re2-exhaustive.txt.bz2")
}
