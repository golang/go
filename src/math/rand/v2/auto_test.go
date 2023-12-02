// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	. "math/rand/v2"
	"testing"
)

// This test is first, in its own file with an alphabetically early name,
// to try to make sure that it runs early. It has the best chance of
// detecting deterministic seeding if it's the first test that runs.

func TestAuto(t *testing.T) {
	// Pull out 10 int64s from the global source
	// and then check that they don't appear in that
	// order in the deterministic seeded result.
	var out []int64
	for i := 0; i < 10; i++ {
		out = append(out, Int64())
	}

	// Look for out in seeded output.
	// Strictly speaking, we should look for them in order,
	// but this is good enough and not significantly more
	// likely to have a false positive.
	r := New(NewPCG(1, 0))
	found := 0
	for i := 0; i < 1000; i++ {
		x := r.Int64()
		if x == out[found] {
			found++
			if found == len(out) {
				t.Fatalf("found unseeded output in Seed(1) output")
			}
		}
	}
}
