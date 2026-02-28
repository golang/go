// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	. "math/rand"
	"testing"
)

// This test is first, in its own file with an alphabetically early name,
// to try to make sure that it runs early. It has the best chance of
// detecting deterministic seeding if it's the first test that runs.

func TestAuto(t *testing.T) {
	// Pull out 10 int64s from the global source
	// and then check that they don't appear in that
	// order in the deterministic Seed(1) result.
	var out []int64
	for i := 0; i < 10; i++ {
		out = append(out, Int63())
	}

	// Look for out in Seed(1)'s output.
	// Strictly speaking, we should look for them in order,
	// but this is good enough and not significantly more
	// likely to have a false positive.
	Seed(1)
	found := 0
	for i := 0; i < 1000; i++ {
		x := Int63()
		if x == out[found] {
			found++
			if found == len(out) {
				t.Fatalf("found unseeded output in Seed(1) output")
			}
		}
	}
}
