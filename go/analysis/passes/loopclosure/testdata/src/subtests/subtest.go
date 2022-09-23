// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests that the loopclosure analyzer detects leaked
// references via parallel subtests.

package subtests

import (
	"testing"
)

// T is used to test that loopclosure only matches T.Run when T is from the
// testing package.
type T struct{}

// Run should not match testing.T.Run. Note that the second argument is
// intentionally a *testing.T, not a *T, so that we can check both
// testing.T.Parallel inside a T.Run, and a T.Parallel inside a testing.T.Run.
func (t *T) Run(string, func(*testing.T)) {
}

func (t *T) Parallel() {}

func _(t *testing.T) {
	for i, test := range []int{1, 2, 3} {
		// Check that parallel subtests are identified.
		t.Run("", func(t *testing.T) {
			t.Parallel()
			println(i)    // want "loop variable i captured by func literal"
			println(test) // want "loop variable test captured by func literal"
		})

		// Check that serial tests are OK.
		t.Run("", func(t *testing.T) {
			println(i)
			println(test)
		})

		// Check that the location of t.Parallel matters.
		t.Run("", func(t *testing.T) {
			println(i)
			println(test)
			t.Parallel()
			println(i)    // want "loop variable i captured by func literal"
			println(test) // want "loop variable test captured by func literal"
		})

		// Check that shadowing the loop variables within the test literal is OK if
		// it occurs before t.Parallel().
		t.Run("", func(t *testing.T) {
			i := i
			test := test
			t.Parallel()
			println(i)
			println(test)
		})

		// Check that shadowing the loop variables within the test literal is Not
		// OK if it occurs after t.Parallel().
		t.Run("", func(t *testing.T) {
			t.Parallel()
			i := i        // want "loop variable i captured by func literal"
			test := test  // want "loop variable test captured by func literal"
			println(i)    // OK
			println(test) // OK
		})

		// Check uses in nested blocks.
		t.Run("", func(t *testing.T) {
			t.Parallel()
			{
				println(i)    // want "loop variable i captured by func literal"
				println(test) // want "loop variable test captured by func literal"
			}
		})

		// Check that we catch uses in nested subtests.
		t.Run("", func(t *testing.T) {
			t.Parallel()
			t.Run("", func(t *testing.T) {
				println(i)    // want "loop variable i captured by func literal"
				println(test) // want "loop variable test captured by func literal"
			})
		})

		// Check that there is no diagnostic if t is not a *testing.T.
		t.Run("", func(_ *testing.T) {
			t := &T{}
			t.Parallel()
			println(i)
			println(test)
		})
	}
}

// Check that there is no diagnostic when loop variables are shadowed within
// the loop body.
func _(t *testing.T) {
	for i, test := range []int{1, 2, 3} {
		i := i
		test := test
		t.Run("", func(t *testing.T) {
			t.Parallel()
			println(i)
			println(test)
		})
	}
}

// Check that t.Run must be *testing.T.Run.
func _(t *T) {
	for i, test := range []int{1, 2, 3} {
		t.Run("", func(t *testing.T) {
			t.Parallel()
			println(i)
			println(test)
		})
	}
}
