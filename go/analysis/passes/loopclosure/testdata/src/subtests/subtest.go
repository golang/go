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

		// Check that *testing.T value matters.
		t.Run("", func(t *testing.T) {
			var x testing.T
			x.Parallel()
			println(i)
			println(test)
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

		// Check that there is no diagnostic when a jump to a label may have caused
		// the call to t.Parallel to have been skipped.
		t.Run("", func(t *testing.T) {
			if true {
				goto Test
			}
			t.Parallel()
		Test:
			println(i)
			println(test)
		})

		// Check that there is no diagnostic when a jump to a label may have caused
		// the loop variable reference to be skipped, but there is a diagnostic
		// when both the call to t.Parallel and the loop variable reference occur
		// after the final label in the block.
		t.Run("", func(t *testing.T) {
			if true {
				goto Test
			}
			t.Parallel()
			println(i) // maybe OK
		Test:
			t.Parallel()
			println(test) // want "loop variable test captured by func literal"
		})

		// Check that multiple labels are handled.
		t.Run("", func(t *testing.T) {
			if true {
				goto Test1
			} else {
				goto Test2
			}
		Test1:
		Test2:
			t.Parallel()
			println(test) // want "loop variable test captured by func literal"
		})

		// Check that we do not have problems when t.Run has a single argument.
		fn := func() (string, func(t *testing.T)) { return "", nil }
		t.Run(fn())
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

// Check that the top-level must be parallel in order to cause a diagnostic.
//
// From https://pkg.go.dev/testing:
//
//	"Run does not return until parallel subtests have completed, providing a
//	way to clean up after a group of parallel tests"
func _(t *testing.T) {
	for _, test := range []int{1, 2, 3} {
		// In this subtest, a/b must complete before the synchronous subtest "a"
		// completes, so the reference to test does not escape the current loop
		// iteration.
		t.Run("a", func(s *testing.T) {
			s.Run("b", func(u *testing.T) {
				u.Parallel()
				println(test)
			})
		})

		// In this subtest, c executes concurrently, so the reference to test may
		// escape the current loop iteration.
		t.Run("c", func(s *testing.T) {
			s.Parallel()
			s.Run("d", func(u *testing.T) {
				println(test) // want "loop variable test captured by func literal"
			})
		})
	}
}
