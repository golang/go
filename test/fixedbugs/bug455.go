// run

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4156: out of fixed registers when chaining method calls.
// Used to happen with 6g.

package main

type test_i interface {
	Test() test_i
	Result() bool
}

type test_t struct {
}

func newTest() *test_t {
	return &test_t{}
}

type testFn func(string) testFn

func main() {
	test := newTest()

	switch {
	case test.
		Test().
		Test().
		Test().
		Test().
		Test().
		Test().
		Test().
		Test().
		Test().
		Test().
		Result():
		// case worked
	default:
		panic("Result returned false unexpectedly")
	}
}

func (t *test_t) Test() test_i {
	return t
}

func (t *test_t) Result() bool {
	return true
}
