// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Some derived-type expressions require the compiler to synthesize
// function literals to plumb sub-dictionaries appropriately.
// However, when these expressions are inlined, we were constructing
// the function literal bodies with the inline-adjusted positions
// instead of the original (inline-free) positions, which could lead
// to infinite loops when unwinding the stack.

package main

import "runtime"

func assert[_ any]() {
	panic(0)
}

func Assert[To any]() func() {
	return assert[To]
}

type asserter[_ any] struct{}

func (asserter[_]) assert() {
	panic(0)
}

func AssertMV[To any]() func() {
	return asserter[To]{}.assert
}

func AssertME[To any]() func(asserter[To]) {
	return asserter[To].assert
}

var me = AssertME[string]()

var tests = []func(){
	Assert[int](),
	AssertMV[int](),
	func() { me(asserter[string]{}) },
}

func main() {
	for _, test := range tests {
		func() {
			defer func() {
				recover()

				// Check that we can unwind the stack without infinite looping.
				runtime.Caller(1000)
			}()
			test()
		}()
	}
}
