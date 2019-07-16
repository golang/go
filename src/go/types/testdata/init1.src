// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// initialization cycles

package init1

// issue 6683 (marked as WorkingAsIntended)

type T0 struct{}

func (T0) m() int { return y0 }

var x0 = T0{}

var y0 /* ERROR initialization cycle */ = x0.m()

type T1 struct{}

func (T1) m() int { return y1 }

var x1 interface {
	m() int
} = T1{}

var y1 = x1.m() // no cycle reported, x1 is of interface type

// issue 6703 (modified)

var x2 /* ERROR initialization cycle */ = T2.m

var y2 = x2

type T2 struct{}

func (T2) m() int {
	_ = y2
	return 0
}

var x3 /* ERROR initialization cycle */ = T3.m(T3{}) // <<<< added (T3{})

var y3 = x3

type T3 struct{}

func (T3) m() int {
	_ = y3
	return 0
}

var x4 /* ERROR initialization cycle */ = T4{}.m // <<<< added {}

var y4 = x4

type T4 struct{}

func (T4) m() int {
	_ = y4
	return 0
}

var x5 /* ERROR initialization cycle */ = T5{}.m() // <<<< added ()

var y5 = x5

type T5 struct{}

func (T5) m() int {
	_ = y5
	return 0
}

// issue 4847
// simplified test case

var x6 = f6
var y6 /* ERROR initialization cycle */ = f6
func f6() { _ = y6 }

// full test case

type (
      E int
      S int
)

type matcher func(s *S) E

func matchList(s *S) E { return matcher(matchAnyFn)(s) }

var foo = matcher(matchList)

var matchAny /* ERROR initialization cycle */ = matcher(matchList)

func matchAnyFn(s *S) (err E) { return matchAny(s) }