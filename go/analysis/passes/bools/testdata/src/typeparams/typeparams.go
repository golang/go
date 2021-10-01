// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the bool checker.

//go:build go1.18

package typeparams

type T[P interface{ ~int }] struct {
	a P
}

func (t T[P]) Foo() int { return int(t.a) }

type FT[P any] func() P

func Sink[Elem any]() chan Elem {
	return make(chan Elem)
}

func RedundantConditions[P interface{ int }]() {
	type _f[P1 any] func() P1

	var f, g _f[P]
	if f() == 0 || f() == 0 { // OK f might have side effects
	}
	var t T[P]
	_ = t.Foo() == 2 || t.Foo() == 2        // OK Foo might have side effects
	if v, w := f(), g(); v == w || v == w { // want `redundant or: v == w \|\| v == w`
	}

	// error messages present type params correctly.
	_ = t == T[P]{2} || t == T[P]{2}                 // want `redundant or: t == T\[P\]\{2\} \|\| t == T\[P\]\{2\}`
	_ = FT[P](f) == nil || FT[P](f) == nil           // want `redundant or: FT\[P\]\(f\) == nil \|\| FT\[P\]\(f\) == nil`
	_ = (func() P)(f) == nil || (func() P)(f) == nil // want `redundant or: \(func\(\) P\)\(f\) == nil \|\| \(func\(\) P\)\(f\) == nil`

	var tint T[int]
	var fint _f[int]
	_ = tint == T[int]{2} || tint == T[int]{2}                 // want `redundant or: tint == T\[int\]\{2\} \|\| tint\ == T\[int\]\{2\}`
	_ = FT[int](fint) == nil || FT[int](fint) == nil           // want `redundant or: FT\[int\]\(fint\) == nil \|\| FT\[int\]\(fint\) == nil`
	_ = (func() int)(fint) == nil || (func() int)(fint) == nil // want `redundant or: \(func\(\) int\)\(fint\) == nil \|\| \(func\(\) int\)\(fint\) == nil`

	c := Sink[P]()
	_ = 0 == <-c || 0 == <-c                                  // OK subsequent receives may yield different values
	for i, j := <-c, <-c; i == j || i == j; i, j = <-c, <-c { // want `redundant or: i == j \|\| i == j`
	}

	var i, j P
	_ = i == 1 || j+1 == i || i == 1 // want `redundant or: i == 1 \|\| i == 1`
	_ = i == 1 || f() == 1 || i == 1 // OK f may alter i as a side effect
	_ = f() == 1 || i == 1 || i == 1 // want `redundant or: i == 1 \|\| i == 1`
}

func SuspectConditions[P interface{ ~int }, S interface{ ~string }]() {
	var i, j P
	_ = i == 0 || i == 1                 // OK
	_ = i+3 != 7 || j+5 == 0 || i+3 != 9 // want `suspect or: i\+3 != 7 \|\| i\+3 != 9`

	var s S
	_ = s != "one" || s != "the other" // want `suspect or: s != .one. \|\| s != .the other.`
}
