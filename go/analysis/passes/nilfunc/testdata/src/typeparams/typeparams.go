// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the lostcancel checker.

//go:build go1.18

package typeparams

func f[P any]() {}

func g[P1 any, P2 any](x P1) {}

var f1 = f[int]

type T1[P any] struct {
	f func() P
}

type T2[P1 any, P2 any] struct {
	g func(P1) P2
}

func Comparison[P any](f2 func()T1[P]) {
	var t1 T1[P]
	var t2 T2[P, int]
	var fn func()
	if fn == nil || f1 == nil || f2 == nil || t1.f == nil || t2.g == nil {
		// no error; these func vars or fields may be nil
	}
	if f[P] == nil { // want "comparison of function f == nil is always false"
		panic("can't happen")
	}
	if f[int] == nil { // want "comparison of function f == nil is always false"
		panic("can't happen")
	}
	if g[P, int] == nil { // want "comparison of function g == nil is always false"
		panic("can't happen")
	}
}

func Index[P any](a [](func()P)) {
	if a[1] == nil {
		// no error
	}
	var t1 []T1[P]
	var t2 [][]T2[P, P]
	if t1[1].f == nil || t2[0][1].g == nil {
		// no error
	}
}