// compile -N

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 25966: liveness code complains autotmp live on
// function entry.

package p

var F = []func(){
	func() func() { return (func())(nil) }(),
}

var A = []int{}

type ss struct {
	string
	float64
	i int
}

var V = A[ss{}.i]
