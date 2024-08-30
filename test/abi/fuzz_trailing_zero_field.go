// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var p0exp = S1{
	F1: complex(float64(2.3640607624715027), float64(-0.2717825524109192)),
	F2: S2{F1: 9},
	F3: 103050709,
}

type S1 struct {
	F1 complex128
	F2 S2
	F3 uint64
}

type S2 struct {
	F1 uint64
	F2 empty
}

type empty struct {
}

//go:noinline
//go:registerparams
func callee(p0 S1) {
	if p0 != p0exp {
		panic("bad p0")
	}
}

func main() {
	callee(p0exp)
}
