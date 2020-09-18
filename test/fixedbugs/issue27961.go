// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 27961: some optimizations generate OffPtr with wrong
// types, which causes invalid bytecode on Wasm.

package main

import "math"

type Vec2 [2]float64

func main() {
	var a Vec2
	a.A().B().C().D()
}

func (v Vec2) A() Vec2 {
	return Vec2{v[0], v[0]}
}

func (v Vec2) B() Vec2 {
	return Vec2{1.0 / v.D(), 0}
}

func (v Vec2) C() Vec2 {
	return Vec2{v[0], v[0]}
}

func (v Vec2) D() float64 {
	return math.Sqrt(v[0])
}
