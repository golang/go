// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
	"./b"
)

type A struct {
	a.W
}

type B struct {
	b.W
}

type I interface {
	M(int)
}

type J interface {
	M()
}

var x I
var y J

func init() {
	x = A{}
	y = B{}
}

func main() {
	x.M(81286)
	y.M()

	if a.Last != 81286 {
		panic(a.Last)
	}
	if !b.Called {
		panic("b.W.M was not called")
	}
}
