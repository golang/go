// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test makes sure that itabs are unique.
// More explicitly, we require that only one itab structure exists for the pair of
// a given compile-time interface type and underlying concrete type.
// Ensuring this invariant enables fixes for 18492 (improve type switch code).

package main

type I interface {
	M()
}
type J interface {
	M()
}

type T struct{}

func (*T) M() {}

func main() {
	test1()
	test2()
}

func test1() {
	t := new(T)
	var i1, i2 I
	var j interface {
		M()
	}
	i1 = t
	j = t
	i2 = j
	if i1 != i2 {
		panic("interfaces not equal")
	}
}

func test2() {
	t := new(T)
	i1 := (I)(t)
	i2 := (I)((interface {
		M()
	})((J)(t)))
	if i1 != i2 {
		panic("interfaces not equal")
	}
}
