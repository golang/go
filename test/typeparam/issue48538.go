// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Testing composite literal for a type param constrained to be a struct or a map.

package p

type C interface {
	~struct{ b1, b2 string }
}

func f[T C]() T {
	return T{
		b1: "a",
		b2: "b",
	}
}

func f2[T ~struct{ b1, b2 string }]() T {
	return T{
		b1: "a",
		b2: "b",
	}
}

type D interface {
	map[string]string | S
}

type S map[string]string

func g[T D]() T {
	b1 := "foo"
	b2 := "bar"
	return T{
		b1: "a",
		b2: "b",
	}
}

func g2[T map[string]string]() T {
	b1 := "foo"
	b2 := "bar"
	return T{
		b1: "a",
		b2: "b",
	}
}

func g3[T S]() T {
	b1 := "foo"
	b2 := "bar"
	return T{
		b1: "a",
		b2: "b",
	}
}
