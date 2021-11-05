// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Constraint[T any] interface {
	~func() T
}

func Foo[T Constraint[T]]() T {
	var t T

	t = func() T {
		return t
	}
	return t
}

func main() {
	type Bar func() Bar
	Foo[Bar]()
}
