// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Iterator[T any] interface {
	Iterate(fn T)
}

type IteratorFunc[T any] func(fn T)

func (f IteratorFunc[T]) Iterate(fn T) {
	f(fn)
}

func Foo[R any]() {
	var _ Iterator[R] = IteratorFunc[R](nil)
}

func main() {
	Foo[int]()
}
