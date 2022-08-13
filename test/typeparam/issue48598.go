// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Iterator[T any] interface {
	Iterate()
}

type IteratorFunc[T any] func(fn func(T) bool)

func (f IteratorFunc[T]) Iterate() {
}

func FromIterator[T any](it Iterator[T]) {
	it.Iterate()
}

func Foo[T, R any]() {
	FromIterator[R](IteratorFunc[R](nil))
}

func main() {
	Foo[int, int]()
}
