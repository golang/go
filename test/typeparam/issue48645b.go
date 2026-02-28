// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Iterator[T any] interface {
	Iterate(fn func(T) bool)
}

type IteratorFunc[T any] func(fn func(T) bool)

func (f IteratorFunc[T]) Iterate(fn func(T) bool) {
	f(fn)
}

type Stream[T any] struct {
	it Iterator[T]
}

func (s Stream[T]) Iterate(fn func(T) bool) {
	if s.it == nil {
		return
	}
	s.it.Iterate(fn)
}

func FromIterator[T any](it Iterator[T]) Stream[T] {
	return Stream[T]{it: it}
}

func (s Stream[T]) DropWhile(fn func(T) bool) Stream[T] {
	return Pipe[T, T](s, func(t T) (T, bool) {
		return t, true
	})
}

func Pipe[T, R any](s Stream[T], op func(d T) (R, bool)) Stream[R] {
	it := func(fn func(R) bool) {
		// XXX Not getting the closure right when converting to interface.
		// s.it.Iterate(func(t T) bool {
		// 	r, ok := op(t)
		// 	if !ok {
		// 		return true
		// 	}

		// 	return fn(r)
		// })
	}

	return FromIterator[R](IteratorFunc[R](it))
}

func Reduce[T, U any](s Stream[T], identity U, acc func(U, T) U) (r U) {
	r = identity
	s.Iterate(func(t T) bool {
		r = acc(r, t)
		return true
	})

	return r
}

type myIterator struct {
}

func (myIterator) Iterate(fn func(int) bool) {
}

func main() {
	s := Stream[int]{}
	s.it = myIterator{}
	s = s.DropWhile(func(i int) bool {
		return false
	})
	Reduce(s, nil, func(acc []int, e int) []int {
		return append(acc, e)
	})
}
