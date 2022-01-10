// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type hello struct{}

func main() {
	_ = Some(hello{})
	res := Applicative2(func(a int, b int) int {
		return 0
	})
	_ = res
}

type NoneType[T any] struct{}

func (r NoneType[T]) Recover() any {
	return nil
}

type Func2[A1, A2, R any] func(a1 A1, a2 A2) R

func Some[T any](v T) any {
	_ = Some2[T](v)
	return NoneType[T]{}.Recover()
}

//go:noinline
func Some2[T any](v T) any {
	return v
}

type Nil struct{}

type ApplicativeFunctor2[H, HT, A1, A2, R any] struct {
	h any
}

func Applicative2[A1, A2, R any](fn Func2[A1, A2, R]) ApplicativeFunctor2[Nil, Nil, A1, A2, R] {
	return ApplicativeFunctor2[Nil, Nil, A1, A2, R]{Some(Nil{})}
}
