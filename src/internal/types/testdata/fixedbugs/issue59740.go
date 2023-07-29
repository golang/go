// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type F[T any] func(func(F[T]))

func f(F[int])      {}
func g[T any](F[T]) {}

func _() {
	g(f /* ERROR "type func(F[int]) of f does not match F[T] (cannot infer T)" */) // type inference/unification must not panic
}

// original test case from issue

type List[T any] func(T, func(T, List[T]) T) T

func nil[T any](n T, _ List[T]) T        { return n }
func cons[T any](h T, t List[T]) List[T] { return func(n T, f func(T, List[T]) T) T { return f(h, t) } }

func nums[T any](t T) List[T] {
	return cons(t, cons(t, nil /* ERROR "type func(n T, _ List[T]) T of nil[T] does not match inferred type List[T] for List[T]" */ [T]))
}
