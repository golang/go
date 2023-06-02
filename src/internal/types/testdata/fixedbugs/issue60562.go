// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type S[T any] struct{}

func (S[T]) m(T) {}

func f0[T any](chan S[T]) {}

func _() {
	var x chan interface{ m(int) }
	f0(x /* ERROR "type chan interface{m(int)} of x does not match chan S[T] (cannot infer T)" */)
}

// variants of the theme

func f1[T any]([]S[T]) {}

func _() {
	var x []interface{ m(int) }
	f1(x /* ERROR "type []interface{m(int)} of x does not match []S[T] (cannot infer T)" */)
}

type I[T any] interface {
	m(T)
}

func f2[T any](func(I[T])) {}

func _() {
	var x func(interface{ m(int) })
	f2(x /* ERROR "type func(interface{m(int)}) of x does not match func(I[T]) (cannot infer T)" */)
}

func f3[T any](func(I[T])) {}

func _() {
	var x func(I[int])
	f3(x) // but this is correct: I[T] and I[int] can be made identical with T == int
}

func f4[T any]([10]I[T]) {}

func _() {
	var x [10]interface{ I[int] }
	f4(x /* ERROR "type [10]interface{I[int]} of x does not match [10]I[T] (cannot infer T)" */)
}

func f5[T any](I[T]) {}

func _() {
	var x interface {
		m(int)
		n()
	}
	f5(x)
	f5[int](x) // ok
}
