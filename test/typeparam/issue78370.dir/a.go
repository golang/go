// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

// These closures have no explicitly declared variables of type T. Their
// dictionaries must survive for executable uses, independently of the
// dictionaries needed to describe generic variables in debug information.

func Nested[T any]() func() func() any {
	return func() func() any {
		return func() any { return []T(nil) }
	}
}

func Siblings[T any]() (func() int, func() any) {
	return func() int { return 7 }, func() any { return []T(nil) }
}

func Assert[T any]() func(any) any {
	return func(x any) any { return x.(T) }
}

func Subcall[T any]() func() any {
	return func() any { return zero[T]() }
}

//go:noinline
func zero[T any]() any {
	return []T(nil)
}

func Method[T interface{ Value() int }]() func() int {
	return func() int { return (*new(T)).Value() }
}

func MethodValue[T interface{ Value() int }]() func() func() int {
	return func() func() int { return (*new(T)).Value }
}

func Map[T comparable]() func() any {
	return func() any { return make(map[T]int) }
}

type Message[T any] struct {
	Index int
}

func GoDefer[T any](ch chan any, next func() int) func() {
	return func() {
		defer send[T](ch, next())
		go send[T](ch, next())
	}
}

//go:noinline
func send[T any](ch chan any, n int) {
	ch <- Message[T]{n}
}

func Range[T any]() func() any {
	return func() any {
		var result any
		for v := range sequence[T] {
			result = v
			break
		}
		return result
	}
}

//go:noinline
func sequence[T any](yield func(any) bool) {
	if yield([]T(nil)) {
		panic("yield did not report break")
	}
}
