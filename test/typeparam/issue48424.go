// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Smoke test for constraint literals with elided interface
// per issue #48424.

package main

func identity[T int](x T) T {
	return x
}

func min[T int | string](x, y T) T {
	if x < y {
		return x
	}
	return y
}

func max[T ~int | ~float64](x, y T) T {
	if x > y {
		return x
	}
	return y
}

func main() {
	if identity(1) != 1 {
		panic("identity(1) failed")
	}

	if min(2, 3) != 2 {
		panic("min(2, 3) failed")
	}

	if min("foo", "bar") != "bar" {
		panic(`min("foo", "bar") failed`)
	}

	if max(2, 3) != 3 {
		panic("max(2, 3) failed")
	}
}

// Some random type parameter lists with elided interfaces.

type (
	_[T struct{}]                     struct{}
	_[M map[K]V, K comparable, V any] struct{}
	_[_ interface{} | int]            struct{}
)
