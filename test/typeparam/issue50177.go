// compile -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type Fn[T any] func(T)
type FnErr[T any] func(T) error

// Test that local generic types across functions don't conflict, and they also don't
// conflict with local non-generic types and local variables.
func caller0() {
	type X[T any] struct {
		fn Fn[int]
	}

	x := X[int]{func(v int) { fmt.Println(v) }}
	x.fn(0)
}

func caller1(val int) {
	type X[T any] struct {
		fn FnErr[int]
	}

	x := X[int]{func(v int) error { fmt.Println(v); return nil }}
	x.fn(0)
}

func caller1a(val int) {
	type X struct {
		fn func(float64) error
	}

	x := X{func(v float64) error { fmt.Println(v); return nil }}
	x.fn(float64(3.2))
}

func caller1b(val int) {
	type Y struct {
		fn func(float64) error
	}

	X := Y{func(v float64) error { fmt.Println(v); return nil }}
	X.fn(float64(3.2))
}

// Test that local generic types within different if clauses don't conflict.
func caller2(val int) {
	if val > 2 {
		type X[T any] struct {
			fn func(v int) float64
		}

		x := X[int]{func(v int) float64 { fmt.Println(v); return 1.5 }}
		x.fn(0)
	} else {
		type X[T any] struct {
			fn func(v int) int
		}
		x := X[int]{func(v int) int { fmt.Println(v); return 5 }}
		x.fn(0)
	}
}

// Test that local generic types within different cases don't conflict with each
// other or with local non-generic types or local variables.
func caller3(val int) {
	switch val {
	case 0:
		type X[T any] struct {
			fn func(v int) float64
		}

		x := X[int]{func(v int) float64 { fmt.Println(v); return 1.5 }}
		x.fn(0)
	case 1:
		type X[T any] struct {
			fn func(v int) int
		}
		x := X[int]{func(v int) int { fmt.Println(v); return 5 }}
		x.fn(0)
	case 2:
		type X struct {
			fn func(v int) bool
		}
		x := X{func(v int) bool { fmt.Println(v); return false }}
		x.fn(0)
	case 3:
		type Y struct {
			fn func(v int) bool
		}
		X := Y{func(v int) bool { fmt.Println(v); return false }}
		X.fn(0)

	}
}
