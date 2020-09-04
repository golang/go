// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package slices implements various slice algorithms.
package slices

// Map turns a []T1 to a []T2 using a mapping function.
func Map[T1, T2 any](s []T1, f func(T1) T2) []T2 {
	r := make([]T2, len(s))
	for i, v := range s {
		r[i] = f(v)
	}
	return r
}

// Reduce reduces a []T1 to a single value using a reduction function.
func Reduce[T1, T2 any](s []T1, initializer T2, f func(T2, T1) T2) T2 {
	r := initializer
	for _, v := range s {
		r = f(r, v)
	}
	return r
}

// Filter filters values from a slice using a filter function.
func Filter[T any](s []T, f func(T) bool) []T {
	var r []T
	for _, v := range s {
		if f(v) {
			r = append(r, v)
		}
	}
	return r
}

// Example uses

func limiter(x int) byte {
	switch {
	case x < 0:
		return 0
	default:
		return byte(x)
	case x > 255:
		return 255
	}
}

var input = []int{-4, 68954, 7, 44, 0, -555, 6945}
var limited1 = Map[int, byte](input, limiter)
var limited2 = Map(input, limiter) // using type inference

func reducer(x float64, y int) float64 {
	return x + float64(y)
}

var reduced1 = Reduce[int, float64](input, 0, reducer)
var reduced2 = Reduce(input, 1i /* ERROR overflows */, reducer) // using type inference
var reduced3 = Reduce(input, 1, reducer) // using type inference

func filter(x int) bool {
	return x&1 != 0
}

var filtered1 = Filter[int](input, filter)
var filtered2 = Filter(input, filter) // using type inference

