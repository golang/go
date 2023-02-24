// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package slices provides functions for basic operations on
// slices of any element type.
package main

import (
	"fmt"
	"math"
	"strings"
)

type Ordered interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 |
		~string
}

type Integer interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

// Max returns the maximum of two values of some ordered type.
func _Max[T Ordered](a, b T) T {
	if a > b {
		return a
	}
	return b
}

// Min returns the minimum of two values of some ordered type.
func _Min[T Ordered](a, b T) T {
	if a < b {
		return a
	}
	return b
}

// _Equal reports whether two slices are equal: the same length and all
// elements equal. All floating point NaNs are considered equal.
func _Equal[Elem comparable](s1, s2 []Elem) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, v1 := range s1 {
		v2 := s2[i]
		if v1 != v2 {
			isNaN := func(f Elem) bool { return f != f }
			if !isNaN(v1) || !isNaN(v2) {
				return false
			}
		}
	}
	return true
}

// _EqualFn reports whether two slices are equal using a comparison
// function on each element.
func _EqualFn[Elem any](s1, s2 []Elem, eq func(Elem, Elem) bool) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, v1 := range s1 {
		v2 := s2[i]
		if !eq(v1, v2) {
			return false
		}
	}
	return true
}

// _Map turns a []Elem1 to a []Elem2 using a mapping function.
func _Map[Elem1, Elem2 any](s []Elem1, f func(Elem1) Elem2) []Elem2 {
	r := make([]Elem2, len(s))
	for i, v := range s {
		r[i] = f(v)
	}
	return r
}

// _Reduce reduces a []Elem1 to a single value of type Elem2 using
// a reduction function.
func _Reduce[Elem1, Elem2 any](s []Elem1, initializer Elem2, f func(Elem2, Elem1) Elem2) Elem2 {
	r := initializer
	for _, v := range s {
		r = f(r, v)
	}
	return r
}

// _Filter filters values from a slice using a filter function.
func _Filter[Elem any](s []Elem, f func(Elem) bool) []Elem {
	var r []Elem
	for _, v := range s {
		if f(v) {
			r = append(r, v)
		}
	}
	return r
}

// _Max returns the maximum element in a slice of some ordered type.
// If the slice is empty it returns the zero value of the element type.
func _SliceMax[Elem Ordered](s []Elem) Elem {
	if len(s) == 0 {
		var zero Elem
		return zero
	}
	return _Reduce(s[1:], s[0], _Max[Elem])
}

// _Min returns the minimum element in a slice of some ordered type.
// If the slice is empty it returns the zero value of the element type.
func _SliceMin[Elem Ordered](s []Elem) Elem {
	if len(s) == 0 {
		var zero Elem
		return zero
	}
	return _Reduce(s[1:], s[0], _Min[Elem])
}

// _Append adds values to the end of a slice, returning a new slice.
// This is like the predeclared append function; it's an example
// of how to write it using generics. We used to write code like
// this before append was added to the language, but we had to write
// a separate copy for each type.
func _Append[T any](s []T, t ...T) []T {
	lens := len(s)
	tot := lens + len(t)
	if tot <= cap(s) {
		s = s[:tot]
	} else {
		news := make([]T, tot, tot+tot/2)
		_Copy(news, s)
		s = news
	}
	_Copy(s[lens:tot], t)
	return s
}

// _Copy copies values from t to s, stopping when either slice is full,
// returning the number of values copied. This is like the predeclared
// copy function; it's an example of how to write it using generics.
func _Copy[T any](s, t []T) int {
	i := 0
	for ; i < len(s) && i < len(t); i++ {
		s[i] = t[i]
	}
	return i
}

func TestEqual() {
	s1 := []int{1, 2, 3}
	if !_Equal(s1, s1) {
		panic(fmt.Sprintf("_Equal(%v, %v) = false, want true", s1, s1))
	}
	s2 := []int{1, 2, 3}
	if !_Equal(s1, s2) {
		panic(fmt.Sprintf("_Equal(%v, %v) = false, want true", s1, s2))
	}
	s2 = append(s2, 4)
	if _Equal(s1, s2) {
		panic(fmt.Sprintf("_Equal(%v, %v) = true, want false", s1, s2))
	}

	s3 := []float64{1, 2, math.NaN()}
	if !_Equal(s3, s3) {
		panic(fmt.Sprintf("_Equal(%v, %v) = false, want true", s3, s3))
	}

	if _Equal(s1, nil) {
		panic(fmt.Sprintf("_Equal(%v, nil) = true, want false", s1))
	}
	if _Equal(nil, s1) {
		panic(fmt.Sprintf("_Equal(nil, %v) = true, want false", s1))
	}
	if !_Equal(s1[:0], nil) {
		panic(fmt.Sprintf("_Equal(%v, nil = false, want true", s1[:0]))
	}
}

func offByOne[Elem Integer](a, b Elem) bool {
	return a == b+1 || a == b-1
}

func TestEqualFn() {
	s1 := []int{1, 2, 3}
	s2 := []int{2, 3, 4}
	if _EqualFn(s1, s1, offByOne[int]) {
		panic(fmt.Sprintf("_EqualFn(%v, %v, offByOne) = true, want false", s1, s1))
	}
	if !_EqualFn(s1, s2, offByOne[int]) {
		panic(fmt.Sprintf("_EqualFn(%v, %v, offByOne) = false, want true", s1, s2))
	}

	if !_EqualFn(s1[:0], nil, offByOne[int]) {
		panic(fmt.Sprintf("_EqualFn(%v, nil, offByOne) = false, want true", s1[:0]))
	}

	s3 := []string{"a", "b", "c"}
	s4 := []string{"A", "B", "C"}
	if !_EqualFn(s3, s4, strings.EqualFold) {
		panic(fmt.Sprintf("_EqualFn(%v, %v, strings.EqualFold) = false, want true", s3, s4))
	}
}

func TestMap() {
	s1 := []int{1, 2, 3}
	s2 := _Map(s1, func(i int) float64 { return float64(i) * 2.5 })
	if want := []float64{2.5, 5, 7.5}; !_Equal(s2, want) {
		panic(fmt.Sprintf("_Map(%v, ...) = %v, want %v", s1, s2, want))
	}

	s3 := []string{"Hello", "World"}
	s4 := _Map(s3, strings.ToLower)
	if want := []string{"hello", "world"}; !_Equal(s4, want) {
		panic(fmt.Sprintf("_Map(%v, strings.ToLower) = %v, want %v", s3, s4, want))
	}

	s5 := _Map(nil, func(i int) int { return i })
	if len(s5) != 0 {
		panic(fmt.Sprintf("_Map(nil, identity) = %v, want empty slice", s5))
	}
}

func TestReduce() {
	s1 := []int{1, 2, 3}
	r := _Reduce(s1, 0, func(f float64, i int) float64 { return float64(i)*2.5 + f })
	if want := 15.0; r != want {
		panic(fmt.Sprintf("_Reduce(%v, 0, ...) = %v, want %v", s1, r, want))
	}

	if got := _Reduce(nil, 0, func(i, j int) int { return i + j }); got != 0 {
		panic(fmt.Sprintf("_Reduce(nil, 0, add) = %v, want 0", got))
	}
}

func TestFilter() {
	s1 := []int{1, 2, 3}
	s2 := _Filter(s1, func(i int) bool { return i%2 == 0 })
	if want := []int{2}; !_Equal(s2, want) {
		panic(fmt.Sprintf("_Filter(%v, even) = %v, want %v", s1, s2, want))
	}

	if s3 := _Filter(s1[:0], func(i int) bool { return true }); len(s3) > 0 {
		panic(fmt.Sprintf("_Filter(%v, identity) = %v, want empty slice", s1[:0], s3))
	}
}

func TestMax() {
	s1 := []int{1, 2, 3, -5}
	if got, want := _SliceMax(s1), 3; got != want {
		panic(fmt.Sprintf("_Max(%v) = %d, want %d", s1, got, want))
	}

	s2 := []string{"aaa", "a", "aa", "aaaa"}
	if got, want := _SliceMax(s2), "aaaa"; got != want {
		panic(fmt.Sprintf("_Max(%v) = %q, want %q", s2, got, want))
	}

	if got, want := _SliceMax(s2[:0]), ""; got != want {
		panic(fmt.Sprintf("_Max(%v) = %q, want %q", s2[:0], got, want))
	}
}

func TestMin() {
	s1 := []int{1, 2, 3, -5}
	if got, want := _SliceMin(s1), -5; got != want {
		panic(fmt.Sprintf("_Min(%v) = %d, want %d", s1, got, want))
	}

	s2 := []string{"aaa", "a", "aa", "aaaa"}
	if got, want := _SliceMin(s2), "a"; got != want {
		panic(fmt.Sprintf("_Min(%v) = %q, want %q", s2, got, want))
	}

	if got, want := _SliceMin(s2[:0]), ""; got != want {
		panic(fmt.Sprintf("_Min(%v) = %q, want %q", s2[:0], got, want))
	}
}

func TestAppend() {
	s := []int{1, 2, 3}
	s = _Append(s, 4, 5, 6)
	want := []int{1, 2, 3, 4, 5, 6}
	if !_Equal(s, want) {
		panic(fmt.Sprintf("after _Append got %v, want %v", s, want))
	}
}

func TestCopy() {
	s1 := []int{1, 2, 3}
	s2 := []int{4, 5}
	if got := _Copy(s1, s2); got != 2 {
		panic(fmt.Sprintf("_Copy returned %d, want 2", got))
	}
	want := []int{4, 5, 3}
	if !_Equal(s1, want) {
		panic(fmt.Sprintf("after _Copy got %v, want %v", s1, want))
	}
}
func main() {
	TestEqual()
	TestEqualFn()
	TestMap()
	TestReduce()
	TestFilter()
	TestMax()
	TestMin()
	TestAppend()
	TestCopy()
}
