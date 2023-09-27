// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
	"fmt"
	"math"
	"strings"
)

type Integer interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

func TestEqual() {
	s1 := []int{1, 2, 3}
	if !a.Equal(s1, s1) {
		panic(fmt.Sprintf("a.Equal(%v, %v) = false, want true", s1, s1))
	}
	s2 := []int{1, 2, 3}
	if !a.Equal(s1, s2) {
		panic(fmt.Sprintf("a.Equal(%v, %v) = false, want true", s1, s2))
	}
	s2 = append(s2, 4)
	if a.Equal(s1, s2) {
		panic(fmt.Sprintf("a.Equal(%v, %v) = true, want false", s1, s2))
	}

	s3 := []float64{1, 2, math.NaN()}
	if !a.Equal(s3, s3) {
		panic(fmt.Sprintf("a.Equal(%v, %v) = false, want true", s3, s3))
	}

	if a.Equal(s1, nil) {
		panic(fmt.Sprintf("a.Equal(%v, nil) = true, want false", s1))
	}
	if a.Equal(nil, s1) {
		panic(fmt.Sprintf("a.Equal(nil, %v) = true, want false", s1))
	}
	if !a.Equal(s1[:0], nil) {
		panic(fmt.Sprintf("a.Equal(%v, nil = false, want true", s1[:0]))
	}
}

func offByOne[Elem Integer](a, b Elem) bool {
	return a == b+1 || a == b-1
}

func TestEqualFn() {
	s1 := []int{1, 2, 3}
	s2 := []int{2, 3, 4}
	if a.EqualFn(s1, s1, offByOne[int]) {
		panic(fmt.Sprintf("a.EqualFn(%v, %v, offByOne) = true, want false", s1, s1))
	}
	if !a.EqualFn(s1, s2, offByOne[int]) {
		panic(fmt.Sprintf("a.EqualFn(%v, %v, offByOne) = false, want true", s1, s2))
	}

	if !a.EqualFn(s1[:0], nil, offByOne[int]) {
		panic(fmt.Sprintf("a.EqualFn(%v, nil, offByOne) = false, want true", s1[:0]))
	}

	s3 := []string{"a", "b", "c"}
	s4 := []string{"A", "B", "C"}
	if !a.EqualFn(s3, s4, strings.EqualFold) {
		panic(fmt.Sprintf("a.EqualFn(%v, %v, strings.EqualFold) = false, want true", s3, s4))
	}
}

func TestMap() {
	s1 := []int{1, 2, 3}
	s2 := a.Map(s1, func(i int) float64 { return float64(i) * 2.5 })
	if want := []float64{2.5, 5, 7.5}; !a.Equal(s2, want) {
		panic(fmt.Sprintf("a.Map(%v, ...) = %v, want %v", s1, s2, want))
	}

	s3 := []string{"Hello", "World"}
	s4 := a.Map(s3, strings.ToLower)
	if want := []string{"hello", "world"}; !a.Equal(s4, want) {
		panic(fmt.Sprintf("a.Map(%v, strings.ToLower) = %v, want %v", s3, s4, want))
	}

	s5 := a.Map(nil, func(i int) int { return i })
	if len(s5) != 0 {
		panic(fmt.Sprintf("a.Map(nil, identity) = %v, want empty slice", s5))
	}
}

func TestReduce() {
	s1 := []int{1, 2, 3}
	r := a.Reduce(s1, 0, func(f float64, i int) float64 { return float64(i)*2.5 + f })
	if want := 15.0; r != want {
		panic(fmt.Sprintf("a.Reduce(%v, 0, ...) = %v, want %v", s1, r, want))
	}

	if got := a.Reduce(nil, 0, func(i, j int) int { return i + j }); got != 0 {
		panic(fmt.Sprintf("a.Reduce(nil, 0, add) = %v, want 0", got))
	}
}

func TestFilter() {
	s1 := []int{1, 2, 3}
	s2 := a.Filter(s1, func(i int) bool { return i%2 == 0 })
	if want := []int{2}; !a.Equal(s2, want) {
		panic(fmt.Sprintf("a.Filter(%v, even) = %v, want %v", s1, s2, want))
	}

	if s3 := a.Filter(s1[:0], func(i int) bool { return true }); len(s3) > 0 {
		panic(fmt.Sprintf("a.Filter(%v, identity) = %v, want empty slice", s1[:0], s3))
	}
}

func TestMax() {
	s1 := []int{1, 2, 3, -5}
	if got, want := a.SliceMax(s1), 3; got != want {
		panic(fmt.Sprintf("a.Max(%v) = %d, want %d", s1, got, want))
	}

	s2 := []string{"aaa", "a", "aa", "aaaa"}
	if got, want := a.SliceMax(s2), "aaaa"; got != want {
		panic(fmt.Sprintf("a.Max(%v) = %q, want %q", s2, got, want))
	}

	if got, want := a.SliceMax(s2[:0]), ""; got != want {
		panic(fmt.Sprintf("a.Max(%v) = %q, want %q", s2[:0], got, want))
	}
}

func TestMin() {
	s1 := []int{1, 2, 3, -5}
	if got, want := a.SliceMin(s1), -5; got != want {
		panic(fmt.Sprintf("a.Min(%v) = %d, want %d", s1, got, want))
	}

	s2 := []string{"aaa", "a", "aa", "aaaa"}
	if got, want := a.SliceMin(s2), "a"; got != want {
		panic(fmt.Sprintf("a.Min(%v) = %q, want %q", s2, got, want))
	}

	if got, want := a.SliceMin(s2[:0]), ""; got != want {
		panic(fmt.Sprintf("a.Min(%v) = %q, want %q", s2[:0], got, want))
	}
}

func TestAppend() {
	s := []int{1, 2, 3}
	s = a.Append(s, 4, 5, 6)
	want := []int{1, 2, 3, 4, 5, 6}
	if !a.Equal(s, want) {
		panic(fmt.Sprintf("after a.Append got %v, want %v", s, want))
	}
}

func TestCopy() {
	s1 := []int{1, 2, 3}
	s2 := []int{4, 5}
	if got := a.Copy(s1, s2); got != 2 {
		panic(fmt.Sprintf("a.Copy returned %d, want 2", got))
	}
	want := []int{4, 5, 3}
	if !a.Equal(s1, want) {
		panic(fmt.Sprintf("after a.Copy got %v, want %v", s1, want))
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
