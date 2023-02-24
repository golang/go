// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
	"fmt"
	"math"
	"sort"
)

var m1 = map[int]int{1: 2, 2: 4, 4: 8, 8: 16}
var m2 = map[int]string{1: "2", 2: "4", 4: "8", 8: "16"}

func TestKeys() {
	want := []int{1, 2, 4, 8}

	got1 := a.Keys(m1)
	sort.Ints(got1)
	if !a.SliceEqual(got1, want) {
		panic(fmt.Sprintf("a.Keys(%v) = %v, want %v", m1, got1, want))
	}

	got2 := a.Keys(m2)
	sort.Ints(got2)
	if !a.SliceEqual(got2, want) {
		panic(fmt.Sprintf("a.Keys(%v) = %v, want %v", m2, got2, want))
	}
}

func TestValues() {
	got1 := a.Values(m1)
	want1 := []int{2, 4, 8, 16}
	sort.Ints(got1)
	if !a.SliceEqual(got1, want1) {
		panic(fmt.Sprintf("a.Values(%v) = %v, want %v", m1, got1, want1))
	}

	got2 := a.Values(m2)
	want2 := []string{"16", "2", "4", "8"}
	sort.Strings(got2)
	if !a.SliceEqual(got2, want2) {
		panic(fmt.Sprintf("a.Values(%v) = %v, want %v", m2, got2, want2))
	}
}

func TestEqual() {
	if !a.Equal(m1, m1) {
		panic(fmt.Sprintf("a.Equal(%v, %v) = false, want true", m1, m1))
	}
	if a.Equal(m1, nil) {
		panic(fmt.Sprintf("a.Equal(%v, nil) = true, want false", m1))
	}
	if a.Equal(nil, m1) {
		panic(fmt.Sprintf("a.Equal(nil, %v) = true, want false", m1))
	}
	if !a.Equal[int, int](nil, nil) {
		panic("a.Equal(nil, nil) = false, want true")
	}
	if ms := map[int]int{1: 2}; a.Equal(m1, ms) {
		panic(fmt.Sprintf("a.Equal(%v, %v) = true, want false", m1, ms))
	}

	// Comparing NaN for equality is expected to fail.
	mf := map[int]float64{1: 0, 2: math.NaN()}
	if a.Equal(mf, mf) {
		panic(fmt.Sprintf("a.Equal(%v, %v) = true, want false", mf, mf))
	}
}

func TestCopy() {
	m2 := a.Copy(m1)
	if !a.Equal(m1, m2) {
		panic(fmt.Sprintf("a.Copy(%v) = %v, want %v", m1, m2, m1))
	}
	m2[16] = 32
	if a.Equal(m1, m2) {
		panic(fmt.Sprintf("a.Equal(%v, %v) = true, want false", m1, m2))
	}
}

func TestAdd() {
	mc := a.Copy(m1)
	a.Add(mc, mc)
	if !a.Equal(mc, m1) {
		panic(fmt.Sprintf("a.Add(%v, %v) = %v, want %v", m1, m1, mc, m1))
	}
	a.Add(mc, map[int]int{16: 32})
	want := map[int]int{1: 2, 2: 4, 4: 8, 8: 16, 16: 32}
	if !a.Equal(mc, want) {
		panic(fmt.Sprintf("a.Add result = %v, want %v", mc, want))
	}
}

func TestSub() {
	mc := a.Copy(m1)
	a.Sub(mc, mc)
	if len(mc) > 0 {
		panic(fmt.Sprintf("a.Sub(%v, %v) = %v, want empty map", m1, m1, mc))
	}
	mc = a.Copy(m1)
	a.Sub(mc, map[int]int{1: 0})
	want := map[int]int{2: 4, 4: 8, 8: 16}
	if !a.Equal(mc, want) {
		panic(fmt.Sprintf("a.Sub result = %v, want %v", mc, want))
	}
}

func TestIntersect() {
	mc := a.Copy(m1)
	a.Intersect(mc, mc)
	if !a.Equal(mc, m1) {
		panic(fmt.Sprintf("a.Intersect(%v, %v) = %v, want %v", m1, m1, mc, m1))
	}
	a.Intersect(mc, map[int]int{1: 0, 2: 0})
	want := map[int]int{1: 2, 2: 4}
	if !a.Equal(mc, want) {
		panic(fmt.Sprintf("a.Intersect result = %v, want %v", mc, want))
	}
}

func TestFilter() {
	mc := a.Copy(m1)
	a.Filter(mc, func(int, int) bool { return true })
	if !a.Equal(mc, m1) {
		panic(fmt.Sprintf("a.Filter(%v, true) = %v, want %v", m1, mc, m1))
	}
	a.Filter(mc, func(k, v int) bool { return k < 3 })
	want := map[int]int{1: 2, 2: 4}
	if !a.Equal(mc, want) {
		panic(fmt.Sprintf("a.Filter result = %v, want %v", mc, want))
	}
}

func TestTransformValues() {
	mc := a.Copy(m1)
	a.TransformValues(mc, func(i int) int { return i / 2 })
	want := map[int]int{1: 1, 2: 2, 4: 4, 8: 8}
	if !a.Equal(mc, want) {
		panic(fmt.Sprintf("a.TransformValues result = %v, want %v", mc, want))
	}
}

func main() {
	TestKeys()
	TestValues()
	TestEqual()
	TestCopy()
	TestAdd()
	TestSub()
	TestIntersect()
	TestFilter()
	TestTransformValues()
}
