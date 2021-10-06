// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"sort"
)

// _Equal reports whether two slices are equal: the same length and all
// elements equal. All floating point NaNs are considered equal.
func _SliceEqual[Elem comparable](s1, s2 []Elem) bool {
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

// _Keys returns the keys of the map m.
// The keys will be an indeterminate order.
func _Keys[K comparable, V any](m map[K]V) []K {
	r := make([]K, 0, len(m))
	for k := range m {
		r = append(r, k)
	}
	return r
}

// _Values returns the values of the map m.
// The values will be in an indeterminate order.
func _Values[K comparable, V any](m map[K]V) []V {
	r := make([]V, 0, len(m))
	for _, v := range m {
		r = append(r, v)
	}
	return r
}

// _Equal reports whether two maps contain the same key/value pairs.
// _Values are compared using ==.
func _Equal[K, V comparable](m1, m2 map[K]V) bool {
	if len(m1) != len(m2) {
		return false
	}
	for k, v1 := range m1 {
		if v2, ok := m2[k]; !ok || v1 != v2 {
			return false
		}
	}
	return true
}

// _Copy returns a copy of m.
func _Copy[K comparable, V any](m map[K]V) map[K]V {
	r := make(map[K]V, len(m))
	for k, v := range m {
		r[k] = v
	}
	return r
}

// _Add adds all key/value pairs in m2 to m1. _Keys in m2 that are already
// present in m1 will be overwritten with the value in m2.
func _Add[K comparable, V any](m1, m2 map[K]V) {
	for k, v := range m2 {
		m1[k] = v
	}
}

// _Sub removes all keys in m2 from m1. _Keys in m2 that are not present
// in m1 are ignored. The values in m2 are ignored.
func _Sub[K comparable, V any](m1, m2 map[K]V) {
	for k := range m2 {
		delete(m1, k)
	}
}

// _Intersect removes all keys from m1 that are not present in m2.
// _Keys in m2 that are not in m1 are ignored. The values in m2 are ignored.
func _Intersect[K comparable, V any](m1, m2 map[K]V) {
	for k := range m1 {
		if _, ok := m2[k]; !ok {
			delete(m1, k)
		}
	}
}

// _Filter deletes any key/value pairs from m for which f returns false.
func _Filter[K comparable, V any](m map[K]V, f func(K, V) bool) {
	for k, v := range m {
		if !f(k, v) {
			delete(m, k)
		}
	}
}

// _TransformValues applies f to each value in m. The keys remain unchanged.
func _TransformValues[K comparable, V any](m map[K]V, f func(V) V) {
	for k, v := range m {
		m[k] = f(v)
	}
}

var m1 = map[int]int{1: 2, 2: 4, 4: 8, 8: 16}
var m2 = map[int]string{1: "2", 2: "4", 4: "8", 8: "16"}

func TestKeys() {
	want := []int{1, 2, 4, 8}

	got1 := _Keys(m1)
	sort.Ints(got1)
	if !_SliceEqual(got1, want) {
		panic(fmt.Sprintf("_Keys(%v) = %v, want %v", m1, got1, want))
	}

	got2 := _Keys(m2)
	sort.Ints(got2)
	if !_SliceEqual(got2, want) {
		panic(fmt.Sprintf("_Keys(%v) = %v, want %v", m2, got2, want))
	}
}

func TestValues() {
	got1 := _Values(m1)
	want1 := []int{2, 4, 8, 16}
	sort.Ints(got1)
	if !_SliceEqual(got1, want1) {
		panic(fmt.Sprintf("_Values(%v) = %v, want %v", m1, got1, want1))
	}

	got2 := _Values(m2)
	want2 := []string{"16", "2", "4", "8"}
	sort.Strings(got2)
	if !_SliceEqual(got2, want2) {
		panic(fmt.Sprintf("_Values(%v) = %v, want %v", m2, got2, want2))
	}
}

func TestEqual() {
	if !_Equal(m1, m1) {
		panic(fmt.Sprintf("_Equal(%v, %v) = false, want true", m1, m1))
	}
	if _Equal(m1, nil) {
		panic(fmt.Sprintf("_Equal(%v, nil) = true, want false", m1))
	}
	if _Equal(nil, m1) {
		panic(fmt.Sprintf("_Equal(nil, %v) = true, want false", m1))
	}
	if !_Equal[int, int](nil, nil) {
		panic("_Equal(nil, nil) = false, want true")
	}
	if ms := map[int]int{1: 2}; _Equal(m1, ms) {
		panic(fmt.Sprintf("_Equal(%v, %v) = true, want false", m1, ms))
	}

	// Comparing NaN for equality is expected to fail.
	mf := map[int]float64{1: 0, 2: math.NaN()}
	if _Equal(mf, mf) {
		panic(fmt.Sprintf("_Equal(%v, %v) = true, want false", mf, mf))
	}
}

func TestCopy() {
	m2 := _Copy(m1)
	if !_Equal(m1, m2) {
		panic(fmt.Sprintf("_Copy(%v) = %v, want %v", m1, m2, m1))
	}
	m2[16] = 32
	if _Equal(m1, m2) {
		panic(fmt.Sprintf("_Equal(%v, %v) = true, want false", m1, m2))
	}
}

func TestAdd() {
	mc := _Copy(m1)
	_Add(mc, mc)
	if !_Equal(mc, m1) {
		panic(fmt.Sprintf("_Add(%v, %v) = %v, want %v", m1, m1, mc, m1))
	}
	_Add(mc, map[int]int{16: 32})
	want := map[int]int{1: 2, 2: 4, 4: 8, 8: 16, 16: 32}
	if !_Equal(mc, want) {
		panic(fmt.Sprintf("_Add result = %v, want %v", mc, want))
	}
}

func TestSub() {
	mc := _Copy(m1)
	_Sub(mc, mc)
	if len(mc) > 0 {
		panic(fmt.Sprintf("_Sub(%v, %v) = %v, want empty map", m1, m1, mc))
	}
	mc = _Copy(m1)
	_Sub(mc, map[int]int{1: 0})
	want := map[int]int{2: 4, 4: 8, 8: 16}
	if !_Equal(mc, want) {
		panic(fmt.Sprintf("_Sub result = %v, want %v", mc, want))
	}
}

func TestIntersect() {
	mc := _Copy(m1)
	_Intersect(mc, mc)
	if !_Equal(mc, m1) {
		panic(fmt.Sprintf("_Intersect(%v, %v) = %v, want %v", m1, m1, mc, m1))
	}
	_Intersect(mc, map[int]int{1: 0, 2: 0})
	want := map[int]int{1: 2, 2: 4}
	if !_Equal(mc, want) {
		panic(fmt.Sprintf("_Intersect result = %v, want %v", mc, want))
	}
}

func TestFilter() {
	mc := _Copy(m1)
	_Filter(mc, func(int, int) bool { return true })
	if !_Equal(mc, m1) {
		panic(fmt.Sprintf("_Filter(%v, true) = %v, want %v", m1, mc, m1))
	}
	_Filter(mc, func(k, v int) bool { return k < 3 })
	want := map[int]int{1: 2, 2: 4}
	if !_Equal(mc, want) {
		panic(fmt.Sprintf("_Filter result = %v, want %v", mc, want))
	}
}

func TestTransformValues() {
	mc := _Copy(m1)
	_TransformValues(mc, func(i int) int { return i / 2 })
	want := map[int]int{1: 1, 2: 2, 4: 4, 8: 8}
	if !_Equal(mc, want) {
		panic(fmt.Sprintf("_TransformValues result = %v, want %v", mc, want))
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
