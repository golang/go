// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"sort"
)

type Ordered interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 |
		~string
}

type orderedSlice[Elem Ordered] []Elem

func (s orderedSlice[Elem]) Len() int { return len(s) }
func (s orderedSlice[Elem]) Less(i, j int) bool {
	if s[i] < s[j] {
		return true
	}
	isNaN := func(f Elem) bool { return f != f }
	if isNaN(s[i]) && !isNaN(s[j]) {
		return true
	}
	return false
}
func (s orderedSlice[Elem]) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

func _OrderedSlice[Elem Ordered](s []Elem) {
	sort.Sort(orderedSlice[Elem](s))
}

var ints = []int{74, 59, 238, -784, 9845, 959, 905, 0, 0, 42, 7586, -5467984, 7586}
var float64s = []float64{74.3, 59.0, math.Inf(1), 238.2, -784.0, 2.3, math.NaN(), math.NaN(), math.Inf(-1), 9845.768, -959.7485, 905, 7.8, 7.8}
var strings = []string{"", "Hello", "foo", "bar", "foo", "f00", "%*&^*&^&", "***"}

func TestSortOrderedInts() bool {
	return testOrdered("ints", ints, sort.Ints)
}

func TestSortOrderedFloat64s() bool {
	return testOrdered("float64s", float64s, sort.Float64s)
}

func TestSortOrderedStrings() bool {
	return testOrdered("strings", strings, sort.Strings)
}

func testOrdered[Elem Ordered](name string, s []Elem, sorter func([]Elem)) bool {
	s1 := make([]Elem, len(s))
	copy(s1, s)
	s2 := make([]Elem, len(s))
	copy(s2, s)
	_OrderedSlice(s1)
	sorter(s2)
	ok := true
	if !sliceEq(s1, s2) {
		fmt.Printf("%s: got %v, want %v", name, s1, s2)
		ok = false
	}
	for i := len(s1) - 1; i > 0; i-- {
		if s1[i] < s1[i-1] {
			fmt.Printf("%s: element %d (%v) < element %d (%v)", name, i, s1[i], i-1, s1[i-1])
			ok = false
		}
	}
	return ok
}

func sliceEq[Elem Ordered](s1, s2 []Elem) bool {
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

func main() {
	if !TestSortOrderedInts() || !TestSortOrderedFloat64s() || !TestSortOrderedStrings() {
		panic("failure")
	}
}
