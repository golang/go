// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"a"
	"fmt"
	"sort"
)

func TestSet() {
	s1 := a.Make[int]()
	if got := s1.Len(); got != 0 {
		panic(fmt.Sprintf("Len of empty set = %d, want 0", got))
	}
	s1.Add(1)
	s1.Add(1)
	s1.Add(1)
	if got := s1.Len(); got != 1 {
		panic(fmt.Sprintf("(%v).Len() == %d, want 1", s1, got))
	}
	s1.Add(2)
	s1.Add(3)
	s1.Add(4)
	if got := s1.Len(); got != 4 {
		panic(fmt.Sprintf("(%v).Len() == %d, want 4", s1, got))
	}
	if !s1.Contains(1) {
		panic(fmt.Sprintf("(%v).Contains(1) == false, want true", s1))
	}
	if s1.Contains(5) {
		panic(fmt.Sprintf("(%v).Contains(5) == true, want false", s1))
	}
	vals := s1.Values()
	sort.Ints(vals)
	w1 := []int{1, 2, 3, 4}
	if !a.SliceEqual(vals, w1) {
		panic(fmt.Sprintf("(%v).Values() == %v, want %v", s1, vals, w1))
	}
}

func TestEqual() {
	s1 := a.Make[string]()
	s2 := a.Make[string]()
	if !a.Equal(s1, s2) {
		panic(fmt.Sprintf("a.Equal(%v, %v) = false, want true", s1, s2))
	}
	s1.Add("hello")
	s1.Add("world")
	if got := s1.Len(); got != 2 {
		panic(fmt.Sprintf("(%v).Len() == %d, want 2", s1, got))
	}
	if a.Equal(s1, s2) {
		panic(fmt.Sprintf("a.Equal(%v, %v) = true, want false", s1, s2))
	}
}

func TestCopy() {
	s1 := a.Make[float64]()
	s1.Add(0)
	s2 := s1.Copy()
	if !a.Equal(s1, s2) {
		panic(fmt.Sprintf("a.Equal(%v, %v) = false, want true", s1, s2))
	}
	s1.Add(1)
	if a.Equal(s1, s2) {
		panic(fmt.Sprintf("a.Equal(%v, %v) = true, want false", s1, s2))
	}
}

func TestAddSet() {
	s1 := a.Make[int]()
	s1.Add(1)
	s1.Add(2)
	s2 := a.Make[int]()
	s2.Add(2)
	s2.Add(3)
	s1.AddSet(s2)
	if got := s1.Len(); got != 3 {
		panic(fmt.Sprintf("(%v).Len() == %d, want 3", s1, got))
	}
	s2.Add(1)
	if !a.Equal(s1, s2) {
		panic(fmt.Sprintf("a.Equal(%v, %v) = false, want true", s1, s2))
	}
}

func TestSubSet() {
	s1 := a.Make[int]()
	s1.Add(1)
	s1.Add(2)
	s2 := a.Make[int]()
	s2.Add(2)
	s2.Add(3)
	s1.SubSet(s2)
	if got := s1.Len(); got != 1 {
		panic(fmt.Sprintf("(%v).Len() == %d, want 1", s1, got))
	}
	if vals, want := s1.Values(), []int{1}; !a.SliceEqual(vals, want) {
		panic(fmt.Sprintf("after SubSet got %v, want %v", vals, want))
	}
}

func TestIntersect() {
	s1 := a.Make[int]()
	s1.Add(1)
	s1.Add(2)
	s2 := a.Make[int]()
	s2.Add(2)
	s2.Add(3)
	s1.Intersect(s2)
	if got := s1.Len(); got != 1 {
		panic(fmt.Sprintf("(%v).Len() == %d, want 1", s1, got))
	}
	if vals, want := s1.Values(), []int{2}; !a.SliceEqual(vals, want) {
		panic(fmt.Sprintf("after Intersect got %v, want %v", vals, want))
	}
}

func TestIterate() {
	s1 := a.Make[int]()
	s1.Add(1)
	s1.Add(2)
	s1.Add(3)
	s1.Add(4)
	tot := 0
	s1.Iterate(func(i int) { tot += i })
	if tot != 10 {
		panic(fmt.Sprintf("total of %v == %d, want 10", s1, tot))
	}
}

func TestFilter() {
	s1 := a.Make[int]()
	s1.Add(1)
	s1.Add(2)
	s1.Add(3)
	s1.Filter(func(v int) bool { return v%2 == 0 })
	if vals, want := s1.Values(), []int{2}; !a.SliceEqual(vals, want) {
		panic(fmt.Sprintf("after Filter got %v, want %v", vals, want))
	}

}

func main() {
	TestSet()
	TestEqual()
	TestCopy()
	TestAddSet()
	TestSubSet()
	TestIntersect()
	TestIterate()
	TestFilter()
}
