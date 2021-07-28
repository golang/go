// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
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

// A _Set is a set of elements of some type.
type _Set[Elem comparable] struct {
	m map[Elem]struct{}
}

// _Make makes a new set.
func _Make[Elem comparable]() _Set[Elem] {
	return _Set[Elem]{m: make(map[Elem]struct{})}
}

// Add adds an element to a set.
func (s _Set[Elem]) Add(v Elem) {
	s.m[v] = struct{}{}
}

// Delete removes an element from a set. If the element is not present
// in the set, this does nothing.
func (s _Set[Elem]) Delete(v Elem) {
	delete(s.m, v)
}

// Contains reports whether v is in the set.
func (s _Set[Elem]) Contains(v Elem) bool {
	_, ok := s.m[v]
	return ok
}

// Len returns the number of elements in the set.
func (s _Set[Elem]) Len() int {
	return len(s.m)
}

// Values returns the values in the set.
// The values will be in an indeterminate order.
func (s _Set[Elem]) Values() []Elem {
	r := make([]Elem, 0, len(s.m))
	for v := range s.m {
		r = append(r, v)
	}
	return r
}

// _Equal reports whether two sets contain the same elements.
func _Equal[Elem comparable](s1, s2 _Set[Elem]) bool {
	if len(s1.m) != len(s2.m) {
		return false
	}
	for v1 := range s1.m {
		if !s2.Contains(v1) {
			return false
		}
	}
	return true
}

// Copy returns a copy of s.
func (s _Set[Elem]) Copy() _Set[Elem] {
	r := _Set[Elem]{m: make(map[Elem]struct{}, len(s.m))}
	for v := range s.m {
		r.m[v] = struct{}{}
	}
	return r
}

// AddSet adds all the elements of s2 to s.
func (s _Set[Elem]) AddSet(s2 _Set[Elem]) {
	for v := range s2.m {
		s.m[v] = struct{}{}
	}
}

// SubSet removes all elements in s2 from s.
// Values in s2 that are not in s are ignored.
func (s _Set[Elem]) SubSet(s2 _Set[Elem]) {
	for v := range s2.m {
		delete(s.m, v)
	}
}

// Intersect removes all elements from s that are not present in s2.
// Values in s2 that are not in s are ignored.
func (s _Set[Elem]) Intersect(s2 _Set[Elem]) {
	for v := range s.m {
		if !s2.Contains(v) {
			delete(s.m, v)
		}
	}
}

// Iterate calls f on every element in the set.
func (s _Set[Elem]) Iterate(f func(Elem)) {
	for v := range s.m {
		f(v)
	}
}

// Filter deletes any elements from s for which f returns false.
func (s _Set[Elem]) Filter(f func(Elem) bool) {
	for v := range s.m {
		if !f(v) {
			delete(s.m, v)
		}
	}
}

func TestSet() {
	s1 := _Make[int]()
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
	if !_SliceEqual(vals, w1) {
		panic(fmt.Sprintf("(%v).Values() == %v, want %v", s1, vals, w1))
	}
}

func TestEqual() {
	s1 := _Make[string]()
	s2 := _Make[string]()
	if !_Equal(s1, s2) {
		panic(fmt.Sprintf("_Equal(%v, %v) = false, want true", s1, s2))
	}
	s1.Add("hello")
	s1.Add("world")
	if got := s1.Len(); got != 2 {
		panic(fmt.Sprintf("(%v).Len() == %d, want 2", s1, got))
	}
	if _Equal(s1, s2) {
		panic(fmt.Sprintf("_Equal(%v, %v) = true, want false", s1, s2))
	}
}

func TestCopy() {
	s1 := _Make[float64]()
	s1.Add(0)
	s2 := s1.Copy()
	if !_Equal(s1, s2) {
		panic(fmt.Sprintf("_Equal(%v, %v) = false, want true", s1, s2))
	}
	s1.Add(1)
	if _Equal(s1, s2) {
		panic(fmt.Sprintf("_Equal(%v, %v) = true, want false", s1, s2))
	}
}

func TestAddSet() {
	s1 := _Make[int]()
	s1.Add(1)
	s1.Add(2)
	s2 := _Make[int]()
	s2.Add(2)
	s2.Add(3)
	s1.AddSet(s2)
	if got := s1.Len(); got != 3 {
		panic(fmt.Sprintf("(%v).Len() == %d, want 3", s1, got))
	}
	s2.Add(1)
	if !_Equal(s1, s2) {
		panic(fmt.Sprintf("_Equal(%v, %v) = false, want true", s1, s2))
	}
}

func TestSubSet() {
	s1 := _Make[int]()
	s1.Add(1)
	s1.Add(2)
	s2 := _Make[int]()
	s2.Add(2)
	s2.Add(3)
	s1.SubSet(s2)
	if got := s1.Len(); got != 1 {
		panic(fmt.Sprintf("(%v).Len() == %d, want 1", s1, got))
	}
	if vals, want := s1.Values(), []int{1}; !_SliceEqual(vals, want) {
		panic(fmt.Sprintf("after SubSet got %v, want %v", vals, want))
	}
}

func TestIntersect() {
	s1 := _Make[int]()
	s1.Add(1)
	s1.Add(2)
	s2 := _Make[int]()
	s2.Add(2)
	s2.Add(3)
	s1.Intersect(s2)
	if got := s1.Len(); got != 1 {
		panic(fmt.Sprintf("(%v).Len() == %d, want 1", s1, got))
	}
	if vals, want := s1.Values(), []int{2}; !_SliceEqual(vals, want) {
		panic(fmt.Sprintf("after Intersect got %v, want %v", vals, want))
	}
}

func TestIterate() {
	s1 := _Make[int]()
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
	s1 := _Make[int]()
	s1.Add(1)
	s1.Add(2)
	s1.Add(3)
	s1.Filter(func(v int) bool { return v%2 == 0 })
	if vals, want := s1.Values(), []int{2}; !_SliceEqual(vals, want) {
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
