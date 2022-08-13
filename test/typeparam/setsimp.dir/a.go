// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

// SliceEqual reports whether two slices are equal: the same length and all
// elements equal. All floating point NaNs are considered equal.
func SliceEqual[Elem comparable](s1, s2 []Elem) bool {
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

// A Set is a set of elements of some type.
type Set[Elem comparable] struct {
	m map[Elem]struct{}
}

// Make makes a new set.
func Make[Elem comparable]() Set[Elem] {
	return Set[Elem]{m: make(map[Elem]struct{})}
}

// Add adds an element to a set.
func (s Set[Elem]) Add(v Elem) {
	s.m[v] = struct{}{}
}

// Delete removes an element from a set. If the element is not present
// in the set, this does nothing.
func (s Set[Elem]) Delete(v Elem) {
	delete(s.m, v)
}

// Contains reports whether v is in the set.
func (s Set[Elem]) Contains(v Elem) bool {
	_, ok := s.m[v]
	return ok
}

// Len returns the number of elements in the set.
func (s Set[Elem]) Len() int {
	return len(s.m)
}

// Values returns the values in the set.
// The values will be in an indeterminate order.
func (s Set[Elem]) Values() []Elem {
	r := make([]Elem, 0, len(s.m))
	for v := range s.m {
		r = append(r, v)
	}
	return r
}

// Equal reports whether two sets contain the same elements.
func Equal[Elem comparable](s1, s2 Set[Elem]) bool {
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
func (s Set[Elem]) Copy() Set[Elem] {
	r := Set[Elem]{m: make(map[Elem]struct{}, len(s.m))}
	for v := range s.m {
		r.m[v] = struct{}{}
	}
	return r
}

// AddSet adds all the elements of s2 to s.
func (s Set[Elem]) AddSet(s2 Set[Elem]) {
	for v := range s2.m {
		s.m[v] = struct{}{}
	}
}

// SubSet removes all elements in s2 from s.
// Values in s2 that are not in s are ignored.
func (s Set[Elem]) SubSet(s2 Set[Elem]) {
	for v := range s2.m {
		delete(s.m, v)
	}
}

// Intersect removes all elements from s that are not present in s2.
// Values in s2 that are not in s are ignored.
func (s Set[Elem]) Intersect(s2 Set[Elem]) {
	for v := range s.m {
		if !s2.Contains(v) {
			delete(s.m, v)
		}
	}
}

// Iterate calls f on every element in the set.
func (s Set[Elem]) Iterate(f func(Elem)) {
	for v := range s.m {
		f(v)
	}
}

// Filter deletes any elements from s for which f returns false.
func (s Set[Elem]) Filter(f func(Elem) bool) {
	for v := range s.m {
		if !f(v) {
			delete(s.m, v)
		}
	}
}
