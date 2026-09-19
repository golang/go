// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "math"

// A BiasedSparseMap is a sparseMap for integers between J and K inclusive,
// where J might be somewhat larger than zero (and K-J is probably much smaller than J).
// (The motivating use case is the line numbers of statements for a single function.)
// Not all features of a SparseMap are exported, and it is also easy to treat a
// BiasedSparseMap like a SparseSet.
type BiasedSparseMap struct {
	s     *SparseMap
	first int
}

// newBiasedSparseMap returns a new biasedSparseMap for values between first and last, inclusive.
func newBiasedSparseMap(first, last int) *BiasedSparseMap {
	if first > last {
		return &BiasedSparseMap{first: math.MaxInt32, s: nil}
	}
	return &BiasedSparseMap{first: first, s: NewSparseMap(1 + last - first)}
}

// cap returns one more than the largest key valid for s
func (s *BiasedSparseMap) cap() int {
	if s == nil || s.s == nil {
		return 0
	}
	return s.s.cap() + s.first
}

// size returns the number of entries stored in s
func (s *BiasedSparseMap) size() int {
	if s == nil || s.s == nil {
		return 0
	}
	return s.s.size()
}

// contains reports whether x is a key in s
func (s *BiasedSparseMap) contains(x uint) bool {
	if s == nil || s.s == nil {
		return false
	}
	if int(x) < s.first {
		return false
	}
	if int(x) >= s.cap() {
		return false
	}
	return s.s.Contains(ID(int(x) - s.first))
}

// get returns the value s maps for key x and true, or
// 0/false if x is not mapped or is out of range for s.
func (s *BiasedSparseMap) get(x uint) (int32, bool) {
	if s == nil || s.s == nil {
		return 0, false
	}
	if int(x) < s.first {
		return 0, false
	}
	if int(x) >= s.cap() {
		return 0, false
	}
	k := ID(int(x) - s.first)
	if !s.s.Contains(k) {
		return 0, false
	}
	return s.s.Get(k)
}

// getEntry returns the i'th key and value stored in s,
// where 0 <= i < s.size()
func (s *BiasedSparseMap) getEntry(i int) (x uint, v int32) {
	e := s.s.contents()[i]
	x = uint(int(e.key) + s.first)
	v = e.val
	return
}

// add inserts x->v into s, provided that x is in the range of keys stored in s.
func (s *BiasedSparseMap) set(x uint, v int32) {
	if int(x) < s.first || int(x) >= s.cap() {
		return
	}
	s.s.Set(ID(int(x)-s.first), v)
}

// remove removes key x from s.
func (s *BiasedSparseMap) remove(x uint) {
	if int(x) < s.first || int(x) >= s.cap() {
		return
	}
	s.s.remove(ID(int(x) - s.first))
}

func (s *BiasedSparseMap) clear() {
	if s.s != nil {
		s.s.Clear()
	}
}
