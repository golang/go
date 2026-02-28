// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// from https://research.swtch.com/sparse
// in turn, from Briggs and Torczon

type sparseSet struct {
	dense  []ID
	sparse []int32
}

// newSparseSet returns a sparseSet that can represent
// integers between 0 and n-1
func newSparseSet(n int) *sparseSet {
	return &sparseSet{dense: nil, sparse: make([]int32, n)}
}

func (s *sparseSet) cap() int {
	return len(s.sparse)
}

func (s *sparseSet) size() int {
	return len(s.dense)
}

func (s *sparseSet) contains(x ID) bool {
	i := s.sparse[x]
	return i < int32(len(s.dense)) && s.dense[i] == x
}

func (s *sparseSet) add(x ID) {
	i := s.sparse[x]
	if i < int32(len(s.dense)) && s.dense[i] == x {
		return
	}
	s.dense = append(s.dense, x)
	s.sparse[x] = int32(len(s.dense)) - 1
}

func (s *sparseSet) addAll(a []ID) {
	for _, x := range a {
		s.add(x)
	}
}

func (s *sparseSet) addAllValues(a []*Value) {
	for _, v := range a {
		s.add(v.ID)
	}
}

func (s *sparseSet) remove(x ID) {
	i := s.sparse[x]
	if i < int32(len(s.dense)) && s.dense[i] == x {
		y := s.dense[len(s.dense)-1]
		s.dense[i] = y
		s.sparse[y] = i
		s.dense = s.dense[:len(s.dense)-1]
	}
}

// pop removes an arbitrary element from the set.
// The set must be nonempty.
func (s *sparseSet) pop() ID {
	x := s.dense[len(s.dense)-1]
	s.dense = s.dense[:len(s.dense)-1]
	return x
}

func (s *sparseSet) clear() {
	s.dense = s.dense[:0]
}

func (s *sparseSet) contents() []ID {
	return s.dense
}
