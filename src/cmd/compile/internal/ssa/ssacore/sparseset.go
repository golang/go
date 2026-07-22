// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

// NewSparseSet returns a sparseSet that can represent
// integers between 0 and n-1.
func NewSparseSet(n int) *SparseSet {
	return &SparseSet{dense: nil, sparse: make([]int32, n)}
}

// from https://research.swtch.com/sparse
// in turn, from Briggs and Torczon

type SparseSet struct {
	dense  []ID
	sparse []int32
}

func (s *SparseSet) cap() int {
	return len(s.sparse)
}

func (s *SparseSet) Size() int {
	return len(s.dense)
}

func (s *SparseSet) Contains(x ID) bool {
	i := s.sparse[x]
	return i < int32(len(s.dense)) && s.dense[i] == x
}

func (s *SparseSet) Add(x ID) {
	i := s.sparse[x]
	if i < int32(len(s.dense)) && s.dense[i] == x {
		return
	}
	s.dense = append(s.dense, x)
	s.sparse[x] = int32(len(s.dense)) - 1
}

func (s *SparseSet) addAll(a []ID) {
	for _, x := range a {
		s.Add(x)
	}
}

func (s *SparseSet) addAllValues(a []*Value) {
	for _, v := range a {
		s.Add(v.ID)
	}
}

func (s *SparseSet) Remove(x ID) {
	i := s.sparse[x]
	if i < int32(len(s.dense)) && s.dense[i] == x {
		y := s.dense[len(s.dense)-1]
		s.dense[i] = y
		s.sparse[y] = i
		s.dense = s.dense[:len(s.dense)-1]
	}
}

// Pop removes an arbitrary element from the set.
// The set must be nonempty.
func (s *SparseSet) Pop() ID {
	x := s.dense[len(s.dense)-1]
	s.dense = s.dense[:len(s.dense)-1]
	return x
}

func (s *SparseSet) Clear() {
	s.dense = s.dense[:0]
}

func (s *SparseSet) Contents() []ID {
	return s.dense
}
