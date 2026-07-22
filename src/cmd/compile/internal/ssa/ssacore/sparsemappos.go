// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

import "cmd/internal/src"

// from https://research.swtch.com/sparse
// in turn, from Briggs and Torczon

type SparseEntryPos struct {
	Key ID
	Val int32
	Pos src.XPos
}

type SparseMapPos struct {
	dense  []SparseEntryPos
	sparse []int32
}

// newSparseMapPos returns a sparseMapPos that can map
// integers between 0 and n-1 to the pair <int32,src.XPos>.
func newSparseMapPos(n int) *SparseMapPos {
	return &SparseMapPos{dense: nil, sparse: make([]int32, n)}
}

func (s *SparseMapPos) cap() int {
	return len(s.sparse)
}

func (s *SparseMapPos) Size() int {
	return len(s.dense)
}

func (s *SparseMapPos) Contains(k ID) bool {
	i := s.sparse[k]
	return i < int32(len(s.dense)) && s.dense[i].Key == k
}

// Get returns the value for key k, or -1 if k does
// not appear in the map.
func (s *SparseMapPos) Get(k ID) int32 {
	i := s.sparse[k]
	if i < int32(len(s.dense)) && s.dense[i].Key == k {
		return s.dense[i].Val
	}
	return -1
}

func (s *SparseMapPos) Set(k ID, v int32, a src.XPos) {
	i := s.sparse[k]
	if i < int32(len(s.dense)) && s.dense[i].Key == k {
		s.dense[i].Val = v
		s.dense[i].Pos = a
		return
	}
	s.dense = append(s.dense, SparseEntryPos{k, v, a})
	s.sparse[k] = int32(len(s.dense)) - 1
}

func (s *SparseMapPos) Remove(k ID) {
	i := s.sparse[k]
	if i < int32(len(s.dense)) && s.dense[i].Key == k {
		y := s.dense[len(s.dense)-1]
		s.dense[i] = y
		s.sparse[y.Key] = i
		s.dense = s.dense[:len(s.dense)-1]
	}
}

func (s *SparseMapPos) Clear() {
	s.dense = s.dense[:0]
}

func (s *SparseMapPos) Contents() []SparseEntryPos {
	return s.dense
}
