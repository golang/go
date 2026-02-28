// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "cmd/internal/src"

// from https://research.swtch.com/sparse
// in turn, from Briggs and Torczon

type sparseEntryPos struct {
	key ID
	val int32
	pos src.XPos
}

type sparseMapPos struct {
	dense  []sparseEntryPos
	sparse []int32
}

// newSparseMapPos returns a sparseMapPos that can map
// integers between 0 and n-1 to the pair <int32,src.XPos>.
func newSparseMapPos(n int) *sparseMapPos {
	return &sparseMapPos{dense: nil, sparse: make([]int32, n)}
}

func (s *sparseMapPos) cap() int {
	return len(s.sparse)
}

func (s *sparseMapPos) size() int {
	return len(s.dense)
}

func (s *sparseMapPos) contains(k ID) bool {
	i := s.sparse[k]
	return i < int32(len(s.dense)) && s.dense[i].key == k
}

// get returns the value for key k, or -1 if k does
// not appear in the map.
func (s *sparseMapPos) get(k ID) int32 {
	i := s.sparse[k]
	if i < int32(len(s.dense)) && s.dense[i].key == k {
		return s.dense[i].val
	}
	return -1
}

func (s *sparseMapPos) set(k ID, v int32, a src.XPos) {
	i := s.sparse[k]
	if i < int32(len(s.dense)) && s.dense[i].key == k {
		s.dense[i].val = v
		s.dense[i].pos = a
		return
	}
	s.dense = append(s.dense, sparseEntryPos{k, v, a})
	s.sparse[k] = int32(len(s.dense)) - 1
}

func (s *sparseMapPos) remove(k ID) {
	i := s.sparse[k]
	if i < int32(len(s.dense)) && s.dense[i].key == k {
		y := s.dense[len(s.dense)-1]
		s.dense[i] = y
		s.sparse[y.key] = i
		s.dense = s.dense[:len(s.dense)-1]
	}
}

func (s *sparseMapPos) clear() {
	s.dense = s.dense[:0]
}

func (s *sparseMapPos) contents() []sparseEntryPos {
	return s.dense
}
