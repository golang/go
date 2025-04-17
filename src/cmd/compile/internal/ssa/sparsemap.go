// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// from https://research.swtch.com/sparse
// in turn, from Briggs and Torczon

type sparseEntry struct {
	key ID
	val int32
}

type sparseMap struct {
	dense  []sparseEntry
	sparse []int32
}

// newSparseMap returns a sparseMap that can map
// integers between 0 and n-1 to int32s.
func newSparseMap(n int) *sparseMap {
	return &sparseMap{dense: nil, sparse: make([]int32, n)}
}

func (s *sparseMap) cap() int {
	return len(s.sparse)
}

func (s *sparseMap) size() int {
	return len(s.dense)
}

func (s *sparseMap) contains(k ID) bool {
	i := s.sparse[k]
	return i < int32(len(s.dense)) && s.dense[i].key == k
}

// get returns the value for key k, or -1 if k does
// not appear in the map.
func (s *sparseMap) get(k ID) int32 {
	i := s.sparse[k]
	if i < int32(len(s.dense)) && s.dense[i].key == k {
		return s.dense[i].val
	}
	return -1
}

func (s *sparseMap) set(k ID, v int32) {
	i := s.sparse[k]
	if i < int32(len(s.dense)) && s.dense[i].key == k {
		s.dense[i].val = v
		return
	}
	s.dense = append(s.dense, sparseEntry{k, v})
	s.sparse[k] = int32(len(s.dense)) - 1
}

// setBit sets the v'th bit of k's value, where 0 <= v < 32
func (s *sparseMap) setBit(k ID, v uint) {
	if v >= 32 {
		panic("bit index too large.")
	}
	i := s.sparse[k]
	if i < int32(len(s.dense)) && s.dense[i].key == k {
		s.dense[i].val |= 1 << v
		return
	}
	s.dense = append(s.dense, sparseEntry{k, 1 << v})
	s.sparse[k] = int32(len(s.dense)) - 1
}

func (s *sparseMap) remove(k ID) {
	i := s.sparse[k]
	if i < int32(len(s.dense)) && s.dense[i].key == k {
		y := s.dense[len(s.dense)-1]
		s.dense[i] = y
		s.sparse[y.key] = i
		s.dense = s.dense[:len(s.dense)-1]
	}
}

func (s *sparseMap) clear() {
	s.dense = s.dense[:0]
}

func (s *sparseMap) contents() []sparseEntry {
	return s.dense
}
