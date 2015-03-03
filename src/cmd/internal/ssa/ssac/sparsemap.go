// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Maintains a map[int]*ssa.Value, but cheaper.

// from http://research.swtch.com/sparse
// in turn, from Briggs and Torczon

import (
	"cmd/internal/ssa"
)

type SparseMap struct {
	dense  []SparseMapEntry
	sparse []int
}
type SparseMapEntry struct {
	Key int
	Val *ssa.Value
}

// NewSparseMap returns a SparseMap that can have
// integers between 0 and n-1 as keys.
func NewSparseMap(n int) *SparseMap {
	return &SparseMap{nil, make([]int, n)}
}

func (s *SparseMap) Get(x int) *ssa.Value {
	i := s.sparse[x]
	if i < len(s.dense) && s.dense[i].Key == x {
		return s.dense[i].Val
	}
	return nil
}

func (s *SparseMap) Put(x int, v *ssa.Value) {
	i := s.sparse[x]
	if i < len(s.dense) && s.dense[i].Key == x {
		s.dense[i].Val = v
		return
	}
	i = len(s.dense)
	s.dense = append(s.dense, SparseMapEntry{x, v})
	s.sparse[x] = i
}

func (s *SparseMap) Remove(x int) {
	i := s.sparse[x]
	if i < len(s.dense) && s.dense[i].Key == x {
		y := s.dense[len(s.dense)-1]
		s.dense[i] = y
		s.sparse[y.Key] = i
		s.dense = s.dense[:len(s.dense)-1]
	}
}

func (s *SparseMap) Clear() {
	s.dense = s.dense[:0]
}

// Contents returns a slice of key/value pairs.
// Caller must not modify any returned entries.
// The return value is invalid after the SparseMap is modified in any way.
func (s *SparseMap) Contents() []SparseMapEntry {
	return s.dense
}
