// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// from https://research.swtch.com/sparse
// in turn, from Briggs and Torczon

// sparseKey needs to be something we can index a slice with.
type sparseKey interface{ ~int | ~int32 }

type sparseEntry[K sparseKey, V any] struct {
	key K
	val V
}

type genericSparseMap[K sparseKey, V any] struct {
	dense  []sparseEntry[K, V]
	sparse []int32
}

// newGenericSparseMap returns a sparseMap that can map
// integers between 0 and n-1 to a value type.
func newGenericSparseMap[K sparseKey, V any](n int) *genericSparseMap[K, V] {
	return &genericSparseMap[K, V]{dense: nil, sparse: make([]int32, n)}
}

func (s *genericSparseMap[K, V]) cap() int {
	return len(s.sparse)
}

func (s *genericSparseMap[K, V]) size() int {
	return len(s.dense)
}

func (s *genericSparseMap[K, V]) contains(k K) bool {
	i := s.sparse[k]
	return i < int32(len(s.dense)) && s.dense[i].key == k
}

// get returns the value for key k, or the zero V
// if k does not appear in the map.
func (s *genericSparseMap[K, V]) get(k K) (V, bool) {
	i := s.sparse[k]
	if i < int32(len(s.dense)) && s.dense[i].key == k {
		return s.dense[i].val, true
	}
	var v V
	return v, false
}

func (s *genericSparseMap[K, V]) set(k K, v V) {
	i := s.sparse[k]
	if i < int32(len(s.dense)) && s.dense[i].key == k {
		s.dense[i].val = v
		return
	}
	s.dense = append(s.dense, sparseEntry[K, V]{k, v})
	s.sparse[k] = int32(len(s.dense)) - 1
}

func (s *genericSparseMap[K, V]) remove(k K) {
	i := s.sparse[k]
	if i < int32(len(s.dense)) && s.dense[i].key == k {
		y := s.dense[len(s.dense)-1]
		s.dense[i] = y
		s.sparse[y.key] = i
		s.dense = s.dense[:len(s.dense)-1]
	}
}

func (s *genericSparseMap[K, V]) clear() {
	s.dense = s.dense[:0]
}

func (s *genericSparseMap[K, V]) contents() []sparseEntry[K, V] {
	return s.dense
}

type sparseMap = genericSparseMap[ID, int32]

// newSparseMap returns a sparseMap that can map
// integers between 0 and n-1 to int32s.
func newSparseMap(n int) *sparseMap {
	return newGenericSparseMap[ID, int32](n)
}
