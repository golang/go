// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Addr struct {
	hi uint64
	lo uint64
	z  *byte
}

func EqualMap[M1, M2 ~map[K]V, K, V comparable](m1 M1, m2 M2) bool {
	for k, v1 := range m1 {
		if v2, ok := m2[k]; !ok || v1 != v2 {
			return false
		}
	}
	return true
}

type Set[T comparable] map[T]struct{}

func NewSet[T comparable](items ...T) Set[T] {
	return nil
}

func (s Set[T]) Equals(other Set[T]) bool {
	return EqualMap(s, other)
}

func main() {
	NewSet[Addr](Addr{0, 0, nil})
}
