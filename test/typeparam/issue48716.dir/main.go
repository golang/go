// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
)

// Creates copy of set
func Copy[T comparable](src MapSet[T]) (dst MapSet[T]) {
	dst = HashSet[T](src.Len())
	Fill(src, dst)
	return
}

// Fill src from dst
func Fill[T any](src, dst MapSet[T]) {
	src.Iterate(func(t T) bool {
		dst.Add(t)
		return true
	})
	return
}

type MapSet[T any] struct {
	m a.Map[T, struct{}]
}

func HashSet[T comparable](capacity int) MapSet[T] {
	return FromMap[T](a.NewHashMap[T, struct{}](capacity))
}

func FromMap[T any](m a.Map[T, struct{}]) MapSet[T] {
	return MapSet[T]{
		m: m,
	}
}

func (s MapSet[T]) Add(t T) {
	s.m.Put(t, struct{}{})
}

func (s MapSet[T]) Len() int {
	return s.m.Len()
}

func (s MapSet[T]) Iterate(cb func(T) bool) {
	s.m.Iterate(func(p a.Pair[T, struct{}]) bool {
		return cb(p.L)
	})
}

func main() {
	x := FromMap[int](a.NewHashMap[int, struct{}](1))
	Copy[int](x)
}
