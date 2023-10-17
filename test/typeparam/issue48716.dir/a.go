// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Pair[L, R any] struct {
	L L
	R R
}

func Two[L, R any](l L, r R) Pair[L, R] {
	return Pair[L, R]{L: l, R: r}
}

type Map[K, V any] interface {
	Put(K, V)
	Len() int
	Iterate(func(Pair[K, V]) bool)
}

type HashMap[K comparable, V any] struct {
	m map[K]V
}

func NewHashMap[K comparable, V any](capacity int) HashMap[K, V] {
	var m map[K]V
	if capacity >= 1 {
		m = make(map[K]V, capacity)
	} else {
		m = map[K]V{}
	}

	return HashMap[K, V]{m: m}
}

func (m HashMap[K, V]) Put(k K, v V) {
	m.m[k] = v
}

func (m HashMap[K, V]) Len() int {
	return len(m.m)
}

func (m HashMap[K, V]) Iterate(cb func(Pair[K, V]) bool) {
	for k, v := range m.m {
		if !cb(Two(k, v)) {
			return
		}
	}
}
