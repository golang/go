// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type I interface {
	M()
}

type S struct {
	str string
}

func (s *S) M() {}

var _ I = &S{}

type CloningMap[K comparable, V any] struct {
	inner map[K]V
}

func (cm CloningMap[K, V]) With(key K, value V) CloningMap[K, V] {
	result := CloneBad(cm.inner)
	result[key] = value
	return CloningMap[K, V]{result}
}

func CloneBad[M ~map[K]V, K comparable, V any](m M) M {
	r := make(M, len(m))
	for k, v := range m {
		r[k] = v
	}
	return r
}

func main() {
	s1 := &S{"one"}
	s2 := &S{"two"}

	m := CloningMap[string, I]{inner: make(map[string]I)}
	m = m.With("a", s1)
	m = m.With("b", s2)

	it, found := m.inner["a"]
	if !found {
		panic("a not found")
	}
	if _, ok := it.(*S); !ok {
		panic(fmt.Sprintf("got %T want *main.S", it))
	}
}
