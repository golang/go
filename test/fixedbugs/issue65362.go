// compile

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Vector[V any] interface {
	ReadVector[V]
}

type ReadVector[V any] interface {
	Comparisons[ReadVector[V], Vector[V]]
}

type Comparisons[RV, V any] interface {
	Diff(RV) V
}

type VectorImpl[V any] struct{}

func (*VectorImpl[V]) Diff(ReadVector[V]) (_ Vector[V]) {
	return
}

func main() {
	var v1 VectorImpl[int]
	var v2 Vector[int]
	_ = v1.Diff(v2)
}
