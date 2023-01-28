// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type C1 interface {
	comparable
}

type C2 interface {
	comparable
	[2]any | int
}

func G1[T C1](t T) { _ = t == t }
func G2[T C2](t T) { _ = t == t }

func F1[V [2]any](v V) {
	_ = G1[V /* ERROR "V does not satisfy comparable" */]
	_ = G1[[2]any]
	_ = G1[int]
}

func F2[V [2]any](v V) {
	_ = G2[V /* ERROR "V does not satisfy C2" */]
	_ = G2[[ /* ERROR "[2]any does not satisfy C2 (C2 mentions [2]any, but [2]any is not in the type set of C2)" */ 2]any]
	_ = G2[int]
}
