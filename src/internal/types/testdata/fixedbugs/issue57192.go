// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type I1[T any] interface {
	m1(T)
}
type I2[T any] interface {
	I1[T]
	m2(T)
}

var V1 I1[int]
var V2 I2[int]

func g[T any](I1[T]) {}
func _() {
	g(V1)
	g(V2)
}
