// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func A() {
	B[int](new(G[int]))
}

func B[T any](iface interface{ M(T) }) {
	x, ok := iface.(*G[T])
	if !ok || iface != x {
		panic("FAIL")
	}
}

type G[T any] struct{}

func (*G[T]) M(T) {}
