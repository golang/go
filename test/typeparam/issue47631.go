// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func g[T any]() {
	type U []T
	type V []int
}

type S[T any] struct {
}

func (s S[T]) m() {
	type U []T
	type V []int
}

func f() {
	type U []int
}

type X struct {
}

func (x X) m() {
	type U []int
}
