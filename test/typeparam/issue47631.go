// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: one day we will support internal type declarations, at which time this test will be removed.

package p

func g[T any]() {
	type U []T // ERROR "type declarations inside generic functions are not currently supported"
	type V []int // ERROR "type declarations inside generic functions are not currently supported"
}

type S[T any] struct {
}

func (s S[T]) m() {
	type U []T // ERROR "type declarations inside generic functions are not currently supported"
	type V []int // ERROR "type declarations inside generic functions are not currently supported"
}


func f() {
	type U []int // ok
}

type X struct {
}

func (x X) m() {
	type U []int // ok
}
