// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "a"

type Model[T any] struct {
	index       a.Index[T]
}

func NewModel[T any](index a.Index[T]) Model[T] {
	return Model[T]{
		index:       index,
	}
}

func main() {
	_ = NewModel[int]((*a.I1[int])(nil))
}
