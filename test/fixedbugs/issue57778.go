// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type FreeListG[T any] struct {
	freelist []*node[T]
}

type node[T any] struct{}

func NewFreeListG[T any](size int) *FreeListG[T] {
	return &FreeListG[T]{freelist: make([]*node[T], 0, size)}
}

var bf = NewFreeListG[*int](1024)
