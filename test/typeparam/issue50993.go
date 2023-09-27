// compile -d=checkptr

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"sync/atomic"
	"unsafe"
)

type Node[T any] struct {
	Next *Node[T]
	// Prev  *Node[T]
}

func LoadPointer[T any](addr **T) (val *T) {
	return (*T)(
		atomic.LoadPointer(
			(*unsafe.Pointer)(unsafe.Pointer(addr)),
		))
}

func (q *Node[T]) Pop() {
	var tail, head *Node[T]
	if head == LoadPointer(&tail) {
	}
}

func main() {
	ch := Node[uint64]{}
	ch.Pop()
}
