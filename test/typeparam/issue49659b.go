// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Testing that AddrTaken logic doesn't cause problems for function instantiations

package main

type A[T interface{ []int | [5]int }] struct {
	val T
}

//go:noinline
func (a A[T]) F() {
	_ = &a.val[2]
}

func main() {
	var x A[[]int]
	x.val = make([]int, 4)
	_ = &x.val[3]
	x.F()
	var y A[[5]int]
	_ = &y.val[3]
	y.F()
}
