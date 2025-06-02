// compile

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type B[T any] struct {
	a A[T]
}

type A[T any] = func(B[T]) bool

func main() {
	var s A[int]
	println(s)
}
