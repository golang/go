// compile -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T[A, B, C any] int

func (T[A, B, C]) m(x int) {
	if x <= 0 {
		return
	}
	T[B, C, A](0).m(x - 1)
}

func main() {
	T[int8, int16, int32](0).m(3)
}
