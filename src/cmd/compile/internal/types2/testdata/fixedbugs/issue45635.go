// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	some /* ERROR "undeclared name" */ [int, int]()
}

type N[T any] struct{}

var _ N[] /* ERROR expected type */

type I interface {
	~[]int
}

func _[T I](i, j int) {
	var m map[int]int
	_ = m[i, j /* ERROR more than one index */ ]

	var a [3]int
	_ = a[i, j /* ERROR more than one index */ ]

	var s []int
	_ = s[i, j /* ERROR more than one index */ ]

	var t T
	_ = t[i, j /* ERROR more than one index */ ]
}
