// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"sort"

	"./a"
)

func main() {
	c := make(chan int, 1)
	d := make(chan int, 1)

	c <- 5
	d <- 6

	var r [2]int
	r[0] = a.F(c, d)
	r[1] = a.F(c, d)
	sort.Ints(r[:])

	if r != [2]int{5, 6} {
		panic("incorrect results")
	}
}
