// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4348. After switch to 64-bit ints the compiler generates
// illegal instructions when using large array bounds or indexes.

package main

// 1<<32 on a 64-bit machine, 1 otherwise.
const LARGE = ^uint(0)>>32 + 1

func A() int {
	var a []int
	return a[LARGE]
}

var b [LARGE]int

func B(i int) int {
	return b[i]
}

func main() {
	n := A()
	B(n)
}
