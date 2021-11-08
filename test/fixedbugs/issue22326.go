// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var (
	_ = d
	_ = f("_", c, b)
	a = f("a")
	b = f("b")
	c = f("c")
	d = f("d")
)

func f(s string, rest ...int) int {
	print(s)
	return 0
}

func main() {
	println()
}
