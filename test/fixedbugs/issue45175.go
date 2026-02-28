// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func f(c bool) int {
	b := true
	x := 0
	y := 1
	for b {
		b = false
		y = x
		x = 2
		if c {
			return 3
		}
	}
	return y
}

func main() {
	if got := f(false); got != 0 {
		panic(got)
	}
}
