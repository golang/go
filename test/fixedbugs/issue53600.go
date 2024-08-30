// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math"

func main() {
	f()
	g()
	h()
	j(math.MinInt64)
}
func f() {
	for i := int64(math.MaxInt64); i <= math.MaxInt64; i++ {
		if i < 0 {
			println("done")
			return
		}
		println(i, i < 0)
	}
}
func g() {
	for i := int64(math.MaxInt64) - 1; i <= math.MaxInt64; i++ {
		if i < 0 {
			println("done")
			return
		}
		println(i, i < 0)
	}
}
func h() {
	for i := int64(math.MaxInt64) - 2; i <= math.MaxInt64; i += 2 {
		if i < 0 {
			println("done")
			return
		}
		println(i, i < 0)
	}
}

//go:noinline
func j(i int64) {
	for j := int64(math.MaxInt64); j <= i-1; j++ {
		if j < 0 {
			break
		}
		println(j)
	}
}
