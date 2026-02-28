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
}
func f() {
	for i := int64(math.MinInt64); i >= math.MinInt64; i-- {
		if i > 0 {
			println("done")
			return
		}
		println(i, i > 0)
	}
}
func g() {
	for i := int64(math.MinInt64) + 1; i >= math.MinInt64; i-- {
		if i > 0 {
			println("done")
			return
		}
		println(i, i > 0)
	}
}
func h() {
	for i := int64(math.MinInt64) + 2; i >= math.MinInt64; i -= 2 {
		if i > 0 {
			println("done")
			return
		}
		println(i, i > 0)
	}
}
