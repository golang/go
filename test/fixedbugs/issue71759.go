// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func f(p *[2]int32) (int64, int64) {
	return int64(p[0]), int64(p[1])
}

func main() {
	p := [2]int32{-1, -1}
	x, y := f(&p)
	if x != -1 || y != -1 {
		println(x, y)
	}
}
