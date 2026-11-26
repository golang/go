// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func f(a, b int8, c, d bool) int8 {
	v := a * b
	if c {
		if d {
			g()
		}
		h(int32(v))
	}
	return v
}

//go:noinline
func g() {}

//go:noinline
func h(x int32) {
	if x != 64 {
		panic(x)
	}
}

func main() {
	f(0x40, 0x41, true, false)
}
