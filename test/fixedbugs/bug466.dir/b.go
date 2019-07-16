// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

func main() {
	s := a.Func()
	if s[0] != 1 {
		println(s[0])
		panic("s[0] != 1")
	}
	if s[1] != 2+3i {
		println(s[1])
		panic("s[1] != 2+3i")
	}
	if s[2] != 4+5i {
		println(s[2])
		panic("s[2] != 4+5i")
	}

	x := 1 + 2i
	y := a.Mul(x)
	if y != (1+2i)*(3+4i) {
		println(y)
		panic("y != (1+2i)*(3+4i)")
	}
}
