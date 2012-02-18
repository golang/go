// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	bad := false
	if (-5 >> 1) != -3 {
		println("-5>>1 =", -5>>1, "want -3")
		bad = true
	}
	if (-4 >> 1) != -2 {
		println("-4>>1 =", -4>>1, "want -2")
		bad = true
	}
	if (-3 >> 1) != -2 {
		println("-3>>1 =", -3>>1, "want -2")
		bad = true
	}
	if (-2 >> 1) != -1 {
		println("-2>>1 =", -2>>1, "want -1")
		bad = true
	}
	if (-1 >> 1) != -1 {
		println("-1>>1 =", -1>>1, "want -1")
		bad = true
	}
	if bad {
		println("errors")
		panic("fail")
	}
}
