// run

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 3907: out of fixed registers in nested byte multiply.
// Used to happen with both 6g and 8g.

package main

func F(a, b, c, d uint8) uint8 {
	return a * (b * (c * (d *
		(a * (b * (c * (d *
			(a * (b * (c * (d *
				a * (b * (c * d)))))))))))))
}

func main() {
	var a, b, c, d uint8 = 1, 1, 1, 1
	x := F(a, b, c, d)
	if x != 1 {
		println(x)
		panic("x != 1")
	}
}
