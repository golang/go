// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to be rejected
// https://golang.org/issue/188

package main

func complexSqrt(i int) (int, int)	{ return 0, 1 }

var re, im = complexSqrt(-1)

func main() {
	if re != 0 || im != 1 {
		println("BUG: bug216: want 0,-1 have ", re, im)
	}
}
