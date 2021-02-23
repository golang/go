// run

// Does not work with gccgo, which uses a smaller (but still permitted)
// exponent size.
// +build !gccgo

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Tests for golang.org/issue/11326.

func main() {
	{
		const n = 1e646456992
		const d = 1e646456991
		x := n / d
		if x != 10.0 {
			println("incorrect value:", x)
		}
	}
	{
		const n = 1e64645699
		const d = 1e64645698
		x := n / d
		if x != 10.0 {
			println("incorrect value:", x)
		}
	}
	{
		const n = 1e6464569
		const d = 1e6464568
		x := n / d
		if x != 10.0 {
			println("incorrect value:", x)
		}
	}
	{
		const n = 1e646456
		const d = 1e646455
		x := n / d
		if x != 10.0 {
			println("incorrect value:", x)
		}
	}
}
