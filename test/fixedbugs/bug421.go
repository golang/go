// errorcheck

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1927.
// gccgo failed to issue the first error below.

package main

func main() {
	println(int(1) == uint(1))	// ERROR "types"
	var x int = 1
	var y uint = 1
	println(x == y)			// ERROR "types"
}
