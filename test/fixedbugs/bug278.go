// errchk $G $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a test case for issue 804.

package main

func f() [10]int {
	return [10]int{}
}

var m map[int][10]int

func main() {
	f()[1] = 2	// ERROR "cannot"
	f()[2:3][0] = 4	// ERROR "cannot"
	var x = "abc"
	x[2] = 3	// ERROR "cannot"
	m[0][5] = 6  // ERROR "cannot"
}
