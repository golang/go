// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gccgo incorrectly rejected an assignment to multiple instances of
// the same variable.

package main

var a int

func F() {
	a, a, a = 1, 2, 3
}

func main() {
	F()
	if a != 3 {
		panic(a)
	}
}
