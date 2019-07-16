// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gccgo used to crash compiling this.

package main

func foo() (int, int) {
	return 1, 2
}

var c = b
var a, b = foo()
var d = b + 1

func main() {
	if a != 1 {
		panic(a)
	}
	if b != 2 {
		panic(b)
	}
	if c != 2 {
		panic(c)
	}
	if d != 3 {
		panic(d)
	}
}
