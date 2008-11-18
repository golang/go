// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import Fmt "fmt"


type T struct {
	x, y int;
	s string;
	next_t *T
}


var (
	A = 5;
	a, b, c int = 0, 0, 0;
	foo = "foo";
)


func main() {
// the prologue
	for i := 0; i <= 10 /* limit */; i++ {
		println(i);  // the index
		println(i + 1);  // the index + 1
		println(i + 1000);  // the index + 1000
		println();
	}
// the epilogue
	println("foo");  // foo
	println("foobar");  // foobar
var x int;
}
