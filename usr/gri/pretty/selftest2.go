// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"array";  // not needed
	"utf8";  // not needed
	Fmt "fmt"
)


const /* enum */ (
	EnumTag0 = iota;
	EnumTag1;
	EnumTag2;
	EnumTag3;
	EnumTag4;
	EnumTag5;
	EnumTag6;
	EnumTag7;
	EnumTag8;
	EnumTag9;
)


type S struct {}


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


func f0(a, b int) int {
	if a < b {
		a = a + 1;  // estimate
	}
	return b;
}


func f1(tag int) {
	switch tag {
	case
		EnumTag0, EnumTag1, EnumTag2, EnumTag3, EnumTag4,
		EnumTag5, EnumTag6, EnumTag7, EnumTag8, EnumTag9: break;
	default:
	}
}


func f2(tag int) {
	type T1 struct {}
	var x T
}


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
var x int;  // declare x
}
