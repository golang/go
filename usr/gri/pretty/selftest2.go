// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"array";  // not needed
	"utf8";  // not needed
	Fmt "fmt"
)


const /* enum1 */ (
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


const /* enum2 */ (
	a, b = iota*2 + 1, iota*2;
	c, d;
	e, f;
)


type S struct {}


type T struct {
	x, y int;
	s string;
	next_t *T
}


var (
	A = 5;
	u, v, w int = 0, 0, 0;
	foo = "foo";
	fixed_array0 = [10]int{};
	fixed_array1 = [10]int{0, 1, 2};
	fixed_array2 = [...]string{"foo", "bar"};
)


func d0() {
	var (
		a string;
		b, c string;
		d, e, f string;
		g, h, i, j string;
		k, l, m, n, o string;
	)
}


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
	type T struct {}
	var x T
}


func f3(a *[]int, m map[string] int) {
	println("A1");
	for i := range a {
		println(i);
	}

	println("A2");
	for i, x := range a {
		println(i, x);
	}

	println("A3");
	for i : x := range a {
		println(i, x);
	}

	println("M1");
	for i range m {
		println(i);
	}

	println("M2");
	for i, x range m {
		println(i, x);
	}

	println("M3");
	var i string;
	var x int;
	for i : x = range m {
		println(i, x);
	}
}


func main() {
// the prologue
	for i := 0; i <= 10 /* limit */; i++ {
		println(i);  // the index
		println(i + 1);  // the index + 1
		println(i + 1000);  // the index + 1000
		println();
	}
	f3(&[]int{2, 3, 5, 7}, map[string]int{"two":2, "three":3, "five":5, "seven":7});
// the epilogue
	println("foo");  // foo
	println("foobar");  // foobar
var x int;  // declare x
}
