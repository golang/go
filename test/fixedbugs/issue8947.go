// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Some uses of zeroed constants in non-assignment
// expressions broke with our more aggressive zeroing
// of assignments (internal compiler errors).

package main

func f1() {
	type T [2]int
	p := T{0, 1}
	switch p {
	case T{0, 0}:
		panic("wrong1")
	case T{0, 1}:
		// ok
	default:
		panic("wrong2")
	}

	if p == (T{0, 0}) {
		panic("wrong3")
	} else if p == (T{0, 1}) {
		// ok
	} else {
		panic("wrong4")
	}
}

type T struct {
	V int
}

var X = T{}.V

func f2() {
	var x = T{}.V
	if x != 0 {
		panic("wrongx")
	}
	if X != 0 {
		panic("wrongX")
	}
}

func main() {
	f1()
	f2()
}
