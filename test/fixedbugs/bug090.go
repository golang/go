// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const (
	c3div2 = 3/2;
	f3div2 = 3./2.;
)

func assert(t bool, s string) {
	if !t {
		panic(s)
	}
}

func main() {
	var i int;
	var f float64;

	assert(c3div2 == 1, "3/2");
	assert(f3div2 == 1.5, "3/2");

	i = c3div2;
	assert(i == c3div2, "i == c3div2");

	f = c3div2;
	assert(f == c3div2, "f == c3div2");

	f = f3div2;
	assert(f == f3div2, "f == f3div2");

	i = f3div2;	// ERROR "truncate"
	assert(i == c3div2, "i == c3div2 from f3div2");
	assert(i != f3div2, "i != f3div2");	// ERROR "truncate"

	const g float64 = 1.0;
	i = g;  // ERROR "convert|incompatible|cannot"

	const h float64 = 3.14;
	i = h;  // ERROR "convert|incompatible|cannot"
	i = int(h);	// ERROR "truncate"
}
