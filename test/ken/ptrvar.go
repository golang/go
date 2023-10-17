// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test pointers and the . (selector) operator on structs.

package main

type	x2	struct { a,b,c int; d int; };
var	g1	x2;
var	g2	struct { a,b,c int; d x2; };

func
main() {
	var x int;
	var s1 *x2;
	var s2 *struct { a,b,c int; d x2; };

	s1 = &g1;
	s2 = &g2;

	s1.a = 1;
	s1.b = 2;
	s1.c = 3;
	s1.d = 5;

	s2.a = 7;
	s2.b = 11;
	s2.c = 13;
	s2.d.a = 17;
	s2.d.b = 19;
	s2.d.c = 23;
	s2.d.d = 20;

	if(s2.d.c != 23) { panic(1); }
	if(g2.d.c != 23) { panic(2); }

	x =	s1.a +
		s1.b +
		s1.c +
		s1.d +

		s2.a +
		s2.b +
		s2.c +
		s2.d.a +
		s2.d.b +
		s2.d.c +
		s2.d.d;

	if(x != 121) { panic(x); }
}
