// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test basic operations on bool.

package main

type s struct {
	a	bool;
	b	bool;
}

func
main() {
	var a,b bool;

	a = true;
	b = false;

	if !a { panic(1); }
	if b { panic(2); }
	if !!!a { panic(3); }
	if !!b { panic(4); }

	a = !b;
	if !a { panic(5); }
	if !!!a { panic(6); }

	var x *s;
	x = new(s);
	x.a = true;
	x.b = false;

	if !x.a { panic(7); }
	if x.b { panic(8); }
	if !!!x.a { panic(9); }
	if !!x.b { panic(10); }

	x.a = !x.b;
	if !x.a { panic(11); }
	if !!!x.a { panic(12); }

	/*
	 * test &&
	 */
	a = true;
	b = true;
	if !(a && b) { panic(21); }
	if a && !b { panic(22); }
	if !a && b { panic(23); }
	if !a && !b { panic(24); }

	a = false;
	b = true;
	if !(!a && b) { panic(31); }
	if !a && !b { panic(32); }
	if a && b { panic(33); }
	if a && !b { panic(34); }

	a = true;
	b = false;
	if !(a && !b) { panic(41); }
	if a && b { panic(41); }
	if !a && !b { panic(41); }
	if !a && b { panic(44); }

	a = false;
	b = false;
	if !(!a && !b) { panic(51); }
	if !a && b { panic(52); }
	if a && !b { panic(53); }
	if a && b { panic(54); }

	/*
	 * test ||
	 */
	a = true;
	b = true;
	if !(a || b) { panic(61); }
	if !(a || !b) { panic(62); }
	if !(!a || b) { panic(63); }
	if !a || !b { panic(64); }

	a = false;
	b = true;
	if !(!a || b) { panic(71); }
	if !(!a || !b) { panic(72); }
	if !(a || b) { panic(73); }
	if a || !b { panic(74); }

	a = true;
	b = false;
	if !(a || !b) { panic(81); }
	if !(a || b) { panic(82); }
	if !(!a || !b) { panic(83); }
	if !a || b { panic(84); }

	a = false;
	b = false;
	if !(!a || !b) { panic(91); }
	if !(!a || b) { panic(92); }
	if !(a || !b) { panic(93); }
	if a || b { panic(94); }
}
