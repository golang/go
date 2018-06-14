// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test method invocation with pointer receivers and function-valued fields.

package main

type C struct {
	a	int;
	x	func(p *C)int;
}

func (this *C) f()int {
	return this.a;
}

func
main() {
	var v int;
	var c *C;

	c = new(C);
	c.a = 6;
	c.x = g;

	v = g(c);
	if v != 6 { panic(v); }

	v = c.x(c);
	if v != 6 { panic(v); }

	v = c.f();
	if v != 6 { panic(v); }
}

func g(p *C)int {
	var v int;

	v = p.a;
	if v != 6 { panic(v); }
	return p.a;
}
