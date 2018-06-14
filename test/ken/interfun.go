// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test interfaces and methods.

package main

type S struct {
	a,b	int;
}

type I1 interface {
	f	()int;
}

type I2 interface {
	g() int;
	f() int;
}

func (this *S) f()int {
	return this.a;
}

func (this *S) g()int {
	return this.b;
}

func
main() {
	var i1 I1;
	var i2 I2;
	var g *S;

	s := new(S);
	s.a = 5;
	s.b = 6;

	// call structure
	if s.f() != 5 { panic(11); }
	if s.g() != 6 { panic(12); }

	i1 = s;		// convert S to I1
	i2 = i1.(I2);	// convert I1 to I2

	// call interface
	if i1.f() != 5 { panic(21); }
	if i2.f() != 5 { panic(22); }
	if i2.g() != 6 { panic(23); }

	g = i1.(*S);		// convert I1 to S
	if g != s { panic(31); }

	g = i2.(*S);		// convert I2 to S
	if g != s { panic(32); }
}
