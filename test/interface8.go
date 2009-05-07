// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $F.go && $L $F.$A && ./$A.out

package main

type T int
var nv, np int

func (t T) V() {
	if t != 42 {
		panic(t)
	}
	nv++
}

func (t *T) P() {
	if *t != 42 {
		panic(t, *t)
	}
	np++
}

type V interface { V() }
type P interface { P(); V() }

type S struct {
	T;
}

type SP struct {
	*T;
}

func main() {
	var t T;
	var v V;
	var p P;

	t = 42;

	t.P();
	t.V();

	v = t;
	v.V();

	p = &t;
	p.P();
	p.V();

	v = &t;
	v.V();

//	p = t;	// ERROR

//	println("--struct--");
	var s S;
	s.T = 42;
	s.P();
	s.V();

	v = s;
	s.V();

	p = &s;
	p.P();
	p.V();

	v = &s;
	v.V();

//	p = s;	// ERROR

//	println("--struct pointer--");
	var sp SP;
	sp.T = &t;
	sp.P();
	sp.V();

	v = sp;
	sp.V();

	p = &sp;
	p.P();
	p.V();

	v = &sp;
	v.V();

	p = sp;	// not error
	p.P();
	p.V();

	if nv != 13 || np != 7 {
		panicln("bad count", nv, np)
	}
}

