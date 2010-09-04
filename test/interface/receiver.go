// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implicit methods for embedded types.
// Mixed pointer and non-pointer receivers.

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
		println(t, *t)
		panic("fail")
	}
	np++
}

type V interface {
	V()
}
type P interface {
	P()
	V()
}

type S struct {
	T
}

type SP struct {
	*T
}

func main() {
	var t T
	var v V
	var p P

	t = 42

	t.P()
	t.V()

	v = t
	v.V()

	p = &t
	p.P()
	p.V()

	v = &t
	v.V()

	//	p = t	// ERROR
	var i interface{} = t
	if _, ok := i.(P); ok {
		println("dynamic i.(P) succeeded incorrectly")
		panic("fail")
	}

	//	println("--struct--");
	var s S
	s.T = 42
	s.P()
	s.V()

	v = s
	s.V()

	p = &s
	p.P()
	p.V()

	v = &s
	v.V()

	//	p = s	// ERROR
	var j interface{} = s
	if _, ok := j.(P); ok {
		println("dynamic j.(P) succeeded incorrectly")
		panic("fail")
	}

	//	println("--struct pointer--");
	var sp SP
	sp.T = &t
	sp.P()
	sp.V()

	v = sp
	sp.V()

	p = &sp
	p.P()
	p.V()

	v = &sp
	v.V()

	p = sp // not error
	p.P()
	p.V()

	if nv != 13 || np != 7 {
		println("bad count", nv, np)
		panic("fail")
	}
}
