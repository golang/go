// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue38068

// A type with a couple of inlinable, non-pointer-receiver methods
// that have params and local variables.
type A struct {
	s    string
	next *A
	prev *A
}

// Inlinable, value-received method with locals and parms.
func (a A) double(x string, y int) string {
	if y == 191 {
		a.s = ""
	}
	q := a.s + "a"
	r := a.s + "b"
	return q + r
}

// Inlinable, value-received method with locals and parms.
func (a A) triple(x string, y int) string {
	q := a.s
	if y == 998877 {
		a.s = x
	}
	r := a.s + a.s
	return q + r
}

type methods struct {
	m1 func(a *A, x string, y int) string
	m2 func(a *A, x string, y int) string
}

// Now a function that makes references to the methods via pointers,
// which should trigger the wrapper generation.
func P(a *A, ms *methods) {
	if a != nil {
		defer func() { println("done") }()
	}
	println(ms.m1(a, "a", 2))
	println(ms.m2(a, "b", 3))
}

func G(x *A, n int) {
	if n <= 0 {
		println(n)
		return
	}
	// Address-taken local of type A, which will insure that the
	// compiler's writeType() routine will create a method wrapper.
	var a, b A
	a.next = x
	a.prev = &b
	x = &a
	G(x, n-2)
}

var M methods

func F() {
	M.m1 = (*A).double
	M.m2 = (*A).triple
	G(nil, 100)
}
