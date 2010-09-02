// errchk $G -e $D/$F.go

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test cases for revised conversion rules.

package main

func main() {
	type NewInt int
	i0 := 0
	var i1 int = 1
	var i2 NewInt = 1
	i0 = i0
	i0 = i1
	i0 = int(i2)
	i1 = i0
	i1 = i1
	i1 = int(i2)
	i2 = NewInt(i0)
	i2 = NewInt(i1)
	i2 = i2

	type A1 [3]int
	type A2 [3]NewInt
	var a0 [3]int
	var a1 A1
	var a2 A2
	a0 = a0
	a0 = a1
	a0 = [3]int(a2) // ERROR "cannot|invalid"
	a1 = a0
	a1 = a1
	a1 = A1(a2) // ERROR "cannot|invalid"
	a2 = A2(a0) // ERROR "cannot|invalid"
	a2 = A2(a1) // ERROR "cannot|invalid"
	a2 = a2

	type S1 struct {
		x int
	}
	type S2 struct {
		x NewInt
	}
	var s0 struct {
		x int
	}
	var s1 S1
	var s2 S2
	s0 = s0
	s0 = s1
	s0 = struct {
		x int
	}(s2) // ERROR "cannot|invalid"
	s1 = s0
	s1 = s1
	s1 = S1(s2) // ERROR "cannot|invalid"
	s2 = S2(s0) // ERROR "cannot|invalid"
	s2 = S2(s1) // ERROR "cannot|invalid"
	s2 = s2

	type P1 *int
	type P2 *NewInt
	var p0 *int
	var p1 P1
	var p2 P2
	p0 = p0
	p0 = p1
	p0 = (*int)(p2) // ERROR "cannot|invalid"
	p1 = p0
	p1 = p1
	p1 = P1(p2) // ERROR "cannot|invalid"
	p2 = P2(p0) // ERROR "cannot|invalid"
	p2 = P2(p1) // ERROR "cannot|invalid"
	p2 = p2

	type Q1 *struct {
		x int
	}
	type Q2 *S1
	var q0 *struct {
		x int
	}
	var q1 Q1
	var q2 Q2
	var ps1 *S1
	q0 = q0
	q0 = q1
	q0 = (*struct {
		x int
	})(ps1) // legal because of special conversion exception for pointers
	q0 = (*struct {
		x int
	})(q2) // ERROR "cannot|invalid"
	q1 = q0
	q1 = q1
	q1 = Q1(q2)    // ERROR "cannot|invalid"
	q2 = (*S1)(q0) // legal because of special conversion exception for pointers
	q2 = Q2(q1)    // ERROR "cannot|invalid"
	q2 = q2

	type F1 func(x NewInt) int
	type F2 func(x int) NewInt
	var f0 func(x NewInt) int
	var f1 F1
	var f2 F2
	f0 = f0
	f0 = f1
	f0 = func(x NewInt) int(f2) // ERROR "cannot|invalid"
	f1 = f0
	f1 = f1
	f1 = F1(f2) // ERROR "cannot|invalid"
	f2 = F2(f0) // ERROR "cannot|invalid"
	f2 = F2(f1) // ERROR "cannot|invalid"
	f2 = f2

	type X1 interface {
		f() int
	}
	type X2 interface {
		f() NewInt
	}
	var x0 interface {
		f() int
	}
	var x1 X1
	var x2 X2
	x0 = x0
	x0 = x1
	x0 = interface {
		f() int
	}(x2) // ERROR "cannot|need type assertion|incompatible"
	x1 = x0
	x1 = x1
	x1 = X1(x2) // ERROR "cannot|need type assertion|incompatible"
	x2 = X2(x0) // ERROR "cannot|need type assertion|incompatible"
	x2 = X2(x1) // ERROR "cannot|need type assertion|incompatible"
	x2 = x2

	type L1 []int
	type L2 []NewInt
	var l0 []int
	var l1 L1
	var l2 L2
	l0 = l0
	l0 = l1
	l0 = []int(l2) // ERROR "cannot|invalid"
	l1 = l0
	l1 = l1
	l1 = L1(l2) // ERROR "cannot|invalid"
	l2 = L2(l0) // ERROR "cannot|invalid"
	l2 = L2(l1) // ERROR "cannot|invalid"
	l2 = l2

	type M1 map[string]int
	type M2 map[string]NewInt
	var m0 []int
	var m1 L1
	var m2 L2
	m0 = m0
	m0 = m1
	m0 = []int(m2) // ERROR "cannot|invalid"
	m1 = m0
	m1 = m1
	m1 = L1(m2) // ERROR "cannot|invalid"
	m2 = L2(m0) // ERROR "cannot|invalid"
	m2 = L2(m1) // ERROR "cannot|invalid"
	m2 = m2

	type C1 chan int
	type C2 chan NewInt
	var c0 chan int
	var c1 C1
	var c2 C2
	c0 = c0
	c0 = c1
	c0 = chan int(c2) // ERROR "cannot|invalid"
	c1 = c0
	c1 = c1
	c1 = C1(c2) // ERROR "cannot|invalid"
	c2 = C2(c0) // ERROR "cannot|invalid"
	c2 = C2(c1) // ERROR "cannot|invalid"
	c2 = c2

	// internal compiler error (6g and gccgo)
	type T interface{}
	var _ T = 17 // assignment compatible
	_ = T(17)    // internal compiler error even though assignment compatible
}
