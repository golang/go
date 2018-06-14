// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify assignment rules are enforced by the compiler.
// Does not compile.

package main

type (
	A [10]int
	B []int
	C chan int
	F func() int
	I interface {
		m() int
	}
	M map[int]int
	P *int
	S struct {
		X int
	}

	A1 [10]int
	B1 []int
	C1 chan int
	F1 func() int
	I1 interface {
		m() int
	}
	M1 map[int]int
	P1 *int
	S1 struct {
		X int
	}
)

var (
	a0 [10]int
	b0 []int
	c0 chan int
	f0 func() int
	i0 interface {
		m() int
	}
	m0 map[int]int
	p0 *int
	s0 struct {
		X int
	}

	a A
	b B
	c C
	f F
	i I
	m M
	p P
	s S

	a1 A1
	b1 B1
	c1 C1
	f1 F1
	i1 I1
	m1 M1
	p1 P1
	s1 S1

	pa0 *[10]int
	pb0 *[]int
	pc0 *chan int
	pf0 *func() int
	pi0 *interface {
		m() int
	}
	pm0 *map[int]int
	pp0 **int
	ps0 *struct {
		X int
	}

	pa *A
	pb *B
	pc *C
	pf *F
	pi *I
	pm *M
	pp *P
	ps *S

	pa1 *A1
	pb1 *B1
	pc1 *C1
	pf1 *F1
	pi1 *I1
	pm1 *M1
	pp1 *P1
	ps1 *S1
)

func main() {
	a0 = a
	a0 = a1
	a = a0
	a = a1 // ERROR "cannot use"
	a1 = a0
	a1 = a // ERROR "cannot use"

	b0 = b
	b0 = b1
	b = b0
	b = b1 // ERROR "cannot use"
	b1 = b0
	b1 = b // ERROR "cannot use"

	c0 = c
	c0 = c1
	c = c0
	c = c1 // ERROR "cannot use"
	c1 = c0
	c1 = c // ERROR "cannot use"

	f0 = f
	f0 = f1
	f = f0
	f = f1 // ERROR "cannot use"
	f1 = f0
	f1 = f // ERROR "cannot use"

	i0 = i
	i0 = i1
	i = i0
	i = i1
	i1 = i0
	i1 = i

	m0 = m
	m0 = m1
	m = m0
	m = m1 // ERROR "cannot use"
	m1 = m0
	m1 = m // ERROR "cannot use"

	p0 = p
	p0 = p1
	p = p0
	p = p1 // ERROR "cannot use"
	p1 = p0
	p1 = p // ERROR "cannot use"

	s0 = s
	s0 = s1
	s = s0
	s = s1 // ERROR "cannot use"
	s1 = s0
	s1 = s // ERROR "cannot use"

	pa0 = pa  // ERROR "cannot use|incompatible"
	pa0 = pa1 // ERROR "cannot use|incompatible"
	pa = pa0  // ERROR "cannot use|incompatible"
	pa = pa1  // ERROR "cannot use|incompatible"
	pa1 = pa0 // ERROR "cannot use|incompatible"
	pa1 = pa  // ERROR "cannot use|incompatible"

	pb0 = pb  // ERROR "cannot use|incompatible"
	pb0 = pb1 // ERROR "cannot use|incompatible"
	pb = pb0  // ERROR "cannot use|incompatible"
	pb = pb1  // ERROR "cannot use|incompatible"
	pb1 = pb0 // ERROR "cannot use|incompatible"
	pb1 = pb  // ERROR "cannot use|incompatible"

	pc0 = pc  // ERROR "cannot use|incompatible"
	pc0 = pc1 // ERROR "cannot use|incompatible"
	pc = pc0  // ERROR "cannot use|incompatible"
	pc = pc1  // ERROR "cannot use|incompatible"
	pc1 = pc0 // ERROR "cannot use|incompatible"
	pc1 = pc  // ERROR "cannot use|incompatible"

	pf0 = pf  // ERROR "cannot use|incompatible"
	pf0 = pf1 // ERROR "cannot use|incompatible"
	pf = pf0  // ERROR "cannot use|incompatible"
	pf = pf1  // ERROR "cannot use|incompatible"
	pf1 = pf0 // ERROR "cannot use|incompatible"
	pf1 = pf  // ERROR "cannot use|incompatible"

	pi0 = pi  // ERROR "cannot use|incompatible"
	pi0 = pi1 // ERROR "cannot use|incompatible"
	pi = pi0  // ERROR "cannot use|incompatible"
	pi = pi1  // ERROR "cannot use|incompatible"
	pi1 = pi0 // ERROR "cannot use|incompatible"
	pi1 = pi  // ERROR "cannot use|incompatible"

	pm0 = pm  // ERROR "cannot use|incompatible"
	pm0 = pm1 // ERROR "cannot use|incompatible"
	pm = pm0  // ERROR "cannot use|incompatible"
	pm = pm1  // ERROR "cannot use|incompatible"
	pm1 = pm0 // ERROR "cannot use|incompatible"
	pm1 = pm  // ERROR "cannot use|incompatible"

	pp0 = pp  // ERROR "cannot use|incompatible"
	pp0 = pp1 // ERROR "cannot use|incompatible"
	pp = pp0  // ERROR "cannot use|incompatible"
	pp = pp1  // ERROR "cannot use|incompatible"
	pp1 = pp0 // ERROR "cannot use|incompatible"
	pp1 = pp  // ERROR "cannot use|incompatible"

	ps0 = ps  // ERROR "cannot use|incompatible"
	ps0 = ps1 // ERROR "cannot use|incompatible"
	ps = ps0  // ERROR "cannot use|incompatible"
	ps = ps1  // ERROR "cannot use|incompatible"
	ps1 = ps0 // ERROR "cannot use|incompatible"
	ps1 = ps  // ERROR "cannot use|incompatible"


	a0 = [10]int(a)
	a0 = [10]int(a1)
	a = A(a0)
	a = A(a1)
	a1 = A1(a0)
	a1 = A1(a)

	b0 = []int(b)
	b0 = []int(b1)
	b = B(b0)
	b = B(b1)
	b1 = B1(b0)
	b1 = B1(b)

	c0 = chan int(c)
	c0 = chan int(c1)
	c = C(c0)
	c = C(c1)
	c1 = C1(c0)
	c1 = C1(c)

	f0 = func() int(f)
	f0 = func() int(f1)
	f = F(f0)
	f = F(f1)
	f1 = F1(f0)
	f1 = F1(f)

	i0 = interface {
		m() int
	}(i)
	i0 = interface {
		m() int
	}(i1)
	i = I(i0)
	i = I(i1)
	i1 = I1(i0)
	i1 = I1(i)

	m0 = map[int]int(m)
	m0 = map[int]int(m1)
	m = M(m0)
	m = M(m1)
	m1 = M1(m0)
	m1 = M1(m)

	p0 = (*int)(p)
	p0 = (*int)(p1)
	p = P(p0)
	p = P(p1)
	p1 = P1(p0)
	p1 = P1(p)

	s0 = struct {
		X int
	}(s)
	s0 = struct {
		X int
	}(s1)
	s = S(s0)
	s = S(s1)
	s1 = S1(s0)
	s1 = S1(s)

	pa0 = (*[10]int)(pa)
	pa0 = (*[10]int)(pa1)
	pa = (*A)(pa0)
	pa = (*A)(pa1)
	pa1 = (*A1)(pa0)
	pa1 = (*A1)(pa)

	pb0 = (*[]int)(pb)
	pb0 = (*[]int)(pb1)
	pb = (*B)(pb0)
	pb = (*B)(pb1)
	pb1 = (*B1)(pb0)
	pb1 = (*B1)(pb)

	pc0 = (*chan int)(pc)
	pc0 = (*chan int)(pc1)
	pc = (*C)(pc0)
	pc = (*C)(pc1)
	pc1 = (*C1)(pc0)
	pc1 = (*C1)(pc)

	pf0 = (*func() int)(pf)
	pf0 = (*func() int)(pf1)
	pf = (*F)(pf0)
	pf = (*F)(pf1)
	pf1 = (*F1)(pf0)
	pf1 = (*F1)(pf)

	pi0 = (*interface {
		m() int
	})(pi)
	pi0 = (*interface {
		m() int
	})(pi1)
	pi = (*I)(pi0)
	pi = (*I)(pi1)
	pi1 = (*I1)(pi0)
	pi1 = (*I1)(pi)

	pm0 = (*map[int]int)(pm)
	pm0 = (*map[int]int)(pm1)
	pm = (*M)(pm0)
	pm = (*M)(pm1)
	pm1 = (*M1)(pm0)
	pm1 = (*M1)(pm)

	pp0 = (**int)(pp)
	pp0 = (**int)(pp1)
	pp = (*P)(pp0)
	pp = (*P)(pp1)
	pp1 = (*P1)(pp0)
	pp1 = (*P1)(pp)

	ps0 = (*struct {
		X int
	})(ps)
	ps0 = (*struct {
		X int
	})(ps1)
	ps = (*S)(ps0)
	ps = (*S)(ps1)
	ps1 = (*S1)(ps0)
	ps1 = (*S1)(ps)

}
