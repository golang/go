// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test embedded fields of structs, including methods.

package main


type I interface {
	test1() int
	test2() int
	test3() int
	test4() int
	test5() int
	test6() int
	test7() int
}

/******
 ******
 ******/

type SubpSubp struct {
	a7 int
	a  int
}

func (p *SubpSubp) test7() int {
	if p.a != p.a7 {
		println("SubpSubp", p, p.a7)
		panic("fail")
	}
	return p.a
}
func (p *SubpSubp) testx() { println("SubpSubp", p, p.a7) }

/******
 ******
 ******/

type SubpSub struct {
	a6 int
	SubpSubp
	a int
}

func (p *SubpSub) test6() int {
	if p.a != p.a6 {
		println("SubpSub", p, p.a6)
		panic("fail")
	}
	return p.a
}
func (p *SubpSub) testx() { println("SubpSub", p, p.a6) }

/******
 ******
 ******/

type SubSubp struct {
	a5 int
	a  int
}

func (p *SubSubp) test5() int {
	if p.a != p.a5 {
		println("SubpSub", p, p.a5)
		panic("fail")
	}
	return p.a
}

/******
 ******
 ******/

type SubSub struct {
	a4 int
	a  int
}

func (p *SubSub) test4() int {
	if p.a != p.a4 {
		println("SubpSub", p, p.a4)
		panic("fail")
	}
	return p.a
}

/******
 ******
 ******/

type Subp struct {
	a3 int
	*SubpSubp
	SubpSub
	a int
}

func (p *Subp) test3() int {
	if p.a != p.a3 {
		println("SubpSub", p, p.a3)
		panic("fail")
	}
	return p.a
}

/******
 ******
 ******/

type Sub struct {
	a2 int
	*SubSubp
	SubSub
	a int
}

func (p *Sub) test2() int {
	if p.a != p.a2 {
		println("SubpSub", p, p.a2)
		panic("fail")
	}
	return p.a
}

/******
 ******
 ******/

type S struct {
	a1 int
	Sub
	*Subp
	a int
}

func (p *S) test1() int {
	if p.a != p.a1 {
		println("SubpSub", p, p.a1)
		panic("fail")
	}
	return p.a
}

/******
 ******
 ******/

func main() {
	var i I
	var s *S

	// allocate
	s = new(S)
	s.Subp = new(Subp)
	s.Sub.SubSubp = new(SubSubp)
	s.Subp.SubpSubp = new(SubpSubp)

	// explicit assignment
	s.a = 1
	s.Sub.a = 2
	s.Subp.a = 3
	s.Sub.SubSub.a = 4
	s.Sub.SubSubp.a = 5
	s.Subp.SubpSub.a = 6
	s.Subp.SubpSubp.a = 7

	// embedded (unique) assignment
	s.a1 = 1
	s.a2 = 2
	s.a3 = 3
	s.a4 = 4
	s.a5 = 5
	s.a6 = 6
	s.a7 = 7

	// unique calls with explicit &
	if s.test1() != 1 {
		println("t1", 1)
		panic("fail")
	}
	if (&s.Sub).test2() != 2 {
		println("t1", 2)
		panic("fail")
	}
	if s.Subp.test3() != 3 {
		println("t1", 3)
		panic("fail")
	}
	if (&s.Sub.SubSub).test4() != 4 {
		println("t1", 4)
		panic("fail")
	}
	if s.Sub.SubSubp.test5() != 5 {
		println("t1", 5)
		panic("fail")
	}
	if (&s.Subp.SubpSub).test6() != 6 {
		println("t1", 6)
		panic("fail")
	}
	if s.Subp.SubpSubp.test7() != 7 {
		println("t1", 7)
		panic("fail")
	}

	// automatic &
	if s.Sub.test2() != 2 {
		println("t2", 2)
		panic("fail")
	}
	if s.Sub.SubSub.test4() != 4 {
		println("t2", 4)
		panic("fail")
	}
	if s.Subp.SubpSub.test6() != 6 {
		println("t2", 6)
		panic("fail")
	}

	// embedded calls
	if s.test1() != s.a1 {
		println("t3", 1)
		panic("fail")
	}
	if s.test2() != s.a2 {
		println("t3", 2)
		panic("fail")
	}
	if s.test3() != s.a3 {
		println("t3", 3)
		panic("fail")
	}
	if s.test4() != s.a4 {
		println("t3", 4)
		panic("fail")
	}
	if s.test5() != s.a5 {
		println("t3", 5)
		panic("fail")
	}
	if s.test6() != s.a6 {
		println("t3", 6)
		panic("fail")
	}
	if s.test7() != s.a7 {
		println("t3", 7)
		panic("fail")
	}

	// run it through an interface
	i = s
	s = i.(*S)

	// same as t3
	if s.test1() != s.a1 {
		println("t4", 1)
		panic("fail")
	}
	if s.test2() != s.a2 {
		println("t4", 2)
		panic("fail")
	}
	if s.test3() != s.a3 {
		println("t4", 3)
		panic("fail")
	}
	if s.test4() != s.a4 {
		println("t4", 4)
		panic("fail")
	}
	if s.test5() != s.a5 {
		println("t4", 5)
		panic("fail")
	}
	if s.test6() != s.a6 {
		println("t4", 6)
		panic("fail")
	}
	if s.test7() != s.a7 {
		println("t4", 7)
		panic("fail")
	}

	// call interface
	if i.test1() != s.test1() {
		println("t5", 1)
		panic("fail")
	}
	if i.test2() != s.test2() {
		println("t5", 2)
		panic("fail")
	}
	if i.test3() != s.test3() {
		println("t5", 3)
		panic("fail")
	}
	if i.test4() != s.test4() {
		println("t5", 4)
		panic("fail")
	}
	if i.test5() != s.test5() {
		println("t5", 5)
		panic("fail")
	}
	if i.test6() != s.test6() {
		println("t5", 6)
		panic("fail")
	}
	if i.test7() != s.test7() {
		println("t5", 7)
		panic("fail")
	}
}
