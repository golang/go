// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main


type
I	interface
{
	test1,
	test2,
	test3,
	test4,
	test5,
	test6,
	test7() int;
};

/******
 ******
 ******/

type
SubpSubp	struct
{
	a7	int;
	a	int;
}
func (p *SubpSubp)
test7() int
{
	if p.a != p.a7 { panicln("SubpSubp", p, p.a7) }
	return p.a
}
func (p *SubpSubp)
testx()
{
	println("SubpSubp", p, p.a7);
}

/******
 ******
 ******/

type
SubpSub	struct
{
	a6	int;
		SubpSubp;
	a	int;
}
func (p *SubpSub)
test6() int
{
	if p.a != p.a6 { panicln("SubpSub", p, p.a6) }
	return p.a
}
func (p *SubpSub)
testx()
{
	println("SubpSub", p, p.a6);
}

/******
 ******
 ******/

type
SubSubp	struct
{
	a5	int;
	a	int;
}
func (p *SubSubp)
test5() int
{
	if p.a != p.a5 { panicln("SubpSub", p, p.a5) }
	return p.a
}

/******
 ******
 ******/

type
SubSub	struct
{
	a4	int;
	a	int;
}
func (p *SubSub)
test4() int
{
	if p.a != p.a4 { panicln("SubpSub", p, p.a4) }
	return p.a
}

/******
 ******
 ******/

type
Subp	struct
{
	a3	int;
		*SubpSubp;
		SubpSub;
	a	int;
}
func (p *Subp)
test3() int
{
	if p.a != p.a3 { panicln("SubpSub", p, p.a3) }
	return p.a
}

/******
 ******
 ******/

type
Sub	struct
{
	a2	int;
		*SubSubp;
		SubSub;
	a	int;
}
func (p *Sub)
test2() int
{
	if p.a != p.a2 { panicln("SubpSub", p, p.a2) }
	return p.a
}

/******
 ******
 ******/

type
S	struct
{
	a1	int;
		Sub;
		*Subp;
	a	int;
}
func (p *S)
test1() int
{
	if p.a != p.a1 { panicln("SubpSub", p, p.a1) }
	return p.a
}

/******
 ******
 ******/

func
main()
{
	var i I;
	var s *S;

	// allocate
	s = new(*S);
	s.Subp = new(*Subp);
	s.Sub.SubSubp = new(*SubSubp);
	s.Subp.SubpSubp = new(*SubpSubp);

	// explicit assignment
	s.a = 1;
	s.Sub.a = 2;
	s.Subp.a = 3;
	s.Sub.SubSub.a = 4;
	s.Sub.SubSubp.a = 5;
	s.Subp.SubpSub.a = 6;
	s.Subp.SubpSubp.a = 7;

	// embedded (unique) assignment
	s.a1 = 1;
	s.a2 = 2;
	s.a3 = 3;
	s.a4 = 4;
	s.a5 = 5;
	s.a6 = 6;
	s.a7 = 7;

	// unique calls with explicit &
	if s.test1() != 1 { panicln("t1", 1) }
	if (&s.Sub).test2() != 2 { panicln("t1", 2) }
	if s.Subp.test3() != 3 { panicln("t1", 3) }
	if (&s.Sub.SubSub).test4() != 4 { panicln("t1", 4) }
	if s.Sub.SubSubp.test5() != 5 { panicln("t1", 5) }
	if (&s.Subp.SubpSub).test6() != 6 { panicln("t1", 6) }
	if s.Subp.SubpSubp.test7() != 7 { panicln("t1", 7) }

	// automatic &
	if s.Sub.test2() != 2 { panicln("t2", 2) }
	if s.Sub.SubSub.test4() != 4 { panicln("t2", 4) }
	if s.Subp.SubpSub.test6() != 6 { panicln("t2", 6) }

	// embedded calls
	if s.test1() != s.a1 { panicln("t3", 1) }
	if s.test2() != s.a2 { panicln("t3", 2) }
	if s.test3() != s.a3 { panicln("t3", 3) }
	if s.test4() != s.a4 { panicln("t3", 4) }
	if s.test5() != s.a5 { panicln("t3", 5) }
	if s.test6() != s.a6 { panicln("t3", 6) }
	if s.test7() != s.a7 { panicln("t3", 7) }

	// run it thru an interface
	i = s;
	s = i;

	// same as t3
	if s.test1() != s.a1 { panicln("t4", 1) }
	if s.test2() != s.a2 { panicln("t4", 2) }
	if s.test3() != s.a3 { panicln("t4", 3) }
	if s.test4() != s.a4 { panicln("t4", 4) }
	if s.test5() != s.a5 { panicln("t4", 5) }
	if s.test6() != s.a6 { panicln("t4", 6) }
	if s.test7() != s.a7 { panicln("t4", 7) }

	// call interface
	if i.test1() != s.test1() { panicln("t5", 1) }
	if i.test2() != s.test2() { panicln("t5", 2) }
	if i.test3() != s.test3() { panicln("t5", 3) }
	if i.test4() != s.test4() { panicln("t5", 4) }
	if i.test5() != s.test5() { panicln("t5", 5) }
	if i.test6() != s.test6() { panicln("t5", 6) }
	if i.test7() != s.test7() { panicln("t5", 7) }
}
