// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG indirect

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var m0 map[string]int
var m1 *map[string]int
var m2 *map[string]int = &m0
var m3 map[string]int = map[string]int{"a": 1}
var m4 *map[string]int = &m3

var s0 string
var s1 *string
var s2 *string = &s0
var s3 string = "a"
var s4 *string = &s3

var a0 [10]int
var a1 *[10]int
var a2 *[10]int = &a0

var b0 []int
var b1 *[]int
var b2 *[]int = &b0
var b3 []int = []int{1, 2, 3}
var b4 *[]int = &b3

func crash()
{
	// these uses of nil pointers
	// would crash but should type check
	println("crash",
		len(m1)+
		len(s1)+
		len(a1)+
		len(b1)+
		cap(b1));
}

func nocrash()
{
	// this is spaced funny so that
	// the compiler will print a different
	// line number for each len call if
	// it decides there are type errors.
	// it might also help in the traceback.
	x :=
		len(m0)+
		len(m2)+
		len(m3)+
		len(m4);
	if x != 2 {
		panicln("wrong maplen");
	}

	x =
		len(s0)+
		len(s2)+
		len(s3)+
		len(s4);
	if x != 2 {
		panicln("wrong stringlen");
	}

	x =
		len(a0)+
		len(a2);
	if x != 20 {
		panicln("wrong arraylen");
	}

	x =
		len(b0)+
		len(b2)+
		len(b3)+
		len(b4);
	if x != 6 {
		panicln("wrong slicelen");
	}

	x =
		cap(b0)+
		cap(b2)+
		cap(b3)+
		cap(b4);
	if x != 6 {
		panicln("wrong slicecap");
	}
}

func main()
{
	nocrash();
}
