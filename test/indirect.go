// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test various safe uses of indirection.

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

func crash() {
	// these uses of nil pointers
	// would crash but should type check
	println("crash",
		len(a1)+cap(a1))
}

func nocrash() {
	// this is spaced funny so that
	// the compiler will print a different
	// line number for each len call if
	// it decides there are type errors.
	// it might also help in the traceback.
	x :=
		len(m0) +
			len(m3)
	if x != 1 {
		println("wrong maplen")
		panic("fail")
	}

	x =
		len(s0) +
			len(s3)
	if x != 1 {
		println("wrong stringlen")
		panic("fail")
	}

	x =
		len(a0) +
			len(a2)
	if x != 20 {
		println("wrong arraylen")
		panic("fail")
	}

	x =
		len(b0) +
			len(b3)
	if x != 3 {
		println("wrong slicelen")
		panic("fail")
	}

	x =
		cap(b0) +
			cap(b3)
	if x != 3 {
		println("wrong slicecap")
		panic("fail")
	}
}

func main() { nocrash() }
