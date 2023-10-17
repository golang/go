// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test big vs. small, pointer vs. value interface methods.

package main

type I interface { M() int64 }

type BigPtr struct { a, b, c, d int64 }
func (z *BigPtr) M() int64 { return z.a+z.b+z.c+z.d }

type SmallPtr struct { a int32 }
func (z *SmallPtr) M() int64 { return int64(z.a) }

type IntPtr int32
func (z *IntPtr) M() int64 { return int64(*z) }

var bad bool

func test(name string, i I) {
	m := i.M()
	if m != 12345 {
		println(name, m)
		bad = true
	}
}

func ptrs() {
	var bigptr BigPtr = BigPtr{ 10000, 2000, 300, 45 }
	var smallptr SmallPtr = SmallPtr{ 12345 }
	var intptr IntPtr = 12345

//	test("bigptr", bigptr)
	test("&bigptr", &bigptr)
//	test("smallptr", smallptr)
	test("&smallptr", &smallptr)
//	test("intptr", intptr)
	test("&intptr", &intptr)
}

type Big struct { a, b, c, d int64 }
func (z Big) M() int64 { return z.a+z.b+z.c+z.d }

type Small struct { a int32 }
func (z Small) M() int64 { return int64(z.a) }

type Int int32
func (z Int) M() int64 { return int64(z) }

func nonptrs() {
	var big Big = Big{ 10000, 2000, 300, 45 }
	var small Small = Small{ 12345 }
	var int Int = 12345

	test("big", big)
	test("&big", &big)
	test("small", small)
	test("&small", &small)
	test("int", int)
	test("&int", &int)
}

func main() {
	ptrs()
	nonptrs()

	if bad {
		println("BUG: interface4")
	}
}
