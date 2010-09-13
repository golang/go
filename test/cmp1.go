// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

func use(bool) {}

func stringptr(s string) uintptr { return *(*uintptr)(unsafe.Pointer(&s)) }

func isfalse(b bool) {
	if b {
		// stack will explain where
		panic("wanted false, got true")
	}
}

func istrue(b bool) {
	if !b {
		// stack will explain where
		panic("wanted true, got false")
	}
}

type T *int

func main() {
	var a []int
	var b map[string]int

	var c string = "hello"
	var d string = "hel" // try to get different pointer
	d = d + "lo"
	if stringptr(c) == stringptr(d) {
		panic("compiler too smart -- got same string")
	}

	var e = make(chan int)

	var ia interface{} = a
	var ib interface{} = b
	var ic interface{} = c
	var id interface{} = d
	var ie interface{} = e

	// these comparisons are okay because
	// string compare is okay and the others
	// are comparisons where the types differ.
	isfalse(ia == ib)
	isfalse(ia == ic)
	isfalse(ia == id)
	isfalse(ib == ic)
	isfalse(ib == id)
	istrue(ic == id)
	istrue(ie == ie)
	
	// these are okay because one side of the
	// comparison need only be assignable to the other.
	isfalse(a == ib)
	isfalse(a == ic)
	isfalse(a == id)
	isfalse(b == ic)
	isfalse(b == id)
	istrue(c == id)
	istrue(e == ie)

	isfalse(ia == b)
	isfalse(ia == c)
	isfalse(ia == d)
	isfalse(ib == c)
	isfalse(ib == d)
	istrue(ic == d)
	istrue(ie == e)

	// 6g used to let this go through as true.
	var g uint64 = 123
	var h int64 = 123
	var ig interface{} = g
	var ih interface{} = h
	isfalse(ig == ih)

	// map of interface should use == on interface values,
	// not memory.
	// TODO: should m[c], m[d] be valid here?
	var m = make(map[interface{}]int)
	m[ic] = 1
	m[id] = 2
	if m[ic] != 2 {
		println("m[ic] = ", m[ic])
		panic("bad m[ic]")
	}
	
	// non-interface comparisons
	{
		c := make(chan int)
		c1 := (<-chan int)(c)
		c2 := (chan<- int)(c)
		istrue(c == c1)
		istrue(c == c2)
		istrue(c1 == c)
		istrue(c2 == c)
		
		d := make(chan int)
		isfalse(c == d)
		isfalse(d == c)
		isfalse(d == c1)
		isfalse(d == c2)
		isfalse(c1 == d)
		isfalse(c2 == d)
	}

	// named types vs not
	{
		var x = new(int)
		var y T
		var z T = x
		
		isfalse(x == y)
		istrue(x == z)
		isfalse(y == z)

		isfalse(y == x)
		istrue(z == x)
		isfalse(z == y)
	}
}
