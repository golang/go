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
}
