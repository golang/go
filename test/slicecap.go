// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

var (
	hello = "hello"
	bytes = []byte{1, 2, 3, 4, 5}
	ints  = []int32{1, 2, 3, 4, 5}

	five = 5

	ok = true
)

func notOK() {
	if ok {
		println("BUG:")
		ok = false
	}
}

func checkString(desc, s string) {
	p1 := *(*uintptr)(unsafe.Pointer(&s))
	p2 := *(*uintptr)(unsafe.Pointer(&hello))
	if p1-p2 >= 5 {
		notOK()
		println("string", desc, "has invalid base")
	}
}

func checkBytes(desc string, s []byte) {
	p1 := *(*uintptr)(unsafe.Pointer(&s))
	p2 := *(*uintptr)(unsafe.Pointer(&bytes))
	if p1-p2 >= 5 {
		println("byte slice", desc, "has invalid base")
	}
}

func checkInts(desc string, s []int32) {
	p1 := *(*uintptr)(unsafe.Pointer(&s))
	p2 := *(*uintptr)(unsafe.Pointer(&ints))
	if p1-p2 >= 5*4 {
		println("int slice", desc, "has invalid base")
	}
}

func main() {
	{
		x := hello
		checkString("x", x)
		checkString("x[5:]", x[5:])
		checkString("x[five:]", x[five:])
		checkString("x[5:five]", x[5:five])
		checkString("x[five:5]", x[five:5])
		checkString("x[five:five]", x[five:five])
		checkString("x[1:][2:][2:]", x[1:][2:][2:])
		y := x[4:]
		checkString("y[1:]", y[1:])
	}
	{
		x := bytes
		checkBytes("x", x)
		checkBytes("x[5:]", x[5:])
		checkBytes("x[five:]", x[five:])
		checkBytes("x[5:five]", x[5:five])
		checkBytes("x[five:5]", x[five:5])
		checkBytes("x[five:five]", x[five:five])
		checkBytes("x[1:][2:][2:]", x[1:][2:][2:])
		y := x[4:]
		checkBytes("y[1:]", y[1:])
	}
	{
		x := ints
		checkInts("x", x)
		checkInts("x[5:]", x[5:])
		checkInts("x[five:]", x[five:])
		checkInts("x[5:five]", x[5:five])
		checkInts("x[five:5]", x[five:5])
		checkInts("x[five:five]", x[five:five])
		checkInts("x[1:][2:][2:]", x[1:][2:][2:])
		y := x[4:]
		checkInts("y[1:]", y[1:])
	}
}
