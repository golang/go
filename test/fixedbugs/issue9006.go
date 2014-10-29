// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T1 struct {
	X int
}

func NewT1(x int) T1 { return T1{x} }

type T2 int

func NewT2(x int) T2 { return T2(x) }

func main() {
	switch (T1{}) {
	case NewT1(1):
		panic("bad1")
	case NewT1(0):
		// ok
	default:
		panic("bad2")
	}

	switch T2(0) {
	case NewT2(2):
		panic("bad3")
	case NewT2(0):
		// ok
	default:
		panic("bad4")
	}
}
