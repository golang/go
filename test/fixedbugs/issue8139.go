// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8139. The x.(T) assertions used to write 1 (unexpected)
// return byte for the 0-byte return value T.

package main

import "fmt"

type T struct{}

func (T) M() {}

type M interface {
	M()
}

var e interface{} = T{}
var i M = T{}
var b bool

func f1() int {
	if b {
		return f1() // convince inliner not to inline
	}
	z := 0x11223344
	_ = e.(T)
	return z
}

func f2() int {
	if b {
		return f1() // convince inliner not to inline
	}
	z := 0x11223344
	_ = i.(T)
	return z
}

func main() {
	x := f1()
	y := f2()
	if x != 0x11223344 || y != 0x11223344 {
		fmt.Printf("BUG: x=%#x y=%#x, want 0x11223344 for both\n", x, y)
	}
}
