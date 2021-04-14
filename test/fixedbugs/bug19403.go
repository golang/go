// run

// Copyright 2017 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for golang.org/issue/19403.
// F15 should not be clobbered by float-to-int conversion on ARM.
// This test requires enough locals that can be put in registers that the compiler can choose to use F15.
package main

var count float32 = 16
var i0 int
var i1 int
var i2 int
var i3 int
var i4 int
var i5 int
var i6 int
var i7 int
var i8 int
var i9 int
var i10 int
var i11 int
var i12 int
var i13 int
var i14 int
var i15 int
var i16 int

func main() {
	var f0 float32 = 0.0
	var f1 float32 = 1.0
	var f2 float32 = 2.0
	var f3 float32 = 3.0
	var f4 float32 = 4.0
	var f5 float32 = 5.0
	var f6 float32 = 6.0
	var f7 float32 = 7.0
	var f8 float32 = 8.0
	var f9 float32 = 9.0
	var f10 float32 = 10.0
	var f11 float32 = 11.0
	var f12 float32 = 12.0
	var f13 float32 = 13.0
	var f14 float32 = 14.0
	var f15 float32 = 15.0
	var f16 float32 = 16.0
	i0 = int(f0)
	i1 = int(f1)
	i2 = int(f2)
	i3 = int(f3)
	i4 = int(f4)
	i5 = int(f5)
	i6 = int(f6)
	i7 = int(f7)
	i8 = int(f8)
	i9 = int(f9)
	i10 = int(f10)
	i11 = int(f11)
	i12 = int(f12)
	i13 = int(f13)
	i14 = int(f14)
	i15 = int(f15)
	i16 = int(f16)
	if f16 != count {
		panic("fail")
	}
	count -= 1
	if f15 != count {
		panic("fail")
	}
	count -= 1
	if f14 != count {
		panic("fail")
	}
	count -= 1
	if f13 != count {
		panic("fail")
	}
	count -= 1
	if f12 != count {
		panic("fail")
	}
	count -= 1
	if f11 != count {
		panic("fail")
	}
	count -= 1
	if f10 != count {
		panic("fail")
	}
	count -= 1
	if f9 != count {
		panic("fail")
	}
	count -= 1
	if f8 != count {
		panic("fail")
	}
	count -= 1
	if f7 != count {
		panic("fail")
	}
	count -= 1
	if f6 != count {
		panic("fail")
	}
	count -= 1
	if f5 != count {
		panic("fail")
	}
	count -= 1
	if f4 != count {
		panic("fail")
	}
	count -= 1
	if f3 != count {
		panic("fail")
	}
	count -= 1
	if f2 != count {
		panic("fail")
	}
	count -= 1
	if f1 != count {
		panic("fail")
	}
	count -= 1
	if f0 != count {
		panic("fail")
	}
	count -= 1
}
