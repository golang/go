// errorcheck -+

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test go:nowritebarrier and related directives.

package p

type t struct {
	f *t
}

var x t
var y *t

//go:nowritebarrier
func a1() {
	x.f = y // ERROR "write barrier prohibited"
	a2()    // no error
}

//go:noinline
func a2() {
	x.f = y
}

//go:nowritebarrierrec
func b1() {
	b2()
}

//go:noinline
func b2() {
	x.f = y // ERROR "write barrier prohibited by caller"
}

// Test recursive cycles through nowritebarrierrec and yeswritebarrierrec.

//go:nowritebarrierrec
func c1() {
	c2()
}

//go:yeswritebarrierrec
func c2() {
	c3()
}

func c3() {
	x.f = y
	c4()
}

//go:nowritebarrierrec
func c4() {
	c2()
}

//go:nowritebarrierrec
func d1() {
	d2()
}

func d2() {
	d3()
}

func d3() {
	x.f = y // ERROR "write barrier prohibited by caller"
	d4()
}

//go:yeswritebarrierrec
func d4() {
	d2()
}

//go:noinline
func systemstack(func()) {}

//go:nowritebarrierrec
func e1() {
	systemstack(e2)
	systemstack(func() {
		x.f = y // ERROR "write barrier prohibited by caller"
	})
}

func e2() {
	x.f = y // ERROR "write barrier prohibited by caller"
}
