// errorcheck -0 -d=ssa/phiopt/debug=3

//go:build amd64 || s390x || arm64

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func f0(a bool) bool {
	x := false
	if a {
		x = true
	} else {
		x = false
	}
	return x // ERROR "converted OpPhi to Copy$"
}

//go:noinline
func f1(a bool) bool {
	x := false
	if a {
		x = false
	} else {
		x = true
	}
	return x // ERROR "converted OpPhi to Not$"
}

//go:noinline
func f2(a, b int) bool {
	x := true
	if a == b {
		x = false
	}
	return x // ERROR "converted OpPhi to Not$"
}

//go:noinline
func f3(a, b int) bool {
	x := false
	if a == b {
		x = true
	}
	return x // ERROR "converted OpPhi to Copy$"
}

//go:noinline
func f4(a, b bool) bool {
	return a || b // ERROR "converted OpPhi to OrB$"
}

//go:noinline
func f5or(a int, b bool) bool {
	var x bool
	if a == 0 {
		x = true
	} else {
		x = b
	}
	return x // ERROR "converted OpPhi to OrB$"
}

//go:noinline
func f5and(a int, b bool) bool {
	var x bool
	if a == 0 {
		x = b
	} else {
		x = false
	}
	return x // ERROR "converted OpPhi to AndB$"
}

//go:noinline
func f6or(a int, b bool) bool {
	x := b
	if a == 0 {
		// f6or has side effects so the OpPhi should not be converted.
		x = f6or(a, b)
	}
	return x
}

//go:noinline
func f6and(a int, b bool) bool {
	x := b
	if a == 0 {
		// f6and has side effects so the OpPhi should not be converted.
		x = f6and(a, b)
	}
	return x
}

//go:noinline
func f7or(a bool, b bool) bool {
	return a || b // ERROR "converted OpPhi to OrB$"
}

//go:noinline
func f7and(a bool, b bool) bool {
	return a && b // ERROR "converted OpPhi to AndB$"
}

//go:noinline
func f8(s string) (string, bool) {
	neg := false
	if s[0] == '-' {    // ERROR "converted OpPhi to Copy$"
		neg = true
		s = s[1:]
	}
	return s, neg
}

var d int

//go:noinline
func f9(a, b int) bool {
	c := false
	if a < 0 {          // ERROR "converted OpPhi to Copy$"
		if b < 0 {
			d = d + 1
		}
		c = true
	}
	return c
}

//go:noinline
func f10and(a bool, b bool) bool {
	var x bool
	if a {
		x = b
	} else {
		x = a
	}
	return x // ERROR "converted OpPhi to AndB$"
}

//go:noinline
func f11or(a bool, b bool) bool {
	var x bool
	if a {
		x = a
	} else {
		x = b
	}
	return x // ERROR "converted OpPhi to OrB$"
}

func main() {
}
