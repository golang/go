// +build amd64
// errorcheck -0 -d=ssa/phiopt/debug=3

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
	return a || b // ERROR "converted OpPhi to Or8$"
}

//go:noinline
func f5(a int, b bool) bool {
	x := b
	if a == 0 {
		x = true
	}
	return x // ERROR "converted OpPhi to Or8$"
}

//go:noinline
func f6(a int, b bool) bool {
	x := b
	if a == 0 {
		// f6 has side effects so the OpPhi should not be converted.
		x = f6(a, b)
	}
	return x
}

func main() {
}
