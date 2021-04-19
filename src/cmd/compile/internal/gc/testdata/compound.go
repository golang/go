// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test compound objects

package main

import "fmt"

func string_ssa(a, b string, x bool) string {
	s := ""
	if x {
		s = a
	} else {
		s = b
	}
	return s
}

func testString() {
	a := "foo"
	b := "barz"
	if want, got := a, string_ssa(a, b, true); got != want {
		fmt.Printf("string_ssa(%v, %v, true) = %v, want %v\n", a, b, got, want)
		failed = true
	}
	if want, got := b, string_ssa(a, b, false); got != want {
		fmt.Printf("string_ssa(%v, %v, false) = %v, want %v\n", a, b, got, want)
		failed = true
	}
}

//go:noinline
func complex64_ssa(a, b complex64, x bool) complex64 {
	var c complex64
	if x {
		c = a
	} else {
		c = b
	}
	return c
}

//go:noinline
func complex128_ssa(a, b complex128, x bool) complex128 {
	var c complex128
	if x {
		c = a
	} else {
		c = b
	}
	return c
}

func testComplex64() {
	var a complex64 = 1 + 2i
	var b complex64 = 3 + 4i

	if want, got := a, complex64_ssa(a, b, true); got != want {
		fmt.Printf("complex64_ssa(%v, %v, true) = %v, want %v\n", a, b, got, want)
		failed = true
	}
	if want, got := b, complex64_ssa(a, b, false); got != want {
		fmt.Printf("complex64_ssa(%v, %v, true) = %v, want %v\n", a, b, got, want)
		failed = true
	}
}

func testComplex128() {
	var a complex128 = 1 + 2i
	var b complex128 = 3 + 4i

	if want, got := a, complex128_ssa(a, b, true); got != want {
		fmt.Printf("complex128_ssa(%v, %v, true) = %v, want %v\n", a, b, got, want)
		failed = true
	}
	if want, got := b, complex128_ssa(a, b, false); got != want {
		fmt.Printf("complex128_ssa(%v, %v, true) = %v, want %v\n", a, b, got, want)
		failed = true
	}
}

func slice_ssa(a, b []byte, x bool) []byte {
	var s []byte
	if x {
		s = a
	} else {
		s = b
	}
	return s
}

func testSlice() {
	a := []byte{3, 4, 5}
	b := []byte{7, 8, 9}
	if want, got := byte(3), slice_ssa(a, b, true)[0]; got != want {
		fmt.Printf("slice_ssa(%v, %v, true) = %v, want %v\n", a, b, got, want)
		failed = true
	}
	if want, got := byte(7), slice_ssa(a, b, false)[0]; got != want {
		fmt.Printf("slice_ssa(%v, %v, false) = %v, want %v\n", a, b, got, want)
		failed = true
	}
}

func interface_ssa(a, b interface{}, x bool) interface{} {
	var s interface{}
	if x {
		s = a
	} else {
		s = b
	}
	return s
}

func testInterface() {
	a := interface{}(3)
	b := interface{}(4)
	if want, got := 3, interface_ssa(a, b, true).(int); got != want {
		fmt.Printf("interface_ssa(%v, %v, true) = %v, want %v\n", a, b, got, want)
		failed = true
	}
	if want, got := 4, interface_ssa(a, b, false).(int); got != want {
		fmt.Printf("interface_ssa(%v, %v, false) = %v, want %v\n", a, b, got, want)
		failed = true
	}
}

var failed = false

func main() {
	testString()
	testSlice()
	testInterface()
	testComplex64()
	testComplex128()
	if failed {
		panic("failed")
	}
}
