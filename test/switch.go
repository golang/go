// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test switch statements.

package main

import "os"

func assert(cond bool, msg string) {
	if !cond {
		print("assertion fail: ", msg, "\n")
		panic(1)
	}
}

func main() {
	i5 := 5
	i7 := 7
	hello := "hello"

	switch true {
	case i5 < 5:
		assert(false, "<")
	case i5 == 5:
		assert(true, "!")
	case i5 > 5:
		assert(false, ">")
	}

	switch {
	case i5 < 5:
		assert(false, "<")
	case i5 == 5:
		assert(true, "!")
	case i5 > 5:
		assert(false, ">")
	}

	switch x := 5; true {
	case i5 < x:
		assert(false, "<")
	case i5 == x:
		assert(true, "!")
	case i5 > x:
		assert(false, ">")
	}

	switch x := 5; true {
	case i5 < x:
		assert(false, "<")
	case i5 == x:
		assert(true, "!")
	case i5 > x:
		assert(false, ">")
	}

	switch i5 {
	case 0:
		assert(false, "0")
	case 1:
		assert(false, "1")
	case 2:
		assert(false, "2")
	case 3:
		assert(false, "3")
	case 4:
		assert(false, "4")
	case 5:
		assert(true, "5")
	case 6:
		assert(false, "6")
	case 7:
		assert(false, "7")
	case 8:
		assert(false, "8")
	case 9:
		assert(false, "9")
	default:
		assert(false, "default")
	}

	switch i5 {
	case 0, 1, 2, 3, 4:
		assert(false, "4")
	case 5:
		assert(true, "5")
	case 6, 7, 8, 9:
		assert(false, "9")
	default:
		assert(false, "default")
	}

	switch i5 {
	case 0:
	case 1:
	case 2:
	case 3:
	case 4:
		assert(false, "4")
	case 5:
		assert(true, "5")
	case 6:
	case 7:
	case 8:
	case 9:
	default:
		assert(i5 == 5, "good")
	}

	switch i5 {
	case 0:
		dummy := 0
		_ = dummy
		fallthrough
	case 1:
		dummy := 0
		_ = dummy
		fallthrough
	case 2:
		dummy := 0
		_ = dummy
		fallthrough
	case 3:
		dummy := 0
		_ = dummy
		fallthrough
	case 4:
		dummy := 0
		_ = dummy
		assert(false, "4")
	case 5:
		dummy := 0
		_ = dummy
		fallthrough
	case 6:
		dummy := 0
		_ = dummy
		fallthrough
	case 7:
		dummy := 0
		_ = dummy
		fallthrough
	case 8:
		dummy := 0
		_ = dummy
		fallthrough
	case 9:
		dummy := 0
		_ = dummy
		fallthrough
	default:
		dummy := 0
		_ = dummy
		assert(i5 == 5, "good")
	}

	fired := false
	switch i5 {
	case 0:
		dummy := 0
		_ = dummy
		fallthrough // tests scoping of cases
	case 1:
		dummy := 0
		_ = dummy
		fallthrough
	case 2:
		dummy := 0
		_ = dummy
		fallthrough
	case 3:
		dummy := 0
		_ = dummy
		fallthrough
	case 4:
		dummy := 0
		_ = dummy
		assert(false, "4")
	case 5:
		dummy := 0
		_ = dummy
		fallthrough
	case 6:
		dummy := 0
		_ = dummy
		fallthrough
	case 7:
		dummy := 0
		_ = dummy
		fallthrough
	case 8:
		dummy := 0
		_ = dummy
		fallthrough
	case 9:
		dummy := 0
		_ = dummy
		fallthrough
	default:
		dummy := 0
		_ = dummy
		fired = !fired
		assert(i5 == 5, "good")
	}
	assert(fired, "fired")

	count := 0
	switch i5 {
	case 0:
		count = count + 1
		fallthrough
	case 1:
		count = count + 1
		fallthrough
	case 2:
		count = count + 1
		fallthrough
	case 3:
		count = count + 1
		fallthrough
	case 4:
		count = count + 1
		assert(false, "4")
	case 5:
		count = count + 1
		fallthrough
	case 6:
		count = count + 1
		fallthrough
	case 7:
		count = count + 1
		fallthrough
	case 8:
		count = count + 1
		fallthrough
	case 9:
		count = count + 1
		fallthrough
	default:
		assert(i5 == count, "good")
	}
	assert(fired, "fired")

	switch hello {
	case "wowie":
		assert(false, "wowie")
	case "hello":
		assert(true, "hello")
	case "jumpn":
		assert(false, "jumpn")
	default:
		assert(false, "default")
	}

	fired = false
	switch i := i5 + 2; i {
	case i7:
		fired = true
	default:
		assert(false, "fail")
	}
	assert(fired, "var")

	// switch on nil-only comparison types
	switch f := func() {}; f {
	case nil:
		assert(false, "f should not be nil")
	default:
	}

	switch m := make(map[int]int); m {
	case nil:
		assert(false, "m should not be nil")
	default:
	}

	switch a := make([]int, 1); a {
	case nil:
		assert(false, "m should not be nil")
	default:
	}

	// switch on interface.
	switch i := interface{}("hello"); i {
	case 42:
		assert(false, `i should be "hello"`)
	case "hello":
		assert(true, "hello")
	default:
		assert(false, `i should be "hello"`)
	}

	// switch on implicit bool converted to interface
	// was broken: see issue 3980
	switch i := interface{}(true); {
	case i:
		assert(true, "true")
	case false:
		assert(false, "i should be true")
	default:
		assert(false, "i should be true")
	}

	// switch on interface with constant cases differing by type.
	// was rejected by compiler: see issue 4781
	type T int
	type B bool
	type F float64
	type S string
	switch i := interface{}(float64(1.0)); i {
	case nil:
		assert(false, "i should be float64(1.0)")
	case (*int)(nil):
		assert(false, "i should be float64(1.0)")
	case 1:
		assert(false, "i should be float64(1.0)")
	case T(1):
		assert(false, "i should be float64(1.0)")
	case F(1.0):
		assert(false, "i should be float64(1.0)")
	case 1.0:
		assert(true, "true")
	case "hello":
		assert(false, "i should be float64(1.0)")
	case S("hello"):
		assert(false, "i should be float64(1.0)")
	case true, B(false):
		assert(false, "i should be float64(1.0)")
	case false, B(true):
		assert(false, "i should be float64(1.0)")
	}

	// switch on array.
	switch ar := [3]int{1, 2, 3}; ar {
	case [3]int{1, 2, 3}:
		assert(true, "[1 2 3]")
	case [3]int{4, 5, 6}:
		assert(false, "ar should be [1 2 3]")
	default:
		assert(false, "ar should be [1 2 3]")
	}

	// switch on channel
	switch c1, c2 := make(chan int), make(chan int); c1 {
	case nil:
		assert(false, "c1 did not match itself")
	case c2:
		assert(false, "c1 did not match itself")
	case c1:
		assert(true, "chan")
	default:
		assert(false, "c1 did not match itself")
	}

	// empty switch
	switch {
	}

	// empty switch with default case.
	fired = false
	switch {
	default:
		fired = true
	}
	assert(fired, "fail")

	// Default and fallthrough.
	count = 0
	switch {
	default:
		count++
		fallthrough
	case false:
		count++
	}
	assert(count == 2, "fail")

	// fallthrough to default, which is not at end.
	count = 0
	switch i5 {
	case 5:
		count++
		fallthrough
	default:
		count++
	case 6:
		count++
	}
	assert(count == 2, "fail")

	i := 0
	switch x := 5; {
	case i < x:
		os.Exit(0)
	case i == x:
	case i > x:
		os.Exit(1)
	}

	// Unified IR converts the tag and all case values to empty
	// interface, when any of the case values aren't assignable to the
	// tag value's type. Make sure that `case nil:` compares against the
	// tag type's nil value (i.e., `(*int)(nil)`), not nil interface
	// (i.e., `any(nil)`).
	switch (*int)(nil) {
	case nil:
		// ok
	case any(nil):
		assert(false, "case any(nil) matched")
	default:
		assert(false, "default matched")
	}
}
