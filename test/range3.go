// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the 'for range' construct ranging over integers.

package main

func testint1() {
	bad := false
	j := 0
	for i := range int(4) {
		if i != j {
			println("range var", i, "want", j)
			bad = true
		}
		j++
	}
	if j != 4 {
		println("wrong count ranging over 4:", j)
		bad = true
	}
	if bad {
		panic("testint1")
	}
}

func testint2() {
	bad := false
	j := 0
	for i := range 4 {
		if i != j {
			println("range var", i, "want", j)
			bad = true
		}
		j++
	}
	if j != 4 {
		println("wrong count ranging over 4:", j)
		bad = true
	}
	if bad {
		panic("testint2")
	}
}

func testint3() {
	bad := false
	type MyInt int
	j := MyInt(0)
	for i := range MyInt(4) {
		if i != j {
			println("range var", i, "want", j)
			bad = true
		}
		j++
	}
	if j != 4 {
		println("wrong count ranging over 4:", j)
		bad = true
	}
	if bad {
		panic("testint3")
	}
}

// Issue #63378.
func testint4() {
	for i := range -1 {
		_ = i
		panic("must not be executed")
	}
}

// Issue #64471.
func testint5() {
	for i := range 'a' {
		var _ *rune = &i // ensure i has type rune
	}
}

func main() {
	testint1()
	testint2()
	testint3()
	testint4()
	testint5()
}
