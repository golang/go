// run -goexperiment range

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the 'for range' construct.

package main

// test range over integers

func testint1() {
	j := 0
	for i := range int(4) {
		if i != j {
			println("range var", i, "want", j)
		}
		j++
	}
	if j != 4 {
		println("wrong count ranging over 4:", j)
	}
}

func testint2() {
	j := 0
	for i := range 4 {
		if i != j {
			println("range var", i, "want", j)
		}
		j++
	}
	if j != 4 {
		println("wrong count ranging over 4:", j)
	}
}

func testint3() {
	type MyInt int

	j := MyInt(0)
	for i := range MyInt(4) {
		if i != j {
			println("range var", i, "want", j)
		}
		j++
	}
	if j != 4 {
		println("wrong count ranging over 4:", j)
	}
}

func main() {
	testint1()
	testint2()
	testint3()
}
