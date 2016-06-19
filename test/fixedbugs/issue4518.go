// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4518. In some circumstances "return F(...)"
// where F has multiple returns is miscompiled by 6g due to
// bold assumptions in componentgen.

package main

//go:noinline
func F(e interface{}) (int, int) {
	return 3, 7
}

//go:noinline
func G() (int, int) {
	return 3, 7
}

func bogus1(d interface{}) (int, int) {
	switch {
	default:
		return F(d)
	}
	return 0, 0
}

func bogus2() (int, int) {
	switch {
	default:
		return F(3)
	}
	return 0, 0
}

func bogus3(d interface{}) (int, int) {
	switch {
	default:
		return G()
	}
	return 0, 0
}

func bogus4() (int, int) {
	switch {
	default:
		return G()
	}
	return 0, 0
}

func check(a, b int) {
	if a != 3 || b != 7 {
		println(a, b)
		panic("a != 3 || b != 7")
	}
}

func main() {
	check(bogus1(42))
	check(bogus2())
}
