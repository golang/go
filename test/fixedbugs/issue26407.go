// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 26407: ensure that stack variables which have
// had their address taken and then used in a comparison,
// but are otherwise unused, are cleared.

package main

func main() {
	poison()
	test()
}

//go:noinline
func poison() {
	// initialise the stack with invalid pointers
	var large [256]uintptr
	for i := range large {
		large[i] = 1
	}
	use(large[:])
}

//go:noinline
func test() {
	a := 2
	x := &a
	if x != compare(&x) {
		panic("not possible")
	}
}

//go:noinline
func compare(x **int) *int {
	var y *int
	if x == &y {
		panic("not possible")
	}
	// grow the stack to trigger a check for invalid pointers
	grow()
	if x == &y {
		panic("not possible")
	}
	return *x
}

//go:noinline
func grow() {
	var large [1 << 16]uintptr
	use(large[:])
}

//go:noinline
func use(_ []uintptr) { }
