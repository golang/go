// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// append_ssa.go tests append operations.
package main

import "fmt"

var failed = false

//go:noinline
func appendOne_ssa(a []int, x int) []int {
	return append(a, x)
}

//go:noinline
func appendThree_ssa(a []int, x, y, z int) []int {
	return append(a, x, y, z)
}

func eq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func expect(got, want []int) {
	if eq(got, want) {
		return
	}
	fmt.Printf("expected %v, got %v\n", want, got)
	failed = true
}

func testAppend() {
	var store [7]int
	a := store[:0]

	a = appendOne_ssa(a, 1)
	expect(a, []int{1})
	a = appendThree_ssa(a, 2, 3, 4)
	expect(a, []int{1, 2, 3, 4})
	a = appendThree_ssa(a, 5, 6, 7)
	expect(a, []int{1, 2, 3, 4, 5, 6, 7})
	if &a[0] != &store[0] {
		fmt.Println("unnecessary grow")
		failed = true
	}
	a = appendOne_ssa(a, 8)
	expect(a, []int{1, 2, 3, 4, 5, 6, 7, 8})
	if &a[0] == &store[0] {
		fmt.Println("didn't grow")
		failed = true
	}
}

func main() {
	testAppend()

	if failed {
		panic("failed")
	}
}
