// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// map_ssa.go tests map operations.
package main

import "fmt"

var failed = false

//go:noinline
func lenMap_ssa(v map[int]int) int {
	return len(v)
}

func testLenMap() {

	v := make(map[int]int)
	v[0] = 0
	v[1] = 0
	v[2] = 0

	if want, got := 3, lenMap_ssa(v); got != want {
		fmt.Printf("expected len(map) = %d, got %d", want, got)
		failed = true
	}
}

func testLenNilMap() {

	var v map[int]int
	if want, got := 0, lenMap_ssa(v); got != want {
		fmt.Printf("expected len(nil) = %d, got %d", want, got)
		failed = true
	}
}
func main() {
	testLenMap()
	testLenNilMap()

	if failed {
		panic("failed")
	}
}
