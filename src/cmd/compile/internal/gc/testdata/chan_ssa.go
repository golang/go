// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// chan_ssa.go tests chan operations.
package main

import "fmt"

var failed = false

//go:noinline
func lenChan_ssa(v chan int) int {
	return len(v)
}

//go:noinline
func capChan_ssa(v chan int) int {
	return cap(v)
}

func testLenChan() {

	v := make(chan int, 10)
	v <- 1
	v <- 1
	v <- 1

	if want, got := 3, lenChan_ssa(v); got != want {
		fmt.Printf("expected len(chan) = %d, got %d", want, got)
		failed = true
	}
}

func testLenNilChan() {

	var v chan int
	if want, got := 0, lenChan_ssa(v); got != want {
		fmt.Printf("expected len(nil) = %d, got %d", want, got)
		failed = true
	}
}

func testCapChan() {

	v := make(chan int, 25)

	if want, got := 25, capChan_ssa(v); got != want {
		fmt.Printf("expected cap(chan) = %d, got %d", want, got)
		failed = true
	}
}

func testCapNilChan() {

	var v chan int
	if want, got := 0, capChan_ssa(v); got != want {
		fmt.Printf("expected cap(nil) = %d, got %d", want, got)
		failed = true
	}
}

func main() {
	testLenChan()
	testLenNilChan()

	testCapChan()
	testCapNilChan()

	if failed {
		panic("failed")
	}
}
