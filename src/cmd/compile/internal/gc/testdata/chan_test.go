// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// chan.go tests chan operations.
package main

import "testing"

//go:noinline
func lenChan_ssa(v chan int) int {
	return len(v)
}

//go:noinline
func capChan_ssa(v chan int) int {
	return cap(v)
}

func testLenChan(t *testing.T) {

	v := make(chan int, 10)
	v <- 1
	v <- 1
	v <- 1

	if want, got := 3, lenChan_ssa(v); got != want {
		t.Errorf("expected len(chan) = %d, got %d", want, got)
	}
}

func testLenNilChan(t *testing.T) {

	var v chan int
	if want, got := 0, lenChan_ssa(v); got != want {
		t.Errorf("expected len(nil) = %d, got %d", want, got)
	}
}

func testCapChan(t *testing.T) {

	v := make(chan int, 25)

	if want, got := 25, capChan_ssa(v); got != want {
		t.Errorf("expected cap(chan) = %d, got %d", want, got)
	}
}

func testCapNilChan(t *testing.T) {

	var v chan int
	if want, got := 0, capChan_ssa(v); got != want {
		t.Errorf("expected cap(nil) = %d, got %d", want, got)
	}
}

func TestChan(t *testing.T) {
	testLenChan(t)
	testLenNilChan(t)

	testCapChan(t)
	testCapNilChan(t)
}
