// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests load/store ordering

package main

import "fmt"

// testLoadStoreOrder tests for reordering of stores/loads.
func testLoadStoreOrder() {
	z := uint32(1000)
	if testLoadStoreOrder_ssa(&z, 100) == 0 {
		println("testLoadStoreOrder failed")
		failed = true
	}
}
func testLoadStoreOrder_ssa(z *uint32, prec uint) int {
	switch {
	}
	old := *z         // load
	*z = uint32(prec) // store
	if *z < old {     // load
		return 1
	}
	return 0
}

func testStoreSize() {
	a := [4]uint16{11, 22, 33, 44}
	testStoreSize_ssa(&a[0], &a[2], 77)
	want := [4]uint16{77, 22, 33, 44}
	if a != want {
		fmt.Println("testStoreSize failed.  want =", want, ", got =", a)
		failed = true
	}
}
func testStoreSize_ssa(p *uint16, q *uint16, v uint32) {
	switch {
	}
	// Test to make sure that (Store ptr (Trunc32to16 val) mem)
	// does not end up as a 32-bit store.  It must stay a 16 bit store
	// even when Trunc32to16 is rewritten to be a nop.
	// To ensure that we get rewrite the Trunc32to16 before
	// we rewrite the Store, we force the truncate into an
	// earlier basic block by using it on both branches.
	w := uint16(v)
	if p != nil {
		*p = w
	} else {
		*q = w
	}
}

var failed = false

func main() {

	testLoadStoreOrder()
	testStoreSize()

	if failed {
		panic("failed")
	}
}
