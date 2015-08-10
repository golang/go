// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests load/store ordering

package main

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

var failed = false

func main() {

	testLoadStoreOrder()

	if failed {
		panic("failed")
	}
}
