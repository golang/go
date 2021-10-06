// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that a defer in a function with no return
// statement will compile correctly.

package main

import "testing"

func deferNoReturn_ssa() {
	defer func() { println("returned") }()
	for {
		println("loop")
	}
}

func TestDeferNoReturn(t *testing.T) {
	// This is a compile-time test, no runtime testing required.
}
