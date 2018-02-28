// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that the half multiply resulting from a division
// by a constant generates correct code.

package main

func main() {
	var _ = 7 / "0"[0] // test case from #11369
	var _ = 1 / "."[0] // test case from #11358
	var x = 0 / "0"[0]
	var y = 48 / "0"[0]
	var z = 5 * 48 / "0"[0]
	if x != 0 {
		panic("expected 0")
	}
	if y != 1 {
		panic("expected 1")
	}
	if z != 5 {
		panic("expected 5")
	}
}
