// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Issue #45624 is the proposal to accept new(expr) in go1.26.
// Here we test its run-time behavior.
func main() {
	{
		p := new(123) // untyped constant expr
		if *p != 123 {
			panic("wrong value")
		}
	}
	{
		x := 42
		p := new(x) // non-constant expr
		if *p != x {
			panic("wrong value")
		}
	}
	{
		x := [2]int{123, 456}
		p := new(x) // composite value
		if *p != x {
			panic("wrong value")
		}
	}
}
