// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test evaluation order in if condition.

package main

var calledf = false

func f() int {
	calledf = true
	return 1
}

func g() int {
	if !calledf {
		panic("BUG: func7 - called g before f")
	}
	return 0
}

func main() {
	// gc used to evaluate g() before f().
	if f() < g() {
		panic("wrong answer")
	}
}
