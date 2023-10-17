// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

func main() {
	var file string
	var line int
	func() {
		defer func() {
			_, file, line, _ = runtime.Caller(1)
		}()
	}() // this is the expected line
	const EXPECTED = 18
	if line != EXPECTED {
		println("Expected line =", EXPECTED, "but got line =", line, "and file =", file)
	}
}
