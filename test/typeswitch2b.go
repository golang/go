// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that various erroneous type switches are caught by the compiler.
// Does not compile.

package main

func notused(x interface{}) {
	// The first t is in a different scope than the 2nd t; it cannot
	// be accessed (=> declared but not used error); but it is legal
	// to declare it.
	switch t := 0; t := x.(type) { // ERROR "declared but not used"
	case int:
		_ = t // this is using the t of "t := x.(type)"
	}
}
