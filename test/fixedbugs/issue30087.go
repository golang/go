// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var a, b = 1    // ERROR "assignment mismatch: 2 variables but 1 values"
	_ = 1, 2        // ERROR "assignment mismatch: 1 variables but 2 values"
	c, d := 1       // ERROR "assignment mismatch: 2 variables but 1 values"
	e, f := 1, 2, 3 // ERROR "assignment mismatch: 2 variables but 3 values"
}
