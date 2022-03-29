// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 7150: array index out of bounds error off by one

package main

func main() {
	_ = [0]int{-1: 50}              // ERROR "index must be non-negative integer constant|index expression is negative|must not be negative"
	_ = [0]int{0: 0}                // ERROR "index 0 out of bounds \[0:0\]|out of range"
	_ = [0]int{5: 25}               // ERROR "index 5 out of bounds \[0:0\]|out of range"
	_ = [10]int{2: 10, 15: 30}      // ERROR "index 15 out of bounds \[0:10\]|out of range"
	_ = [10]int{5: 5, 1: 1, 12: 12} // ERROR "index 12 out of bounds \[0:10\]|out of range"
}
