// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that make and new arguments requirements are enforced by the
// compiler.

package main

func main() {
	_ = make()      // ERROR "missing argument|not enough arguments"
	_ = make(int)   // ERROR "cannot make type|cannot make int"
	_ = make([]int) // ERROR "missing len argument|expects 2 or 3 arguments"

	_ = new()       // ERROR "missing argument|not enough arguments"
	_ = new(int, 2) // ERROR "too many arguments"
}
