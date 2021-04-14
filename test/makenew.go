// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that make and new arguments requirements are enforced by the
// compiler.

package main

func main() {
	_ = make()      // ERROR "missing argument"
	_ = make(int)   // ERROR "cannot make type"
	_ = make([]int) // ERROR "missing len argument"

	_ = new()       // ERROR "missing argument"
	_ = new(int, 2) // ERROR "too many arguments"
}
