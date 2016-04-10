// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that built-in types don't get printed with
// (empty) package qualification.

package main

func main() {
	true = false // ERROR "cannot assign to true"
	byte = 0     // ERROR "not an expression" "cannot assign to byte"
}
