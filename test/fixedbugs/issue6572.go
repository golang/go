// errorcheck

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func foo() (T, T) { // ERROR "undefined"
	return 0, 0
}

func bar() (T, string, T) { // ERROR "undefined"
	return 0, "", 0
}

func main() {
	var x, y, z int
	x, y = foo()
	x, y, z = bar() // ERROR "cannot (use type|assign) string"
}
