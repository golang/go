// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that result parameters are in the same scope as regular parameters.
// Does not compile.

package main

func f1(a int) (int, float32) {
	return 7, 7.0
}


func f2(a int) (a int, b float32) { // ERROR "duplicate argument a|definition"
	return 8, 8.0
}
