// errchk $G $F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// does not compile and should not compile

package main

func f1(a int) (int, float32) { // BUG (not caught by compiler): multiple return values must have names
	return 7, 7.0
}


func f2(a int) (a int, b float32) { // ERROR "redeclared|definition"
	return 8, 8.0
}
