// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a package for testing purposes.
//
package main

import 	"fmt"	// fmt

const c0	= 0	// zero
const (
	c1	= iota;	// c1
	c2	// c2
)


// The T type.
type T struct {
	a, b, c	int	// 3 fields
}

// This comment group should be separated
// with a newline from the next comment
// group.

// This comment should NOT be associated with the next declaration.

var x int	// x
var ()


// This comment SHOULD be associated with the next declaration.
func f0() {
	const pi			= 3.14;					// pi
	var s1 struct {}	/* an empty struct */	/* foo */
	// a struct constructor
	// --------------------
	var s2 struct {}	= struct {}{};
	x := pi
}
//
// NO SPACE HERE
//
func f1() {
	f0();
	/* 1 */
	// 2
	/* 3 */
	/* 4 */
	f0()
}
