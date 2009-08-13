// Copyright 2009 The Go Authors. All rights reserved.
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

func abs(x int) int {
	if x < 0 {	// the tab printed before this comment's // must not affect the remaining lines
		return -x	// this statement should be properly indented
	}
	return x
}

func typeswitch(x interface {}) {
	switch v := x.(type) {
	case bool, int, float:
	case string:
	default:
	}
	switch x.(type) {}
	switch v0, ok := x.(int); v := x.(type) {}
	switch v0, ok := x.(int); x.(type) {
	case bool, int, float:
	case string:
	default:
	}
}
