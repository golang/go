// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S struct {
	A, B, C, X, Y, Z int
}

type T struct {
	S
}

var x = 1
var a1 = S { 0, X: 1 }	// ERROR "mixture|undefined"
var a2 = S { Y: 3, Z: 2, Y: 3 } // ERROR "duplicate"
var a3 = T { 1, 2, 3, 4, 5, 6 }	// ERROR "convert|too many"
var a4 = [5]byte{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }	// ERROR "index|too many"
var a5 = []byte { x: 2 }	// ERROR "index"

var ok1 = S { }	// should be ok
var ok2 = T { S: ok1 }	// should be ok
