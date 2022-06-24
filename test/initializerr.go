// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that erroneous initialization expressions are caught by the compiler
// Does not compile.

package main

type S struct {
	A, B, C, X, Y, Z int
}

type T struct {
	S
}

var x = 1
var a1 = S{0, X: 1}                             // ERROR "mixture|undefined" "too few values"
var a2 = S{Y: 3, Z: 2, Y: 3}                    // ERROR "duplicate"
var a3 = T{S{}, 2, 3, 4, 5, 6}                  // ERROR "convert|too many"
var a4 = [5]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10} // ERROR "index|too many"
var a5 = []byte{x: 2}                           // ERROR "index"
var a6 = []byte{1: 1, 2: 2, 1: 3}               // ERROR "duplicate"

var ok1 = S{}       // should be ok
var ok2 = T{S: ok1} // should be ok

// These keys can be computed at compile time but they are
// not constants as defined by the spec, so they do not trigger
// compile-time errors about duplicate key values.
// See issue 4555.

type Key struct{ X, Y int }

var _ = map[Key]string{
	Key{1, 2}: "hello",
	Key{1, 2}: "world",
}
