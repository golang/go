// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// https://golang.org/issue/3351

package main

// struct with four fields of basic type
type S struct {a, b, c, d int}

// struct with five fields of basic type
type T struct {a, b, c, d, e int}

// array with four elements
type A [4]int

// array with five elements
type B [5]int

func main() {
	var i interface{}

	var s1, s2 S
	i = s1 == s2

	var t1, t2 T
	i = t1 == t2

	var a1, a2 A
	i = a1 == a2

	var b1, b2 B
	i = b1 == b2

	_ = i
}
