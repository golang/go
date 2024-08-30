// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify allowed and disallowed conversions.
// Does not compile.

package main

// everything here is legal except the ERROR line

var c chan int
var d1 chan<- int = c
var d2 = (chan<- int)(c)

var e *[4]int
var f1 []int = e[0:]
var f2 = []int(e[0:])

var g = []int(nil)

type H []int
type J []int

var h H
var j1 J = h // ERROR "compat|illegal|cannot"
var j2 = J(h)
