// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T int
type U int

var x int

var t T = int(0)	// ERROR "cannot use|incompatible"
var t1 T = int(x)	// ERROR "cannot use|incompatible"
var u U = int(0)	// ERROR "cannot use|incompatible"
var u1 U = int(x)	// ERROR "cannot use|incompatible"

type S string
var s S

var s1 = s + "hello"
var s2 = "hello" + s
var s3 = s + string("hello")	// ERROR "invalid operation|incompatible"
var s4 = string("hello") + s	// ERROR "invalid operation|incompatible"

var r string

var r1 = r + "hello"
var r2 = "hello" + r
var r3 = r + string("hello")
var r4 = string("hello") + r

