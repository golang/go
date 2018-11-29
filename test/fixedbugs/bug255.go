// errorcheck

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a [10]int      // ok
var b [1e1]int     // ok
var c [1.5]int     // ERROR "truncated"
var d ["abc"]int   // ERROR "invalid array bound|not numeric"
var e [nil]int     // ERROR "use of untyped nil|invalid array bound|not numeric"
var f [e]int       // ok: error already reported for e
var g [1 << 65]int // ERROR "array bound is too large|overflows"
var h [len(a)]int  // ok

func ff() string

var i [len([1]string{ff()})]int // ERROR "non-constant array bound|not constant"
