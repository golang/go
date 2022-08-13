// errorcheck

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a [10]int    // ok
var b [1e1]int   // ok
var c [1.5]int   // ERROR "truncated|must be integer"
var d ["abc"]int // ERROR "invalid array bound|not numeric|must be integer"
var e [nil]int   // ERROR "use of untyped nil|invalid array (bound|length)|not numeric|must be constant"
// var f [e]int  // ok with Go 1.17 because an error was reported for e; leads to an error for Go 1.18
var f [ee]int      // ERROR "undefined|undeclared"
var g [1 << 65]int // ERROR "array bound is too large|overflows|must be integer"
var h [len(a)]int  // ok

func ff() string

var i [len([1]string{ff()})]int // ERROR "non-constant array bound|not constant|must be constant"
