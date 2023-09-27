// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var x int

var a = []int{x: 1}    // ERROR "constant"
var b = [...]int{x: 1} // ERROR "constant"
var c = map[int]int{x: 1}
