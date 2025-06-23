// errorcheck

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var a []int = []int{1: 1}
var b []int = []int{-1: 1} // ERROR "must be non-negative integer constant|index expression is negative|must not be negative"

var c []int = []int{2.0: 2}
var d []int = []int{-2.0: 2} // ERROR "must be non-negative integer constant|index expression is negative|must not be negative"

var e []int = []int{3 + 0i: 3}
var f []int = []int{3i: 3} // ERROR "truncated to integer|index expression is not integer constant|truncated to int"

var g []int = []int{"a": 4} // ERROR "must be non-negative integer constant|index expression is not integer constant|cannot convert"
