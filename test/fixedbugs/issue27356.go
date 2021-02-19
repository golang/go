// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 27356: function parameter hiding built-in function results in compiler crash

package p

var a = []int{1,2,3}

func _(len int) {
	_ =  len(a) // ERROR "cannot call non-function|expected function"
}

var cap = false
var _ = cap(a) // ERROR "cannot call non-function|expected function"

