// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 3824.
// Method calls are ignored when deciding initialization
// order.

package main

type T int

func (r T) Method1() int { return a }
func (r T) Method2() int { return b }

// dummy1 and dummy2 must be initialized after a and b.
var dummy1 = T(0).Method1()
var dummy2 = T.Method2(0)

// Use a function call to force generating code.
var a = identity(1)
var b = identity(2)

func identity(a int) int { return a }

func main() {
	if dummy1 != 1 {
		panic("dummy1 != 1")
	}
	if dummy2 != 2 {
		panic("dummy2 != 2")
	}
}

