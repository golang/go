// errorcheck

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 1664

package main

func main() {
	var i uint = 33
	var a = (1<<i) + 4.5  // ERROR "shift of type float64|invalid.*shift"
	println(a)
	
	var b = (1<<i) + 4.0  // ERROR "shift of type float64|invalid.*shift"
	println(b)

	var c int64 = (1<<i) + 4.0  // ok - it's all int64
	println(c)
}
