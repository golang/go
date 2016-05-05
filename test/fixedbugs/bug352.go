// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var x [10][0]byte
var y = make([]struct{}, 10)

func main() {
	if &x[1] != &x[2] {
		println("BUG: bug352 [0]byte")
	}
	if &y[1] != &y[2] {
		println("BUG: bug352 struct{}")
	}
}
