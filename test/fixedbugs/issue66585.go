// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var x = 0
var a = foo()
var b = x

func foo() int {
	x++
	return x
}

func main() {
	if a != 1 {
		panic("unexpected a value")
	}
	if b != 1 {
		panic("unexpected b value")
	}
}
