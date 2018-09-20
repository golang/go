// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var s uint
	var x = interface{}(1<<s + 1<<s) // compiler must not crash here
	if x.(int) != 2 {
		panic("x not int or not 2")
	}

	var y interface{}
	y = 1<<s + 1 // compiler must not crash here
	if y.(int) != 2 {
		panic("y not int or not 2")
	}
}
