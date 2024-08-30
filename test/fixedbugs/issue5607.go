// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 5607: generation of init() function incorrectly
// uses initializers of blank variables inside closures.

package main

var Test = func() {
	var mymap = map[string]string{"a": "b"}

	var innerTest = func() {
		// Used to crash trying to compile this line as
		// part of init() (funcdepth mismatch).
		var _, x = mymap["a"]
		println(x)
	}
	innerTest()
}

var Test2 = func() {
	// The following initializer should not be part of init()
	// The compiler used to generate a call to Panic() in init().
	var _, x = Panic()
	_ = x
}

func Panic() (int, int) {
	panic("omg")
	return 1, 2
}

func main() {}
