// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that calling a function shadowing a built-in provides a good
// error message.

package main

func F() {
	slice := []int{1, 2, 3}
	_ = slice
	len := int(2)
	println(len(slice)) // ERROR "cannot call non-function len .type int., declared at LINE-1|expected function|cannot call non-function len"
	const iota = 1
	println(iota(slice)) // ERROR "cannot call non-function iota .type int., declared at LINE-1|expected function|cannot call non-function iota"
}
