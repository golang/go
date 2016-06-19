// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to crash generating hash and == functions for struct
// with leading _ field.  Issue 3607.

package main

type T struct {
	_ int
	X interface{}
	_ string
	Y float64
}

func main() {
	m := map[T]int{}
	m[T{X: 1, Y: 2}] = 1
	m[T{X: 2, Y: 3}] = 2
	m[T{X: 1, Y: 2}] = 3  // overwrites first entry
	if len(m) != 2 {
		println("BUG")
	}
}
