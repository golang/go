// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a = 1

func main() {
	defer func() {
		recover()
		if a != 2 {
			println("BUG a =", a)
		}
	}()
	a = 2
	b := a - a
	c := 4
	a = c / b
	a = 3
}
