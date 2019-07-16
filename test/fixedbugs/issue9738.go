// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func F() (x int) {
	defer func() {
		if x != 42 {
			println("BUG: x =", x)
		}
	}()
	return 42
}

func main() {
	F()
}
