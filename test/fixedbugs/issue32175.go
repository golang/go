// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This used to print 0, because x was incorrectly captured by value.

func f() (x int) {
	defer func() func() {
		return func() {
			println(x)
		}
	}()()
	return 42
}

func main() {
	f()
}
