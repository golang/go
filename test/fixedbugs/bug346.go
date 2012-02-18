// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"

func main() {
	x := 4
	a, b, c, d := func(i int) (p int, q int, r int, s int) { return 1, i, 3, x }(2)

	if a != 1 || b != 2 || c != 3 || d != 4 {
		println("abcd: expected 1 2 3 4 got", a, b, c, d)
		os.Exit(1)
	}
}
