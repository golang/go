// run

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ensure that inlined struct/array comparisons have the right side-effects.

package main

import "os"

func main() {
	var x int
	f := func() (r [4]int) {
		x++
		return
	}
	_ = f() == f()
	if x != 2 {
		println("f evaluated ", x, " times, want 2")
		os.Exit(1)
	}
}
