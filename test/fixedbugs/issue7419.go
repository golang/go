// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7419: odd behavior for float constants underflowing to 0

package main

import (
	"os"
)

var x = 1e-779137
var y = 1e-779138

func main() {
	if x != 0 {
		os.Exit(1)
	}
	if y != 0 {
		os.Exit(2)
	}
}
