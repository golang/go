// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 6902: confusing printing of large floating point constants

package main

import (
	"os"
)

var x = -1e-10000

func main() {
	if x != 0 {
		os.Exit(1)
	}
}
