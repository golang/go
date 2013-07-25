// build

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 5841: 8g produces invalid CMPL $0, $0.
// Similar to issue 5002, used to fail at link time.

package main

func main() {
	var y int
	if y%1 == 0 {
	}
}
