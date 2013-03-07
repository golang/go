// build

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 5002: 8g produces invalid CMPL $0, $0.
// Used to fail at link time.

package main

func main() {
	var y int64
	if y%1 == 0 {
	}
}
