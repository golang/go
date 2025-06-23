// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for https://golang.org/issue/700

package main

import "os"

func f() (e int) {
	_ = &e
	return 999
}

func main() {
	if f() != 999 {
		os.Exit(1)
	}
}
