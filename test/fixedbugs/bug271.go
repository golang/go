// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// http://code.google.com/p/go/issues/detail?id=662

package main

import "fmt"

func f() (int, int) { return 1, 2 }

func main() {
	s := fmt.Sprint(f())
	if s != "1 2" {	// with bug, was "{1 2}"
		println("BUG")
	}
}
