// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7995: globals not flushed quickly enough.

package main

import "fmt"

var (
	p = 1
	q = &p
)

func main() {
	p = 50
	*q = 100
	s := fmt.Sprintln(p, *q)
	if s != "100 100\n" {
		println("BUG:", s)
	}
}
