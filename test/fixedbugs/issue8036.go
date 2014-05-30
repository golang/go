// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8036. Stores necessary for stack scan being eliminated as redundant by optimizer.

package main

import "runtime"

type T struct {
	X *int
	Y *int
	Z *int
}

type TI [3]uintptr

func G() (t TI) {
	t[0] = 1
	t[1] = 2
	t[2] = 3
	runtime.GC() // prevent inlining
	return
}

func F() (t T) {
	t.X = newint()
	t.Y = t.X
	t.Z = t.Y
	runtime.GC() // prevent inlining
	return
}

func newint() *int {
	runtime.GC()
	return nil
}

func main() {
	G() // leave non-pointers where F's return values go
	F()
}
