// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// http://code.google.com/p/go/issues/detail?id=843

package main

import "unsafe"

type T struct {
	X, Y uint8
}

func main() {
	var t T
	if unsafe.Offsetof(t.X) != 0 || unsafe.Offsetof(t.Y) != 1 {
		println("BUG", unsafe.Offsetof(t.X), unsafe.Offsetof(t.Y))
	}
}
