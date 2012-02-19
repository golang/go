// run

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that returning &T{} from a function causes an allocation.

package main

type T struct {
	int
}

func f() *T {
	return &T{1}
}

func main() {
	x := f()
	y := f()
	if x == y {
		panic("not allocating & composite literals")
	}
}
