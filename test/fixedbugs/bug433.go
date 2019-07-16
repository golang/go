// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that initializing struct fields out of order still runs
// functions in the right order.  This failed with gccgo.

package main

type S struct {
	i1, i2, i3 int
}

var G int

func v(i int) int {
	if i != G {
		panic(i)
	}
	G = i + 1
	return G
}

func F() S {
	return S{
		i1: v(0),
		i3: v(1),
		i2: v(2),
	}
}

func main() {
	s := F()
	if s != (S{1, 3, 2}) {
		panic(s)
	}
}
