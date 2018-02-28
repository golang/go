// compile

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gccgo compiler had a bug: mentioning a function type in an
// expression in a function literal messed up the list of variables
// referenced in enclosing functions.

package main

func main() {
	v1, v2 := 0, 0
	f := func() {
		a := v1
		g := (func())(nil)
		b := v2
		_, _, _ = a, g, b
	}
	_, _, _ = v1, v2, f
}
