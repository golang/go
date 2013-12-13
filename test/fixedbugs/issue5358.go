// errorcheck

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 5358: incorrect error message when using f(g()) form on ... args.

package main

func f(x int, y ...int) {}

func g() (int, []int)

func main() {
	f(g()) // ERROR "as type int in|incompatible type"
}
