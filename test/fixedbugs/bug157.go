// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f() {}

func main() {
	x := 0;

	// this compiles
	switch x {
	case 0: f();
	default: f();
	}

	// this doesn't but it should
	// (semicolons are not needed at the end of a statement list)
	switch x {
	case 0: f()
	default: f()
	}
}


/*
bug157.go:20: syntax error near default
bug157.go:20: first switch statement must be a case
*/
