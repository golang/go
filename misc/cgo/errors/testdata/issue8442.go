// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Issue 8442.  Cgo output unhelpful error messages for
// invalid C preambles.

/*
void issue8442foo(UNDEF*); // ERROR HERE
*/
import "C"

func main() {
	C.issue8442foo(nil)
}
