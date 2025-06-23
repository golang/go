// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#cgo LDFLAGS: -L/nonexist

void test() {
	xxx;		// ERROR HERE
}

// Issue 8442.  Cgo output unhelpful error messages for
// invalid C preambles.
void issue8442foo(UNDEF*); // ERROR HERE
*/
import "C"

func main() {
	C.test()
}
