// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
void foo() {}
*/
import "C"

func main() {
	C.foo = C.foo // ERROR HERE
}
