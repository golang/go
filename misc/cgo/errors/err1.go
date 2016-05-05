// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#cgo LDFLAGS: -c

void test() {
	xxx;		// ERROR HERE
}
*/
import "C"

func main() {
	C.test()
}
