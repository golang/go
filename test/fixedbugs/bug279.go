// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// https://golang.org/issue/799

package main

import "unsafe"

func main() {
	n := unsafe.Sizeof(0)
	if n != 4 && n != 8 {
		println("BUG sizeof 0", n)
		return
	}
	n = unsafe.Alignof(0)
	if n != 4 && n != 8 {
		println("BUG alignof 0", n)
		return
	}
	
	n = unsafe.Sizeof("")
	if n != 8 && n != 16 {
		println("BUG sizeof \"\"", n)
		return
	}
	n = unsafe.Alignof("")
	if n != 4 && n != 8 {
		println("BUG alignof \"\"", n)
		return
	}
}

