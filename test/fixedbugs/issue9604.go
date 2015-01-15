// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var x uint16 = 0xffff
var y uint16 = 0xfffe
var a uint16 = 0x7000
var b uint16 = 0x9000

func main() {
	// Make sure we truncate to smaller-width types after evaluating expressions.
	// This is a problem for arm where there is no 16-bit comparison op.
	if ^x != 0 {
		panic("^uint16(0xffff) != 0")
	}
	if ^y != 1 {
		panic("^uint16(0xfffe) != 1")
	}
	if -x != 1 {
		panic("-uint16(0xffff) != 1")
	}
	if a+b != 0 {
		panic("0x7000+0x9000 != 0")
	}
}
