// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 10320: 7g failed to compile a program because it attempted
// to use ZR as register. Other programs compiled but failed to
// execute correctly because they clobbered the g register.

package main

func main() {
	var x00, x01, x02, x03, x04, x05, x06, x07, x08, x09 int
	var x10, x11, x12, x13, x14, x15, x16, x17, x18, x19 int
	var x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 int
	var x30, x31, x32 int

	_ = x00
	_ = x01
	_ = x02
	_ = x03
	_ = x04
	_ = x05
	_ = x06
	_ = x07
	_ = x08
	_ = x09

	_ = x10
	_ = x11
	_ = x12
	_ = x13
	_ = x14
	_ = x15
	_ = x16
	_ = x17
	_ = x18
	_ = x19

	_ = x20
	_ = x21
	_ = x22
	_ = x23
	_ = x24
	_ = x25
	_ = x26
	_ = x27
	_ = x28
	_ = x29

	_ = x30
	_ = x31
	_ = x32
}
