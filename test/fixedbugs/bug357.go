// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 1993.
// error used to have last line number in file

package main

func bla1() bool {
	return false
}

func bla5() bool {
	_ = 1
	false  // ERROR "false evaluated but not used|value computed is not used|is not used"
	_ = 2
	return false
}

func main() {
	x := bla1()
	_ = x
}
