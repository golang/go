// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that incorrect uses of the blank identifer are caught.
// Does not compile.

package _	// ERROR "invalid package name _"

var t struct {
	_ int
}

func main() {
	_()	// ERROR "cannot use _ as value"
	x := _+1	// ERROR "cannot use _ as value"
	_ = x
	_ = t._ // ERROR "cannot refer to blank field"
}
