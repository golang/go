// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we're compiling "_" functions at least enough
// to get to an error which is generated during walk.

package main

func _() {
	x := 7 // ERROR ".*x.* declared but not used"
}
