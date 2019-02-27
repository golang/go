// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the suspicious shift checker.

package shift

func ShiftTest() {
	var i8 int8
	_ = i8 << 7
	_ = (i8 + 1) << 8 // ERROR ".i8 . 1. .8 bits. too small for shift of 8"
}
