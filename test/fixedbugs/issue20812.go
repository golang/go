// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() {
	_ = int("1")      // ERROR "cannot convert"
	_ = bool(0)       // ERROR "cannot convert"
	_ = bool("false") // ERROR "cannot convert"
	_ = int(false)    // ERROR "cannot convert"
	_ = string(true)  // ERROR "cannot convert"
}
