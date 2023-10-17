// errorcheck

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4359: wrong handling of broken struct fields
// causes "internal compiler error: lookdot badwidth".

package main

type T struct {
	x T1 // ERROR "undefined"
}

func f() {
	var t *T
	_ = t.x
}
