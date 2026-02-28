// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that erroneous labels are caught by the compiler.
// This set is caught by pass 1.
// Does not compile.

package main

var x int

func f() {
L1: // ERROR "label .*L1.* defined and not used"
	for {
	}
L2: // ERROR "label .*L2.* defined and not used"
	select {}
L3: // ERROR "label .*L3.* defined and not used"
	switch {
	}
L4: // ERROR "label .*L4.* defined and not used"
	if true {
	}
L5: // ERROR "label .*L5.* defined and not used"
	f()
L6: // GCCGO_ERROR "previous"
	f()
L6: // ERROR "label .*L6.* already defined"
	f()
	if x == 20 {
		goto L6
	}

L7:
	for {
		break L7
	}

L8:
	for {
		if x == 21 {
			continue L8
		}
	}

L9:
	switch {
	case true:
		break L9
	defalt: // ERROR "label .*defalt.* defined and not used"
	}

L10:
	select {
	default:
		break L10
	}

	goto L10

	goto go2 // ERROR "label go2 not defined|reference to undefined label .*go2"
}
