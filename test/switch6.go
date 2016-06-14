// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check the compiler's switch handling that happens
// at typechecking time.
// This must be separate from other checks,
// because errors during typechecking
// prevent other errors from being discovered.

package main

// Verify that type switch statements with impossible cases are detected by the compiler.
func f0(e error) {
	switch e.(type) {
	case int: // ERROR "impossible type switch case: e \(type error\) cannot have dynamic type int \(missing Error method\)"
	}
}

// Verify that the compiler rejects multiple default cases.
func f1(e interface{}) {
	switch e { // ERROR "multiple defaults in switch"
	default:
	default:
	}
	switch e.(type) { // ERROR "multiple defaults in switch"
	default:
	default:
	}
}
