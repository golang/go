// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

func main() {
	// Test that inlined type switches without short variable
	// declarations work correctly.
	check(0, a.F(nil)) // ERROR "inlining call to a.F"
	check(1, a.F(0))   // ERROR "inlining call to a.F" "does not escape"
	check(2, a.F(0.0)) // ERROR "inlining call to a.F" "does not escape"
	check(3, a.F(""))  // ERROR "inlining call to a.F" "does not escape"

	// Test that inlined type switches with short variable
	// declarations work correctly.
	_ = a.G(nil).(*interface{})                       // ERROR "inlining call to a.G"
	_ = a.G(1).(*int)                                 // ERROR "inlining call to a.G" "does not escape"
	_ = a.G(2.0).(*float64)                           // ERROR "inlining call to a.G" "does not escape"
	_ = (*a.G("").(*interface{})).(string)            // ERROR "inlining call to a.G" "does not escape"
	_ = (*a.G(([]byte)(nil)).(*interface{})).([]byte) // ERROR "inlining call to a.G" "does not escape"
	_ = (*a.G(true).(*interface{})).(bool)            // ERROR "inlining call to a.G" "does not escape"
}

//go:noinline
func check(want, got int) {
	if want != got {
		println("want", want, "but got", got)
	}
}
