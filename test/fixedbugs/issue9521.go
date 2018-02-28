// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that an incorrect use of the blank identifer is caught.
// Does not compile.

package main

func f() (_, _ []int)         { return }
func g() (x []int, y float64) { return }

func main() {
	_ = append(f()) // ERROR "cannot append \[\]int value to \[\]int"
	_ = append(g()) // ERROR "cannot append float64 value to \[\]int"
}
