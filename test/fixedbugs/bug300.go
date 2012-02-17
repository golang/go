// errorcheck

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	x, y *T
}

func main() {
	// legal composite literals
	_ = struct{}{}
	_ = [42]int{}
	_ = [...]int{}
	_ = []int{}
	_ = map[int]int{}
	_ = T{}

	// illegal composite literals: parentheses not allowed around literal type
	_ = (struct{}){}    // ERROR "parenthesize"
	_ = ([42]int){}     // ERROR "parenthesize"
	_ = ([...]int){}    // ERROR "parenthesize"
	_ = ([]int){}       // ERROR "parenthesize"
	_ = (map[int]int){} // ERROR "parenthesize"
	_ = (T){}           // ERROR "parenthesize"
}
