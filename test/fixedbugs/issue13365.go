// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 13365: confusing error message (array vs slice)

package main

var t struct{}

func main() {
	_ = []int{-1: 0}    // ERROR "index must be non\-negative integer constant"
	_ = [10]int{-1: 0}  // ERROR "index must be non\-negative integer constant"
	_ = [...]int{-1: 0} // ERROR "index must be non\-negative integer constant"

	_ = []int{100: 0}
	_ = [10]int{100: 0} // ERROR "array index 100 out of bounds"
	_ = [...]int{100: 0}

	_ = []int{t}    // ERROR "cannot use .* as type int in array or slice literal"
	_ = [10]int{t}  // ERROR "cannot use .* as type int in array or slice literal"
	_ = [...]int{t} // ERROR "cannot use .* as type int in array or slice literal"
}
