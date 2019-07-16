// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Internal compiler crash used to stop errors during second copy.

package main

func main() {
	_ = copy(nil, []int{}) // ERROR "use of untyped nil"
	_ = copy([]int{}, nil) // ERROR "use of untyped nil"
	_ = 1+true // ERROR "cannot convert true" "mismatched types int and bool"
}
