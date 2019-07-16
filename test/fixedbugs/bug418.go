// errorcheck

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 3044.
// Multiple valued expressions in return lists.

package p

func Two() (a, b int)

// F used to compile.
func F() (x interface{}, y int) {
	return Two(), 0 // ERROR "single-value context"
}

// Recursive used to trigger an internal compiler error.
func Recursive() (x interface{}, y int) {
	return Recursive(), 0 // ERROR "single-value context"
}
