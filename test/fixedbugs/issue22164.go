// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test error recovery after missing closing parentheses in lists.

package p

func f() {
	x := f(g() // ERROR "unexpected newline"
	y := 1
}

func g() {
}

func h() {
	x := f(g() // ERROR "unexpected newline"
}

func i() {
	x := []int{1, 2, 3 // ERROR "unexpected newline"
	y := 0
}