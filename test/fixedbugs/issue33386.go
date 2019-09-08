// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that we don't get spurious follow-on errors
// after a missing expression. Specifically, the parser
// shouldn't skip over closing parentheses of any kind.

package p

func _() {
	go func() { // no error here about goroutine
		send <-
	}() // ERROR "expecting expression"
}

func _() {
	defer func() { // no error here about deferred function
		1 +
	}() // ERROR "expecting expression"
}

func _() {
	_ = (1 +)             // ERROR "expecting expression"
	_ = a[2 +]            // ERROR "expecting expression"
	_ = []int{1, 2, 3 + } // ERROR "expecting expression"
}
