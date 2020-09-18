// errorcheck -0 -m -l

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for goto statements.

package escape

var x bool

func _() {
	var p *int
loop:
	if x {
		goto loop
	}
	// BAD: We should be able to recognize that there
	// aren't any more "goto loop" after here.
	p = new(int) // ERROR "escapes to heap"
	_ = p
}

func _() {
	var p *int
	if x {
	loop:
		goto loop
	} else {
		p = new(int) // ERROR "does not escape"
	}
	_ = p
}

func _() {
	var p *int
	if x {
	loop:
		goto loop
	}
	p = new(int) // ERROR "does not escape"
	_ = p
}
