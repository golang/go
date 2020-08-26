// errorcheck -+

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test walk errors for go:notinheap.

package p

//go:notinheap
type nih struct {
	next *nih
}

// Global variables are okay.

var x nih

// Stack variables are not okay.

func f() {
	var y nih // ERROR "nih is go:notinheap; stack allocation disallowed"
	x = y
}

// Heap allocation is not okay.

var y *nih
var y2 *struct{ x nih }
var y3 *[1]nih
var z []nih
var w []nih
var n int

func g() {
	y = new(nih)              // ERROR "heap allocation disallowed"
	y2 = new(struct{ x nih }) // ERROR "heap allocation disallowed"
	y3 = new([1]nih)          // ERROR "heap allocation disallowed"
	z = make([]nih, 1)        // ERROR "heap allocation disallowed"
	z = append(z, x)          // ERROR "heap allocation disallowed"
	// Test for special case of OMAKESLICECOPY
	x := make([]nih, n) // ERROR "heap allocation disallowed"
	copy(x, z)
	z = x
}

// Writes don't produce write barriers.

var p *nih

//go:nowritebarrier
func h() {
	y.next = p.next
}
