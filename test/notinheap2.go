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

// Globals and stack variables are okay.

var x nih

func f() {
	var y nih
	x = y
}

// Heap allocation is not okay.

var y *nih
var z []nih

func g() {
	y = new(nih)       // ERROR "heap allocation disallowed"
	z = make([]nih, 1) // ERROR "heap allocation disallowed"
	z = append(z, x)   // ERROR "heap allocation disallowed"
}

// Writes don't produce write barriers.

var p *nih

//go:nowritebarrier
func h() {
	y.next = p.next
}
