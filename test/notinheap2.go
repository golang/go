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
	var y nih // ERROR "nih is incomplete \(or unallocatable\); stack allocation disallowed"
	x = y
}

// Heap allocation is not okay.

var y *nih
var y2 *struct{ x nih }
var y3 *[1]nih
var z []nih
var w []nih
var n int
var sink interface{}

type embed1 struct { // implicitly notinheap
	x nih
}

type embed2 [1]nih // implicitly notinheap

type embed3 struct { // implicitly notinheap
	x [1]nih
}

// Type aliases inherit the go:notinheap-ness of the type they alias.
type nihAlias = nih

type embedAlias1 struct { // implicitly notinheap
	x nihAlias
}
type embedAlias2 [1]nihAlias // implicitly notinheap

func g() {
	y = new(nih)              // ERROR "can't be allocated in Go"
	y2 = new(struct{ x nih }) // ERROR "can't be allocated in Go"
	y3 = new([1]nih)          // ERROR "can't be allocated in Go"
	z = make([]nih, 1)        // ERROR "can't be allocated in Go"
	z = append(z, x)          // ERROR "can't be allocated in Go"

	sink = new(embed1)      // ERROR "can't be allocated in Go"
	sink = new(embed2)      // ERROR "can't be allocated in Go"
	sink = new(embed3)      // ERROR "can't be allocated in Go"
	sink = new(embedAlias1) // ERROR "can't be allocated in Go"
	sink = new(embedAlias2) // ERROR "can't be allocated in Go"

	// Test for special case of OMAKESLICECOPY
	x := make([]nih, n) // ERROR "can't be allocated in Go"
	copy(x, z)
	z = x
}

// Writes don't produce write barriers.

var p *nih

//go:nowritebarrier
func h() {
	y.next = p.next
}
