// errorcheck -+ -0 -l -d=wb

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test write barrier elimination for notinheap.

package p

type t1 struct {
	x *nih
	s []nih
	y [1024]byte // Prevent write decomposition
}

type t2 struct {
	x *ih
	s []ih
	y [1024]byte
}

//go:notinheap
type nih struct {
	x uintptr
}

type ih struct { // In-heap type
	x uintptr
}

var (
	v1 t1
	v2 t2

	v1s []t1
	v2s []t2
)

func f() {
	// Test direct writes
	v1.x = nil        // no barrier
	v2.x = nil        // ERROR "write barrier"
	v1.s = []nih(nil) // no barrier
	v2.s = []ih(nil)  // ERROR "write barrier"
}

func g() {
	// Test aggregate writes
	v1 = t1{x: nil} // no barrier
	v2 = t2{x: nil} // ERROR "write barrier"
}

func h() {
	// Test copies and appends.
	copy(v1s, v1s[1:])      // no barrier
	copy(v2s, v2s[1:])      // ERROR "write barrier"
	_ = append(v1s, v1s...) // no barrier
	_ = append(v2s, v2s...) // ERROR "write barrier"
}

// Slice clearing

var (
	sliceIH  []*ih
	sliceNIH []*nih
)

func sliceClear() {
	for i := range sliceIH {
		sliceIH[i] = nil // ERROR "write barrier"
	}
	for i := range sliceNIH {
		sliceNIH[i] = nil // no barrier
	}
}
