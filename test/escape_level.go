// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test indirection level computation in escape analysis.

package escape

var sink interface{}

func level0() {
	i := 0     // ERROR "moved to heap: i"
	p0 := &i   // ERROR "moved to heap: p0" "&i escapes to heap"
	p1 := &p0  // ERROR "moved to heap: p1" "&p0 escapes to heap"
	p2 := &p1  // ERROR "moved to heap: p2" "&p1 escapes to heap"
	sink = &p2 // ERROR "&p2 escapes to heap"
}

func level1() {
	i := 0    // ERROR "moved to heap: i"
	p0 := &i  // ERROR "moved to heap: p0" "&i escapes to heap"
	p1 := &p0 // ERROR "moved to heap: p1" "&p0 escapes to heap"
	p2 := &p1 // ERROR "&p1 escapes to heap"
	sink = p2 // ERROR "p2 escapes to heap"
}

func level2() {
	i := 0     // ERROR "moved to heap: i"
	p0 := &i   // ERROR "moved to heap: p0" "&i escapes to heap"
	p1 := &p0  // ERROR "&p0 escapes to heap"
	p2 := &p1  // ERROR "&p1 does not escape"
	sink = *p2 // ERROR "\*p2 escapes to heap"
}

func level3() {
	i := 0      // ERROR "moved to heap: i"
	p0 := &i    // ERROR "&i escapes to heap"
	p1 := &p0   // ERROR "&p0 does not escape"
	p2 := &p1   // ERROR "&p1 does not escape"
	sink = **p2 // ERROR "\* \(\*p2\) escapes to heap"
}

func level4() {
	i := 0     // ERROR "moved to heap: i"
	p0 := &i   // ERROR "moved to heap: p0" "&i escapes to heap"
	p1 := &p0  // ERROR "&p0 escapes to heap"
	p2 := p1   // ERROR "moved to heap: p2"
	sink = &p2 // ERROR "&p2 escapes to heap"
}

func level5() {
	i := 0    // ERROR "moved to heap: i"
	p0 := &i  // ERROR "moved to heap: p0" "&i escapes to heap"
	p1 := &p0 // ERROR "&p0 escapes to heap"
	p2 := p1
	sink = p2 // ERROR "p2 escapes to heap"
}

func level6() {
	i := 0    // ERROR "moved to heap: i"
	p0 := &i  // ERROR "&i escapes to heap"
	p1 := &p0 // ERROR "&p0 does not escape"
	p2 := p1
	sink = *p2 // ERROR "\*p2 escapes to heap"
}

func level7() {
	i := 0    // ERROR "moved to heap: i"
	p0 := &i  // ERROR "&i escapes to heap"
	p1 := &p0 // ERROR "&p0 does not escape"
	// note *p1 == &i
	p2 := *p1  // ERROR "moved to heap: p2"
	sink = &p2 // ERROR "&p2 escapes to heap"
}

func level8() {
	i := 0    // ERROR "moved to heap: i"
	p0 := &i  // ERROR "&i escapes to heap"
	p1 := &p0 // ERROR "&p0 does not escape"
	p2 := *p1
	sink = p2 // ERROR "p2 escapes to heap"
}

func level9() {
	i := 0
	p0 := &i  // ERROR "&i does not escape"
	p1 := &p0 // ERROR "&p0 does not escape"
	p2 := *p1
	sink = *p2 // ERROR "\*p2 escapes to heap"
}

func level10() {
	i := 0
	p0 := &i // ERROR "&i does not escape"
	p1 := *p0
	p2 := &p1  // ERROR "&p1 does not escape"
	sink = *p2 // ERROR "\*p2 escapes to heap"
}

func level11() {
	i := 0
	p0 := &i   // ERROR "&i does not escape"
	p1 := &p0  // ERROR "&p0 does not escape"
	p2 := **p1 // ERROR "moved to heap: p2"
	sink = &p2 // ERROR "&p2 escapes to heap"
}
