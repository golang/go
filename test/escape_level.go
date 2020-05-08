// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test indirection level computation in escape analysis.

package escape

var sink interface{}

func level0() {
	i := 0     // ERROR "moved to heap: i"
	p0 := &i   // ERROR "moved to heap: p0"
	p1 := &p0  // ERROR "moved to heap: p1"
	p2 := &p1  // ERROR "moved to heap: p2"
	sink = &p2
}

func level1() {
	i := 0    // ERROR "moved to heap: i"
	p0 := &i  // ERROR "moved to heap: p0"
	p1 := &p0 // ERROR "moved to heap: p1"
	p2 := &p1
	sink = p2
}

func level2() {
	i := 0     // ERROR "moved to heap: i"
	p0 := &i   // ERROR "moved to heap: p0"
	p1 := &p0
	p2 := &p1
	sink = *p2
}

func level3() {
	i := 0      // ERROR "moved to heap: i"
	p0 := &i
	p1 := &p0
	p2 := &p1
	sink = **p2
}

func level4() {
	i := 0     // ERROR "moved to heap: i"
	p0 := &i   // ERROR "moved to heap: p0"
	p1 := &p0
	p2 := p1   // ERROR "moved to heap: p2"
	sink = &p2
}

func level5() {
	i := 0    // ERROR "moved to heap: i"
	p0 := &i  // ERROR "moved to heap: p0"
	p1 := &p0
	p2 := p1
	sink = p2
}

func level6() {
	i := 0    // ERROR "moved to heap: i"
	p0 := &i
	p1 := &p0
	p2 := p1
	sink = *p2
}

func level7() {
	i := 0    // ERROR "moved to heap: i"
	p0 := &i
	p1 := &p0
	// note *p1 == &i
	p2 := *p1  // ERROR "moved to heap: p2"
	sink = &p2
}

func level8() {
	i := 0    // ERROR "moved to heap: i"
	p0 := &i
	p1 := &p0
	p2 := *p1
	sink = p2
}

func level9() {
	i := 0
	p0 := &i
	p1 := &p0
	p2 := *p1
	sink = *p2 // ERROR "\*p2 escapes to heap"
}

func level10() {
	i := 0
	p0 := &i
	p1 := *p0
	p2 := &p1
	sink = *p2 // ERROR "\*p2 escapes to heap"
}

func level11() {
	i := 0
	p0 := &i
	p1 := &p0
	p2 := **p1 // ERROR "moved to heap: p2"
	sink = &p2
}
