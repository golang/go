// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for maps.

package escape

var sink interface{}

func map0() {
	m := make(map[*int]*int) // ERROR "make\(map\[\*int\]\*int\) does not escape"
	// BAD: i should not escape
	i := 0 // ERROR "moved to heap: i"
	// BAD: j should not escape
	j := 0     // ERROR "moved to heap: j"
	m[&i] = &j // ERROR "&i escapes to heap" "&j escapes to heap"
	_ = m
}

func map1() *int {
	m := make(map[*int]*int) // ERROR "make\(map\[\*int\]\*int\) does not escape"
	// BAD: i should not escape
	i := 0       // ERROR "moved to heap: i"
	j := 0       // ERROR "moved to heap: j"
	m[&i] = &j   // ERROR "&i escapes to heap" "&j escapes to heap"
	return m[&i] // ERROR "&i does not escape"
}

func map2() map[*int]*int {
	m := make(map[*int]*int) // ERROR "make\(map\[\*int\]\*int\) escapes to heap"
	i := 0                   // ERROR "moved to heap: i"
	j := 0                   // ERROR "moved to heap: j"
	m[&i] = &j               // ERROR "&i escapes to heap" "&j escapes to heap"
	return m
}

func map3() []*int {
	m := make(map[*int]*int) // ERROR "make\(map\[\*int\]\*int\) does not escape"
	i := 0                   // ERROR "moved to heap: i"
	// BAD: j should not escape
	j := 0     // ERROR "moved to heap: j"
	m[&i] = &j // ERROR "&i escapes to heap" "&j escapes to heap"
	var r []*int
	for k := range m {
		r = append(r, k)
	}
	return r
}

func map4() []*int {
	m := make(map[*int]*int) // ERROR "make\(map\[\*int\]\*int\) does not escape"
	// BAD: i should not escape
	i := 0     // ERROR "moved to heap: i"
	j := 0     // ERROR "moved to heap: j"
	m[&i] = &j // ERROR "&i escapes to heap" "&j escapes to heap"
	var r []*int
	for k, v := range m {
		// We want to test exactly "for k, v := range m" rather than "for _, v := range m".
		// The following if is merely to use (but not leak) k.
		if k != nil {
			r = append(r, v)
		}
	}
	return r
}

func map5(m map[*int]*int) { // ERROR "m does not escape"
	i := 0     // ERROR "moved to heap: i"
	j := 0     // ERROR "moved to heap: j"
	m[&i] = &j // ERROR "&i escapes to heap" "&j escapes to heap"
}

func map6(m map[*int]*int) { // ERROR "m does not escape"
	if m != nil {
		m = make(map[*int]*int) // ERROR "make\(map\[\*int\]\*int\) does not escape"
	}
	i := 0     // ERROR "moved to heap: i"
	j := 0     // ERROR "moved to heap: j"
	m[&i] = &j // ERROR "&i escapes to heap" "&j escapes to heap"
}

func map7() {
	// BAD: i should not escape
	i := 0 // ERROR "moved to heap: i"
	// BAD: j should not escape
	j := 0                     // ERROR "moved to heap: j"
	m := map[*int]*int{&i: &j} // ERROR "&i escapes to heap" "&j escapes to heap" "literal does not escape"
	_ = m
}

func map8() {
	i := 0                     // ERROR "moved to heap: i"
	j := 0                     // ERROR "moved to heap: j"
	m := map[*int]*int{&i: &j} // ERROR "&i escapes to heap" "&j escapes to heap" "literal escapes to heap"
	sink = m // ERROR "m escapes to heap"
}

func map9() *int {
	// BAD: i should not escape
	i := 0                     // ERROR "moved to heap: i"
	j := 0                     // ERROR "moved to heap: j"
	m := map[*int]*int{&i: &j} // ERROR "&i escapes to heap" "&j escapes to heap" "literal does not escape"
	return m[nil]
}
