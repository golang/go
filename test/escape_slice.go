// errorcheck -0 -m -l

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for slices.

package escape

var sink interface{}

func slice0() {
	var s []*int
	// BAD: i should not escape
	i := 0            // ERROR "moved to heap: i"
	s = append(s, &i) // ERROR "&i escapes to heap"
	_ = s
}

func slice1() *int {
	var s []*int
	i := 0            // ERROR "moved to heap: i"
	s = append(s, &i) // ERROR "&i escapes to heap"
	return s[0]
}

func slice2() []*int {
	var s []*int
	i := 0            // ERROR "moved to heap: i"
	s = append(s, &i) // ERROR "&i escapes to heap"
	return s
}

func slice3() *int {
	var s []*int
	i := 0            // ERROR "moved to heap: i"
	s = append(s, &i) // ERROR "&i escapes to heap"
	for _, p := range s {
		return p
	}
	return nil
}

func slice4(s []*int) { // ERROR "s does not escape"
	i := 0    // ERROR "moved to heap: i"
	s[0] = &i // ERROR "&i escapes to heap"
}

func slice5(s []*int) { // ERROR "s does not escape"
	if s != nil {
		s = make([]*int, 10) // ERROR "make\(\[\]\*int, 10\) does not escape"
	}
	i := 0    // ERROR "moved to heap: i"
	s[0] = &i // ERROR "&i escapes to heap"
}

func slice6() {
	s := make([]*int, 10) // ERROR "make\(\[\]\*int, 10\) does not escape"
	// BAD: i should not escape
	i := 0    // ERROR "moved to heap: i"
	s[0] = &i // ERROR "&i escapes to heap"
	_ = s
}

func slice7() *int {
	s := make([]*int, 10) // ERROR "make\(\[\]\*int, 10\) does not escape"
	i := 0                // ERROR "moved to heap: i"
	s[0] = &i             // ERROR "&i escapes to heap"
	return s[0]
}

func slice8() {
	// BAD: i should not escape here
	i := 0          // ERROR "moved to heap: i"
	s := []*int{&i} // ERROR "&i escapes to heap" "literal does not escape"
	_ = s
}

func slice9() *int {
	i := 0          // ERROR "moved to heap: i"
	s := []*int{&i} // ERROR "&i escapes to heap" "literal does not escape"
	return s[0]
}

func slice10() []*int {
	i := 0          // ERROR "moved to heap: i"
	s := []*int{&i} // ERROR "&i escapes to heap" "literal escapes to heap"
	return s
}
