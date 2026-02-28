// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that switch statements with duplicate cases are detected by the compiler.
// Does not compile.

package main

func f0(x int) {
	switch x {
	case 0:
	case 0: // ERROR "duplicate case (0 in switch)?"
	}

	switch x {
	case 0:
	case int(0): // ERROR "duplicate case (int.0. .value 0. in switch)?"
	}
}

func f1(x float32) {
	switch x {
	case 5:
	case 5: // ERROR "duplicate case (5 in switch)?"
	case 5.0: // ERROR "duplicate case (5 in switch)?"
	}
}

func f2(s string) {
	switch s {
	case "":
	case "": // ERROR "duplicate case (.. in switch)?"
	case "abc":
	case "abc": // ERROR "duplicate case (.abc. in switch)?"
	}
}

func f3(e interface{}) {
	switch e {
	case 0:
	case 0: // ERROR "duplicate case (0 in switch)?"
	case int64(0):
	case float32(10):
	case float32(10): // ERROR "duplicate case (float32\(10\) .value 10. in switch)?"
	case float64(10):
	case float64(10): // ERROR "duplicate case (float64\(10\) .value 10. in switch)?"
	}
}

func f5(a [1]int) {
	switch a {
	case [1]int{0}:
	case [1]int{0}: // OK -- see issue 15896
	}
}

// Ensure duplicate const bool clauses are accepted.
func f6() int {
	switch {
	case 0 == 0:
		return 0
	case 1 == 1: // Intentionally OK, even though a duplicate of the above const true
		return 1
	}
	return 2
}

// Ensure duplicates in ranges are detected (issue #17517).
func f7(a int) {
	switch a {
	case 0:
	case 0, 1: // ERROR "duplicate case 0"
	case 1, 2, 3, 4: // ERROR "duplicate case 1"
	}
}

// Ensure duplicates with simple literals are printed as they were
// written, not just their values. Particularly useful for runes.
func f8(r rune) {
	const x = 10
	switch r {
	case 33, 33: // ERROR "duplicate case (33 in switch)?"
	case 34, '"': // ERROR "duplicate case '"' .value 34. in switch"
	case 35, rune('#'): // ERROR "duplicate case (rune.'#'. .value 35. in switch)?"
	case 36, rune(36): // ERROR "duplicate case (rune.36. .value 36. in switch)?"
	case 37, '$'+1: // ERROR "duplicate case ('\$' \+ 1 .value 37. in switch)?"
	case 'b':
	case 'a', 'b', 'c', 'd': // ERROR "duplicate case ('b' .value 98.)?"
	case x, x: // ERROR "duplicate case (x .value 10.)?"
	}
}
