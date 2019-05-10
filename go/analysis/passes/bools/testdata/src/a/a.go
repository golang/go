// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the bool checker.

package a

import "io"

type T int

func (t T) Foo() int { return int(t) }

type FT func() int

var S []int

func RatherStupidConditions() {
	var f, g func() int
	if f() == 0 || f() == 0 { // OK f might have side effects
	}
	var t T
	_ = t.Foo() == 2 || t.Foo() == 2        // OK Foo might have side effects
	if v, w := f(), g(); v == w || v == w { // want `redundant or: v == w \|\| v == w`
	}
	_ = f == nil || f == nil // want `redundant or: f == nil \|\| f == nil`

	var B byte
	_ = B == byte(1) || B == byte(1) // want `redundant or: B == byte\(1\) \|\| B == byte\(1\)`
	_ = t == T(2) || t == T(2)       // want `redundant or: t == T\(2\) \|\| t == T\(2\)`
	_ = FT(f) == nil || FT(f) == nil // want `redundant or: FT\(f\) == nil \|\| FT\(f\) == nil`

	_ = (func() int)(f) == nil || (func() int)(f) == nil // want `redundant or: \(func\(\) int\)\(f\) == nil \|\| \(func\(\) int\)\(f\) == nil`
	_ = append(S, 3) == nil || append(S, 3) == nil       // OK append has side effects

	var namedFuncVar FT
	_ = namedFuncVar() == namedFuncVar() // OK still func calls

	var c chan int
	_ = 0 == <-c || 0 == <-c                                  // OK subsequent receives may yield different values
	for i, j := <-c, <-c; i == j || i == j; i, j = <-c, <-c { // want `redundant or: i == j \|\| i == j`
	}

	var i, j, k int
	_ = i+1 == 1 || i+1 == 1         // want `redundant or: i\+1 == 1 \|\| i\+1 == 1`
	_ = i == 1 || j+1 == i || i == 1 // want `redundant or: i == 1 \|\| i == 1`

	_ = i == 1 || i == 1 || f() == 1 // want `redundant or: i == 1 \|\| i == 1`
	_ = i == 1 || f() == 1 || i == 1 // OK f may alter i as a side effect
	_ = f() == 1 || i == 1 || i == 1 // want `redundant or: i == 1 \|\| i == 1`

	// Test partition edge cases
	_ = f() == 1 || i == 1 || i == 1 || j == 1 // want `redundant or: i == 1 \|\| i == 1`
	_ = f() == 1 || j == 1 || i == 1 || i == 1 // want `redundant or: i == 1 \|\| i == 1`
	_ = i == 1 || f() == 1 || i == 1 || i == 1 // want `redundant or: i == 1 \|\| i == 1`
	_ = i == 1 || i == 1 || f() == 1 || i == 1 // want `redundant or: i == 1 \|\| i == 1`
	_ = i == 1 || i == 1 || j == 1 || f() == 1 // want `redundant or: i == 1 \|\| i == 1`
	_ = j == 1 || i == 1 || i == 1 || f() == 1 // want `redundant or: i == 1 \|\| i == 1`
	_ = i == 1 || f() == 1 || f() == 1 || i == 1

	_ = i == 1 || (i == 1 || i == 2)             // want `redundant or: i == 1 \|\| i == 1`
	_ = i == 1 || (f() == 1 || i == 1)           // OK f may alter i as a side effect
	_ = i == 1 || (i == 1 || f() == 1)           // want `redundant or: i == 1 \|\| i == 1`
	_ = i == 1 || (i == 2 || (i == 1 || i == 3)) // want `redundant or: i == 1 \|\| i == 1`

	var a, b bool
	_ = i == 1 || (a || (i == 1 || b)) // want `redundant or: i == 1 \|\| i == 1`

	// Check that all redundant ors are flagged
	_ = j == 0 ||
		i == 1 ||
		f() == 1 ||
		j == 0 || // want `redundant or: j == 0 \|\| j == 0`
		i == 1 || // want `redundant or: i == 1 \|\| i == 1`
		i == 1 || // want `redundant or: i == 1 \|\| i == 1`
		i == 1 ||
		j == 0 ||
		k == 0

	_ = i == 1*2*3 || i == 1*2*3 // want `redundant or: i == 1\*2\*3 \|\| i == 1\*2\*3`

	// These test that redundant, suspect expressions do not trigger multiple errors.
	_ = i != 0 || i != 0 // want `redundant or: i != 0 \|\| i != 0`
	_ = i == 0 && i == 0 // want `redundant and: i == 0 && i == 0`

	// and is dual to or; check the basics and
	// let the or tests pull the rest of the weight.
	_ = 0 != <-c && 0 != <-c         // OK subsequent receives may yield different values
	_ = f() != 0 && f() != 0         // OK f might have side effects
	_ = f != nil && f != nil         // want `redundant and: f != nil && f != nil`
	_ = i != 1 && i != 1 && f() != 1 // want `redundant and: i != 1 && i != 1`
	_ = i != 1 && f() != 1 && i != 1 // OK f may alter i as a side effect
	_ = f() != 1 && i != 1 && i != 1 // want `redundant and: i != 1 && i != 1`
}

func RoyallySuspectConditions() {
	var i, j int

	_ = i == 0 || i == 1 // OK
	_ = i != 0 || i != 1 // want `suspect or: i != 0 \|\| i != 1`
	_ = i != 0 || 1 != i // want `suspect or: i != 0 \|\| 1 != i`
	_ = 0 != i || 1 != i // want `suspect or: 0 != i \|\| 1 != i`
	_ = 0 != i || i != 1 // want `suspect or: 0 != i \|\| i != 1`

	_ = (0 != i) || i != 1 // want `suspect or: 0 != i \|\| i != 1`

	_ = i+3 != 7 || j+5 == 0 || i+3 != 9 // want `suspect or: i\+3 != 7 \|\| i\+3 != 9`

	_ = i != 0 || j == 0 || i != 1 // want `suspect or: i != 0 \|\| i != 1`

	_ = i != 0 || i != 1<<4 // want `suspect or: i != 0 \|\| i != 1<<4`

	_ = i != 0 || j != 0
	_ = 0 != i || 0 != j

	var s string
	_ = s != "one" || s != "the other" // want `suspect or: s != .one. \|\| s != .the other.`

	_ = "et" != "alii" || "et" != "cetera"         // want `suspect or: .et. != .alii. \|\| .et. != .cetera.`
	_ = "me gustas" != "tu" || "le gustas" != "tu" // OK we could catch this case, but it's not worth the code

	var err error
	_ = err != nil || err != io.EOF // TODO catch this case?

	// Sanity check and.
	_ = i != 0 && i != 1 // OK
	_ = i == 0 && i == 1 // want `suspect and: i == 0 && i == 1`
	_ = i == 0 && 1 == i // want `suspect and: i == 0 && 1 == i`
	_ = 0 == i && 1 == i // want `suspect and: 0 == i && 1 == i`
	_ = 0 == i && i == 1 // want `suspect and: 0 == i && i == 1`
}
