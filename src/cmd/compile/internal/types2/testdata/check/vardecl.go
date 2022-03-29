// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vardecl

// Prerequisites.
import "math"
func f() {}
func g() (x, y int) { return }
var m map[string]int

// Var decls must have a type or an initializer.
var _ int
var _, _ int

var _ /* ERROR "expecting type" */
var _, _ /* ERROR "expecting type" */
var _, _, _ /* ERROR "expecting type" */

// The initializer must be an expression.
var _ = int /* ERROR "not an expression" */
var _ = f /* ERROR "used as value" */ ()

// Identifier and expression arity must match.
var _, _ = 1, 2
var _ = 1, 2 /* ERROR "extra init expr 2" */
var _, _ = 1 /* ERROR "cannot initialize [0-9]+ variables with [0-9]+ values" */
var _, _, _ /* ERROR "missing init expr for _" */ = 1, 2

var _ = g /* ERROR "2-valued g" */ ()
var _, _ = g()
var _, _, _ = g /* ERROR "cannot initialize [0-9]+ variables with [0-9]+ values" */ ()

var _ = m["foo"]
var _, _ = m["foo"]
var _, _, _ = m  /* ERROR "cannot initialize [0-9]+ variables with [0-9]+ values" */ ["foo"]

var _, _ int = 1, 2
var _ int = 1, 2 /* ERROR "extra init expr 2" */
var _, _ int = 1 /* ERROR "cannot initialize [0-9]+ variables with [0-9]+ values" */
var _, _, _ /* ERROR "missing init expr for _" */ int = 1, 2

var (
	_, _ = 1, 2
	_ = 1, 2 /* ERROR "extra init expr 2" */
	_, _ = 1 /* ERROR "cannot initialize [0-9]+ variables with [0-9]+ values" */
	_, _, _ /* ERROR "missing init expr for _" */ = 1, 2

	_ = g /* ERROR "2-valued g" */ ()
	_, _ = g()
	_, _, _ = g /* ERROR "cannot initialize [0-9]+ variables with [0-9]+ values" */ ()

	_ = m["foo"]
	_, _ = m["foo"]
	_, _, _ = m /* ERROR "cannot initialize [0-9]+ variables with [0-9]+ values" */ ["foo"]

	_, _ int = 1, 2
	_ int = 1, 2 /* ERROR "extra init expr 2" */
	_, _ int = 1 /* ERROR "cannot initialize [0-9]+ variables with [0-9]+ values" */
	_, _, _ /* ERROR "missing init expr for _" */ int = 1, 2
)

// Variables declared in function bodies must be 'used'.
type T struct{}
func (r T) _(a, b, c int) (u, v, w int) {
	var x1 /* ERROR "declared but not used" */ int
	var x2 /* ERROR "declared but not used" */ int
	x1 = 1
	(x2) = 2

	y1 /* ERROR "declared but not used" */ := 1
	y2 /* ERROR "declared but not used" */ := 2
	y1 = 1
	(y1) = 2

	{
		var x1 /* ERROR "declared but not used" */ int
		var x2 /* ERROR "declared but not used" */ int
		x1 = 1
		(x2) = 2

		y1 /* ERROR "declared but not used" */ := 1
		y2 /* ERROR "declared but not used" */ := 2
		y1 = 1
		(y1) = 2
	}

	if x /* ERROR "declared but not used" */ := 0; a < b {}

	switch x /* ERROR "declared but not used" */, y := 0, 1; a {
	case 0:
		_ = y
	case 1:
		x /* ERROR "declared but not used" */ := 0
	}

	var t interface{}
	switch t /* ERROR "declared but not used" */ := t.(type) {}

	switch t /* ERROR "declared but not used" */ := t.(type) {
	case int:
	}

	switch t /* ERROR "declared but not used" */ := t.(type) {
	case int:
	case float32, complex64:
		t = nil
	}

	switch t := t.(type) {
	case int:
	case float32, complex64:
		_ = t
	}

	switch t := t.(type) {
	case int:
	case float32:
	case string:
		_ = func() string {
			return t
		}
	}

	switch t := t; t /* ERROR "declared but not used" */ := t.(type) {}

	var z1 /* ERROR "declared but not used" */ int
	var z2 int
	_ = func(a, b, c int) (u, v, w int) {
		z1 = a
		(z1) = b
		a = z2
		return
	}

	var s []int
	var i /* ERROR "declared but not used" */ , j int
	for i, j = range s {
		_ = j
	}

	for i, j /* ERROR "declared but not used" */ := range s {
		_ = func() int {
			return i
		}
	}
	return
}

// Unused variables in function literals must lead to only one error (issue #22524).
func _() {
	_ = func() {
		var x /* ERROR declared but not used */ int
	}
}

// Invalid variable declarations must not lead to "declared but not used errors".
func _() {
	var a x                        // ERROR undeclared name: x
	var b = x                      // ERROR undeclared name: x
	var c int = x                  // ERROR undeclared name: x
	var d, e, f x                  /* ERROR x */ /* ERROR x */ /* ERROR x */
	var g, h, i = x, x, x          /* ERROR x */ /* ERROR x */ /* ERROR x */
	var j, k, l float32 = x, x, x  /* ERROR x */ /* ERROR x */ /* ERROR x */
	// but no "declared but not used" errors
}

// Invalid (unused) expressions must not lead to spurious "declared but not used errors".
func _() {
	var a, b, c int
	var x, y int
	x, y = a /* ERROR cannot assign [0-9]+ values to [0-9]+ variables */ , b, c
	_ = x
	_ = y
}

func _() {
	var x int
	return x /* ERROR too many return values */
	return math /* ERROR too many return values */ .Sin(0)
}

func _() int {
	var x, y int
	return x, y /* ERROR too many return values */
}

// Short variable declarations must declare at least one new non-blank variable.
func _() {
	_ := /* ERROR no new variables */ 0
	_, a := 0, 1
	_, a := /* ERROR no new variables */ 0, 1
	_, a, b := 0, 1, 2
	_, _, _ := /* ERROR no new variables */ 0, 1, 2

	_ = a
	_ = b
}

// Test case for variables depending on function literals (see also #22992).
var A /* ERROR initialization cycle */ = func() int { return A }()

func _() {
	// The function literal below must not see a.
	var a = func() int { return a /* ERROR "undeclared name" */ }()
	var _ = func() int { return a }()

	// The function literal below must not see x, y, or z.
	var x, y, z = 0, 1, func() int { return x /* ERROR "undeclared name" */ + y /* ERROR "undeclared name" */ + z /* ERROR "undeclared name" */ }()
	_, _, _ = x, y, z
}

// TODO(gri) consolidate other var decl checks in this file