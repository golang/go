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

var _; /* ERROR "expected type" */
var _, _; /* ERROR "expected type" */
var _, _, _; /* ERROR "expected type" */

// The initializer must be an expression.
var _ = int /* ERROR "not an expression" */
var _ = f /* ERROR "used as value" */ ()

// Identifier and expression arity must match.
var _, _ = 1, 2
var _ = 1, 2 /* ERROR "extra init expr 2" */
var _, _ = 1 /* ERRORx `assignment mismatch: [1-9]+ variables but.*[1-9]+ value(s)?` */
var _, _, _ /* ERROR "missing init expr for _" */ = 1, 2

var _ = g /* ERROR "multiple-value g" */ ()
var _, _ = g()
var _, _, _ = g /* ERRORx `assignment mismatch: [1-9]+ variables but.*[1-9]+ value(s)?` */ ()

var _ = m["foo"]
var _, _ = m["foo"]
var _, _, _ = m  /* ERRORx `assignment mismatch: [1-9]+ variables but.*[1-9]+ value(s)?` */ ["foo"]

var _, _ int = 1, 2
var _ int = 1, 2 /* ERROR "extra init expr 2" */
var _, _ int = 1 /* ERRORx `assignment mismatch: [1-9]+ variables but.*[1-9]+ value(s)?` */
var _, _, _ /* ERROR "missing init expr for _" */ int = 1, 2

var (
	_, _ = 1, 2
	_ = 1, 2 /* ERROR "extra init expr 2" */
	_, _ = 1 /* ERRORx `assignment mismatch: [1-9]+ variables but.*[1-9]+ value(s)?` */
	_, _, _ /* ERROR "missing init expr for _" */ = 1, 2

	_ = g /* ERROR "multiple-value g" */ ()
	_, _ = g()
	_, _, _ = g /* ERRORx `assignment mismatch: [1-9]+ variables but.*[1-9]+ value(s)?` */ ()

	_ = m["foo"]
	_, _ = m["foo"]
	_, _, _ = m /* ERRORx `assignment mismatch: [1-9]+ variables but.*[1-9]+ value(s)?` */ ["foo"]

	_, _ int = 1, 2
	_ int = 1, 2 /* ERROR "extra init expr 2" */
	_, _ int = 1 /* ERRORx `assignment mismatch: [1-9]+ variables but.*[1-9]+ value(s)?` */
	_, _, _ /* ERROR "missing init expr for _" */ int = 1, 2
)

// Variables declared in function bodies must be 'used'.
type T struct{}
func (r T) _(a, b, c int) (u, v, w int) {
	var x1 /* ERROR "declared and not used" */ int
	var x2 /* ERROR "declared and not used" */ int
	x1 = 1
	(x2) = 2

	y1 /* ERROR "declared and not used" */ := 1
	y2 /* ERROR "declared and not used" */ := 2
	y1 = 1
	(y1) = 2

	{
		var x1 /* ERROR "declared and not used" */ int
		var x2 /* ERROR "declared and not used" */ int
		x1 = 1
		(x2) = 2

		y1 /* ERROR "declared and not used" */ := 1
		y2 /* ERROR "declared and not used" */ := 2
		y1 = 1
		(y1) = 2
	}

	if x /* ERROR "declared and not used" */ := 0; a < b {}

	switch x /* ERROR "declared and not used" */, y := 0, 1; a {
	case 0:
		_ = y
	case 1:
		x /* ERROR "declared and not used" */ := 0
	}

	var t interface{}
	switch t /* ERROR "declared and not used" */ := t.(type) {}

	switch t /* ERROR "declared and not used" */ := t.(type) {
	case int:
	}

	switch t /* ERROR "declared and not used" */ := t.(type) {
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

	switch t := t; t /* ERROR "declared and not used" */ := t.(type) {}

	var z1 /* ERROR "declared and not used" */ int
	var z2 int
	_ = func(a, b, c int) (u, v, w int) {
		z1 = a
		(z1) = b
		a = z2
		return
	}

	var s []int
	var i /* ERROR "declared and not used" */ , j int
	for i, j = range s {
		_ = j
	}

	for i, j /* ERROR "declared and not used" */ := range s {
		_ = func() int {
			return i
		}
	}
	return
}

// Unused variables in function literals must lead to only one error (issue #22524).
func _() {
	_ = func() {
		var x /* ERROR "declared and not used" */ int
	}
}

// Invalid variable declarations must not lead to "declared and not used errors".
// TODO(gri) enable these tests once go/types follows types2 logic for declared and not used variables
// func _() {
//	var a x                        // DISABLED_ERROR undefined: x
//	var b = x                      // DISABLED_ERROR undefined: x
//	var c int = x                  // DISABLED_ERROR undefined: x
//	var d, e, f x                  /* DISABLED_ERROR x */ /* DISABLED_ERROR x */ /* DISABLED_ERROR x */
//	var g, h, i = x, x, x          /* DISABLED_ERROR x */ /* DISABLED_ERROR x */ /* DISABLED_ERROR x */
//	var j, k, l float32 = x, x, x  /* DISABLED_ERROR x */ /* DISABLED_ERROR x */ /* DISABLED_ERROR x */
//	// but no "declared and not used" errors
// }

// Invalid (unused) expressions must not lead to spurious "declared and not used errors".
func _() {
	var a, b, c int
	var x, y int
	x, y = a /* ERRORx `assignment mismatch: [1-9]+ variables but.*[1-9]+ value(s)?` */ , b, c
	_ = x
	_ = y
}

func _() {
	var x int
	return x /* ERROR "too many return values" */
	return math /* ERROR "too many return values" */ .Sin(0)
}

func _() int {
	var x, y int
	return x, y /* ERROR "too many return values" */
}

// Short variable declarations must declare at least one new non-blank variable.
func _() {
	_ := /* ERROR "no new variables" */ 0
	_, a := 0, 1
	_, a := /* ERROR "no new variables" */ 0, 1
	_, a, b := 0, 1, 2
	_, _, _ := /* ERROR "no new variables" */ 0, 1, 2

	_ = a
	_ = b
}

// Test case for variables depending on function literals (see also #22992).
var A /* ERROR "initialization cycle" */ = func() int { return A }()

func _() {
	// The function literal below must not see a.
	var a = func() int { return a /* ERROR "undefined" */ }()
	var _ = func() int { return a }()

	// The function literal below must not see x, y, or z.
	var x, y, z = 0, 1, func() int { return x /* ERROR "undefined" */ + y /* ERROR "undefined" */ + z /* ERROR "undefined" */ }()
	_, _, _ = x, y, z
}

// TODO(gri) consolidate other var decl checks in this file