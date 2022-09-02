// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package constdecl

import "math"
import "unsafe"

var v int

// Const decls must be initialized by constants.
const _ = v /* ERROR "not constant" */
const _ = math /* ERROR "not constant" */ .Sin(0)
const _ = int /* ERROR "not an expression" */

func _() {
	const _ = v /* ERROR "not constant" */
	const _ = math /* ERROR "not constant" */ .Sin(0)
	const _ = int /* ERROR "not an expression" */
}

// Identifier and expression arity must match.
const _ /* ERROR "missing init expr for _" */
const _ = 1, 2 /* ERROR "extra init expr 2" */

const _ /* ERROR "missing init expr for _" */ int
const _ int = 1, 2 /* ERROR "extra init expr 2" */

const (
	_ /* ERROR "missing init expr for _" */
	_ = 1, 2 /* ERROR "extra init expr 2" */

	_ /* ERROR "missing init expr for _" */ int
	_ int = 1, 2 /* ERROR "extra init expr 2" */
)

const (
	_ = 1
	_
	_, _ /* ERROR "missing init expr for _" */
	_
)

const (
	_, _ = 1, 2
	_, _
	_ /* ERROR "extra init expr at" */
	_, _
	_, _, _ /* ERROR "missing init expr for _" */
	_, _
)

func _() {
	const _ /* ERROR "missing init expr for _" */
	const _ = 1, 2 /* ERROR "extra init expr 2" */

	const _ /* ERROR "missing init expr for _" */ int
	const _ int = 1, 2 /* ERROR "extra init expr 2" */

	const (
		_ /* ERROR "missing init expr for _" */
		_ = 1, 2 /* ERROR "extra init expr 2" */

		_ /* ERROR "missing init expr for _" */ int
		_ int = 1, 2 /* ERROR "extra init expr 2" */
	)

	const (
		_ = 1
		_
		_, _ /* ERROR "missing init expr for _" */
		_
	)

	const (
		_, _ = 1, 2
		_, _
		_ /* ERROR "extra init expr at" */
		_, _
		_, _, _ /* ERROR "missing init expr for _" */
		_, _
	)
}

// Test case for constant with invalid initialization.
// Caused panic because the constant value was not set up (gri - 7/8/2014).
func _() {
	const (
	    x string = missing /* ERROR "undeclared name" */
	    y = x + ""
	)
}

// Test case for constants depending on function literals (see also #22992).
const A /* ERROR initialization cycle */ = unsafe.Sizeof(func() { _ = A })

func _() {
	// The function literal below must not see a.
	const a = unsafe.Sizeof(func() { _ = a /* ERROR "undeclared name" */ })
	const b = unsafe.Sizeof(func() { _ = a })

	// The function literal below must not see x, y, or z.
	const x, y, z = 0, 1, unsafe.Sizeof(func() { _ = x /* ERROR "undeclared name" */ + y /* ERROR "undeclared name" */ + z /* ERROR "undeclared name" */ })
}

// Test cases for errors in inherited constant initialization expressions.
// Errors related to inherited initialization expressions must appear at
// the constant identifier being declared, not at the original expression
// (issues #42991, #42992).
const (
	_ byte = 255 + iota
	/* some gap */
	_ // ERROR overflows
	/* some gap */
	/* some gap */ _ /* ERROR overflows */; _ /* ERROR overflows */
	/* some gap */
	_ = 255 + iota
	_ = byte /* ERROR overflows */ (255) + iota
	_ /* ERROR overflows */
)

// Test cases from issue.
const (
	ok = byte(iota + 253)
	bad
	barn
	bard // ERROR cannot convert
)

const (
	c = len([1 - iota]int{})
	d
	e // ERROR invalid array length
	f // ERROR invalid array length
)

// Test that identifiers in implicit (omitted) RHS
// expressions of constant declarations are resolved
// in the correct context; see issues #49157, #53585.
const X = 2

func _() {
	const (
		A    = iota // 0
		iota = iota // 1
		B           // 1 (iota is declared locally on prev. line)
		C           // 1
	)
	assert(A == 0 && B == 1 && C == 1)

	const (
		X = X + X
		Y
		Z = iota
	)
	assert(X == 4 && Y == 8 && Z == 1)
}

// TODO(gri) move extra tests from testdata/const0.src into here
