// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// constant declarations

package const0

import "unsafe"

// constants declarations must be initialized by constants
var x = 0
const c0 = x /* ERROR "not constant" */

// typed constants must have constant types
const _ interface /* ERROR invalid constant type */ {} = 0

func _ () {
	const _ interface /* ERROR invalid constant type */ {} = 0
	for i := 0; i < 10; i++ {} // don't crash with non-nil iota here
}

// untyped constants
const (
	// boolean values
	ub0 = false
	ub1 = true
	ub2 = 2 < 1
	ub3 = ui1 == uf1
	ub4 = true /* ERROR "mismatched types untyped bool and untyped int" */ == 0

	// integer values
	ui0 = 0
	ui1 = 1
	ui2 = 42
	ui3 = 3141592653589793238462643383279502884197169399375105820974944592307816406286
	ui4 = -10

	ui5 = ui0 + ui1
	ui6 = ui1 - ui1
	ui7 = ui2 * ui1
	ui8 = ui3 / ui3
	ui9 = ui3 % ui3

	ui10 = 1 / 0 /* ERROR "division by zero" */
	ui11 = ui1 / 0 /* ERROR "division by zero" */
	ui12 = ui3 / ui0 /* ERROR "division by zero" */
	ui13 = 1 % 0 /* ERROR "division by zero" */
	ui14 = ui1 % 0 /* ERROR "division by zero" */
	ui15 = ui3 % ui0 /* ERROR "division by zero" */

	ui16 = ui2 & ui3
	ui17 = ui2 | ui3
	ui18 = ui2 ^ ui3
	ui19 = 1 /* ERROR "invalid operation" */ % 1.0

	// floating point values
	uf0 = 0.
	uf1 = 1.
	uf2 = 4.2e1
	uf3 = 3.141592653589793238462643383279502884197169399375105820974944592307816406286
	uf4 = 1e-1

	uf5 = uf0 + uf1
	uf6 = uf1 - uf1
	uf7 = uf2 * uf1
	uf8 = uf3 / uf3
	uf9 = uf3 /* ERROR "not defined" */ % uf3

	uf10 = 1 / 0 /* ERROR "division by zero" */
	uf11 = uf1 / 0 /* ERROR "division by zero" */
	uf12 = uf3 / uf0 /* ERROR "division by zero" */

	uf16 = uf2 /* ERROR "not defined" */ & uf3
	uf17 = uf2 /* ERROR "not defined" */ | uf3
	uf18 = uf2 /* ERROR "not defined" */ ^ uf3

	// complex values
	uc0 = 0.i
	uc1 = 1.i
	uc2 = 4.2e1i
	uc3 = 3.141592653589793238462643383279502884197169399375105820974944592307816406286i
	uc4 = 1e-1i

	uc5 = uc0 + uc1
	uc6 = uc1 - uc1
	uc7 = uc2 * uc1
	uc8 = uc3 / uc3
	uc9 = uc3 /* ERROR "not defined" */ % uc3

	uc10 = 1 / 0 /* ERROR "division by zero" */
	uc11 = uc1 / 0 /* ERROR "division by zero" */
	uc12 = uc3 / uc0 /* ERROR "division by zero" */

	uc16 = uc2 /* ERROR "not defined" */ & uc3
	uc17 = uc2 /* ERROR "not defined" */ | uc3
	uc18 = uc2 /* ERROR "not defined" */ ^ uc3
)

type (
	mybool bool
	myint int
	myfloat float64
	mycomplex complex128
)

// typed constants
const (
	// boolean values
	tb0 bool = false
	tb1 bool = true
	tb2 mybool = 2 < 1
	tb3 mybool = ti1 /* ERROR "mismatched types" */ == tf1

	// integer values
	ti0 int8 = ui0
	ti1 int32 = ui1
	ti2 int64 = ui2
	ti3 myint = ui3 /* ERROR "overflows" */
	ti4 myint = ui4

	ti5 = ti0 /* ERROR "mismatched types" */ + ti1
	ti6 = ti1 - ti1
	ti7 = ti2 /* ERROR "mismatched types" */ * ti1
	ti8 = ti3 / ti3
	ti9 = ti3 % ti3

	ti10 = 1 / 0 /* ERROR "division by zero" */
	ti11 = ti1 / 0 /* ERROR "division by zero" */
	ti12 = ti3 /* ERROR "mismatched types" */ / ti0
	ti13 = 1 % 0 /* ERROR "division by zero" */
	ti14 = ti1 % 0 /* ERROR "division by zero" */
	ti15 = ti3 /* ERROR "mismatched types" */ % ti0

	ti16 = ti2 /* ERROR "mismatched types" */ & ti3
	ti17 = ti2 /* ERROR "mismatched types" */ | ti4
	ti18 = ti2 ^ ti5 // no mismatched types error because the type of ti5 is unknown

	// floating point values
	tf0 float32 = 0.
	tf1 float32 = 1.
	tf2 float64 = 4.2e1
	tf3 myfloat = 3.141592653589793238462643383279502884197169399375105820974944592307816406286
	tf4 myfloat = 1e-1

	tf5 = tf0 + tf1
	tf6 = tf1 - tf1
	tf7 = tf2 /* ERROR "mismatched types" */ * tf1
	tf8 = tf3 / tf3
	tf9 = tf3 /* ERROR "not defined" */ % tf3

	tf10 = 1 / 0 /* ERROR "division by zero" */
	tf11 = tf1 / 0 /* ERROR "division by zero" */
	tf12 = tf3 /* ERROR "mismatched types" */ / tf0

	tf16 = tf2 /* ERROR "mismatched types" */ & tf3
	tf17 = tf2 /* ERROR "mismatched types" */ | tf3
	tf18 = tf2 /* ERROR "mismatched types" */ ^ tf3

	// complex values
	tc0 = 0.i
	tc1 = 1.i
	tc2 = 4.2e1i
	tc3 = 3.141592653589793238462643383279502884197169399375105820974944592307816406286i
	tc4 = 1e-1i

	tc5 = tc0 + tc1
	tc6 = tc1 - tc1
	tc7 = tc2 * tc1
	tc8 = tc3 / tc3
	tc9 = tc3 /* ERROR "not defined" */ % tc3

	tc10 = 1 / 0 /* ERROR "division by zero" */
	tc11 = tc1 / 0 /* ERROR "division by zero" */
	tc12 = tc3 / tc0 /* ERROR "division by zero" */

	tc16 = tc2 /* ERROR "not defined" */ & tc3
	tc17 = tc2 /* ERROR "not defined" */ | tc3
	tc18 = tc2 /* ERROR "not defined" */ ^ tc3
)

// initialization cycles
const (
	a /* ERROR "initialization cycle" */ = a
	b /* ERROR "initialization cycle" */ , c /* ERROR "initialization cycle" */, d, e = e, d, c, b // TODO(gri) should only have one cycle error
	f float64 = d
)

// multiple initialization
const (
	a1, a2, a3 = 7, 3.1415926, "foo"
	b1, b2, b3 = b3, b1, 42
	c1, c2, c3  /* ERROR "missing init expr for c3" */ = 1, 2
	d1, d2, d3 = 1, 2, 3, 4 /* ERROR "extra init expr 4" */
	_p0 = assert(a1 == 7)
	_p1 = assert(a2 == 3.1415926)
	_p2 = assert(a3 == "foo")
	_p3 = assert(b1 == 42)
	_p4 = assert(b2 == 42)
	_p5 = assert(b3 == 42)
)

func _() {
	const (
		a1, a2, a3 = 7, 3.1415926, "foo"
		b1, b2, b3 = b3, b1, 42
		c1, c2, c3  /* ERROR "missing init expr for c3" */ = 1, 2
		d1, d2, d3 = 1, 2, 3, 4 /* ERROR "extra init expr 4" */
		_p0 = assert(a1 == 7)
		_p1 = assert(a2 == 3.1415926)
		_p2 = assert(a3 == "foo")
		_p3 = assert(b1 == 42)
		_p4 = assert(b2 == 42)
		_p5 = assert(b3 == 42)
	)
}

// iota
const (
	iota0 = iota
	iota1 = iota
	iota2 = iota*2
	_a0 = assert(iota0 == 0)
	_a1 = assert(iota1 == 1)
	_a2 = assert(iota2 == 4)
	iota6 = iota*3

	iota7
	iota8
	_a3 = assert(iota7 == 21)
	_a4 = assert(iota8 == 24)
)

const (
	_b0 = iota
	_b1 = assert(iota + iota2 == 5)
	_b2 = len([iota]int{}) // iota may appear in a type!
	_b3 = assert(_b2 == 2)
	_b4 = len(A{})
)

type A [iota /* ERROR "cannot use iota" */ ]int

// constant expressions with operands across different
// constant declarations must use the right iota values
const (
	_c0 = iota
	_c1
	_c2
	_x = _c2 + _d1 + _e0 // 3
)

const (
	_d0 = iota
	_d1
)

const (
	_e0 = iota
)

var _ = assert(_x == 3)

// special cases
const (
	_n0 = nil /* ERROR "not constant" */
	_n1 = [ /* ERROR "not constant" */ ]int{}
)

// iotas must not be usable in expressions outside constant declarations
type _ [iota /* ERROR "iota outside constant decl" */ ]byte
var _ = iota /* ERROR "iota outside constant decl" */
func _() {
	_ = iota /* ERROR "iota outside constant decl" */
	const _ = iota
	_ = iota /* ERROR "iota outside constant decl" */
}

func _() {
	iota := 123
	const x = iota /* ERROR "is not constant" */
	var y = iota
	_ = y
}

// iotas are usable inside closures in constant declarations (#22345)
const (
	_ = iota
	_ = len([iota]byte{})
	_ = unsafe.Sizeof(iota)
	_ = unsafe.Sizeof(func() { _ = iota })
	_ = unsafe.Sizeof(func() { var _ = iota })
	_ = unsafe.Sizeof(func() { const _ = iota })
	_ = unsafe.Sizeof(func() { type _ [iota]byte })
	_ = unsafe.Sizeof(func() { func() int { return iota }() })
)

// verify inner and outer const declarations have distinct iotas
const (
	zero = iota
	one  = iota
	_    = unsafe.Sizeof(func() {
		var x [iota]int // [2]int
		const (
			Zero = iota
			One
			Two
			_ = unsafe.Sizeof([iota-1]int{} == x) // assert types are equal
			_ = unsafe.Sizeof([Two]int{} == x)    // assert types are equal
		)
		var z [iota]int                           // [2]int
		_ = unsafe.Sizeof([2]int{} == z)          // assert types are equal
	})
	three = iota // the sequence continues
)
var _ [three]int = [3]int{} // assert 'three' has correct value

var (
	_ = iota /* ERROR "iota outside constant decl" */
	_ = unsafe.Sizeof(iota  /* ERROR "iota outside constant decl" */ )
	_ = unsafe.Sizeof(func() { _ = iota /* ERROR "iota outside constant decl" */ })
	_ = unsafe.Sizeof(func() { var _ = iota /* ERROR "iota outside constant decl" */ })
	_ = unsafe.Sizeof(func() { type _ [iota /* ERROR "iota outside constant decl" */ ]byte })
	_ = unsafe.Sizeof(func() { func() int { return iota /* ERROR "iota outside constant decl" */ }() })
)

// constant arithmetic precision and rounding must lead to expected (integer) results
var _ = []int64{
	0.0005 * 1e9,
	0.001 * 1e9,
	0.005 * 1e9,
	0.01 * 1e9,
	0.05 * 1e9,
	0.1 * 1e9,
	0.5 * 1e9,
	1 * 1e9,
	5 * 1e9,
}

const _ = unsafe.Sizeof(func() {
	const _ = 0
	_ = iota

	const (
	   zero = iota
	   one
	)
	assert(one == 1)
	assert(iota == 0)
})

// issue #52438
const i1 = iota
const i2 = iota
const i3 = iota

func _() {
	assert(i1 == 0)
	assert(i2 == 0)
	assert(i3 == 0)

	const i4 = iota
	const i5 = iota
	const i6 = iota

	assert(i4 == 0)
	assert(i5 == 0)
	assert(i6 == 0)
}

// untyped constants must not get arbitrarily large
const prec = 512 // internal maximum precision for integers
const maxInt = (1<<(prec/2) - 1) * (1<<(prec/2) + 1) // == 1<<prec - 1

const _ = maxInt + /* ERROR constant addition overflow */ 1
const _ = -maxInt - /* ERROR constant subtraction overflow */ 1
const _ = maxInt ^ /* ERROR constant bitwise XOR overflow */ -1
const _ = maxInt * /* ERROR constant multiplication overflow */ 2
const _ = maxInt << /* ERROR constant shift overflow */ 2
const _ = 1 << /* ERROR constant shift overflow */ prec

const _ = ^ /* ERROR constant bitwise complement overflow */ maxInt
