// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// conversions

package conversions

import "unsafe"

// argument count
var (
	_ = int() /* ERROR "missing argument" */
	_ = int(1, 2 /* ERROR "too many arguments" */ )
)

// numeric constant conversions are in const1.src.

func string_conversions() {
	const A = string(65)
	assert(A == "A")
	const E = string(-1)
	assert(E == "\uFFFD")
	assert(E == string(1234567890))

	type myint int
	assert(A == string(myint(65)))

	type mystring string
	const _ mystring = mystring("foo")

	const _ = string(true /* ERROR "cannot convert" */ )
	const _ = string(1.2 /* ERROR "cannot convert" */ )
	const _ = string(nil /* ERROR "cannot convert" */ )

	// issues 11357, 11353: argument must be of integer type
	_ = string(0.0 /* ERROR "cannot convert" */ )
	_ = string(0i /* ERROR "cannot convert" */ )
	_ = string(1 /* ERROR "cannot convert" */ + 2i)
}

func interface_conversions() {
	type E interface{}

	type I1 interface{
		m1()
	}

	type I2 interface{
		m1()
		m2(x int)
	}

	type I3 interface{
		m1()
		m2() int
	}

	var e E
	var i1 I1
	var i2 I2
	var i3 I3

	_ = E(0)
	_ = E(nil)
	_ = E(e)
	_ = E(i1)
	_ = E(i2)

	_ = I1(0 /* ERROR "cannot convert" */ )
	_ = I1(nil)
	_ = I1(i1)
	_ = I1(e /* ERROR "cannot convert" */ )
	_ = I1(i2)

	_ = I2(nil)
	_ = I2(i1 /* ERROR "cannot convert" */ )
	_ = I2(i2)
	_ = I2(i3 /* ERROR "cannot convert" */ )

	_ = I3(nil)
	_ = I3(i1 /* ERROR "cannot convert" */ )
	_ = I3(i2 /* ERROR "cannot convert" */ )
	_ = I3(i3)

	// TODO(gri) add more tests, improve error message
}

func issue6326() {
	type T unsafe.Pointer
	var x T
	_ = uintptr(x) // see issue 6326
}
