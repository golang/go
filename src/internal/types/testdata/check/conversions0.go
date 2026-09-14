// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// conversions

package conversions

import "unsafe"

// argument count
var (
	_ = int() /* ERROR "missing argument" */
	_ = int(1, 2 /* ERROR "too many arguments" */)
)

// numeric constant conversions are in const1.src.

func string_conversions() {
	const _ = string(65 /* ERROR "argument must be untyped rune constant or have type byte or rune with go1.28 or later" */)
	const _ = string(- /* ERROR "argument must be untyped rune constant or have type byte or rune with go1.28 or later" */ 1)
	const _ = string(1234567890 /* ERROR "argument must be untyped rune constant or have type byte or rune with go1.28 or later" */)

	type myint int
	const _ = string(myint /* ERROR "argument must be untyped rune constant or have type byte or rune with go1.28 or later" */ (65))

	var i int
	_ = string(i /* ERROR "argument must have type byte or rune with go1.28 or later" */)

	var b byte
	_ = string(b)

	var r rune
	_ = string(r)

	const _ = string('a')
	const _ = string('\xf8')
	const _ = string(byte(65))
	const _ = string(rune(65))
	const _ = string(uint8(65)) // uint8 and byte are aliases
	const _ = string(int32(65)) // int32 and rune are aliases

	type mystring string
	const _ mystring = mystring("foo")

	const _ = string(true /* ERROR "cannot convert" */)
	const _ = string(1.2 /* ERROR "cannot convert" */)
	const _ = string(nil /* ERROR "cannot convert" */)

	// issues 11357, 11353: argument must be of integer type
	_ = string(0.0 /* ERROR "cannot convert" */)
	_ = string(0i /* ERROR "cannot convert" */)
	_ = string(1 /* ERROR "cannot convert" */ + 2i)
}

func interface_conversions() {
	type E interface{}

	type I1 interface {
		m1()
	}

	type I2 interface {
		m1()
		m2(x int)
	}

	type I3 interface {
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

	_ = I1(0 /* ERROR "cannot convert" */)
	_ = I1(nil)
	_ = I1(i1)
	_ = I1(e /* ERROR "cannot convert" */)
	_ = I1(i2)

	_ = I2(nil)
	_ = I2(i1 /* ERROR "cannot convert" */)
	_ = I2(i2)
	_ = I2(i3 /* ERROR "cannot convert" */)

	_ = I3(nil)
	_ = I3(i1 /* ERROR "cannot convert" */)
	_ = I3(i2 /* ERROR "cannot convert" */)
	_ = I3(i3)

	// TODO(gri) add more tests, improve error message
}

func issue6326() {
	type T unsafe.Pointer
	var x T
	_ = uintptr(x) // see issue 6326
}
