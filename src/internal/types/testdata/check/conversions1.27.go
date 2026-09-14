// -lang=go1.27

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a replica of the string conversions in conversions0.go at Go 1.27
// to ensure that integer-to-string conversions are accepted without error in Go 1.27.

package conversions

func string_conversions() {
	const A = string(65)
	assert(A == "A")
	const E = string(-1)
	assert(E == "\uFFFD")
	assert(E == string(1234567890))

	type myint int
	assert(A == string(myint(65)))

	var i int
	_ = string(i)

	var b byte
	_ = string(b)

	var r rune
	_ = string(r)

	const _ = string('a')
	const _ = string('\xf8')
	const _ = string(byte(65))
	const _ = string(rune(65))

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
