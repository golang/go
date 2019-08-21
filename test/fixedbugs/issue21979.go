// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() {
	// old error: "(type untyped string)"
	_ = bool("") // ERROR "cannot convert .. \(type string\) to type bool"

	// old error: "(type untyped number)"
	_ = bool(1) // ERROR "cannot convert 1 \(type int\) to type bool"

	// old error: "(type untyped number)"
	_ = bool(1.0) // ERROR "cannot convert 1 \(type float64\) to type bool"

	// old error: "(type untyped number)"
	_ = bool(-4 + 2i) // ERROR "cannot convert -4 \+ 2i \(type complex128\) to type bool"

	// old error: "(type untyped bool)"
	_ = string(true) // ERROR "cannot convert true \(type bool\) to type string"

	_ = string(-1)

	// old error: "(type untyped number)"
	_ = string(1.0) // ERROR "cannot convert 1 \(type float64\) to type string"

	// old error: "(type untyped number)"
	_ = string(-4 + 2i) // ERROR "cannot convert -4 \+ 2i \(type complex128\) to type string"

	// old error: "(type untyped string)"
	_ = int("") // ERROR "cannot convert .. \(type string\) to type int"

	// old error: "(type untyped bool)"
	_ = int(true) // ERROR "cannot convert true \(type bool\) to type int"

	_ = int(-1)
	_ = int(1)
	_ = int(1.0)
	_ = int(-4 + 2i) // ERROR "truncated to integer"

	// old error: "(type untyped string)"
	_ = uint("") // ERROR "cannot convert .. \(type string\) to type uint"

	// old error: "(type untyped bool)"
	_ = uint(true) // ERROR "cannot convert true \(type bool\) to type uint"

	_ = uint(-1) // ERROR "constant -1 overflows uint"
	_ = uint(1)
	_ = uint(1.0)
	_ = uint(-4 + 2i) // ERROR "constant -4 overflows uint" "truncated to integer"

	// old error: "(type untyped string)"
	_ = float64("") // ERROR "cannot convert .. \(type string\) to type float64"

	// old error: "(type untyped bool)"
	_ = float64(true) // ERROR "cannot convert true \(type bool\) to type float64"

	_ = float64(-1)
	_ = float64(1)
	_ = float64(1.0)
	_ = float64(-4 + 2i) // ERROR "truncated to real"

	// old error: "(type untyped string)"
	_ = complex128("") // ERROR "cannot convert .. \(type string\) to type complex128"

	// old error: "(type untyped bool)"
	_ = complex128(true) // ERROR "cannot convert true \(type bool\) to type complex128"

	_ = complex128(-1)
	_ = complex128(1)
	_ = complex128(1.0)
}
