// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() {
	_ = bool("")      // ERROR "cannot convert .. \(type untyped string\) to type bool"
	_ = bool(1)       // ERROR "cannot convert 1 \(type untyped int\) to type bool"
	_ = bool(1.0)     // ERROR "cannot convert 1 \(type untyped float\) to type bool"
	_ = bool(-4 + 2i) // ERROR "cannot convert -4 \+ 2i \(type untyped complex\) to type bool"

	_ = string(true) // ERROR "cannot convert true \(type untyped bool\) to type string"
	_ = string(-1)
	_ = string(1.0)     // ERROR "cannot convert 1 \(type untyped float\) to type string"
	_ = string(-4 + 2i) // ERROR "cannot convert -4 \+ 2i \(type untyped complex\) to type string"

	_ = int("")   // ERROR "cannot convert .. \(type untyped string\) to type int"
	_ = int(true) // ERROR "cannot convert true \(type untyped bool\) to type int"
	_ = int(-1)
	_ = int(1)
	_ = int(1.0)
	_ = int(-4 + 2i) // ERROR "truncated to integer"

	_ = uint("")   // ERROR "cannot convert .. \(type untyped string\) to type uint"
	_ = uint(true) // ERROR "cannot convert true \(type untyped bool\) to type uint"
	_ = uint(-1)   // ERROR "constant -1 overflows uint"
	_ = uint(1)
	_ = uint(1.0)
	_ = uint(-4 + 2i) // ERROR "constant -4 overflows uint" "truncated to integer"

	_ = float64("")   // ERROR "cannot convert .. \(type untyped string\) to type float64"
	_ = float64(true) // ERROR "cannot convert true \(type untyped bool\) to type float64"
	_ = float64(-1)
	_ = float64(1)
	_ = float64(1.0)
	_ = float64(-4 + 2i) // ERROR "truncated to real"

	_ = complex128("")   // ERROR "cannot convert .. \(type untyped string\) to type complex128"
	_ = complex128(true) // ERROR "cannot convert true \(type untyped bool\) to type complex128"
	_ = complex128(-1)
	_ = complex128(1)
	_ = complex128(1.0)
}
