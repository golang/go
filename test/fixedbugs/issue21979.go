// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() {
	_ = bool("")      // ERROR "cannot convert .. \(.*untyped string.*\) to type bool|invalid type conversion"
	_ = bool(1)       // ERROR "cannot convert 1 \(.*untyped int.*\) to type bool|invalid type conversion"
	_ = bool(1.0)     // ERROR "cannot convert 1.* \(.*untyped float.*\) to type bool|invalid type conversion"
	_ = bool(-4 + 2i) // ERROR "cannot convert -4 \+ 2i \(.*untyped complex.*\) to type bool|invalid type conversion"

	_ = string(true) // ERROR "cannot convert true \(.*untyped bool.*\) to type string|invalid type conversion"
	_ = string(-1)
	_ = string(1.0)     // ERROR "cannot convert 1.* \(.*untyped float.*\) to type string|invalid type conversion"
	_ = string(-4 + 2i) // ERROR "cannot convert -4 \+ 2i \(.*untyped complex.*\) to type string|invalid type conversion"

	_ = int("")   // ERROR "cannot convert .. \(.*untyped string.*\) to type int|invalid type conversion"
	_ = int(true) // ERROR "cannot convert true \(.*untyped bool.*\) to type int|invalid type conversion"
	_ = int(-1)
	_ = int(1)
	_ = int(1.0)
	_ = int(-4 + 2i) // ERROR "truncated to integer|cannot convert -4 \+ 2i \(.*untyped complex.*\) to type int"

	_ = uint("")   // ERROR "cannot convert .. \(.*untyped string.*\) to type uint|invalid type conversion"
	_ = uint(true) // ERROR "cannot convert true \(.*untyped bool.*\) to type uint|invalid type conversion"
	_ = uint(-1)   // ERROR "constant -1 overflows uint|integer constant overflow|cannot convert -1 \(untyped int constant\) to type uint"
	_ = uint(1)
	_ = uint(1.0)
	// types1 reports extra error "truncated to integer"
	_ = uint(-4 + 2i) // ERROR "constant -4 overflows uint|truncated to integer|cannot convert -4 \+ 2i \(untyped complex constant.*\) to type uint"

	_ = float64("")   // ERROR "cannot convert .. \(.*untyped string.*\) to type float64|invalid type conversion"
	_ = float64(true) // ERROR "cannot convert true \(.*untyped bool.*\) to type float64|invalid type conversion"
	_ = float64(-1)
	_ = float64(1)
	_ = float64(1.0)
	_ = float64(-4 + 2i) // ERROR "truncated to|cannot convert -4 \+ 2i \(.*untyped complex.*\) to type float64"

	_ = complex128("")   // ERROR "cannot convert .. \(.*untyped string.*\) to type complex128|invalid type conversion"
	_ = complex128(true) // ERROR "cannot convert true \(.*untyped bool.*\) to type complex128|invalid type conversion"
	_ = complex128(-1)
	_ = complex128(1)
	_ = complex128(1.0)
}
