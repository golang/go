// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ensure that typed non-integer, negative and too large
// values are not accepted as size argument in make for
// maps.

package main

type T map[int]int

var sink T

func main() {
	sink = make(T, -1)            // ERROR "negative size argument in make.*"
	sink = make(T, uint64(1<<63)) // ERROR "size argument too large in make.*"

	sink = make(T, 0.5) // ERROR "constant 0.5 truncated to integer"
	sink = make(T, 1.0)
	sink = make(T, float32(1.0)) // ERROR "non-integer size argument in make.*"
	sink = make(T, float64(1.0)) // ERROR "non-integer size argument in make.*"
	sink = make(T, 1.0)
	sink = make(T, float32(1.0)) // ERROR "non-integer size argument in make.*"
	sink = make(T, float64(1.0)) // ERROR "non-integer size argument in make.*"
	sink = make(T, 1+0i)
	sink = make(T, complex64(1+0i))  // ERROR "non-integer size argument in make.*"
	sink = make(T, complex128(1+0i)) // ERROR "non-integer size argument in make.*"
	sink = make(T, 1+0i)
	sink = make(T, complex64(1+0i))  // ERROR "non-integer size argument in make.*"
	sink = make(T, complex128(1+0i)) // ERROR "non-integer size argument in make.*"
}
