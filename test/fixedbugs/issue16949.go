// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ensure that typed non-integer len and cap make arguments are not accepted.

package main

var sink []byte

func main() {
	sink = make([]byte, 1.0)
	sink = make([]byte, float32(1.0)) // ERROR "non-integer.*len|must be integer"
	sink = make([]byte, float64(1.0)) // ERROR "non-integer.*len|must be integer"

	sink = make([]byte, 0, 1.0)
	sink = make([]byte, 0, float32(1.0)) // ERROR "non-integer.*cap|must be integer"
	sink = make([]byte, 0, float64(1.0)) // ERROR "non-integer.*cap|must be integer"

	sink = make([]byte, 1+0i)
	sink = make([]byte, complex64(1+0i))  // ERROR "non-integer.*len|must be integer"
	sink = make([]byte, complex128(1+0i)) // ERROR "non-integer.*len|must be integer"

	sink = make([]byte, 0, 1+0i)
	sink = make([]byte, 0, complex64(1+0i))  // ERROR "non-integer.*cap|must be integer"
	sink = make([]byte, 0, complex128(1+0i)) // ERROR "non-integer.*cap|must be integer"

}
