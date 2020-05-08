// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check closure in const declaration group can be compiled
// and set correct value

package main

import "unsafe"

const (
	x = unsafe.Sizeof(func() {})
	y
)

func main() {
	const (
		z = unsafe.Sizeof(func() {})
		t
	)

	// x and y must be equal
	println(x == y)
	// size must be greater than zero
	println(y > 0)

	// Same logic as x, y above
	println(z == t)
	println(t > 0)
}
