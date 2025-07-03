// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test to make sure that we don't try using larger loads for
// generated equality functions on architectures that can't do
// unaligned loads.

package main

// T has a big field that wants to be compared with larger loads/stores.
// T is "special" because of the unnamed field, so it needs a generated equality function.
// T is an odd number of bytes in size and has alignment 1.
type T struct {
	src [8]byte
	_   byte
}

// U contains 8 copies of T, each at a different %8 alignment.
type U [8]T

//go:noinline
func f(x, y *U) bool {
	return *x == *y
}

func main() {
	var a U
	_ = f(&a, &a)
}
