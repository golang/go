// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Comparing a wrapped product difference against zero must respect the
// sign of the wrapped value, even when it wraps to MinInt.

package main

import "fmt"

// int64 (not int) so the cross product is 64-bit on every GOARCH.
type point struct{ x, y int64 }

//go:noinline
func sign(p1, p2, p3 point) bool {
	return (p1.x-p3.x)*(p2.y-p3.y)-(p2.x-p3.x)*(p1.y-p3.y) < 0
}

type point32 struct{ x, y int32 }

//go:noinline
func sign32(p1, p2, p3 point32) bool {
	if (p1.x-p3.x)*(p2.y-p3.y)-(p2.x-p3.x)*(p1.y-p3.y) < 0 {
		return true
	}
	return false
}

const minInt64 = -1 << 63

//go:noinline
func msub64(x, y int64) bool {
	if minInt64-x*y < 0 {
		return true
	}
	return false
}

const minInt32 = -1 << 31

//go:noinline
func msub32(x, y int32) bool {
	if minInt32-x*y < 0 {
		return true
	}
	return false
}

const minUint32 = uint32(1 << 31)

//go:noinline
func umod32(y uint32) bool {
	if int32(minUint32%y) < 0 {
		return true
	}
	return false
}

func main() {
	var bad bool
	check := func(name string, got, want bool) {
		if got != want {
			fmt.Printf("%s = %v, want %v\n", name, got, want)
			bad = true
		}
	}

	check("sign", sign(point{0, 2}, point{1 << 62, 0}, point{0, 0}), true)           // wraps to MinInt64
	check("sign32", sign32(point32{0, 2}, point32{1 << 30, 0}, point32{0, 0}), true) // wraps to MinInt32
	check("msub64", msub64(0, 123), true)                                            // minInt64 - 0
	check("msub32", msub32(0, 123), true)                                            // minInt32 - 0
	check("umod32", umod32(0xffffffff), true)                                        // minUint32 % big == minUint32

	if bad {
		panic("InvertFlags noov MinInt miscompilation")
	}
}
