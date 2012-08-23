// run

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 3835: 8g tries to optimize arithmetic involving integer
// constants, but can run out of registers in the process.

package main

var a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, A, B, C, D, E, F, G int

func foo() int {
	return a + 1 + b + 2 + c + 3 + d + 4 + e + 5 + f + 6 + g + 7 + h + 8 + i + 9 + j + 10 +
		k + 1 + l + 2 + m + 3 + n + 4 + o + 5 + p + 6 + q + 7 + r + 8 + s + 9 + t + 10 +
		u + 1 + v + 2 + w + 3 + x + 4 + y + 5 + z + 6 + A + 7 + B + 8 + C + 9 + D + 10 +
		E + 1 + F + 2 + G + 3
}

func bar() int8 {
	var (
		W int16
		X int32
		Y int32
		Z int32
	)
	return int8(W+int16(X+3)+3) * int8(Y+3+Z*3)
}

func main() {
	if foo() == 0 {
		panic("foo")
	}
	if bar() == 0 {
		panic("bar")
	}
}
