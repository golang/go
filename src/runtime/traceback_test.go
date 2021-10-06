// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"runtime"
	"testing"
)

var testTracebackArgsBuf [1000]byte

func TestTracebackArgs(t *testing.T) {
	tests := []struct {
		fn     func() int
		expect string
	}{
		// simple ints
		{
			func() int { return testTracebackArgs1(1, 2, 3, 4, 5) },
			"testTracebackArgs1(0x1, 0x2, 0x3, 0x4, 0x5)",
		},
		// some aggregates
		{
			func() int {
				return testTracebackArgs2(false, struct {
					a, b, c int
					x       [2]int
				}{1, 2, 3, [2]int{4, 5}}, [0]int{}, [3]byte{6, 7, 8})
			},
			"testTracebackArgs2(0x0, {0x1, 0x2, 0x3, {0x4, 0x5}}, {}, {0x6, 0x7, 0x8})",
		},
		{
			func() int { return testTracebackArgs3([3]byte{1, 2, 3}, 4, 5, 6, [3]byte{7, 8, 9}) },
			"testTracebackArgs3({0x1, 0x2, 0x3}, 0x4, 0x5, 0x6, {0x7, 0x8, 0x9})",
		},
		// too deeply nested type
		{
			func() int { return testTracebackArgs4(false, [1][1][1][1][1][1][1][1][1][1]int{}) },
			"testTracebackArgs4(0x0, {{{{{...}}}}})",
		},
		// a lot of zero-sized type
		{
			func() int {
				z := [0]int{}
				return testTracebackArgs5(false, struct {
					x int
					y [0]int
					z [2][0]int
				}{1, z, [2][0]int{}}, z, z, z, z, z, z, z, z, z, z, z, z)
			},
			"testTracebackArgs5(0x0, {0x1, {}, {{}, {}}}, {}, {}, {}, {}, {}, ...)",
		},

		// edge cases for ...
		// no ... for 10 args
		{
			func() int { return testTracebackArgs6a(1, 2, 3, 4, 5, 6, 7, 8, 9, 10) },
			"testTracebackArgs6a(0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa)",
		},
		// has ... for 11 args
		{
			func() int { return testTracebackArgs6b(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) },
			"testTracebackArgs6b(0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, ...)",
		},
		// no ... for aggregates with 10 words
		{
			func() int { return testTracebackArgs7a([10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) },
			"testTracebackArgs7a({0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa})",
		},
		// has ... for aggregates with 11 words
		{
			func() int { return testTracebackArgs7b([11]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) },
			"testTracebackArgs7b({0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, ...})",
		},
		// no ... for aggregates, but with more args
		{
			func() int { return testTracebackArgs7c([10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 11) },
			"testTracebackArgs7c({0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa}, ...)",
		},
		// has ... for aggregates and also for more args
		{
			func() int { return testTracebackArgs7d([11]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 12) },
			"testTracebackArgs7d({0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, ...}, ...)",
		},
		// nested aggregates, no ...
		{
			func() int { return testTracebackArgs8a(testArgsType8a{1, 2, 3, 4, 5, 6, 7, 8, [2]int{9, 10}}) },
			"testTracebackArgs8a({0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, {0x9, 0xa}})",
		},
		// nested aggregates, ... in inner but not outer
		{
			func() int { return testTracebackArgs8b(testArgsType8b{1, 2, 3, 4, 5, 6, 7, 8, [3]int{9, 10, 11}}) },
			"testTracebackArgs8b({0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, {0x9, 0xa, ...}})",
		},
		// nested aggregates, ... in outer but not inner
		{
			func() int { return testTracebackArgs8c(testArgsType8c{1, 2, 3, 4, 5, 6, 7, 8, [2]int{9, 10}, 11}) },
			"testTracebackArgs8c({0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, {0x9, 0xa}, ...})",
		},
		// nested aggregates, ... in both inner and outer
		{
			func() int { return testTracebackArgs8d(testArgsType8d{1, 2, 3, 4, 5, 6, 7, 8, [3]int{9, 10, 11}, 12}) },
			"testTracebackArgs8d({0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, {0x9, 0xa, ...}, ...})",
		},
	}
	for _, test := range tests {
		n := test.fn()
		got := testTracebackArgsBuf[:n]
		if !bytes.Contains(got, []byte(test.expect)) {
			t.Errorf("traceback does not contain expected string: want %q, got\n%s", test.expect, got)
		}
	}
}

//go:noinline
func testTracebackArgs1(a, b, c, d, e int) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a < 0 {
		// use in-reg args to keep them alive
		return a + b + c + d + e
	}
	return n
}

//go:noinline
func testTracebackArgs2(a bool, b struct {
	a, b, c int
	x       [2]int
}, _ [0]int, d [3]byte) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a {
		// use in-reg args to keep them alive
		return b.a + b.b + b.c + b.x[0] + b.x[1] + int(d[0]) + int(d[1]) + int(d[2])
	}
	return n

}

//go:noinline
//go:registerparams
func testTracebackArgs3(x [3]byte, a, b, c int, y [3]byte) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a < 0 {
		// use in-reg args to keep them alive
		return int(x[0]) + int(x[1]) + int(x[2]) + a + b + c + int(y[0]) + int(y[1]) + int(y[2])
	}
	return n
}

//go:noinline
func testTracebackArgs4(a bool, x [1][1][1][1][1][1][1][1][1][1]int) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a {
		panic(x) // use args to keep them alive
	}
	return n
}

//go:noinline
func testTracebackArgs5(a bool, x struct {
	x int
	y [0]int
	z [2][0]int
}, _, _, _, _, _, _, _, _, _, _, _, _ [0]int) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a {
		panic(x) // use args to keep them alive
	}
	return n
}

//go:noinline
func testTracebackArgs6a(a, b, c, d, e, f, g, h, i, j int) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a < 0 {
		// use in-reg args to keep them alive
		return a + b + c + d + e + f + g + h + i + j
	}
	return n
}

//go:noinline
func testTracebackArgs6b(a, b, c, d, e, f, g, h, i, j, k int) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a < 0 {
		// use in-reg args to keep them alive
		return a + b + c + d + e + f + g + h + i + j + k
	}
	return n
}

//go:noinline
func testTracebackArgs7a(a [10]int) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a[0] < 0 {
		// use in-reg args to keep them alive
		return a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] + a[8] + a[9]
	}
	return n
}

//go:noinline
func testTracebackArgs7b(a [11]int) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a[0] < 0 {
		// use in-reg args to keep them alive
		return a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] + a[8] + a[9] + a[10]
	}
	return n
}

//go:noinline
func testTracebackArgs7c(a [10]int, b int) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a[0] < 0 {
		// use in-reg args to keep them alive
		return a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] + a[8] + a[9] + b
	}
	return n
}

//go:noinline
func testTracebackArgs7d(a [11]int, b int) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a[0] < 0 {
		// use in-reg args to keep them alive
		return a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] + a[8] + a[9] + a[10] + b
	}
	return n
}

type testArgsType8a struct {
	a, b, c, d, e, f, g, h int
	i                      [2]int
}
type testArgsType8b struct {
	a, b, c, d, e, f, g, h int
	i                      [3]int
}
type testArgsType8c struct {
	a, b, c, d, e, f, g, h int
	i                      [2]int
	j                      int
}
type testArgsType8d struct {
	a, b, c, d, e, f, g, h int
	i                      [3]int
	j                      int
}

//go:noinline
func testTracebackArgs8a(a testArgsType8a) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a.a < 0 {
		// use in-reg args to keep them alive
		return a.b + a.c + a.d + a.e + a.f + a.g + a.h + a.i[0] + a.i[1]
	}
	return n
}

//go:noinline
func testTracebackArgs8b(a testArgsType8b) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a.a < 0 {
		// use in-reg args to keep them alive
		return a.b + a.c + a.d + a.e + a.f + a.g + a.h + a.i[0] + a.i[1] + a.i[2]
	}
	return n
}

//go:noinline
func testTracebackArgs8c(a testArgsType8c) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a.a < 0 {
		// use in-reg args to keep them alive
		return a.b + a.c + a.d + a.e + a.f + a.g + a.h + a.i[0] + a.i[1] + a.j
	}
	return n
}

//go:noinline
func testTracebackArgs8d(a testArgsType8d) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a.a < 0 {
		// use in-reg args to keep them alive
		return a.b + a.c + a.d + a.e + a.f + a.g + a.h + a.i[0] + a.i[1] + a.i[2] + a.j
	}
	return n
}
