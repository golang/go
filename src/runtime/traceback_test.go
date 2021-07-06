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
			func() int { return testTracebackArgs1(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) },
			"testTracebackArgs1(0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, ...)",
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
func testTracebackArgs1(a, b, c, d, e, f, g, h, i, j, k, l int) int {
	n := runtime.Stack(testTracebackArgsBuf[:], false)
	if a < 0 {
		// use in-reg args to keep them alive
		return a + b + c + d + e + f + g + h + i + j + k + l
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
