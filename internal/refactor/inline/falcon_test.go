// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline_test

import "testing"

// Testcases mostly come in pairs, of a success and a failure
// to substitute based on specific constant argument values.

func TestFalconStringIndex(t *testing.T) {
	runTests(t, []testcase{
		{
			"Non-negative string index.",
			`func f(i int) byte { return s[i] }; var s string`,
			`func _() { f(0) }`,
			`func _() { _ = s[0] }`,
		},
		{
			"Negative string index.",
			`func f(i int) byte { return s[i] }; var s string`,
			`func _() { f(-1) }`,
			`func _() {
	var i int = -1
	_ = s[i]
}`,
		},
		{
			"String index in range.",
			`func f(s string, i int) byte { return s[i] }`,
			`func _() { f("-", 0) }`,
			`func _() { _ = "-"[0] }`,
		},
		{
			"String index out of range.",
			`func f(s string, i int) byte { return s[i] }`,
			`func _() { f("-", 1) }`,
			`func _() {
	var (
		s string = "-"
		i int    = 1
	)
	_ = s[i]
}`,
		},
		{
			"Remove known prefix (OK)",
			`func f(s, prefix string) string { return s[:len(prefix)] }`,
			`func _() { f("", "") }`,
			`func _() { _ = ""[:len("")] }`,
		},
		{
			"Remove not-a-prefix (out of range)",
			`func f(s, prefix string) string { return s[:len(prefix)] }`,
			`func _() { f("", "pre") }`,
			`func _() {
	var s, prefix string = "", "pre"
	_ = s[:len(prefix)]
}`,
		},
	})
}

func TestFalconSliceIndices(t *testing.T) {
	runTests(t, []testcase{
		{
			"Monotonic (0<=i<=j) slice indices (len unknown).",
			`func f(i, j int) []int { return s[i:j] }; var s []int`,
			`func _() { f(0, 1) }`,
			`func _() { _ = s[0:1] }`,
		},
		{
			"Non-monotonic slice indices (len unknown).",
			`func f(i, j int) []int { return s[i:j] }; var s []int`,
			`func _() { f(1, 0) }`,
			`func _() {
	var i, j int = 1, 0
	_ = s[i:j]
}`,
		},
		{
			"Negative slice index.",
			`func f(i, j int) []int { return s[i:j] }; var s []int`,
			`func _() { f(-1, 1) }`,
			`func _() {
	var i, j int = -1, 1
	_ = s[i:j]
}`,
		},
	})
}

func TestFalconMapKeys(t *testing.T) {
	runTests(t, []testcase{
		{
			"Unique map keys (int)",
			`func f(x int) { _ = map[int]bool{1: true, x: true} }`,
			`func _() { f(2) }`,
			`func _() { _ = map[int]bool{1: true, 2: true} }`,
		},
		{
			"Duplicate map keys (int)",
			`func f(x int) { _ = map[int]bool{1: true, x: true} }`,
			`func _() { f(1) }`,
			`func _() {
	var x int = 1
	_ = map[int]bool{1: true, x: true}
}`,
		},
		{
			"Unique map keys (varied built-in types)",
			`func f(x int16) { _ = map[any]bool{1: true, x: true} }`,
			`func _() { f(2) }`,
			`func _() { _ = map[any]bool{1: true, int16(2): true} }`,
		},
		{
			"Duplicate map keys (varied built-in types)",
			`func f(x int16) { _ = map[any]bool{1: true, x: true} }`,
			`func _() { f(1) }`,
			`func _() { _ = map[any]bool{1: true, int16(1): true} }`,
		},
		{
			"Unique map keys (varied user-defined types)",
			`func f(x myint) { _ = map[any]bool{1: true, x: true} }; type myint int`,
			`func _() { f(2) }`,
			`func _() { _ = map[any]bool{1: true, myint(2): true} }`,
		},
		{
			"Duplicate map keys (varied user-defined types)",
			`func f(x myint, y myint2) { _ = map[any]bool{x: true, y: true} }; type (myint int; myint2 int)`,
			`func _() { f(1, 1) }`,
			`func _() {
	var (
		x myint  = 1
		y myint2 = 1
	)
	_ = map[any]bool{x: true, y: true}
}`,
		},
		{
			"Duplicate map keys (user-defined alias to built-in)",
			`func f(x myint, y int) { _ = map[any]bool{x: true, y: true} }; type myint = int`,
			`func _() { f(1, 1) }`,
			`func _() {
	var (
		x myint = 1
		y int   = 1
	)
	_ = map[any]bool{x: true, y: true}
}`,
		},
	})
}

func TestFalconSwitchCases(t *testing.T) {
	runTests(t, []testcase{
		{
			"Unique switch cases (int).",
			`func f(x int) { switch 0 { case x: case 1: } }`,
			`func _() { f(2) }`,
			`func _() {
	switch 0 {
	case 2:
	case 1:
	}
}`,
		},
		{
			"Duplicate switch cases (int).",
			`func f(x int) { switch 0 { case x: case 1: } }`,
			`func _() { f(1) }`,
			`func _() {
	var x int = 1
	switch 0 {
	case x:
	case 1:
	}
}`,
		},
		{
			"Unique switch cases (varied built-in types).",
			`func f(x int) { switch any(nil) { case x: case int16(1): } }`,
			`func _() { f(2) }`,
			`func _() {
	switch any(nil) {
	case 2:
	case int16(1):
	}
}`,
		},
		{
			"Duplicate switch cases (varied built-in types).",
			`func f(x int) { switch any(nil) { case x: case int16(1): } }`,
			`func _() { f(1) }`,
			`func _() {
	switch any(nil) {
	case 1:
	case int16(1):
	}
}`,
		},
	})
}

func TestFalconDivision(t *testing.T) {
	runTests(t, []testcase{
		{
			"Division by two.",
			`func f(x, y int) int { return x / y }`,
			`func _() { f(1, 2) }`,
			`func _() { _ = 1 / 2 }`,
		},
		{
			"Division by zero.",
			`func f(x, y int) int { return x / y }`,
			`func _() { f(1, 0) }`,
			`func _() {
	var x, y int = 1, 0
	_ = x / y
}`,
		},
		{
			"Division by two (statement).",
			`func f(x, y int) { x /= y }`,
			`func _() { f(1, 2) }`,
			`func _() {
	var x int = 1
	x /= 2
}`,
		},
		{
			"Division by zero (statement).",
			`func f(x, y int) { x /= y }`,
			`func _() { f(1, 0) }`,
			`func _() {
	var x, y int = 1, 0
	x /= y
}`,
		},
		{
			"Division of minint by two (ok).",
			`func f(x, y int32) { _ = x / y }`,
			`func _() { f(-0x80000000, 2) }`,
			`func _() { _ = int32(-0x80000000) / int32(2) }`,
		},
		{
			"Division of minint by -1 (overflow).",
			`func f(x, y int32) { _ = x / y }`,
			`func _() { f(-0x80000000, -1) }`,
			`func _() {
	var x, y int32 = -0x80000000, -1
	_ = x / y
}`,
		},
	})
}

func TestFalconMinusMinInt(t *testing.T) {
	runTests(t, []testcase{
		{
			"Negation of maxint.",
			`func f(x int32) int32 { return -x }`,
			`func _() { f(0x7fffffff) }`,
			`func _() { _ = -int32(0x7fffffff) }`,
		},
		{
			"Negation of minint.",
			`func f(x int32) int32 { return -x }`,
			`func _() { f(-0x80000000) }`,
			`func _() {
	var x int32 = -0x80000000
	_ = -x
}`,
		},
	})
}

func TestFalconArithmeticOverflow(t *testing.T) {
	runTests(t, []testcase{
		{
			"Addition without overflow.",
			`func f(x, y int32) int32 { return x + y }`,
			`func _() { f(100, 200) }`,
			`func _() { _ = int32(100) + int32(200) }`,
		},
		{
			"Addition with overflow.",
			`func f(x, y int32) int32 { return x + y }`,
			`func _() { f(1<<30, 1<<30) }`,
			`func _() {
	var x, y int32 = 1 << 30, 1 << 30
	_ = x + y
}`,
		},
		{
			"Conversion in range.",
			`func f(x int) int8 { return int8(x) }`,
			`func _() { f(123) }`,
			`func _() { _ = int8(123) }`,
		},
		{
			"Conversion out of range.",
			`func f(x int) int8 { return int8(x) }`,
			`func _() { f(456) }`,
			`func _() {
	var x int = 456
	_ = int8(x)
}`,
		},
	})
}

func TestFalconComplex(t *testing.T) {
	runTests(t, []testcase{
		{
			"Complex arithmetic (good).",
			`func f(re, im float64, z complex128) byte { return "x"[int(real(complex(re, im)*complex(re, -im)-z))] }`,
			`func _() { f(1, 2, 5+0i) }`,
			`func _() { _ = "x"[int(real(complex(float64(1), float64(2))*complex(float64(1), -float64(2))-(5+0i)))] }`,
		},
		{
			"Complex arithmetic (bad).",
			`func f(re, im float64, z complex128) byte { return "x"[int(real(complex(re, im)*complex(re, -im)-z))] }`,
			`func _() { f(1, 3, 5+0i) }`,
			`func _() {
	var (
		re, im float64    = 1, 3
		z      complex128 = 5 + 0i
	)
	_ = "x"[int(real(complex(re, im)*complex(re, -im)-z))]
}`,
		},
	})
}
func TestFalconMisc(t *testing.T) {
	runTests(t, []testcase{
		{
			"Compound constant expression (good).",
			`func f(x, y string, i, j int) byte { return x[i*len(y)+j] }`,
			`func _() { f("abc", "xy", 2, -3) }`,
			`func _() { _ = "abc"[2*len("xy")+-3] }`,
		},
		{
			"Compound constant expression (index out of range).",
			`func f(x, y string, i, j int) byte { return x[i*len(y)+j] }`,
			`func _() { f("abc", "xy", 4, -3) }`,
			`func _() {
	var (
		x, y string = "abc", "xy"
		i, j int    = 4, -3
	)
	_ = x[i*len(y)+j]
}`,
		},
		{
			"Constraints within nested functions (good).",
			`func f(x int) { _ = func() { _ = [1]int{}[x] } }`,
			`func _() { f(0) }`,
			`func _() { _ = func() { _ = [1]int{}[0] } }`,
		},
		{
			"Constraints within nested functions (bad).",
			`func f(x int) { _ = func() { _ = [1]int{}[x] } }`,
			`func _() { f(1) }`,
			`func _() {
	var x int = 1
	_ = func() { _ = [1]int{}[x] }
}`,
		},
		{
			"Falcon violation rejects only the constant arguments (x, z).",
			`func f(x, y, z string) string { return x[:2] + y + z[:2] }; var b string`,
			`func _() { f("a", b, "c") }`,
			`func _() {
	var x, z string = "a", "c"
	_ = x[:2] + b + z[:2]
}`,
		},
	})
}
