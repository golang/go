// errorcheck -0 -m

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

const globalConstSize = 128

var globalVarSize = 128

//go:noinline
func testSlices() {
	{
		size := 128
		_ = make([]byte, size) // ERROR "make\(\[\]byte, 128\) does not escape"
	}

	{
		s := 128
		size := s
		_ = make([]byte, size) // ERROR "make\(\[\]byte, 128\) does not escape"
	}

	{
		size := 128
		_ = make([]byte, size) // ERROR "make\(\[\]byte, 128\) does not escape"
	}

	{
		s := 128
		size := s
		_ = make([]byte, size) // ERROR "make\(\[\]byte, 128\) does not escape"
	}

	{
		s1 := 128
		s2 := 256
		_ = make([]byte, s2, s1) // ERROR "make\(\[\]byte, s2, 128\) does not escape"
	}

	allocLen(256) // ERROR "make\(\[\]byte, 256\) does not escape" "inlining call"
	allocCap(256) // ERROR "make\(\[\]byte, 0, 256\) does not escape" "inlining call"
	_ = newT(256) // ERROR "make\(\[\]byte, 256\) does not escape" "inlining call"

	{
		size := globalConstSize
		_ = make([]byte, size) // ERROR "make\(\[\]byte, 128\) does not escape"
	}

	allocLen(globalConstSize) // ERROR "make\(\[\]byte, 128\) does not escape" "inlining call"
	allocCap(globalConstSize) // ERROR "make\(\[\]byte, 0, 128\) does not escape" "inlining call"
	_ = newT(globalConstSize) // ERROR "make\(\[\]byte, 128\) does not escape" "inlining call"

	{
		c := 128
		s := 256
		_ = make([]byte, s, c) // ERROR "make\(\[\]byte, s, 128\) does not escape"
	}

	{
		s := 256
		_ = make([]byte, s, globalConstSize) // ERROR "make\(\[\]byte, s, 128\) does not escape"
	}

	{
		_ = make([]byte, globalVarSize)                  // ERROR "make\(\[\]byte, globalVarSize\) does not escape"
		_ = make([]byte, globalVarSize, globalConstSize) // ERROR "make\(\[\]byte, globalVarSize, 128\) does not escape"
	}
}

func allocLen(l int) []byte { // ERROR "can inline"
	return make([]byte, l) // ERROR "escapes to heap"
}

func allocCap(l int) []byte { // ERROR "can inline"
	return make([]byte, 0, l) // ERROR "escapes to heap"
}

type t struct {
	s []byte
}

func newT(l int) t { // ERROR "can inline"
	return t{make([]byte, l)} // ERROR "make.*escapes to heap"
}

//go:noinline
func testMaps() {
	size := 128
	_ = make(map[string]int, size) // ERROR "does not escape"

	_ = allocMapLen(128) // ERROR "does not escape" "inlining call"
	_ = newM(128)        // ERROR "does not escape" "inlining call"
}

func allocMapLen(l int) map[string]int { // ERROR "can inline"
	return make(map[string]int, l) // ERROR "escapes to heap"
}

type m struct {
	m map[string]int
}

func newM(l int) m { // ERROR "can inline"
	return m{make(map[string]int, l)} // ERROR "make.*escapes to heap"
}

//go:noinline
func testLenOfSliceLit() {
	ints := []int{0, 1, 2, 3, 4, 5} // ERROR "\[\]int\{\.\.\.\} does not escape"'
	_ = make([]int, len(ints))      // ERROR "make\(\[\]int, 6\) does not escape"
	_ = allocLenOf(ints)            // ERROR "inlining call", "make\(\[\]int, 6\) does not escape"

	_ = make([]int, 2, len(ints))  // ERROR "make\(\[\]int, 2, 6\) does not escape"
	_ = make([]int, len(ints), 2)  // ERROR "make\(\[\]int, len\(ints\), 2\) does not escape"
	_ = make([]int, 10, len(ints)) // ERROR "make\(\[\]int, 10, 6\) does not escape"
}

func allocLenOf(s []int) []int { // ERROR "can inline" "s does not escape"
	return make([]int, len(s)) // ERROR "escapes to heap"
}
