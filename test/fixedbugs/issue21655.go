// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure assembly offsets don't get too large.

// To trigger issue21655, the index offset needs to be small
// enough to fit into an int32 (to get rewritten to an ADDQconst)
// but large enough to overflow an int32 after multiplying by the stride.

package main

func f1(a []int64, i int64) int64 {
	return a[i+1<<30]
}
func f2(a []int32, i int64) int32 {
	return a[i+1<<30]
}
func f3(a []int16, i int64) int16 {
	return a[i+1<<30]
}
func f4(a []int8, i int64) int8 {
	return a[i+1<<31]
}
func f5(a []float64, i int64) float64 {
	return a[i+1<<30]
}
func f6(a []float32, i int64) float32 {
	return a[i+1<<30]
}

// Note: Before the fix for issue 21655, f{1,2,5,6} made
// the compiler crash. f3 silently generated the wrong
// code, using an offset of -1<<31 instead of 1<<31.
// (This is due to the assembler accepting offsets
// like 0x80000000 and silently using them as
// signed 32 bit offsets.)
// f4 was ok, but testing it can't hurt.

func f7(ss []*string, i int) string {
	const offset = 3 << 29 // 3<<29 * 4 = 3<<31 = 1<<31 mod 1<<32.
	if i > offset {
		return *ss[i-offset]
	}
	return ""
}
func f8(ss []*string, i int) string {
	const offset = 3<<29 + 10
	if i > offset {
		return *ss[i-offset]
	}
	return ""
}
func f9(ss []*string, i int) string {
	const offset = 3<<29 - 10
	if i > offset {
		return *ss[i-offset]
	}
	return ""
}
