// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !wasm && !386 && !arm && !mips

// TODO fix this to work for wasm and 32-bit architectures.
// Doing more than this, however, expands the change.

package main

import (
	"fmt"
	"runtime"
)

// This test checks that conversion from floats to (unsigned) 32 and 64-bit
// integers has the same sensible behavior for corner cases, and that the
// conversions to smaller integers agree.  Because outliers are platform-
// independent, the "golden test" for smaller integers is more like of
// a "gold-ish test" and subject to change.

//go:noinline
func id[T any](x T) T {
	return x
}

//go:noinline
func want[T comparable](name string, x, y T) {
	if x != y {
		_, _, line, _ := runtime.Caller(1)
		fmt.Println("FAIL at line", line, "var =", name, "got =", x, "want =", y)
	}
}

//go:noinline
func log[T comparable](name string, x T) {
	fmt.Println(name, x)
}

const (
	// pX = max positive signed X bit
	// nX = min negative signed X bit
	// uX = max unsigned X bit
	// tX = two to the X
	p32 = 2147483647
	n32 = -2147483648
	u32 = 4294967295
	p64 = 9223372036854775807
	n64 = -9223372036854775808
	u64 = 18446744073709551615
	t44 = 1 << 44
)

func main() {
	one := 1.0
	minus1_32 := id(float32(-1.0))
	minus1_64 := id(float64(-1.0))
	p32_plus4k_plus1 := id(float32(p32 + 4096 + 1)) // want this to be precise and fit in 24 bits mantissa
	p64_plus4k_plus1 := id(float64(p64 + 4096 + 1)) // want this to be precise and fit in 53 bits mantissa
	n32_minus4k := id(float32(n32 - 4096))
	n64_minus4k := id(float64(n64 - 4096))
	inf_32 := id(float32(one / 0))
	inf_64 := id(float64(one / 0))
	ninf_32 := id(float32(-one / 0))
	ninf_64 := id(float64(-one / 0))

	// int32 conversions
	int32Tests := []struct {
		name     string
		input    any // Use any to handle both float32 and float64
		expected int32
	}{
		{"minus1_32", minus1_32, -1},
		{"minus1_64", minus1_64, -1},
		{"p32_plus4k_plus1", p32_plus4k_plus1, p32},
		{"p64_plus4k_plus1", p64_plus4k_plus1, p32},
		{"n32_minus4k", n32_minus4k, n32},
		{"n64_minus4k", n64_minus4k, n32},
		{"inf_32", inf_32, p32},
		{"inf_64", inf_64, p32},
		{"ninf_32", ninf_32, n32},
		{"ninf_64", ninf_64, n32},
	}

	for _, test := range int32Tests {
		var converted int32
		switch v := test.input.(type) {
		case float32:
			converted = int32(v)
		case float64:
			converted = int32(v)
		}
		want(test.name, converted, test.expected)
	}

	// int64 conversions
	int64Tests := []struct {
		name     string
		input    any
		expected int64
	}{
		{"minus1_32", minus1_32, -1},
		{"minus1_64", minus1_64, -1},
		{"p32_plus4k_plus1", p32_plus4k_plus1, p32 + 4096 + 1},
		{"p64_plus4k_plus1", p64_plus4k_plus1, p64},
		{"n32_minus4k", n32_minus4k, n32 - 4096},
		{"n64_minus4k", n64_minus4k, n64},
		{"inf_32", inf_32, p64},
		{"inf_64", inf_64, p64},
		{"ninf_32", ninf_32, n64},
		{"ninf_64", ninf_64, n64},
	}

	for _, test := range int64Tests {
		var converted int64
		switch v := test.input.(type) {
		case float32:
			converted = int64(v)
		case float64:
			converted = int64(v)
		}
		want(test.name, converted, test.expected)
	}

	// uint32 conversions
	uint32Tests := []struct {
		name     string
		input    any
		expected uint32
	}{
		{"minus1_32", minus1_32, 0},
		{"minus1_64", minus1_64, 0},
		{"p32_plus4k_plus1", p32_plus4k_plus1, p32 + 4096 + 1},
		{"p64_plus4k_plus1", p64_plus4k_plus1, u32},
		{"n32_minus4k", n32_minus4k, 0},
		{"n64_minus4k", n64_minus4k, 0},
		{"inf_32", inf_32, u32},
		{"inf_64", inf_64, u32},
		{"ninf_32", ninf_32, 0},
		{"ninf_64", ninf_64, 0},
	}

	for _, test := range uint32Tests {
		var converted uint32
		switch v := test.input.(type) {
		case float32:
			converted = uint32(v)
		case float64:
			converted = uint32(v)
		}
		want(test.name, converted, test.expected)
	}

	u64_plus4k_plus1_64 := id(float64(u64 + 4096 + 1))
	u64_plust44_plus1_32 := id(float32(u64 + t44 + 1))

	// uint64 conversions
	uint64Tests := []struct {
		name     string
		input    any
		expected uint64
	}{
		{"minus1_32", minus1_32, 0},
		{"minus1_64", minus1_64, 0},
		{"p32_plus4k_plus1", p32_plus4k_plus1, p32 + 4096 + 1},
		{"p64_plus4k_plus1", p64_plus4k_plus1, p64 + 4096 + 1},
		{"n32_minus4k", n32_minus4k, 0},
		{"n64_minus4k", n64_minus4k, 0},
		{"inf_32", inf_32, u64},
		{"inf_64", inf_64, u64},
		{"ninf_32", ninf_32, 0},
		{"ninf_64", ninf_64, 0},
		{"u64_plus4k_plus1_64", u64_plus4k_plus1_64, u64},
		{"u64_plust44_plus1_32", u64_plust44_plus1_32, u64},
	}

	for _, test := range uint64Tests {
		var converted uint64
		switch v := test.input.(type) {
		case float32:
			converted = uint64(v)
		case float64:
			converted = uint64(v)
		}
		want(test.name, converted, test.expected)
	}

	// for smaller integer types
	// TODO the overflow behavior is dubious, maybe we should fix it to be more sensible, e.g. saturating.
	fmt.Println("Below this are 'golden' results to check for consistency across platforms.  Overflow behavior is not necessarily what we want")

	u8plus2 := id(float64(257))
	p8minus1 := id(float32(126))
	n8plus2 := id(float64(-126))
	n8minusone := id(float32(-129))

	fmt.Println("\nuint8 conversions")
	uint8Tests := []struct {
		name  string
		input any
	}{
		{"minus1_32", minus1_32},
		{"minus1_64", minus1_64},
		{"p32_plus4k_plus1", p32_plus4k_plus1},
		{"p64_plus4k_plus1", p64_plus4k_plus1},
		{"n32_minus4k", n32_minus4k},
		{"n64_minus4k", n64_minus4k},
		{"inf_32", inf_32},
		{"inf_64", inf_64},
		{"ninf_32", ninf_32},
		{"ninf_64", ninf_64},
		{"u64_plus4k_plus1_64", u64_plus4k_plus1_64},
		{"u64_plust44_plus1_32", u64_plust44_plus1_32},
		{"u8plus2", u8plus2},
		{"p8minus1", p8minus1},
		{"n8plus2", n8plus2},
		{"n8minusone", n8minusone},
	}

	for _, test := range uint8Tests {
		var converted uint8
		switch v := test.input.(type) {
		case float32:
			converted = uint8(v)
		case float64:
			converted = uint8(v)
		}
		log(test.name, converted)
	}

	fmt.Println("\nint8 conversions")
	int8Tests := []struct {
		name  string
		input any
	}{
		{"minus1_32", minus1_32},
		{"minus1_64", minus1_64},
		{"p32_plus4k_plus1", p32_plus4k_plus1},
		{"p64_plus4k_plus1", p64_plus4k_plus1},
		{"n32_minus4k", n32_minus4k},
		{"n64_minus4k", n64_minus4k},
		{"inf_32", inf_32},
		{"inf_64", inf_64},
		{"ninf_32", ninf_32},
		{"ninf_64", ninf_64},
		{"u64_plus4k_plus1_64", u64_plus4k_plus1_64},
		{"u64_plust44_plus1_32", u64_plust44_plus1_32},
		{"u8plus2", u8plus2},
		{"p8minus1", p8minus1},
		{"n8plus2", n8plus2},
		{"n8minusone", n8minusone},
	}

	for _, test := range int8Tests {
		var converted int8
		switch v := test.input.(type) {
		case float32:
			converted = int8(v)
		case float64:
			converted = int8(v)
		}
		log(test.name, converted)
	}

}
