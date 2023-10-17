// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"testing"
)

var res int64
var ures uint64

func TestFloatTruncation(t *testing.T) {
	testdata := []struct {
		input      float64
		convInt64  int64
		convUInt64 uint64
		overflow   bool
	}{
		// max +- 1
		{
			input:      0x7fffffffffffffff,
			convInt64:  -0x8000000000000000,
			convUInt64: 0x8000000000000000,
		},
		// For out-of-bounds conversion, the result is implementation-dependent.
		// This test verifies the implementation of wasm architecture.
		{
			input:      0x8000000000000000,
			convInt64:  -0x8000000000000000,
			convUInt64: 0x8000000000000000,
		},
		{
			input:      0x7ffffffffffffffe,
			convInt64:  -0x8000000000000000,
			convUInt64: 0x8000000000000000,
		},
		// neg max +- 1
		{
			input:      -0x8000000000000000,
			convInt64:  -0x8000000000000000,
			convUInt64: 0x8000000000000000,
		},
		{
			input:      -0x8000000000000001,
			convInt64:  -0x8000000000000000,
			convUInt64: 0x8000000000000000,
		},
		{
			input:      -0x7fffffffffffffff,
			convInt64:  -0x8000000000000000,
			convUInt64: 0x8000000000000000,
		},
		// trunc point +- 1
		{
			input:      0x7ffffffffffffdff,
			convInt64:  0x7ffffffffffffc00,
			convUInt64: 0x7ffffffffffffc00,
		},
		{
			input:      0x7ffffffffffffe00,
			convInt64:  -0x8000000000000000,
			convUInt64: 0x8000000000000000,
		},
		{
			input:      0x7ffffffffffffdfe,
			convInt64:  0x7ffffffffffffc00,
			convUInt64: 0x7ffffffffffffc00,
		},
		// neg trunc point +- 1
		{
			input:      -0x7ffffffffffffdff,
			convInt64:  -0x7ffffffffffffc00,
			convUInt64: 0x8000000000000000,
		},
		{
			input:      -0x7ffffffffffffe00,
			convInt64:  -0x8000000000000000,
			convUInt64: 0x8000000000000000,
		},
		{
			input:      -0x7ffffffffffffdfe,
			convInt64:  -0x7ffffffffffffc00,
			convUInt64: 0x8000000000000000,
		},
		// umax +- 1
		{
			input:      0xffffffffffffffff,
			convInt64:  -0x8000000000000000,
			convUInt64: 0x8000000000000000,
		},
		{
			input:      0x10000000000000000,
			convInt64:  -0x8000000000000000,
			convUInt64: 0x8000000000000000,
		},
		{
			input:      0xfffffffffffffffe,
			convInt64:  -0x8000000000000000,
			convUInt64: 0x8000000000000000,
		},
		// umax trunc +- 1
		{
			input:      0xfffffffffffffbff,
			convInt64:  -0x8000000000000000,
			convUInt64: 0xfffffffffffff800,
		},
		{
			input:      0xfffffffffffffc00,
			convInt64:  -0x8000000000000000,
			convUInt64: 0x8000000000000000,
		},
		{
			input:      0xfffffffffffffbfe,
			convInt64:  -0x8000000000000000,
			convUInt64: 0xfffffffffffff800,
		},
	}
	for _, item := range testdata {
		if got, want := int64(item.input), item.convInt64; got != want {
			t.Errorf("int64(%f): got %x, want %x", item.input, got, want)
		}
		if got, want := uint64(item.input), item.convUInt64; got != want {
			t.Errorf("uint64(%f): got %x, want %x", item.input, got, want)
		}
	}
}
