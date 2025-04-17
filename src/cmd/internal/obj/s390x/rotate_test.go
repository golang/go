// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390x

import (
	"testing"
)

func TestRotateParamsMask(t *testing.T) {
	tests := []struct {
		start, end, amount uint8
		inMask, outMask    uint64
	}{
		// start before end, no rotation
		{start: 0, end: 63, amount: 0, inMask: ^uint64(0), outMask: ^uint64(0)},
		{start: 1, end: 63, amount: 0, inMask: ^uint64(0) >> 1, outMask: ^uint64(0) >> 1},
		{start: 0, end: 62, amount: 0, inMask: ^uint64(1), outMask: ^uint64(1)},
		{start: 1, end: 62, amount: 0, inMask: ^uint64(3) >> 1, outMask: ^uint64(3) >> 1},

		// end before start, no rotation
		{start: 63, end: 0, amount: 0, inMask: 1<<63 | 1, outMask: 1<<63 | 1},
		{start: 62, end: 0, amount: 0, inMask: 1<<63 | 3, outMask: 1<<63 | 3},
		{start: 63, end: 1, amount: 0, inMask: 3<<62 | 1, outMask: 3<<62 | 1},
		{start: 62, end: 1, amount: 0, inMask: 3<<62 | 3, outMask: 3<<62 | 3},

		// rotation
		{start: 32, end: 63, amount: 32, inMask: 0xffffffff00000000, outMask: 0x00000000ffffffff},
		{start: 48, end: 15, amount: 16, inMask: 0xffffffff00000000, outMask: 0xffff00000000ffff},
		{start: 0, end: 7, amount: -8 & 63, inMask: 0xff, outMask: 0xff << 56},
	}
	for i, test := range tests {
		r := NewRotateParams(test.start, test.end, test.amount)
		if m := r.OutMask(); m != test.outMask {
			t.Errorf("out mask %v: want %#x, got %#x", i, test.outMask, m)
		}
		if m := r.InMask(); m != test.inMask {
			t.Errorf("in mask %v: want %#x, got %#x", i, test.inMask, m)
		}
	}
}

func TestRotateParamsMerge(t *testing.T) {
	tests := []struct {
		// inputs
		src  RotateParams
		mask uint64

		// results
		in  *RotateParams
		out *RotateParams
	}{
		{
			src:  RotateParams{Start: 48, End: 15, Amount: 16},
			mask: 0xffffffffffffffff,
			in:   &RotateParams{Start: 48, End: 15, Amount: 16},
			out:  &RotateParams{Start: 48, End: 15, Amount: 16},
		},
		{
			src:  RotateParams{Start: 16, End: 47, Amount: 0},
			mask: 0x00000000ffffffff,
			in:   &RotateParams{Start: 32, End: 47, Amount: 0},
			out:  &RotateParams{Start: 32, End: 47, Amount: 0},
		},
		{
			src:  RotateParams{Start: 16, End: 47, Amount: 0},
			mask: 0xffff00000000ffff,
			in:   nil,
			out:  nil,
		},
		{
			src:  RotateParams{Start: 0, End: 63, Amount: 0},
			mask: 0xf7f0000000000000,
			in:   nil,
			out:  nil,
		},
		{
			src:  RotateParams{Start: 0, End: 63, Amount: 1},
			mask: 0x000000000000ff00,
			in:   &RotateParams{Start: 47, End: 54, Amount: 1},
			out:  &RotateParams{Start: 48, End: 55, Amount: 1},
		},
		{
			src:  RotateParams{Start: 32, End: 63, Amount: 32},
			mask: 0xffff00000000ffff,
			in:   &RotateParams{Start: 32, End: 47, Amount: 32},
			out:  &RotateParams{Start: 48, End: 63, Amount: 32},
		},
		{
			src:  RotateParams{Start: 0, End: 31, Amount: 32},
			mask: 0x8000000000000000,
			in:   nil,
			out:  &RotateParams{Start: 0, End: 0, Amount: 32},
		},
		{
			src:  RotateParams{Start: 0, End: 31, Amount: 32},
			mask: 0x0000000080000000,
			in:   &RotateParams{Start: 0, End: 0, Amount: 32},
			out:  nil,
		},
	}

	eq := func(x, y *RotateParams) bool {
		if x == nil && y == nil {
			return true
		}
		if x == nil || y == nil {
			return false
		}
		return *x == *y
	}

	for _, test := range tests {
		if r := test.src.InMerge(test.mask); !eq(r, test.in) {
			t.Errorf("%v merged with %#x (input): want %v, got %v", test.src, test.mask, test.in, r)
		}
		if r := test.src.OutMerge(test.mask); !eq(r, test.out) {
			t.Errorf("%v merged with %#x (output): want %v, got %v", test.src, test.mask, test.out, r)
		}
	}
}
