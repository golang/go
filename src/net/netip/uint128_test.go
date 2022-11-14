// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netip

import (
	"testing"
)

func TestUint128AddSub(t *testing.T) {
	const add1 = 1
	const sub1 = -1
	tests := []struct {
		in   uint128
		op   int // +1 or -1 to add vs subtract
		want uint128
	}{
		{uint128{0, 0}, add1, uint128{0, 1}},
		{uint128{0, 1}, add1, uint128{0, 2}},
		{uint128{1, 0}, add1, uint128{1, 1}},
		{uint128{0, ^uint64(0)}, add1, uint128{1, 0}},
		{uint128{^uint64(0), ^uint64(0)}, add1, uint128{0, 0}},

		{uint128{0, 0}, sub1, uint128{^uint64(0), ^uint64(0)}},
		{uint128{0, 1}, sub1, uint128{0, 0}},
		{uint128{0, 2}, sub1, uint128{0, 1}},
		{uint128{1, 0}, sub1, uint128{0, ^uint64(0)}},
		{uint128{1, 1}, sub1, uint128{1, 0}},
	}
	for _, tt := range tests {
		var got uint128
		switch tt.op {
		case add1:
			got = tt.in.addOne()
		case sub1:
			got = tt.in.subOne()
		default:
			panic("bogus op")
		}
		if got != tt.want {
			t.Errorf("%v add %d = %v; want %v", tt.in, tt.op, got, tt.want)
		}
	}
}

func TestBitsSetFrom(t *testing.T) {
	tests := []struct {
		bit  uint8
		want uint128
	}{
		{0, uint128{^uint64(0), ^uint64(0)}},
		{1, uint128{^uint64(0) >> 1, ^uint64(0)}},
		{63, uint128{1, ^uint64(0)}},
		{64, uint128{0, ^uint64(0)}},
		{65, uint128{0, ^uint64(0) >> 1}},
		{127, uint128{0, 1}},
		{128, uint128{0, 0}},
	}
	for _, tt := range tests {
		var zero uint128
		got := zero.bitsSetFrom(tt.bit)
		if got != tt.want {
			t.Errorf("0.bitsSetFrom(%d) = %064b want %064b", tt.bit, got, tt.want)
		}
	}
}

func TestBitsClearedFrom(t *testing.T) {
	tests := []struct {
		bit  uint8
		want uint128
	}{
		{0, uint128{0, 0}},
		{1, uint128{1 << 63, 0}},
		{63, uint128{^uint64(0) &^ 1, 0}},
		{64, uint128{^uint64(0), 0}},
		{65, uint128{^uint64(0), 1 << 63}},
		{127, uint128{^uint64(0), ^uint64(0) &^ 1}},
		{128, uint128{^uint64(0), ^uint64(0)}},
	}
	for _, tt := range tests {
		ones := uint128{^uint64(0), ^uint64(0)}
		got := ones.bitsClearedFrom(tt.bit)
		if got != tt.want {
			t.Errorf("ones.bitsClearedFrom(%d) = %064b want %064b", tt.bit, got, tt.want)
		}
	}
}
