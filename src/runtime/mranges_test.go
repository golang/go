// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"testing"
)

func TestAddrRangesFindSucc(t *testing.T) {
	var large []AddrRange
	for i := 0; i < 100; i++ {
		large = append(large, MakeAddrRange(5+uintptr(i)*5, 5+uintptr(i)*5+3))
	}

	type testt struct {
		name   string
		base   uintptr
		expect int
		ranges []AddrRange
	}
	tests := []testt{
		{
			name:   "Empty",
			base:   12,
			expect: 0,
			ranges: []AddrRange{},
		},
		{
			name:   "OneBefore",
			base:   12,
			expect: 0,
			ranges: []AddrRange{
				MakeAddrRange(14, 16),
			},
		},
		{
			name:   "OneWithin",
			base:   14,
			expect: 1,
			ranges: []AddrRange{
				MakeAddrRange(14, 16),
			},
		},
		{
			name:   "OneAfterLimit",
			base:   16,
			expect: 1,
			ranges: []AddrRange{
				MakeAddrRange(14, 16),
			},
		},
		{
			name:   "OneAfter",
			base:   17,
			expect: 1,
			ranges: []AddrRange{
				MakeAddrRange(14, 16),
			},
		},
		{
			name:   "ThreeBefore",
			base:   3,
			expect: 0,
			ranges: []AddrRange{
				MakeAddrRange(6, 10),
				MakeAddrRange(12, 16),
				MakeAddrRange(19, 22),
			},
		},
		{
			name:   "ThreeAfter",
			base:   24,
			expect: 3,
			ranges: []AddrRange{
				MakeAddrRange(6, 10),
				MakeAddrRange(12, 16),
				MakeAddrRange(19, 22),
			},
		},
		{
			name:   "ThreeBetween",
			base:   11,
			expect: 1,
			ranges: []AddrRange{
				MakeAddrRange(6, 10),
				MakeAddrRange(12, 16),
				MakeAddrRange(19, 22),
			},
		},
		{
			name:   "ThreeWithin",
			base:   9,
			expect: 1,
			ranges: []AddrRange{
				MakeAddrRange(6, 10),
				MakeAddrRange(12, 16),
				MakeAddrRange(19, 22),
			},
		},
		{
			name:   "Zero",
			base:   0,
			expect: 1,
			ranges: []AddrRange{
				MakeAddrRange(0, 10),
			},
		},
		{
			name:   "Max",
			base:   ^uintptr(0),
			expect: 1,
			ranges: []AddrRange{
				MakeAddrRange(^uintptr(0)-5, ^uintptr(0)),
			},
		},
		{
			name:   "LargeBefore",
			base:   2,
			expect: 0,
			ranges: large,
		},
		{
			name:   "LargeAfter",
			base:   5 + uintptr(len(large))*5 + 30,
			expect: len(large),
			ranges: large,
		},
		{
			name:   "LargeBetweenLow",
			base:   14,
			expect: 2,
			ranges: large,
		},
		{
			name:   "LargeBetweenHigh",
			base:   249,
			expect: 49,
			ranges: large,
		},
		{
			name:   "LargeWithinLow",
			base:   25,
			expect: 5,
			ranges: large,
		},
		{
			name:   "LargeWithinHigh",
			base:   396,
			expect: 79,
			ranges: large,
		},
		{
			name:   "LargeWithinMiddle",
			base:   250,
			expect: 50,
			ranges: large,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := MakeAddrRanges(test.ranges...)
			i := a.FindSucc(test.base)
			if i != test.expect {
				t.Fatalf("expected %d, got %d", test.expect, i)
			}
		})
	}
}
