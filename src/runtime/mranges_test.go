// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"testing"
)

func validateAddrRanges(t *testing.T, a *AddrRanges, want ...AddrRange) {
	ranges := a.Ranges()
	if len(ranges) != len(want) {
		t.Errorf("want %v, got %v", want, ranges)
		t.Fatal("different lengths")
	}
	gotTotalBytes := uintptr(0)
	wantTotalBytes := uintptr(0)
	for i := range ranges {
		gotTotalBytes += ranges[i].Size()
		wantTotalBytes += want[i].Size()
		if ranges[i].Base() >= ranges[i].Limit() {
			t.Error("empty range found")
		}
		// Ensure this is equivalent to what we want.
		if !ranges[i].Equals(want[i]) {
			t.Errorf("range %d: got [0x%x, 0x%x), want [0x%x, 0x%x)", i,
				ranges[i].Base(), ranges[i].Limit(),
				want[i].Base(), want[i].Limit(),
			)
		}
		if i != 0 {
			// Ensure the ranges are sorted.
			if ranges[i-1].Base() >= ranges[i].Base() {
				t.Errorf("ranges %d and %d are out of sorted order", i-1, i)
			}
			// Check for a failure to coalesce.
			if ranges[i-1].Limit() == ranges[i].Base() {
				t.Errorf("ranges %d and %d should have coalesced", i-1, i)
			}
			// Check if any ranges overlap. Because the ranges are sorted
			// by base, it's sufficient to just check neighbors.
			if ranges[i-1].Limit() > ranges[i].Base() {
				t.Errorf("ranges %d and %d overlap", i-1, i)
			}
		}
	}
	if wantTotalBytes != gotTotalBytes {
		t.Errorf("expected %d total bytes, got %d", wantTotalBytes, gotTotalBytes)
	}
	if b := a.TotalBytes(); b != gotTotalBytes {
		t.Errorf("inconsistent total bytes: want %d, got %d", gotTotalBytes, b)
	}
	if t.Failed() {
		t.Errorf("addrRanges: %v", ranges)
		t.Fatal("detected bad addrRanges")
	}
}

func TestAddrRangesAdd(t *testing.T) {
	a := NewAddrRanges()

	// First range.
	a.Add(MakeAddrRange(512, 1024))
	validateAddrRanges(t, &a,
		MakeAddrRange(512, 1024),
	)

	// Coalesce up.
	a.Add(MakeAddrRange(1024, 2048))
	validateAddrRanges(t, &a,
		MakeAddrRange(512, 2048),
	)

	// Add new independent range.
	a.Add(MakeAddrRange(4096, 8192))
	validateAddrRanges(t, &a,
		MakeAddrRange(512, 2048),
		MakeAddrRange(4096, 8192),
	)

	// Coalesce down.
	a.Add(MakeAddrRange(3776, 4096))
	validateAddrRanges(t, &a,
		MakeAddrRange(512, 2048),
		MakeAddrRange(3776, 8192),
	)

	// Coalesce up and down.
	a.Add(MakeAddrRange(2048, 3776))
	validateAddrRanges(t, &a,
		MakeAddrRange(512, 8192),
	)

	// Push a bunch of independent ranges to the end to try and force growth.
	expectedRanges := []AddrRange{MakeAddrRange(512, 8192)}
	for i := uintptr(0); i < 64; i++ {
		dRange := MakeAddrRange(8192+(i+1)*2048, 8192+(i+1)*2048+10)
		a.Add(dRange)
		expectedRanges = append(expectedRanges, dRange)
		validateAddrRanges(t, &a, expectedRanges...)
	}

	// Push a bunch of independent ranges to the beginning to try and force growth.
	var bottomRanges []AddrRange
	for i := uintptr(0); i < 63; i++ {
		dRange := MakeAddrRange(8+i*8, 8+i*8+4)
		a.Add(dRange)
		bottomRanges = append(bottomRanges, dRange)
		validateAddrRanges(t, &a, append(bottomRanges, expectedRanges...)...)
	}
}

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
		t.Run(test.name, func { t ->
			a := MakeAddrRanges(test.ranges...)
			i := a.FindSucc(test.base)
			if i != test.expect {
				t.Fatalf("expected %d, got %d", test.expect, i)
			}
		})
	}
}
