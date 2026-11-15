// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

import "testing"

func placementExpectedSum(n int) int {
	sum := 0
	for i := range n {
		sum += (i*7 + 11) & 0xff
	}
	return sum
}

func TestFastcallSlicePlacement(t *testing.T) {
	n := placementCapacity
	expected := placementExpectedSum(n)

	if got := placementChecksumDynamic(); got != expected {
		t.Fatalf("dynamic checksum = %d, want %d (n=%d)", got, expected, n)
	}

	buf := make([]placementBytes, n)
	control := placementControl(buf)
	if got := control(); got != expected {
		t.Fatalf("control checksum = %d, want %d (n=%d)", got, expected, n)
	}

	if testUnderRace {
		t.Skip("allocation gates need quiet accounting; the typed path " +
			"still runs under race in TestFastcallTypedSliceCorrectness.")
	}

	dynamicAllocs := testing.AllocsPerRun(100, func() { placementChecksumDynamic() })
	controlAllocs := testing.AllocsPerRun(100, func() { control() })
	if controlAllocs != 0 {
		t.Fatalf("control allocs = %v, want exactly 0", controlAllocs)
	}
	if delta := dynamicAllocs - controlAllocs; delta != 1 {
		t.Fatalf("placement delta = %v - %v = %v, want exactly 1",
			dynamicAllocs, controlAllocs, delta)
	}
}

func TestFastcallTypedSliceCorrectness(t *testing.T) {
	n := placementCapacity
	expected := placementExpectedSum(n)

	buf := make([]placementBytes, n)
	control := placementControl(buf)
	for i := range 16 {
		if got := placementChecksumDynamic(); got != expected {
			t.Fatalf("run %d: dynamic checksum = %d, want %d (n=%d)",
				i, got, expected, n)
		}
		if got := control(); got != expected {
			t.Fatalf("run %d: control checksum = %d, want %d (n=%d)",
				i, got, expected, n)
		}
	}
}
