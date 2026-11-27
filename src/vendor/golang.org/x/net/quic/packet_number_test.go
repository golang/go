// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import "testing"

func TestPacketNumberRange(t *testing.T) {
	if got := decodePacketNumber(maxPacketNumber, 0, 1); validPacketNumber(got) {
		t.Fatalf("decodePacketNumber(%d, 0, 1) = %d, want an out-of-range value", maxPacketNumber, got)
	}
	for _, tt := range []struct {
		pnum packetNumber
		want bool
	}{
		{-1, false},
		{0, true},
		{maxPacketNumber, true},
		{maxPacketNumber + 1, false},
		{packetNumber(1) << 62, false},
	} {
		if got := validPacketNumber(tt.pnum); got != tt.want {
			t.Errorf("validPacketNumber(%d) = %v, want %v", tt.pnum, got, tt.want)
		}
	}
}
