// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"testing"
)

func TestIssue48807(t *testing.T) {
	for _, i := range []uint64{
		0x8234508000000001, // from issue48807
		1<<56 + 1<<32 + 1,
	} {
		got := float32(i)
		dontwant := float32(float64(i))
		if got == dontwant {
			// The test cases above should be uint64s such that
			// this equality doesn't hold. These examples trigger
			// the case where using an intermediate float64 doesn't work.
			t.Errorf("direct float32 conversion doesn't work: arg=%x got=%x dontwant=%x", i, got, dontwant)
		}
	}
}

func TestUint64ToFloat32Rounding(t *testing.T) {
	for _, tt := range []struct {
		in   uint64
		want float32
	}{
		{1<<46 + 1<<22, 1 << 46},
		{1<<46 + 1<<22 + 1, 1<<46 + 1<<23},
		{1<<47 + 1<<24 + 1, 1<<47 + 1<<24},
		{1<<47 + 1<<23 + 1, 1<<47 + 1<<24},
	} {
		if got := float32(tt.in); got != tt.want {
			t.Errorf("float32(%#x) = %x, want %x", tt.in, got, tt.want)
		}
		if got := float32(-int64(tt.in)); got != -tt.want {
			t.Errorf("float32(%d) = %x, want %x", -int64(tt.in), got, -tt.want)
		}
	}
}
