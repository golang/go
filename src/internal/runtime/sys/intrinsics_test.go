// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys_test

import (
	"internal/runtime/sys"
	"testing"
)

func TestTrailingZeros64(t *testing.T) {
	for i := 0; i <= 64; i++ {
		x := uint64(5) << uint(i)
		if got := sys.TrailingZeros64(x); got != i {
			t.Errorf("TrailingZeros64(%d)=%d, want %d", x, got, i)
		}
	}
}
func TestTrailingZeros32(t *testing.T) {
	for i := 0; i <= 32; i++ {
		x := uint32(5) << uint(i)
		if got := sys.TrailingZeros32(x); got != i {
			t.Errorf("TrailingZeros32(%d)=%d, want %d", x, got, i)
		}
	}
}

func TestBswap64(t *testing.T) {
	x := uint64(0x1122334455667788)
	y := sys.Bswap64(x)
	if y != 0x8877665544332211 {
		t.Errorf("Bswap(%x)=%x, want 0x8877665544332211", x, y)
	}
}
func TestBswap32(t *testing.T) {
	x := uint32(0x11223344)
	y := sys.Bswap32(x)
	if y != 0x44332211 {
		t.Errorf("Bswap(%x)=%x, want 0x44332211", x, y)
	}
}
