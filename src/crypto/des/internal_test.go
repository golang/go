// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package des

import "testing"

func TestInitialPermute(t *testing.T) {
	for i := uint(0); i < 64; i++ {
		bit := uint64(1) << i
		got := permuteInitialBlock(bit)
		want := uint64(1) << finalPermutation[63-i]
		if got != want {
			t.Errorf("permute(%x) = %x, want %x", bit, got, want)
		}
	}
}

func TestFinalPermute(t *testing.T) {
	for i := uint(0); i < 64; i++ {
		bit := uint64(1) << i
		got := permuteFinalBlock(bit)
		want := uint64(1) << initialPermutation[63-i]
		if got != want {
			t.Errorf("permute(%x) = %x, want %x", bit, got, want)
		}
	}
}
