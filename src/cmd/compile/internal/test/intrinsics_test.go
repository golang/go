// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"math/bits"
	"testing"
)

func TestBitLen64(t *testing.T) {
	for i := 0; i <= 64; i++ {
		got := bits.Len64(1 << i)
		want := i + 1
		if want == 65 {
			want = 0
		}
		if got != want {
			t.Errorf("Len64(1<<%d) = %d, want %d", i, got, want)
		}
	}
}

func TestBitLen32(t *testing.T) {
	for i := 0; i <= 32; i++ {
		got := bits.Len32(1 << i)
		want := i + 1
		if want == 33 {
			want = 0
		}
		if got != want {
			t.Errorf("Len32(1<<%d) = %d, want %d", i, got, want)
		}
	}
}

func TestBitLen16(t *testing.T) {
	for i := 0; i <= 16; i++ {
		got := bits.Len16(1 << i)
		want := i + 1
		if want == 17 {
			want = 0
		}
		if got != want {
			t.Errorf("Len16(1<<%d) = %d, want %d", i, got, want)
		}
	}
}

func TestBitLen8(t *testing.T) {
	for i := 0; i <= 8; i++ {
		got := bits.Len8(1 << i)
		want := i + 1
		if want == 9 {
			want = 0
		}
		if got != want {
			t.Errorf("Len8(1<<%d) = %d, want %d", i, got, want)
		}
	}
}
