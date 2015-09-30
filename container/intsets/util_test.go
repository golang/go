// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package intsets

import (
	"math/rand"
	"testing"
)

func TestNLZ(t *testing.T) {
	// Test the platform-specific edge case.
	// NB: v must be a var (not const) so that the word() conversion is dynamic.
	// Otherwise the compiler will report an error.
	v := uint64(0x0000801000000000)
	n := nlz(word(v))
	want := 32 // (on 32-bit)
	if bitsPerWord == 64 {
		want = 16
	}
	if n != want {
		t.Errorf("%d-bit nlz(%d) = %d, want %d", bitsPerWord, v, n, want)
	}
}

// Backdoor for testing.
func (s *Sparse) Check() error { return s.check() }

func dumbPopcount(x word) int {
	var popcnt int
	for i := uint(0); i < bitsPerWord; i++ {
		if x&(1<<i) != 0 {
			popcnt++
		}
	}
	return popcnt
}

func TestPopcount(t *testing.T) {
	for i := 0; i < 1e5; i++ {
		x := word(rand.Uint32())
		if bitsPerWord == 64 {
			x = x | (word(rand.Uint32()) << 32)
		}
		want := dumbPopcount(x)
		got := popcount(x)
		if got != want {
			t.Errorf("popcount(%d) = %d, want %d", x, got, want)
		}
	}
}

func BenchmarkPopcount(b *testing.B) {
	for i := 0; i < b.N; i++ {
		popcount(word(i))
	}
}
