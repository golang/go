// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package intsets

import "testing"

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
