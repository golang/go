// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !race && !asan && !msan

package test

import (
	"internal/testenv"
	"testing"
)

func TestAppendOfMake(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	for n := 32; n < 33; n++ { // avoid stack allocation of make()
		b := make([]byte, n)
		f := func() {
			b = append(b[:0], make([]byte, n)...)
		}
		if n := testing.AllocsPerRun(10, f); n > 0 {
			t.Errorf("got %f allocs, want 0", n)
		}
		type S []byte

		s := make(S, n)
		g := func() {
			s = append(s[:0], make(S, n)...)
		}
		if n := testing.AllocsPerRun(10, g); n > 0 {
			t.Errorf("got %f allocs, want 0", n)
		}
		h := func() {
			s = append(s[:0], make([]byte, n)...)
		}
		if n := testing.AllocsPerRun(10, h); n > 0 {
			t.Errorf("got %f allocs, want 0", n)
		}
		i := func() {
			b = append(b[:0], make(S, n)...)
		}
		if n := testing.AllocsPerRun(10, i); n > 0 {
			t.Errorf("got %f allocs, want 0", n)
		}
	}
}
