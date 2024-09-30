// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noopt

package sha3_test

import (
	"crypto/internal/fips/sha3"
	"runtime"
	"testing"
)

var sink byte

func TestAllocations(t *testing.T) {
	want := 0.0

	if runtime.GOARCH == "s390x" {
		// On s390x the returned hash.Hash is conditional so it escapes.
		want = 3.0
	}

	t.Run("New", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(10, func() {
			h := sha3.New256()
			b := []byte("ABC")
			h.Write(b)
			out := make([]byte, 0, 32)
			out = h.Sum(out)
			sink ^= out[0]
		}); allocs > want {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
	t.Run("NewShake", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(10, func() {
			h := sha3.NewShake128()
			b := []byte("ABC")
			h.Write(b)
			out := make([]byte, 0, 32)
			out = h.Sum(out)
			sink ^= out[0]
			h.Read(out)
			sink ^= out[0]
		}); allocs > want {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
	t.Run("Sum", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(10, func() {
			b := []byte("ABC")
			out := sha3.Sum256(b)
			sink ^= out[0]
		}); allocs > want {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
}
