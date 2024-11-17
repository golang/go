// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

import (
	"crypto/internal/cryptotest"
	. "crypto/internal/fips/edwards25519"
	"testing"
)

var testAllocationsSink byte

func TestEdwards25519Allocations(t *testing.T) {
	cryptotest.SkipTestAllocations(t)
	if allocs := testing.AllocsPerRun(100, func() {
		p := NewIdentityPoint()
		p.Add(p, NewGeneratorPoint())
		s := NewScalar()
		testAllocationsSink ^= s.Bytes()[0]
		testAllocationsSink ^= p.Bytes()[0]
	}); allocs > 0 {
		t.Errorf("expected zero allocations, got %0.1v", allocs)
	}
}
