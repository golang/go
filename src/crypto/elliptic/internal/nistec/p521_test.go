// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nistec_test

import (
	"crypto/elliptic/internal/nistec"
	"math/rand"
	"os"
	"strings"
	"testing"
)

func TestP521Allocations(t *testing.T) {
	if strings.HasSuffix(os.Getenv("GO_BUILDER_NAME"), "-noopt") {
		t.Skip("skipping allocations test without relevant optimizations")
	}
	if allocs := testing.AllocsPerRun(100, func() {
		p := nistec.NewP521Generator()
		scalar := make([]byte, 66)
		rand.Read(scalar)
		p.ScalarMult(p, scalar)
		out := p.Bytes()
		if _, err := p.SetBytes(out); err != nil {
			t.Fatal(err)
		}
	}); allocs > 0 {
		t.Errorf("expected zero allocations, got %0.1f", allocs)
	}
}

func BenchmarkScalarMult(b *testing.B) {
	b.Run("P521", func(b *testing.B) {
		scalar := make([]byte, 66)
		rand.Read(scalar)
		p := nistec.NewP521Generator()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p.ScalarMult(p, scalar)
		}
	})
}
