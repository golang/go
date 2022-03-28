// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nistec_test

import (
	"bytes"
	"crypto/elliptic"
	"crypto/elliptic/internal/nistec"
	"math/big"
	"math/rand"
	"os"
	"strings"
	"testing"
)

func TestAllocations(t *testing.T) {
	if strings.HasSuffix(os.Getenv("GO_BUILDER_NAME"), "-noopt") {
		t.Skip("skipping allocations test without relevant optimizations")
	}
	t.Run("P224", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(100, func() {
			p := nistec.NewP224Generator()
			scalar := make([]byte, 28)
			rand.Read(scalar)
			p.ScalarBaseMult(scalar)
			p.ScalarMult(p, scalar)
			out := p.Bytes()
			if _, err := nistec.NewP224Point().SetBytes(out); err != nil {
				t.Fatal(err)
			}
		}); allocs > 0 {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
	t.Run("P256", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(100, func() {
			p := nistec.NewP256Generator()
			scalar := make([]byte, 32)
			rand.Read(scalar)
			p.ScalarBaseMult(scalar)
			p.ScalarMult(p, scalar)
			out := p.Bytes()
			if _, err := nistec.NewP256Point().SetBytes(out); err != nil {
				t.Fatal(err)
			}
		}); allocs > 0 {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
	t.Run("P384", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(100, func() {
			p := nistec.NewP384Generator()
			scalar := make([]byte, 48)
			rand.Read(scalar)
			p.ScalarBaseMult(scalar)
			p.ScalarMult(p, scalar)
			out := p.Bytes()
			if _, err := nistec.NewP384Point().SetBytes(out); err != nil {
				t.Fatal(err)
			}
		}); allocs > 0 {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
	t.Run("P521", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(100, func() {
			p := nistec.NewP521Generator()
			scalar := make([]byte, 66)
			rand.Read(scalar)
			p.ScalarBaseMult(scalar)
			p.ScalarMult(p, scalar)
			out := p.Bytes()
			if _, err := nistec.NewP521Point().SetBytes(out); err != nil {
				t.Fatal(err)
			}
		}); allocs > 0 {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
}

type nistPoint[T any] interface {
	Bytes() []byte
	SetBytes([]byte) (T, error)
	Add(T, T) T
	Double(T) T
	ScalarMult(T, []byte) (T, error)
	ScalarBaseMult([]byte) (T, error)
}

func TestEquivalents(t *testing.T) {
	t.Run("P224", func(t *testing.T) {
		testEquivalents(t, nistec.NewP224Point, nistec.NewP224Generator, elliptic.P224())
	})
	t.Run("P256", func(t *testing.T) {
		testEquivalents(t, nistec.NewP256Point, nistec.NewP256Generator, elliptic.P256())
	})
	t.Run("P384", func(t *testing.T) {
		testEquivalents(t, nistec.NewP384Point, nistec.NewP384Generator, elliptic.P384())
	})
	t.Run("P521", func(t *testing.T) {
		testEquivalents(t, nistec.NewP521Point, nistec.NewP521Generator, elliptic.P521())
	})
}

func testEquivalents[P nistPoint[P]](t *testing.T, newPoint, newGenerator func() P, c elliptic.Curve) {
	p := newGenerator()

	elementSize := (c.Params().BitSize + 7) / 8
	two := make([]byte, elementSize)
	two[len(two)-1] = 2
	nPlusTwo := make([]byte, elementSize)
	new(big.Int).Add(c.Params().N, big.NewInt(2)).FillBytes(nPlusTwo)

	p1 := newPoint().Double(p)
	p2 := newPoint().Add(p, p)
	p3, err := newPoint().ScalarMult(p, two)
	if err != nil {
		t.Fatal(err)
	}
	p4, err := newPoint().ScalarBaseMult(two)
	if err != nil {
		t.Fatal(err)
	}
	p5, err := newPoint().ScalarMult(p, nPlusTwo)
	if err != nil {
		t.Fatal(err)
	}
	p6, err := newPoint().ScalarBaseMult(nPlusTwo)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(p1.Bytes(), p2.Bytes()) {
		t.Error("P+P != 2*P")
	}
	if !bytes.Equal(p1.Bytes(), p3.Bytes()) {
		t.Error("P+P != [2]P")
	}
	if !bytes.Equal(p1.Bytes(), p4.Bytes()) {
		t.Error("G+G != [2]G")
	}
	if !bytes.Equal(p1.Bytes(), p5.Bytes()) {
		t.Error("P+P != [N+2]P")
	}
	if !bytes.Equal(p1.Bytes(), p6.Bytes()) {
		t.Error("G+G != [N+2]G")
	}
}

func BenchmarkScalarMult(b *testing.B) {
	b.Run("P224", func(b *testing.B) {
		benchmarkScalarMult(b, nistec.NewP224Generator(), 28)
	})
	b.Run("P256", func(b *testing.B) {
		benchmarkScalarMult(b, nistec.NewP256Generator(), 32)
	})
	b.Run("P384", func(b *testing.B) {
		benchmarkScalarMult(b, nistec.NewP384Generator(), 48)
	})
	b.Run("P521", func(b *testing.B) {
		benchmarkScalarMult(b, nistec.NewP521Generator(), 66)
	})
}

func benchmarkScalarMult[P nistPoint[P]](b *testing.B, p P, scalarSize int) {
	scalar := make([]byte, scalarSize)
	rand.Read(scalar)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.ScalarMult(p, scalar)
	}
}

func BenchmarkScalarBaseMult(b *testing.B) {
	b.Run("P224", func(b *testing.B) {
		benchmarkScalarBaseMult(b, nistec.NewP224Generator(), 28)
	})
	b.Run("P256", func(b *testing.B) {
		benchmarkScalarBaseMult(b, nistec.NewP256Generator(), 32)
	})
	b.Run("P384", func(b *testing.B) {
		benchmarkScalarBaseMult(b, nistec.NewP384Generator(), 48)
	})
	b.Run("P521", func(b *testing.B) {
		benchmarkScalarBaseMult(b, nistec.NewP521Generator(), 66)
	})
}

func benchmarkScalarBaseMult[P nistPoint[P]](b *testing.B, p P, scalarSize int) {
	scalar := make([]byte, scalarSize)
	rand.Read(scalar)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.ScalarBaseMult(scalar)
	}
}
