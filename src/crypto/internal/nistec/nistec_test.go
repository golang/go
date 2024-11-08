// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nistec_test

import (
	"bytes"
	"crypto/elliptic"
	"crypto/internal/cryptotest"
	"crypto/internal/nistec"
	"fmt"
	"math/big"
	"math/rand"
	"testing"
)

func TestAllocations(t *testing.T) {
	cryptotest.SkipTestAllocations(t)
	t.Run("P224", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(10, func() {
			p := nistec.NewP224Point().SetGenerator()
			scalar := make([]byte, 28)
			rand.Read(scalar)
			p.ScalarBaseMult(scalar)
			p.ScalarMult(p, scalar)
			out := p.Bytes()
			if _, err := nistec.NewP224Point().SetBytes(out); err != nil {
				t.Fatal(err)
			}
			out = p.BytesCompressed()
			if _, err := p.SetBytes(out); err != nil {
				t.Fatal(err)
			}
		}); allocs > 0 {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
	t.Run("P256", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(10, func() {
			p := nistec.NewP256Point().SetGenerator()
			scalar := make([]byte, 32)
			rand.Read(scalar)
			p.ScalarBaseMult(scalar)
			p.ScalarMult(p, scalar)
			out := p.Bytes()
			if _, err := nistec.NewP256Point().SetBytes(out); err != nil {
				t.Fatal(err)
			}
			out = p.BytesCompressed()
			if _, err := p.SetBytes(out); err != nil {
				t.Fatal(err)
			}
		}); allocs > 0 {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
	t.Run("P384", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(10, func() {
			p := nistec.NewP384Point().SetGenerator()
			scalar := make([]byte, 48)
			rand.Read(scalar)
			p.ScalarBaseMult(scalar)
			p.ScalarMult(p, scalar)
			out := p.Bytes()
			if _, err := nistec.NewP384Point().SetBytes(out); err != nil {
				t.Fatal(err)
			}
			out = p.BytesCompressed()
			if _, err := p.SetBytes(out); err != nil {
				t.Fatal(err)
			}
		}); allocs > 0 {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
	t.Run("P521", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(10, func() {
			p := nistec.NewP521Point().SetGenerator()
			scalar := make([]byte, 66)
			rand.Read(scalar)
			p.ScalarBaseMult(scalar)
			p.ScalarMult(p, scalar)
			out := p.Bytes()
			if _, err := nistec.NewP521Point().SetBytes(out); err != nil {
				t.Fatal(err)
			}
			out = p.BytesCompressed()
			if _, err := p.SetBytes(out); err != nil {
				t.Fatal(err)
			}
		}); allocs > 0 {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
}

type nistPoint[T any] interface {
	Bytes() []byte
	SetGenerator() T
	SetBytes([]byte) (T, error)
	Add(T, T) T
	Double(T) T
	ScalarMult(T, []byte) (T, error)
	ScalarBaseMult([]byte) (T, error)
}

func TestEquivalents(t *testing.T) {
	t.Run("P224", func(t *testing.T) {
		testEquivalents(t, nistec.NewP224Point, elliptic.P224())
	})
	t.Run("P256", func(t *testing.T) {
		testEquivalents(t, nistec.NewP256Point, elliptic.P256())
	})
	t.Run("P384", func(t *testing.T) {
		testEquivalents(t, nistec.NewP384Point, elliptic.P384())
	})
	t.Run("P521", func(t *testing.T) {
		testEquivalents(t, nistec.NewP521Point, elliptic.P521())
	})
}

func testEquivalents[P nistPoint[P]](t *testing.T, newPoint func() P, c elliptic.Curve) {
	p := newPoint().SetGenerator()

	elementSize := (c.Params().BitSize + 7) / 8
	two := make([]byte, elementSize)
	two[len(two)-1] = 2
	nPlusTwo := make([]byte, elementSize)
	new(big.Int).Add(c.Params().N, big.NewInt(2)).FillBytes(nPlusTwo)

	p1 := newPoint().Double(p)
	p2 := newPoint().Add(p, p)
	p3, err := newPoint().ScalarMult(p, two)
	fatalIfErr(t, err)
	p4, err := newPoint().ScalarBaseMult(two)
	fatalIfErr(t, err)
	p5, err := newPoint().ScalarMult(p, nPlusTwo)
	fatalIfErr(t, err)
	p6, err := newPoint().ScalarBaseMult(nPlusTwo)
	fatalIfErr(t, err)

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

func TestScalarMult(t *testing.T) {
	t.Run("P224", func(t *testing.T) {
		testScalarMult(t, nistec.NewP224Point, elliptic.P224())
	})
	t.Run("P256", func(t *testing.T) {
		testScalarMult(t, nistec.NewP256Point, elliptic.P256())
	})
	t.Run("P384", func(t *testing.T) {
		testScalarMult(t, nistec.NewP384Point, elliptic.P384())
	})
	t.Run("P521", func(t *testing.T) {
		testScalarMult(t, nistec.NewP521Point, elliptic.P521())
	})
}

func testScalarMult[P nistPoint[P]](t *testing.T, newPoint func() P, c elliptic.Curve) {
	G := newPoint().SetGenerator()
	checkScalar := func(t *testing.T, scalar []byte) {
		p1, err := newPoint().ScalarBaseMult(scalar)
		fatalIfErr(t, err)
		p2, err := newPoint().ScalarMult(G, scalar)
		fatalIfErr(t, err)
		if !bytes.Equal(p1.Bytes(), p2.Bytes()) {
			t.Error("[k]G != ScalarBaseMult(k)")
		}

		expectInfinity := new(big.Int).Mod(new(big.Int).SetBytes(scalar), c.Params().N).Sign() == 0
		if expectInfinity {
			if !bytes.Equal(p1.Bytes(), newPoint().Bytes()) {
				t.Error("ScalarBaseMult(k) != ∞")
			}
			if !bytes.Equal(p2.Bytes(), newPoint().Bytes()) {
				t.Error("[k]G != ∞")
			}
		} else {
			if bytes.Equal(p1.Bytes(), newPoint().Bytes()) {
				t.Error("ScalarBaseMult(k) == ∞")
			}
			if bytes.Equal(p2.Bytes(), newPoint().Bytes()) {
				t.Error("[k]G == ∞")
			}
		}

		d := new(big.Int).SetBytes(scalar)
		d.Sub(c.Params().N, d)
		d.Mod(d, c.Params().N)
		g1, err := newPoint().ScalarBaseMult(d.FillBytes(make([]byte, len(scalar))))
		fatalIfErr(t, err)
		g1.Add(g1, p1)
		if !bytes.Equal(g1.Bytes(), newPoint().Bytes()) {
			t.Error("[N - k]G + [k]G != ∞")
		}
	}

	byteLen := len(c.Params().N.Bytes())
	bitLen := c.Params().N.BitLen()
	t.Run("0", func(t *testing.T) { checkScalar(t, make([]byte, byteLen)) })
	t.Run("1", func(t *testing.T) {
		checkScalar(t, big.NewInt(1).FillBytes(make([]byte, byteLen)))
	})
	t.Run("N-1", func(t *testing.T) {
		checkScalar(t, new(big.Int).Sub(c.Params().N, big.NewInt(1)).Bytes())
	})
	t.Run("N", func(t *testing.T) { checkScalar(t, c.Params().N.Bytes()) })
	t.Run("N+1", func(t *testing.T) {
		checkScalar(t, new(big.Int).Add(c.Params().N, big.NewInt(1)).Bytes())
	})
	t.Run("all1s", func(t *testing.T) {
		s := new(big.Int).Lsh(big.NewInt(1), uint(bitLen))
		s.Sub(s, big.NewInt(1))
		checkScalar(t, s.Bytes())
	})
	if testing.Short() {
		return
	}
	for i := 0; i < bitLen; i++ {
		t.Run(fmt.Sprintf("1<<%d", i), func(t *testing.T) {
			s := new(big.Int).Lsh(big.NewInt(1), uint(i))
			checkScalar(t, s.FillBytes(make([]byte, byteLen)))
		})
	}
	for i := 0; i <= 64; i++ {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			checkScalar(t, big.NewInt(int64(i)).FillBytes(make([]byte, byteLen)))
		})
	}
	// Test N-64...N+64 since they risk overlapping with precomputed table values
	// in the final additions.
	for i := int64(-64); i <= 64; i++ {
		t.Run(fmt.Sprintf("N%+d", i), func(t *testing.T) {
			checkScalar(t, new(big.Int).Add(c.Params().N, big.NewInt(i)).Bytes())
		})
	}
}

func fatalIfErr(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func BenchmarkScalarMult(b *testing.B) {
	b.Run("P224", func(b *testing.B) {
		benchmarkScalarMult(b, nistec.NewP224Point().SetGenerator(), 28)
	})
	b.Run("P256", func(b *testing.B) {
		benchmarkScalarMult(b, nistec.NewP256Point().SetGenerator(), 32)
	})
	b.Run("P384", func(b *testing.B) {
		benchmarkScalarMult(b, nistec.NewP384Point().SetGenerator(), 48)
	})
	b.Run("P521", func(b *testing.B) {
		benchmarkScalarMult(b, nistec.NewP521Point().SetGenerator(), 66)
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
		benchmarkScalarBaseMult(b, nistec.NewP224Point().SetGenerator(), 28)
	})
	b.Run("P256", func(b *testing.B) {
		benchmarkScalarBaseMult(b, nistec.NewP256Point().SetGenerator(), 32)
	})
	b.Run("P384", func(b *testing.B) {
		benchmarkScalarBaseMult(b, nistec.NewP384Point().SetGenerator(), 48)
	})
	b.Run("P521", func(b *testing.B) {
		benchmarkScalarBaseMult(b, nistec.NewP521Point().SetGenerator(), 66)
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
