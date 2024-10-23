// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mlkem

import (
	"bytes"
	"crypto/internal/fips/sha3"
	"crypto/rand"
	_ "embed"
	"encoding/hex"
	"flag"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	dk, err := GenerateKey768()
	if err != nil {
		t.Fatal(err)
	}
	c, Ke := dk.EncapsulationKey().Encapsulate()
	Kd, err := dk.Decapsulate(c)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(Ke, Kd) {
		t.Fail()
	}

	dk1, err := GenerateKey768()
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(dk.EncapsulationKey().Bytes(), dk1.EncapsulationKey().Bytes()) {
		t.Fail()
	}
	if bytes.Equal(dk.Bytes(), dk1.Bytes()) {
		t.Fail()
	}

	c1, Ke1 := dk.EncapsulationKey().Encapsulate()
	if bytes.Equal(c, c1) {
		t.Fail()
	}
	if bytes.Equal(Ke, Ke1) {
		t.Fail()
	}
}

func TestBadLengths(t *testing.T) {
	dk, err := GenerateKey768()
	if err != nil {
		t.Fatal(err)
	}
	ek := dk.EncapsulationKey()
	ekBytes := dk.EncapsulationKey().Bytes()
	c, _ := ek.Encapsulate()

	for i := 0; i < len(ekBytes)-1; i++ {
		if _, err := NewEncapsulationKey768(ekBytes[:i]); err == nil {
			t.Errorf("expected error for ek length %d", i)
		}
	}
	ekLong := ekBytes
	for i := 0; i < 100; i++ {
		ekLong = append(ekLong, 0)
		if _, err := NewEncapsulationKey768(ekLong); err == nil {
			t.Errorf("expected error for ek length %d", len(ekLong))
		}
	}

	for i := 0; i < len(c)-1; i++ {
		if _, err := dk.Decapsulate(c[:i]); err == nil {
			t.Errorf("expected error for c length %d", i)
		}
	}
	cLong := c
	for i := 0; i < 100; i++ {
		cLong = append(cLong, 0)
		if _, err := dk.Decapsulate(cLong); err == nil {
			t.Errorf("expected error for c length %d", len(cLong))
		}
	}
}

var millionFlag = flag.Bool("million", false, "run the million vector test")

// TestAccumulated accumulates 10k (or 100, or 1M) random vectors and checks the
// hash of the result, to avoid checking in 150MB of test vectors.
func TestAccumulated(t *testing.T) {
	n := 10000
	expected := "8a518cc63da366322a8e7a818c7a0d63483cb3528d34a4cf42f35d5ad73f22fc"
	if testing.Short() {
		n = 100
		expected = "1114b1b6699ed191734fa339376afa7e285c9e6acf6ff0177d346696ce564415"
	}
	if *millionFlag {
		n = 1000000
		expected = "424bf8f0e8ae99b78d788a6e2e8e9cdaf9773fc0c08a6f433507cb559edfd0f0"
	}

	s := sha3.NewShake128()
	o := sha3.NewShake128()
	seed := make([]byte, SeedSize)
	var msg [messageSize]byte
	ct1 := make([]byte, CiphertextSize768)

	for i := 0; i < n; i++ {
		s.Read(seed)
		dk, err := NewDecapsulationKey768(seed)
		if err != nil {
			t.Fatal(err)
		}
		ek := dk.EncapsulationKey()
		o.Write(ek.Bytes())

		s.Read(msg[:])
		ct, k := kemEncaps(nil, ek, &msg)
		o.Write(ct)
		o.Write(k)

		kk, err := dk.Decapsulate(ct)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(kk, k) {
			t.Errorf("k: got %x, expected %x", kk, k)
		}

		s.Read(ct1)
		k1, err := dk.Decapsulate(ct1)
		if err != nil {
			t.Fatal(err)
		}
		o.Write(k1)
	}

	got := hex.EncodeToString(o.Sum(nil))
	if got != expected {
		t.Errorf("got %s, expected %s", got, expected)
	}
}

var sink byte

func BenchmarkKeyGen(b *testing.B) {
	var dk DecapsulationKey768
	var d, z [32]byte
	rand.Read(d[:])
	rand.Read(z[:])
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dk := kemKeyGen(&dk, &d, &z)
		sink ^= dk.EncapsulationKey().Bytes()[0]
	}
}

func BenchmarkEncaps(b *testing.B) {
	seed := make([]byte, SeedSize)
	rand.Read(seed)
	var m [messageSize]byte
	rand.Read(m[:])
	dk, err := NewDecapsulationKey768(seed)
	if err != nil {
		b.Fatal(err)
	}
	ekBytes := dk.EncapsulationKey().Bytes()
	var c [CiphertextSize768]byte
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ek, err := NewEncapsulationKey768(ekBytes)
		if err != nil {
			b.Fatal(err)
		}
		c, K := kemEncaps(&c, ek, &m)
		sink ^= c[0] ^ K[0]
	}
}

func BenchmarkDecaps(b *testing.B) {
	dk, err := GenerateKey768()
	if err != nil {
		b.Fatal(err)
	}
	ek := dk.EncapsulationKey()
	c, _ := ek.Encapsulate()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		K := kemDecaps(dk, (*[CiphertextSize768]byte)(c))
		sink ^= K[0]
	}
}

func BenchmarkRoundTrip(b *testing.B) {
	dk, err := GenerateKey768()
	if err != nil {
		b.Fatal(err)
	}
	ek := dk.EncapsulationKey()
	ekBytes := ek.Bytes()
	c, _ := ek.Encapsulate()
	if err != nil {
		b.Fatal(err)
	}
	b.Run("Alice", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			dkS, err := GenerateKey768()
			if err != nil {
				b.Fatal(err)
			}
			ekS := dkS.EncapsulationKey().Bytes()
			sink ^= ekS[0]

			Ks, err := dk.Decapsulate(c)
			if err != nil {
				b.Fatal(err)
			}
			sink ^= Ks[0]
		}
	})
	b.Run("Bob", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ek, err := NewEncapsulationKey768(ekBytes)
			if err != nil {
				b.Fatal(err)
			}
			cS, Ks := ek.Encapsulate()
			if err != nil {
				b.Fatal(err)
			}
			sink ^= cS[0] ^ Ks[0]
		}
	})
}
