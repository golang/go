// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mlkem768

import (
	"bytes"
	"crypto/rand"
	_ "embed"
	"encoding/hex"
	"errors"
	"flag"
	"math/big"
	"strconv"
	"testing"

	"golang.org/x/crypto/sha3"
)

func TestFieldReduce(t *testing.T) {
	for a := uint32(0); a < 2*q*q; a++ {
		got := fieldReduce(a)
		exp := fieldElement(a % q)
		if got != exp {
			t.Fatalf("reduce(%d) = %d, expected %d", a, got, exp)
		}
	}
}

func TestFieldAdd(t *testing.T) {
	for a := fieldElement(0); a < q; a++ {
		for b := fieldElement(0); b < q; b++ {
			got := fieldAdd(a, b)
			exp := (a + b) % q
			if got != exp {
				t.Fatalf("%d + %d = %d, expected %d", a, b, got, exp)
			}
		}
	}
}

func TestFieldSub(t *testing.T) {
	for a := fieldElement(0); a < q; a++ {
		for b := fieldElement(0); b < q; b++ {
			got := fieldSub(a, b)
			exp := (a - b + q) % q
			if got != exp {
				t.Fatalf("%d - %d = %d, expected %d", a, b, got, exp)
			}
		}
	}
}

func TestFieldMul(t *testing.T) {
	for a := fieldElement(0); a < q; a++ {
		for b := fieldElement(0); b < q; b++ {
			got := fieldMul(a, b)
			exp := fieldElement((uint32(a) * uint32(b)) % q)
			if got != exp {
				t.Fatalf("%d * %d = %d, expected %d", a, b, got, exp)
			}
		}
	}
}

func TestDecompressCompress(t *testing.T) {
	for _, bits := range []uint8{1, 4, 10} {
		for a := uint16(0); a < 1<<bits; a++ {
			f := decompress(a, bits)
			if f >= q {
				t.Fatalf("decompress(%d, %d) = %d >= q", a, bits, f)
			}
			got := compress(f, bits)
			if got != a {
				t.Fatalf("compress(decompress(%d, %d), %d) = %d", a, bits, bits, got)
			}
		}

		for a := fieldElement(0); a < q; a++ {
			c := compress(a, bits)
			if c >= 1<<bits {
				t.Fatalf("compress(%d, %d) = %d >= 2^bits", a, bits, c)
			}
			got := decompress(c, bits)
			diff := min(a-got, got-a, a-got+q, got-a+q)
			ceil := q / (1 << bits)
			if diff > fieldElement(ceil) {
				t.Fatalf("decompress(compress(%d, %d), %d) = %d (diff %d, max diff %d)",
					a, bits, bits, got, diff, ceil)
			}
		}
	}
}

func CompressRat(x fieldElement, d uint8) uint16 {
	if x >= q {
		panic("x out of range")
	}
	if d <= 0 || d >= 12 {
		panic("d out of range")
	}

	precise := big.NewRat((1<<d)*int64(x), q) // (2ᵈ / q) * x == (2ᵈ * x) / q

	// FloatString rounds halves away from 0, and our result should always be positive,
	// so it should work as we expect. (There's no direct way to round a Rat.)
	rounded, err := strconv.ParseInt(precise.FloatString(0), 10, 64)
	if err != nil {
		panic(err)
	}

	// If we rounded up, `rounded` may be equal to 2ᵈ, so we perform a final reduction.
	return uint16(rounded % (1 << d))
}

func TestCompress(t *testing.T) {
	for d := 1; d < 12; d++ {
		for n := 0; n < q; n++ {
			expected := CompressRat(fieldElement(n), uint8(d))
			result := compress(fieldElement(n), uint8(d))
			if result != expected {
				t.Errorf("compress(%d, %d): got %d, expected %d", n, d, result, expected)
			}
		}
	}
}

func DecompressRat(y uint16, d uint8) fieldElement {
	if y >= 1<<d {
		panic("y out of range")
	}
	if d <= 0 || d >= 12 {
		panic("d out of range")
	}

	precise := big.NewRat(q*int64(y), 1<<d) // (q / 2ᵈ) * y  ==  (q * y) / 2ᵈ

	// FloatString rounds halves away from 0, and our result should always be positive,
	// so it should work as we expect. (There's no direct way to round a Rat.)
	rounded, err := strconv.ParseInt(precise.FloatString(0), 10, 64)
	if err != nil {
		panic(err)
	}

	// If we rounded up, `rounded` may be equal to q, so we perform a final reduction.
	return fieldElement(rounded % q)
}

func TestDecompress(t *testing.T) {
	for d := 1; d < 12; d++ {
		for n := 0; n < (1 << d); n++ {
			expected := DecompressRat(uint16(n), uint8(d))
			result := decompress(uint16(n), uint8(d))
			if result != expected {
				t.Errorf("decompress(%d, %d): got %d, expected %d", n, d, result, expected)
			}
		}
	}
}

func BitRev7(n uint8) uint8 {
	if n>>7 != 0 {
		panic("not 7 bits")
	}
	var r uint8
	r |= n >> 6 & 0b0000_0001
	r |= n >> 4 & 0b0000_0010
	r |= n >> 2 & 0b0000_0100
	r |= n /**/ & 0b0000_1000
	r |= n << 2 & 0b0001_0000
	r |= n << 4 & 0b0010_0000
	r |= n << 6 & 0b0100_0000
	return r
}

func TestZetas(t *testing.T) {
	ζ := big.NewInt(17)
	q := big.NewInt(q)
	for k, zeta := range zetas {
		// ζ^BitRev7(k) mod q
		exp := new(big.Int).Exp(ζ, big.NewInt(int64(BitRev7(uint8(k)))), q)
		if big.NewInt(int64(zeta)).Cmp(exp) != 0 {
			t.Errorf("zetas[%d] = %v, expected %v", k, zeta, exp)
		}
	}
}

func TestGammas(t *testing.T) {
	ζ := big.NewInt(17)
	q := big.NewInt(q)
	for k, gamma := range gammas {
		// ζ^2BitRev7(i)+1
		exp := new(big.Int).Exp(ζ, big.NewInt(int64(BitRev7(uint8(k)))*2+1), q)
		if big.NewInt(int64(gamma)).Cmp(exp) != 0 {
			t.Errorf("gammas[%d] = %v, expected %v", k, gamma, exp)
		}
	}
}

func TestRoundTrip(t *testing.T) {
	dk, err := GenerateKey()
	if err != nil {
		t.Fatal(err)
	}
	c, Ke, err := Encapsulate(dk.EncapsulationKey())
	if err != nil {
		t.Fatal(err)
	}
	Kd, err := Decapsulate(dk, c)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(Ke, Kd) {
		t.Fail()
	}

	dk1, err := GenerateKey()
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(dk.EncapsulationKey(), dk1.EncapsulationKey()) {
		t.Fail()
	}
	if bytes.Equal(dk.Bytes(), dk1.Bytes()) {
		t.Fail()
	}
	if bytes.Equal(dk.Bytes()[EncapsulationKeySize-32:], dk1.Bytes()[EncapsulationKeySize-32:]) {
		t.Fail()
	}

	c1, Ke1, err := Encapsulate(dk.EncapsulationKey())
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(c, c1) {
		t.Fail()
	}
	if bytes.Equal(Ke, Ke1) {
		t.Fail()
	}
}

func TestBadLengths(t *testing.T) {
	dk, err := GenerateKey()
	if err != nil {
		t.Fatal(err)
	}
	ek := dk.EncapsulationKey()

	for i := 0; i < len(ek)-1; i++ {
		if _, _, err := Encapsulate(ek[:i]); err == nil {
			t.Errorf("expected error for ek length %d", i)
		}
	}
	ekLong := ek
	for i := 0; i < 100; i++ {
		ekLong = append(ekLong, 0)
		if _, _, err := Encapsulate(ekLong); err == nil {
			t.Errorf("expected error for ek length %d", len(ekLong))
		}
	}

	c, _, err := Encapsulate(ek)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < len(dk.Bytes())-1; i++ {
		if _, err := NewKeyFromExtendedEncoding(dk.Bytes()[:i]); err == nil {
			t.Errorf("expected error for dk length %d", i)
		}
	}
	dkLong := dk.Bytes()
	for i := 0; i < 100; i++ {
		dkLong = append(dkLong, 0)
		if _, err := NewKeyFromExtendedEncoding(dkLong); err == nil {
			t.Errorf("expected error for dk length %d", len(dkLong))
		}
	}

	for i := 0; i < len(c)-1; i++ {
		if _, err := Decapsulate(dk, c[:i]); err == nil {
			t.Errorf("expected error for c length %d", i)
		}
	}
	cLong := c
	for i := 0; i < 100; i++ {
		cLong = append(cLong, 0)
		if _, err := Decapsulate(dk, cLong); err == nil {
			t.Errorf("expected error for c length %d", len(cLong))
		}
	}
}

func EncapsulateDerand(ek, m []byte) (c, K []byte, err error) {
	if len(m) != messageSize {
		return nil, nil, errors.New("bad message length")
	}
	return kemEncaps(nil, ek, (*[messageSize]byte)(m))
}

func DecapsulateFromBytes(dkBytes []byte, c []byte) ([]byte, error) {
	dk, err := NewKeyFromExtendedEncoding(dkBytes)
	if err != nil {
		return nil, err
	}
	return Decapsulate(dk, c)
}

func GenerateKeyDerand(t testing.TB, d, z []byte) ([]byte, *DecapsulationKey) {
	if len(d) != 32 || len(z) != 32 {
		t.Fatal("bad length")
	}
	dk := kemKeyGen(nil, (*[32]byte)(d), (*[32]byte)(z))
	return dk.EncapsulationKey(), dk
}

var millionFlag = flag.Bool("million", false, "run the million vector test")

// TestPQCrystalsAccumulated accumulates the 10k vectors generated by the
// reference implementation and checks the hash of the result, to avoid checking
// in 150MB of test vectors.
func TestPQCrystalsAccumulated(t *testing.T) {
	n := 10000
	expected := "f7db260e1137a742e05fe0db9525012812b004d29040a5b606aad3d134b548d3"
	if testing.Short() {
		n = 100
		expected = "8d0c478ead6037897a0da6be21e5399545babf5fc6dd10c061c99b7dee2bf0dc"
	}
	if *millionFlag {
		n = 1000000
		expected = "70090cc5842aad0ec43d5042c783fae9bc320c047b5dafcb6e134821db02384d"
	}

	s := sha3.NewShake128()
	o := sha3.NewShake128()
	d := make([]byte, 32)
	z := make([]byte, 32)
	msg := make([]byte, 32)
	ct1 := make([]byte, CiphertextSize)

	for i := 0; i < n; i++ {
		s.Read(d)
		s.Read(z)
		ek, dk := GenerateKeyDerand(t, d, z)
		o.Write(ek)
		o.Write(dk.Bytes())

		s.Read(msg)
		ct, k, err := EncapsulateDerand(ek, msg)
		if err != nil {
			t.Fatal(err)
		}
		o.Write(ct)
		o.Write(k)

		kk, err := Decapsulate(dk, ct)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(kk, k) {
			t.Errorf("k: got %x, expected %x", kk, k)
		}

		s.Read(ct1)
		k1, err := Decapsulate(dk, ct1)
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
	var dk DecapsulationKey
	var d, z [32]byte
	rand.Read(d[:])
	rand.Read(z[:])
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dk := kemKeyGen(&dk, &d, &z)
		sink ^= dk.EncapsulationKey()[0]
	}
}

func BenchmarkEncaps(b *testing.B) {
	d := make([]byte, 32)
	rand.Read(d)
	z := make([]byte, 32)
	rand.Read(z)
	var m [messageSize]byte
	rand.Read(m[:])
	ek, _ := GenerateKeyDerand(b, d, z)
	var c [CiphertextSize]byte
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c, K, err := kemEncaps(&c, ek, &m)
		if err != nil {
			b.Fatal(err)
		}
		sink ^= c[0] ^ K[0]
	}
}

func BenchmarkDecaps(b *testing.B) {
	d := make([]byte, 32)
	rand.Read(d)
	z := make([]byte, 32)
	rand.Read(z)
	m := make([]byte, 32)
	rand.Read(m)
	ek, dk := GenerateKeyDerand(b, d, z)
	c, _, err := EncapsulateDerand(ek, m)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		K := kemDecaps(dk, (*[CiphertextSize]byte)(c))
		sink ^= K[0]
	}
}

func BenchmarkRoundTrip(b *testing.B) {
	dk, err := GenerateKey()
	if err != nil {
		b.Fatal(err)
	}
	ek := dk.EncapsulationKey()
	c, _, err := Encapsulate(ek)
	if err != nil {
		b.Fatal(err)
	}
	b.Run("Alice", func { b ->
		for i := 0; i < b.N; i++ {
			dkS, err := GenerateKey()
			if err != nil {
				b.Fatal(err)
			}
			ekS := dkS.EncapsulationKey()
			sink ^= ekS[0]

			Ks, err := Decapsulate(dk, c)
			if err != nil {
				b.Fatal(err)
			}
			sink ^= Ks[0]
		}
	})
	b.Run("Bob", func { b ->
		for i := 0; i < b.N; i++ {
			cS, Ks, err := Encapsulate(ek)
			if err != nil {
				b.Fatal(err)
			}
			sink ^= cS[0] ^ Ks[0]
		}
	})
}
