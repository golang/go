// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mlkem

import (
	"bytes"
	"crypto/rand"
	"math/big"
	mathrand "math/rand/v2"
	"strconv"
	"testing"
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

func randomRingElement() ringElement {
	var r ringElement
	for i := range r {
		r[i] = fieldElement(mathrand.IntN(q))
	}
	return r
}

// ringCompressAndEncode appends an encoding of a ring element to s,
// compressing each coefficient to d bits.
//
// It implements Compress, according to FIPS 203, Definition 4.7,
// followed by ByteEncode, according to FIPS 203, Algorithm 5.
func ringCompressAndEncode(s []byte, f ringElement, d uint8) []byte {
	var b byte
	var bIdx uint8
	for i := 0; i < n; i++ {
		c := compress(f[i], d)
		var cIdx uint8
		for cIdx < d {
			b |= byte(c>>cIdx) << bIdx
			bits := min(8-bIdx, d-cIdx)
			bIdx += bits
			cIdx += bits
			if bIdx == 8 {
				s = append(s, b)
				b = 0
				bIdx = 0
			}
		}
	}
	if bIdx != 0 {
		panic("mlkem: internal error: bitsFilled != 0")
	}
	return s
}

// ringDecodeAndDecompress decodes an encoding of a ring element where
// each d bits are mapped to an equidistant distribution.
//
// It implements ByteDecode, according to FIPS 203, Algorithm 6,
// followed by Decompress, according to FIPS 203, Definition 4.8.
func ringDecodeAndDecompress(b []byte, d uint8) ringElement {
	var f ringElement
	var bIdx uint8
	for i := 0; i < n; i++ {
		var c uint16
		var cIdx uint8
		for cIdx < d {
			c |= uint16(b[0]>>bIdx) << cIdx
			c &= (1 << d) - 1
			bits := min(8-bIdx, d-cIdx)
			bIdx += bits
			cIdx += bits
			if bIdx == 8 {
				b = b[1:]
				bIdx = 0
			}
		}
		f[i] = fieldElement(decompress(c, d))
	}
	if len(b) != 0 {
		panic("mlkem: internal error: leftover bytes")
	}
	return f
}

func TestEncodeDecode(t *testing.T) {
	f := randomRingElement()
	b := make([]byte, 12*n/8)
	rand.Read(b)

	// Compare ringCompressAndEncode to ringCompressAndEncodeN.
	e1 := ringCompressAndEncode(nil, f, 10)
	e2 := ringCompressAndEncode10(nil, f)
	if !bytes.Equal(e1, e2) {
		t.Errorf("ringCompressAndEncode = %x, ringCompressAndEncode10 = %x", e1, e2)
	}
	e1 = ringCompressAndEncode(nil, f, 4)
	e2 = ringCompressAndEncode4(nil, f)
	if !bytes.Equal(e1, e2) {
		t.Errorf("ringCompressAndEncode = %x, ringCompressAndEncode4 = %x", e1, e2)
	}
	e1 = ringCompressAndEncode(nil, f, 1)
	e2 = ringCompressAndEncode1(nil, f)
	if !bytes.Equal(e1, e2) {
		t.Errorf("ringCompressAndEncode = %x, ringCompressAndEncode1 = %x", e1, e2)
	}
	e1 = ringCompressAndEncode(nil, f, 11)
	e2 = ringCompressAndEncode11(nil, f)
	if !bytes.Equal(e1, e2) {
		t.Errorf("ringCompressAndEncode = %x, ringCompressAndEncode11 = %x", e1, e2)
	}
	e1 = ringCompressAndEncode(nil, f, 5)
	e2 = ringCompressAndEncode5(nil, f)
	if !bytes.Equal(e1, e2) {
		t.Errorf("ringCompressAndEncode = %x, ringCompressAndEncode5 = %x", e1, e2)
	}

	// Compare ringDecodeAndDecompress to ringDecodeAndDecompressN.
	g1 := ringDecodeAndDecompress(b[:encodingSize10], 10)
	g2 := ringDecodeAndDecompress10((*[encodingSize10]byte)(b))
	if g1 != g2 {
		t.Errorf("ringDecodeAndDecompress = %v, ringDecodeAndDecompress10 = %v", g1, g2)
	}
	g1 = ringDecodeAndDecompress(b[:encodingSize4], 4)
	g2 = ringDecodeAndDecompress4((*[encodingSize4]byte)(b))
	if g1 != g2 {
		t.Errorf("ringDecodeAndDecompress = %v, ringDecodeAndDecompress4 = %v", g1, g2)
	}
	g1 = ringDecodeAndDecompress(b[:encodingSize1], 1)
	g2 = ringDecodeAndDecompress1((*[encodingSize1]byte)(b))
	if g1 != g2 {
		t.Errorf("ringDecodeAndDecompress = %v, ringDecodeAndDecompress1 = %v", g1, g2)
	}
	g1 = ringDecodeAndDecompress(b[:encodingSize11], 11)
	g2 = ringDecodeAndDecompress11((*[encodingSize11]byte)(b))
	if g1 != g2 {
		t.Errorf("ringDecodeAndDecompress = %v, ringDecodeAndDecompress11 = %v", g1, g2)
	}
	g1 = ringDecodeAndDecompress(b[:encodingSize5], 5)
	g2 = ringDecodeAndDecompress5((*[encodingSize5]byte)(b))
	if g1 != g2 {
		t.Errorf("ringDecodeAndDecompress = %v, ringDecodeAndDecompress5 = %v", g1, g2)
	}

	// Round-trip ringCompressAndEncode and ringDecodeAndDecompress.
	for d := 1; d < 12; d++ {
		encodingSize := d * n / 8
		g := ringDecodeAndDecompress(b[:encodingSize], uint8(d))
		out := ringCompressAndEncode(nil, g, uint8(d))
		if !bytes.Equal(out, b[:encodingSize]) {
			t.Errorf("roundtrip failed for d = %d", d)
		}
	}

	// Round-trip ringCompressAndEncodeN and ringDecodeAndDecompressN.
	g := ringDecodeAndDecompress10((*[encodingSize10]byte)(b))
	out := ringCompressAndEncode10(nil, g)
	if !bytes.Equal(out, b[:encodingSize10]) {
		t.Errorf("roundtrip failed for specialized 10")
	}
	g = ringDecodeAndDecompress4((*[encodingSize4]byte)(b))
	out = ringCompressAndEncode4(nil, g)
	if !bytes.Equal(out, b[:encodingSize4]) {
		t.Errorf("roundtrip failed for specialized 4")
	}
	g = ringDecodeAndDecompress1((*[encodingSize1]byte)(b))
	out = ringCompressAndEncode1(nil, g)
	if !bytes.Equal(out, b[:encodingSize1]) {
		t.Errorf("roundtrip failed for specialized 1")
	}
	g = ringDecodeAndDecompress11((*[encodingSize11]byte)(b))
	out = ringCompressAndEncode11(nil, g)
	if !bytes.Equal(out, b[:encodingSize11]) {
		t.Errorf("roundtrip failed for specialized 11")
	}
	g = ringDecodeAndDecompress5((*[encodingSize5]byte)(b))
	out = ringCompressAndEncode5(nil, g)
	if !bytes.Equal(out, b[:encodingSize5]) {
		t.Errorf("roundtrip failed for specialized 5")
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

// nttMul multiplies two nttElements.
//
// It implements MultiplyNTTs, according to FIPS 203, Algorithm 11.
func nttMul(f, g nttElement) nttElement {
	var h nttElement
	for i := 0; i < n; i += 2 {
		a0, a1 := f[i], f[i+1]
		b0, b1 := g[i], g[i+1]
		h[i] = fieldAddMul(a0, b0, fieldMul(a1, b1), gammas[i/2])
		h[i+1] = fieldAddMul(a0, b1, a1, b0)
	}
	return h
}

func TestNTTMulAdd(t *testing.T) {
	impls := []struct {
		name string
		fn   func(h, f, g *nttElement)
	}{
		{"nttMulAdd", nttMulAdd},
		{"nttMulAddPrecomputed", func(h, f, g *nttElement) {
			var gamma [n / 2]fieldElement
			nttMulPrecompute(&gamma, g)
			nttMulAddPrecomputed(h, f, g, &gamma)
		}},
	}

	var zero, one, max nttElement
	for i := range max {
		one[i] = 1
		max[i] = q - 1
	}
	inputs := [][3]nttElement{
		{max, max, max},
		{zero, zero, zero},
		{one, one, one},
		{
			nttElement(randomRingElement()),
			nttElement(randomRingElement()),
			nttElement(randomRingElement()),
		},
	}

	for _, impl := range impls {
		t.Run(impl.name, func(t *testing.T) {
			for _, in := range inputs {
				h, f, g := in[0], in[1], in[2]

				x, y, z := h, f, g
				impl.fn(&x, &y, &z)
				if exp := polyAdd(h, nttMul(f, g)); x != exp {
					t.Fatalf("h += f*g: got %v, expected %v", x, exp)
				}
				if y != f || z != g {
					t.Errorf("h += f*g: inputs were modified")
				}

				x, z = f, g
				impl.fn(&x, &x, &z)
				if exp := polyAdd(f, nttMul(f, g)); x != exp {
					t.Fatalf("f += f*g: got %v, expected %v", x, exp)
				}
				if z != g {
					t.Errorf("f += f*g: g was modified")
				}

				x, y = g, f
				impl.fn(&x, &y, &x)
				if exp := polyAdd(g, nttMul(f, g)); x != exp {
					t.Fatalf("g += f*g: got %v, expected %v", x, exp)
				}
				if y != f {
					t.Errorf("g += f*g: f was modified")
				}

				x = f
				impl.fn(&x, &x, &x)
				if exp := polyAdd(f, nttMul(f, f)); x != exp {
					t.Fatalf("f += f*f: got %v, expected %v", x, exp)
				}
			}
		})
	}
}
