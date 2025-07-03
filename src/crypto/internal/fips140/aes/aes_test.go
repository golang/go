// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aes

import "testing"

// See const.go for overview of math here.

// Test that powx is initialized correctly.
// (Can adapt this code to generate it too.)
func TestPowx(t *testing.T) {
	p := 1
	for i := 0; i < len(powx); i++ {
		if powx[i] != byte(p) {
			t.Errorf("powx[%d] = %#x, want %#x", i, powx[i], p)
		}
		p <<= 1
		if p&0x100 != 0 {
			p ^= poly
		}
	}
}

// Multiply b and c as GF(2) polynomials modulo poly
func mul(b, c uint32) uint32 {
	i := b
	j := c
	s := uint32(0)
	for k := uint32(1); k < 0x100 && j != 0; k <<= 1 {
		// Invariant: k == 1<<n, i == b * xⁿ

		if j&k != 0 {
			// s += i in GF(2); xor in binary
			s ^= i
			j ^= k // turn off bit to end loop early
		}

		// i *= x in GF(2) modulo the polynomial
		i <<= 1
		if i&0x100 != 0 {
			i ^= poly
		}
	}
	return s
}

// Test all mul inputs against bit-by-bit n² algorithm.
func TestMul(t *testing.T) {
	for i := uint32(0); i < 256; i++ {
		for j := uint32(0); j < 256; j++ {
			// Multiply i, j bit by bit.
			s := uint8(0)
			for k := uint(0); k < 8; k++ {
				for l := uint(0); l < 8; l++ {
					if i&(1<<k) != 0 && j&(1<<l) != 0 {
						s ^= powx[k+l]
					}
				}
			}
			if x := mul(i, j); x != uint32(s) {
				t.Fatalf("mul(%#x, %#x) = %#x, want %#x", i, j, x, s)
			}
		}
	}
}

// Check that S-boxes are inverses of each other.
// They have more structure that we could test,
// but if this sanity check passes, we'll assume
// the cut and paste from the FIPS PDF worked.
func TestSboxes(t *testing.T) {
	for i := 0; i < 256; i++ {
		if j := sbox0[sbox1[i]]; j != byte(i) {
			t.Errorf("sbox0[sbox1[%#x]] = %#x", i, j)
		}
		if j := sbox1[sbox0[i]]; j != byte(i) {
			t.Errorf("sbox1[sbox0[%#x]] = %#x", i, j)
		}
	}
}

// Test that encryption tables are correct.
// (Can adapt this code to generate them too.)
func TestTe(t *testing.T) {
	for i := 0; i < 256; i++ {
		s := uint32(sbox0[i])
		s2 := mul(s, 2)
		s3 := mul(s, 3)
		w := s2<<24 | s<<16 | s<<8 | s3
		te := [][256]uint32{te0, te1, te2, te3}
		for j := 0; j < 4; j++ {
			if x := te[j][i]; x != w {
				t.Fatalf("te[%d][%d] = %#x, want %#x", j, i, x, w)
			}
			w = w<<24 | w>>8
		}
	}
}

// Test that decryption tables are correct.
// (Can adapt this code to generate them too.)
func TestTd(t *testing.T) {
	for i := 0; i < 256; i++ {
		s := uint32(sbox1[i])
		s9 := mul(s, 0x9)
		sb := mul(s, 0xb)
		sd := mul(s, 0xd)
		se := mul(s, 0xe)
		w := se<<24 | s9<<16 | sd<<8 | sb
		td := [][256]uint32{td0, td1, td2, td3}
		for j := 0; j < 4; j++ {
			if x := td[j][i]; x != w {
				t.Fatalf("td[%d][%d] = %#x, want %#x", j, i, x, w)
			}
			w = w<<24 | w>>8
		}
	}
}
