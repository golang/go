// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aes

import (
	"crypto/aes";
	"fmt";
	"testing";
)

// See const.go for overview of math here.

// Test that powx is initialized correctly.
// (Can adapt this code to generate it too.)
func TestPowx(t *testing.T) {
	p := 1;
	for i := 0; i < len(powx); i++ {
		if powx[i] != byte(p) {
			t.Errorf("powx[%d] = %#x, want %#x", i, powx[i], p);
		}
		p <<= 1;
		if p & 0x100 != 0 {
			p ^= poly;
		}
	}
}

// Multiply b and c as GF(2) polynomials modulo poly
func mul(b, c uint32) uint32 {
	i := b;
	j := c;
	s := uint32(0);
	for k := uint32(1); k < 0x100 && j != 0; k <<= 1 {
		// Invariant: k == 1<<n, i == b * xⁿ

		if j & k != 0 {
			// s += i in GF(2); xor in binary
			s ^= i;
			j ^= k;	// turn off bit to end loop early
		}

		// i *= x in GF(2) modulo the polynomial
		i <<= 1;
		if i & 0x100 != 0 {
			i ^= poly;
		}
	}
	return s;
}

// Test all mul inputs against bit-by-bit n² algorithm.
func TestMul(t *testing.T) {
	for i := uint32(0); i < 256; i++ {
		for j := uint32(0); j < 256; j++ {
			// Multiply i, j bit by bit.
			s := uint8(0);
			for k := uint(0); k < 8; k++ {
				for l := uint(0); l < 8; l++ {
					if i & (1<<k) != 0 && j & (1<<l) != 0 {
						s ^= powx[k+l];
					}
				}
			}
			if x := mul(i, j); x != uint32(s) {
				t.Fatalf("mul(%#x, %#x) = %#x, want %#x", i, j, x, s);
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
			t.Errorf("sbox0[sbox1[%#x]] = %#x", i, j);
		}
		if j := sbox1[sbox0[i]]; j != byte(i) {
			t.Errorf("sbox1[sbox0[%#x]] = %#x", i, j);
		}
	}
}

// Test that encryption tables are correct.
// (Can adapt this code to generate them too.)
func TestTe(t *testing.T) {
	for i := 0; i < 256; i++ {
		s := uint32(sbox0[i]);
		s2 := mul(s, 2);
		s3 := mul(s, 3);
		w := s2<<24 | s<<16 | s<<8 | s3;
		for j := 0; j < 4; j++ {
			if x := te[j][i]; x != w {
				t.Fatalf("te[%d][%d] = %#x, want %#x", j, i, x, w);
			}
			w = w<<24 | w>>8;
		}
	}
}

// Test that decryption tables are correct.
// (Can adapt this code to generate them too.)
func TestTd(t *testing.T) {
	for i := 0; i < 256; i++ {
		s := uint32(sbox1[i]);
		s9 := mul(s, 0x9);
		sb := mul(s, 0xb);
		sd := mul(s, 0xd);
		se := mul(s, 0xe);
		w := se<<24 | s9<<16 | sd<<8 | sb;
		for j := 0; j < 4; j++ {
			if x := td[j][i]; x != w {
				t.Fatalf("td[%d][%d] = %#x, want %#x", j, i, x, w);
			}
			w = w<<24 | w>>8;
		}
	}
}

// Test vectors are from FIPS 197:
//	http://www.csrc.nist.gov/publications/fips/fips197/fips-197.pdf

// Appendix A of FIPS 197: Key expansion examples
type KeyTest struct {
	key []uint32;
	enc []uint32;
	dec []uint32;	// decryption expansion; not in FIPS 197, computed from C implementation.
}

var keyTests = []KeyTest {
	KeyTest {
		// A.1.  Expansion of a 128-bit Cipher Key
		[]uint32 {
			0x2b7e1516, 0x28aed2a6, 0xabf71588, 0x09cf4f3c
		},
		[]uint32 {
			0x2b7e1516, 0x28aed2a6, 0xabf71588, 0x09cf4f3c,
			0xa0fafe17, 0x88542cb1, 0x23a33939, 0x2a6c7605,
			0xf2c295f2, 0x7a96b943, 0x5935807a, 0x7359f67f,
			0x3d80477d, 0x4716fe3e, 0x1e237e44, 0x6d7a883b,
			0xef44a541, 0xa8525b7f, 0xb671253b, 0xdb0bad00,
			0xd4d1c6f8, 0x7c839d87, 0xcaf2b8bc, 0x11f915bc,
			0x6d88a37a, 0x110b3efd, 0xdbf98641, 0xca0093fd,
			0x4e54f70e, 0x5f5fc9f3, 0x84a64fb2, 0x4ea6dc4f,
			0xead27321, 0xb58dbad2, 0x312bf560, 0x7f8d292f,
			0xac7766f3, 0x19fadc21, 0x28d12941, 0x575c006e,
			0xd014f9a8, 0xc9ee2589, 0xe13f0cc8, 0xb6630ca6,
		},
		[]uint32 {
			0xd014f9a8, 0xc9ee2589, 0xe13f0cc8, 0xb6630ca6,
			0xc7b5a63, 0x1319eafe, 0xb0398890, 0x664cfbb4,
			0xdf7d925a, 0x1f62b09d, 0xa320626e, 0xd6757324,
			0x12c07647, 0xc01f22c7, 0xbc42d2f3, 0x7555114a,
			0x6efcd876, 0xd2df5480, 0x7c5df034, 0xc917c3b9,
			0x6ea30afc, 0xbc238cf6, 0xae82a4b4, 0xb54a338d,
			0x90884413, 0xd280860a, 0x12a12842, 0x1bc89739,
			0x7c1f13f7, 0x4208c219, 0xc021ae48, 0x969bf7b,
			0xcc7505eb, 0x3e17d1ee, 0x82296c51, 0xc9481133,
			0x2b3708a7, 0xf262d405, 0xbc3ebdbf, 0x4b617d62,
			0x2b7e1516, 0x28aed2a6, 0xabf71588, 0x9cf4f3c,
		},
	},
	KeyTest {
		// A.2.  Expansion of a 192-bit Cipher Key
		[]uint32 {
			0x8e73b0f7, 0xda0e6452, 0xc810f32b, 0x809079e5,
			0x62f8ead2, 0x522c6b7b,
		},
		[]uint32 {
			0x8e73b0f7, 0xda0e6452, 0xc810f32b, 0x809079e5,
			0x62f8ead2, 0x522c6b7b, 0xfe0c91f7, 0x2402f5a5,
			0xec12068e, 0x6c827f6b, 0x0e7a95b9, 0x5c56fec2,
			0x4db7b4bd, 0x69b54118, 0x85a74796, 0xe92538fd,
			0xe75fad44, 0xbb095386, 0x485af057, 0x21efb14f,
			0xa448f6d9, 0x4d6dce24, 0xaa326360, 0x113b30e6,
			0xa25e7ed5, 0x83b1cf9a, 0x27f93943, 0x6a94f767,
			0xc0a69407, 0xd19da4e1, 0xec1786eb, 0x6fa64971,
			0x485f7032, 0x22cb8755, 0xe26d1352, 0x33f0b7b3,
			0x40beeb28, 0x2f18a259, 0x6747d26b, 0x458c553e,
			0xa7e1466c, 0x9411f1df, 0x821f750a, 0xad07d753,
			0xca400538, 0x8fcc5006, 0x282d166a, 0xbc3ce7b5,
			0xe98ba06f, 0x448c773c, 0x8ecc7204, 0x01002202,
		},
		nil,
	},
	KeyTest {
		// A.3.  Expansion of a 256-bit Cipher Key
		[]uint32 {
			0x603deb10, 0x15ca71be, 0x2b73aef0, 0x857d7781,
			0x1f352c07, 0x3b6108d7, 0x2d9810a3, 0x0914dff4,
		},
		[]uint32 {
			0x603deb10, 0x15ca71be, 0x2b73aef0, 0x857d7781,
			0x1f352c07, 0x3b6108d7, 0x2d9810a3, 0x0914dff4,
			0x9ba35411, 0x8e6925af, 0xa51a8b5f, 0x2067fcde,
			0xa8b09c1a, 0x93d194cd, 0xbe49846e, 0xb75d5b9a,
			0xd59aecb8, 0x5bf3c917, 0xfee94248, 0xde8ebe96,
			0xb5a9328a, 0x2678a647, 0x98312229, 0x2f6c79b3,
			0x812c81ad, 0xdadf48ba, 0x24360af2, 0xfab8b464,
			0x98c5bfc9, 0xbebd198e, 0x268c3ba7, 0x09e04214,
			0x68007bac, 0xb2df3316, 0x96e939e4, 0x6c518d80,
			0xc814e204, 0x76a9fb8a, 0x5025c02d, 0x59c58239,
			0xde136967, 0x6ccc5a71, 0xfa256395, 0x9674ee15,
			0x5886ca5d, 0x2e2f31d7, 0x7e0af1fa, 0x27cf73c3,
			0x749c47ab, 0x18501dda, 0xe2757e4f, 0x7401905a,
			0xcafaaae3, 0xe4d59b34, 0x9adf6ace, 0xbd10190d,
			0xfe4890d1, 0xe6188d0b, 0x046df344, 0x706c631e,
		},
		nil,
	},
}

// Test key expansion against FIPS 197 examples.
func TestExpandKey(t *testing.T) {
L:
	for i, tt := range keyTests {
		enc := make([]uint32, len(tt.enc));
		var dec []uint32;
		if tt.dec != nil {
			dec = make([]uint32, len(tt.dec));
		}
		expandKey(tt.key, enc, dec);
		for j, v := range enc {
			if v != tt.enc[j] {
				t.Errorf("key %d: enc[%d] = %#x, want %#x", i, j, v, tt.enc[j]);
				continue L;
			}
		}
		if dec != nil {
			for j, v := range dec {
				if v != tt.dec[j] {
					t.Errorf("key %d: dec[%d] = %#x, want %#x", i, j, v, tt.dec[j]);
					continue L;
				}
			}
		}
	}
}

// Appendix B, C of FIPS 197: Cipher examples, Example vectors.
type CryptTest struct {
	key []uint32;
	in []uint32;
	out []uint32;
}

var encryptTests = []CryptTest {
	CryptTest {
		// Appendix B.
		[]uint32 { 0x2b7e1516, 0x28aed2a6, 0xabf71588, 0x09cf4f3c, },
		[]uint32 { 0x3243f6a8, 0x885a308d, 0x313198a2, 0xe0370734, },
		[]uint32 { 0x3925841d, 0x02dc09fb, 0xdc118597, 0x196a0b32, },
	},
	CryptTest {
		// Appendix C.1.  AES-128
		[]uint32 { 0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f, },
		[]uint32 { 0x00112233, 0x44556677, 0x8899aabb, 0xccddeeff, },
		[]uint32 { 0x69c4e0d8, 0x6a7b0430, 0xd8cdb780, 0x70b4c55a, },
	},
	CryptTest {
		// Appendix C.2.  AES-192
		[]uint32 { 0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f,
			0x10111213, 0x14151617, },
		[]uint32 { 0x00112233, 0x44556677, 0x8899aabb, 0xccddeeff, },
		[]uint32 { 0xdda97ca4, 0x864cdfe0, 0x6eaf70a0, 0xec0d7191, },
	},
	CryptTest {
		// Appendix C.3.  AES-256
		[]uint32 { 0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f,
			0x10111213, 0x14151617, 0x18191a1b, 0x1c1d1e1f, },
		[]uint32 { 0x00112233, 0x44556677, 0x8899aabb, 0xccddeeff, },
		[]uint32 { 0x8ea2b7ca, 0x516745bf, 0xeafc4990, 0x4b496089, },
	},
}

// Test encryption against FIPS 197 examples.
func TestEncrypt(t *testing.T) {
	for i, tt := range encryptTests {
		n := 4*(len(tt.key) + 7);
		enc := make([]uint32, n);
		dec := make([]uint32, n);
		expandKey(tt.key, enc, dec);
		out := make([]uint32, len(tt.in));
		encryptBlock(enc, tt.in, out);
		for j, v := range out {
			if v != tt.out[j] {
				t.Errorf("encrypt %d: out[%d] = %#x, want %#x", i, j, v, tt.out[j]);
				break;
			}
		}
	}
}

// Test decryption against FIPS 197 examples.
func TestDecrypt(t *testing.T) {
	for i, tt := range encryptTests {
		n := 4*(len(tt.key) + 7);
		enc := make([]uint32, n);
		dec := make([]uint32, n);
		expandKey(tt.key, enc, dec);
		plain := make([]uint32, len(tt.in));
		decryptBlock(dec, tt.out, plain);
		for j, v := range plain {
			if v != tt.in[j] {
				t.Errorf("decrypt %d: plain[%d] = %#x, want %#x", i, j, v, tt.in[j]);
				break;
			}
		}
	}
}

