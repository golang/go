// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements Bruce Schneier's Twofish encryption algorithm.
package twofish

// Twofish is defined in http://www.schneier.com/paper-twofish-paper.pdf [TWOFISH]

// This code is a port of the LibTom C implementation.
// See http://libtom.org/?page=features&newsitems=5&whatfile=crypt.
// LibTomCrypt is free for all purposes under the public domain.
// It was heavily inspired by the go blowfish package.

import (
	"os"
	"strconv"
)

// BlockSize is the constant block size of Twofish.
const BlockSize = 16

const mdsPolynomial = 0x169 // x^8 + x^6 + x^5 + x^3 + 1, see [TWOFISH] 4.2
const rsPolynomial = 0x14d  // x^8 + x^6 + x^3 + x^2 + 1, see [TWOFISH] 4.3

// A Cipher is an instance of Twofish encryption using a particular key.
type Cipher struct {
	s [4][256]uint32
	k [40]uint32
}

type KeySizeError int

func (k KeySizeError) String() string {
	return "crypto/twofish: invalid key size " + strconv.Itoa(int(k))
}

// NewCipher creates and returns a Cipher.
// The key argument should be the Twofish key, 16, 24 or 32 bytes.
func NewCipher(key []byte) (*Cipher, os.Error) {
	keylen := len(key)

	if keylen != 16 && keylen != 24 && keylen != 32 {
		return nil, KeySizeError(keylen)
	}

	// k is the number of 64 bit words in key
	k := keylen / 8

	// Create the S[..] words
	var S [4 * 4]byte
	for i := 0; i < k; i++ {
		// Computes [y0 y1 y2 y3] = rs . [x0 x1 x2 x3 x4 x5 x6 x7]
		for j, rsRow := range rs {
			for k, rsVal := range rsRow {
				S[4*i+j] ^= gfMult(key[8*i+k], rsVal, rsPolynomial)
			}
		}
	}

	// Calculate subkeys
	c := new(Cipher)
	var tmp [4]byte
	for i := byte(0); i < 20; i++ {
		// A = h(p * 2x, Me)
		for j := range tmp {
			tmp[j] = 2 * i
		}
		A := h(tmp[:], key, 0)

		// B = rolc(h(p * (2x + 1), Mo), 8)
		for j := range tmp {
			tmp[j] = 2*i + 1
		}
		B := h(tmp[:], key, 1)
		B = rol(B, 8)

		c.k[2*i] = A + B

		// K[2i+1] = (A + 2B) <<< 9
		c.k[2*i+1] = rol(2*B+A, 9)
	}

	// Calculate sboxes
	switch k {
	case 2:
		for i := range c.s[0] {
			c.s[0][i] = mdsColumnMult(sbox[1][sbox[0][sbox[0][byte(i)]^S[0]]^S[4]], 0)
			c.s[1][i] = mdsColumnMult(sbox[0][sbox[0][sbox[1][byte(i)]^S[1]]^S[5]], 1)
			c.s[2][i] = mdsColumnMult(sbox[1][sbox[1][sbox[0][byte(i)]^S[2]]^S[6]], 2)
			c.s[3][i] = mdsColumnMult(sbox[0][sbox[1][sbox[1][byte(i)]^S[3]]^S[7]], 3)
		}
	case 3:
		for i := range c.s[0] {
			c.s[0][i] = mdsColumnMult(sbox[1][sbox[0][sbox[0][sbox[1][byte(i)]^S[0]]^S[4]]^S[8]], 0)
			c.s[1][i] = mdsColumnMult(sbox[0][sbox[0][sbox[1][sbox[1][byte(i)]^S[1]]^S[5]]^S[9]], 1)
			c.s[2][i] = mdsColumnMult(sbox[1][sbox[1][sbox[0][sbox[0][byte(i)]^S[2]]^S[6]]^S[10]], 2)
			c.s[3][i] = mdsColumnMult(sbox[0][sbox[1][sbox[1][sbox[0][byte(i)]^S[3]]^S[7]]^S[11]], 3)
		}
	default:
		for i := range c.s[0] {
			c.s[0][i] = mdsColumnMult(sbox[1][sbox[0][sbox[0][sbox[1][sbox[1][byte(i)]^S[0]]^S[4]]^S[8]]^S[12]], 0)
			c.s[1][i] = mdsColumnMult(sbox[0][sbox[0][sbox[1][sbox[1][sbox[0][byte(i)]^S[1]]^S[5]]^S[9]]^S[13]], 1)
			c.s[2][i] = mdsColumnMult(sbox[1][sbox[1][sbox[0][sbox[0][sbox[0][byte(i)]^S[2]]^S[6]]^S[10]]^S[14]], 2)
			c.s[3][i] = mdsColumnMult(sbox[0][sbox[1][sbox[1][sbox[0][sbox[1][byte(i)]^S[3]]^S[7]]^S[11]]^S[15]], 3)
		}
	}

	return c, nil
}

// Reset zeros the key data, so that it will no longer appear in the process's
// memory.
func (c *Cipher) Reset() {
	for i := range c.k {
		c.k[i] = 0
	}
	for i := range c.s {
		for j := 0; j < 265; j++ {
			c.s[i][j] = 0
		}
	}
}

// BlockSize returns the Twofish block size, 16 bytes.
func (c *Cipher) BlockSize() int { return BlockSize }

// store32l stores src in dst in little-endian form.
func store32l(dst []byte, src uint32) {
	dst[0] = byte(src)
	dst[1] = byte(src >> 8)
	dst[2] = byte(src >> 16)
	dst[3] = byte(src >> 24)
	return
}

// load32l reads a little-endian uint32 from src.
func load32l(src []byte) uint32 {
	return uint32(src[0]) | uint32(src[1])<<8 | uint32(src[2])<<16 | uint32(src[3])<<24
}

// rol returns x after a left circular rotation of y bits.
func rol(x, y uint32) uint32 {
	return (x << (y & 31)) | (x >> (32 - (y & 31)))
}

// ror returns x after a right circular rotation of y bits.
func ror(x, y uint32) uint32 {
	return (x >> (y & 31)) | (x << (32 - (y & 31)))
}

// The RS matrix. See [TWOFISH] 4.3
var rs = [4][8]byte{
	{0x01, 0xA4, 0x55, 0x87, 0x5A, 0x58, 0xDB, 0x9E},
	{0xA4, 0x56, 0x82, 0xF3, 0x1E, 0xC6, 0x68, 0xE5},
	{0x02, 0xA1, 0xFC, 0xC1, 0x47, 0xAE, 0x3D, 0x19},
	{0xA4, 0x55, 0x87, 0x5A, 0x58, 0xDB, 0x9E, 0x03},
}

// sbox tables
var sbox = [2][256]byte{
	{
		0xa9, 0x67, 0xb3, 0xe8, 0x04, 0xfd, 0xa3, 0x76, 0x9a, 0x92, 0x80, 0x78, 0xe4, 0xdd, 0xd1, 0x38,
		0x0d, 0xc6, 0x35, 0x98, 0x18, 0xf7, 0xec, 0x6c, 0x43, 0x75, 0x37, 0x26, 0xfa, 0x13, 0x94, 0x48,
		0xf2, 0xd0, 0x8b, 0x30, 0x84, 0x54, 0xdf, 0x23, 0x19, 0x5b, 0x3d, 0x59, 0xf3, 0xae, 0xa2, 0x82,
		0x63, 0x01, 0x83, 0x2e, 0xd9, 0x51, 0x9b, 0x7c, 0xa6, 0xeb, 0xa5, 0xbe, 0x16, 0x0c, 0xe3, 0x61,
		0xc0, 0x8c, 0x3a, 0xf5, 0x73, 0x2c, 0x25, 0x0b, 0xbb, 0x4e, 0x89, 0x6b, 0x53, 0x6a, 0xb4, 0xf1,
		0xe1, 0xe6, 0xbd, 0x45, 0xe2, 0xf4, 0xb6, 0x66, 0xcc, 0x95, 0x03, 0x56, 0xd4, 0x1c, 0x1e, 0xd7,
		0xfb, 0xc3, 0x8e, 0xb5, 0xe9, 0xcf, 0xbf, 0xba, 0xea, 0x77, 0x39, 0xaf, 0x33, 0xc9, 0x62, 0x71,
		0x81, 0x79, 0x09, 0xad, 0x24, 0xcd, 0xf9, 0xd8, 0xe5, 0xc5, 0xb9, 0x4d, 0x44, 0x08, 0x86, 0xe7,
		0xa1, 0x1d, 0xaa, 0xed, 0x06, 0x70, 0xb2, 0xd2, 0x41, 0x7b, 0xa0, 0x11, 0x31, 0xc2, 0x27, 0x90,
		0x20, 0xf6, 0x60, 0xff, 0x96, 0x5c, 0xb1, 0xab, 0x9e, 0x9c, 0x52, 0x1b, 0x5f, 0x93, 0x0a, 0xef,
		0x91, 0x85, 0x49, 0xee, 0x2d, 0x4f, 0x8f, 0x3b, 0x47, 0x87, 0x6d, 0x46, 0xd6, 0x3e, 0x69, 0x64,
		0x2a, 0xce, 0xcb, 0x2f, 0xfc, 0x97, 0x05, 0x7a, 0xac, 0x7f, 0xd5, 0x1a, 0x4b, 0x0e, 0xa7, 0x5a,
		0x28, 0x14, 0x3f, 0x29, 0x88, 0x3c, 0x4c, 0x02, 0xb8, 0xda, 0xb0, 0x17, 0x55, 0x1f, 0x8a, 0x7d,
		0x57, 0xc7, 0x8d, 0x74, 0xb7, 0xc4, 0x9f, 0x72, 0x7e, 0x15, 0x22, 0x12, 0x58, 0x07, 0x99, 0x34,
		0x6e, 0x50, 0xde, 0x68, 0x65, 0xbc, 0xdb, 0xf8, 0xc8, 0xa8, 0x2b, 0x40, 0xdc, 0xfe, 0x32, 0xa4,
		0xca, 0x10, 0x21, 0xf0, 0xd3, 0x5d, 0x0f, 0x00, 0x6f, 0x9d, 0x36, 0x42, 0x4a, 0x5e, 0xc1, 0xe0,
	},
	{
		0x75, 0xf3, 0xc6, 0xf4, 0xdb, 0x7b, 0xfb, 0xc8, 0x4a, 0xd3, 0xe6, 0x6b, 0x45, 0x7d, 0xe8, 0x4b,
		0xd6, 0x32, 0xd8, 0xfd, 0x37, 0x71, 0xf1, 0xe1, 0x30, 0x0f, 0xf8, 0x1b, 0x87, 0xfa, 0x06, 0x3f,
		0x5e, 0xba, 0xae, 0x5b, 0x8a, 0x00, 0xbc, 0x9d, 0x6d, 0xc1, 0xb1, 0x0e, 0x80, 0x5d, 0xd2, 0xd5,
		0xa0, 0x84, 0x07, 0x14, 0xb5, 0x90, 0x2c, 0xa3, 0xb2, 0x73, 0x4c, 0x54, 0x92, 0x74, 0x36, 0x51,
		0x38, 0xb0, 0xbd, 0x5a, 0xfc, 0x60, 0x62, 0x96, 0x6c, 0x42, 0xf7, 0x10, 0x7c, 0x28, 0x27, 0x8c,
		0x13, 0x95, 0x9c, 0xc7, 0x24, 0x46, 0x3b, 0x70, 0xca, 0xe3, 0x85, 0xcb, 0x11, 0xd0, 0x93, 0xb8,
		0xa6, 0x83, 0x20, 0xff, 0x9f, 0x77, 0xc3, 0xcc, 0x03, 0x6f, 0x08, 0xbf, 0x40, 0xe7, 0x2b, 0xe2,
		0x79, 0x0c, 0xaa, 0x82, 0x41, 0x3a, 0xea, 0xb9, 0xe4, 0x9a, 0xa4, 0x97, 0x7e, 0xda, 0x7a, 0x17,
		0x66, 0x94, 0xa1, 0x1d, 0x3d, 0xf0, 0xde, 0xb3, 0x0b, 0x72, 0xa7, 0x1c, 0xef, 0xd1, 0x53, 0x3e,
		0x8f, 0x33, 0x26, 0x5f, 0xec, 0x76, 0x2a, 0x49, 0x81, 0x88, 0xee, 0x21, 0xc4, 0x1a, 0xeb, 0xd9,
		0xc5, 0x39, 0x99, 0xcd, 0xad, 0x31, 0x8b, 0x01, 0x18, 0x23, 0xdd, 0x1f, 0x4e, 0x2d, 0xf9, 0x48,
		0x4f, 0xf2, 0x65, 0x8e, 0x78, 0x5c, 0x58, 0x19, 0x8d, 0xe5, 0x98, 0x57, 0x67, 0x7f, 0x05, 0x64,
		0xaf, 0x63, 0xb6, 0xfe, 0xf5, 0xb7, 0x3c, 0xa5, 0xce, 0xe9, 0x68, 0x44, 0xe0, 0x4d, 0x43, 0x69,
		0x29, 0x2e, 0xac, 0x15, 0x59, 0xa8, 0x0a, 0x9e, 0x6e, 0x47, 0xdf, 0x34, 0x35, 0x6a, 0xcf, 0xdc,
		0x22, 0xc9, 0xc0, 0x9b, 0x89, 0xd4, 0xed, 0xab, 0x12, 0xa2, 0x0d, 0x52, 0xbb, 0x02, 0x2f, 0xa9,
		0xd7, 0x61, 0x1e, 0xb4, 0x50, 0x04, 0xf6, 0xc2, 0x16, 0x25, 0x86, 0x56, 0x55, 0x09, 0xbe, 0x91,
	},
}

// gfMult returns a·b in GF(2^8)/p
func gfMult(a, b byte, p uint32) byte {
	B := [2]uint32{0, uint32(b)}
	P := [2]uint32{0, p}
	var result uint32

	// branchless GF multiplier
	for i := 0; i < 7; i++ {
		result ^= B[a&1]
		a >>= 1
		B[1] = P[B[1]>>7] ^ (B[1] << 1)
	}
	result ^= B[a&1]
	return byte(result)
}

// mdsColumnMult calculates y{col} where [y0 y1 y2 y3] = MDS · [x0]
func mdsColumnMult(in byte, col int) uint32 {
	mul01 := in
	mul5B := gfMult(in, 0x5B, mdsPolynomial)
	mulEF := gfMult(in, 0xEF, mdsPolynomial)

	switch col {
	case 0:
		return uint32(mul01) | uint32(mul5B)<<8 | uint32(mulEF)<<16 | uint32(mulEF)<<24
	case 1:
		return uint32(mulEF) | uint32(mulEF)<<8 | uint32(mul5B)<<16 | uint32(mul01)<<24
	case 2:
		return uint32(mul5B) | uint32(mulEF)<<8 | uint32(mul01)<<16 | uint32(mulEF)<<24
	case 3:
		return uint32(mul5B) | uint32(mul01)<<8 | uint32(mulEF)<<16 | uint32(mul5B)<<24
	}

	panic("unreachable")
}

// h implements the S-box generation function. See [TWOFISH] 4.3.5
func h(in, key []byte, offset int) uint32 {
	var y [4]byte
	for x := range y {
		y[x] = in[x]
	}
	switch len(key) / 8 {
	case 4:
		y[0] = sbox[1][y[0]] ^ key[4*(6+offset)+0]
		y[1] = sbox[0][y[1]] ^ key[4*(6+offset)+1]
		y[2] = sbox[0][y[2]] ^ key[4*(6+offset)+2]
		y[3] = sbox[1][y[3]] ^ key[4*(6+offset)+3]
		fallthrough
	case 3:
		y[0] = sbox[1][y[0]] ^ key[4*(4+offset)+0]
		y[1] = sbox[1][y[1]] ^ key[4*(4+offset)+1]
		y[2] = sbox[0][y[2]] ^ key[4*(4+offset)+2]
		y[3] = sbox[0][y[3]] ^ key[4*(4+offset)+3]
		fallthrough
	case 2:
		y[0] = sbox[1][sbox[0][sbox[0][y[0]]^key[4*(2+offset)+0]]^key[4*(0+offset)+0]]
		y[1] = sbox[0][sbox[0][sbox[1][y[1]]^key[4*(2+offset)+1]]^key[4*(0+offset)+1]]
		y[2] = sbox[1][sbox[1][sbox[0][y[2]]^key[4*(2+offset)+2]]^key[4*(0+offset)+2]]
		y[3] = sbox[0][sbox[1][sbox[1][y[3]]^key[4*(2+offset)+3]]^key[4*(0+offset)+3]]
	}
	// [y0 y1 y2 y3] = MDS . [x0 x1 x2 x3]
	var mdsMult uint32
	for i := range y {
		mdsMult ^= mdsColumnMult(y[i], i)
	}
	return mdsMult
}

// Encrypt encrypts a 16-byte block from src to dst, which may overlap.
// Note that for amounts of data larger than a block,
// it is not safe to just call Encrypt on successive blocks;
// instead, use an encryption mode like CBC (see crypto/block/cbc.go).
func (c *Cipher) Encrypt(dst, src []byte) {
	S1 := c.s[0]
	S2 := c.s[1]
	S3 := c.s[2]
	S4 := c.s[3]

	// Load input
	ia := load32l(src[0:4])
	ib := load32l(src[4:8])
	ic := load32l(src[8:12])
	id := load32l(src[12:16])

	// Pre-whitening
	ia ^= c.k[0]
	ib ^= c.k[1]
	ic ^= c.k[2]
	id ^= c.k[3]

	for i := 0; i < 8; i++ {
		k := c.k[8+i*4 : 12+i*4]
		t2 := S2[byte(ib)] ^ S3[byte(ib>>8)] ^ S4[byte(ib>>16)] ^ S1[byte(ib>>24)]
		t1 := S1[byte(ia)] ^ S2[byte(ia>>8)] ^ S3[byte(ia>>16)] ^ S4[byte(ia>>24)] + t2
		ic = ror(ic^(t1+k[0]), 1)
		id = rol(id, 1) ^ (t2 + t1 + k[1])

		t2 = S2[byte(id)] ^ S3[byte(id>>8)] ^ S4[byte(id>>16)] ^ S1[byte(id>>24)]
		t1 = S1[byte(ic)] ^ S2[byte(ic>>8)] ^ S3[byte(ic>>16)] ^ S4[byte(ic>>24)] + t2
		ia = ror(ia^(t1+k[2]), 1)
		ib = rol(ib, 1) ^ (t2 + t1 + k[3])
	}

	// Output with "undo last swap"
	ta := ic ^ c.k[4]
	tb := id ^ c.k[5]
	tc := ia ^ c.k[6]
	td := ib ^ c.k[7]

	store32l(dst[0:4], ta)
	store32l(dst[4:8], tb)
	store32l(dst[8:12], tc)
	store32l(dst[12:16], td)
}

// Decrypt decrypts a 16-byte block from src to dst, which may overlap.
func (c *Cipher) Decrypt(dst, src []byte) {
	S1 := c.s[0]
	S2 := c.s[1]
	S3 := c.s[2]
	S4 := c.s[3]

	// Load input
	ta := load32l(src[0:4])
	tb := load32l(src[4:8])
	tc := load32l(src[8:12])
	td := load32l(src[12:16])

	// Undo undo final swap
	ia := tc ^ c.k[6]
	ib := td ^ c.k[7]
	ic := ta ^ c.k[4]
	id := tb ^ c.k[5]

	for i := 8; i > 0; i-- {
		k := c.k[4+i*4 : 8+i*4]
		t2 := S2[byte(id)] ^ S3[byte(id>>8)] ^ S4[byte(id>>16)] ^ S1[byte(id>>24)]
		t1 := S1[byte(ic)] ^ S2[byte(ic>>8)] ^ S3[byte(ic>>16)] ^ S4[byte(ic>>24)] + t2
		ia = rol(ia, 1) ^ (t1 + k[2])
		ib = ror(ib^(t2+t1+k[3]), 1)

		t2 = S2[byte(ib)] ^ S3[byte(ib>>8)] ^ S4[byte(ib>>16)] ^ S1[byte(ib>>24)]
		t1 = S1[byte(ia)] ^ S2[byte(ia>>8)] ^ S3[byte(ia>>16)] ^ S4[byte(ia>>24)] + t2
		ic = rol(ic, 1) ^ (t1 + k[0])
		id = ror(id^(t2+t1+k[1]), 1)
	}

	// Undo pre-whitening
	ia ^= c.k[0]
	ib ^= c.k[1]
	ic ^= c.k[2]
	id ^= c.k[3]

	store32l(dst[0:4], ia)
	store32l(dst[4:8], ib)
	store32l(dst[8:12], ic)
	store32l(dst[12:16], id)
}
