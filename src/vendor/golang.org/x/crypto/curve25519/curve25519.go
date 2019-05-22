// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// We have an implementation in amd64 assembly so this code is only run on
// non-amd64 platforms. The amd64 assembly does not support gccgo.
// +build !amd64 gccgo appengine

package curve25519

import (
	"encoding/binary"
)

// This code is a port of the public domain, "ref10" implementation of
// curve25519 from SUPERCOP 20130419 by D. J. Bernstein.

// fieldElement represents an element of the field GF(2^255 - 19). An element
// t, entries t[0]...t[9], represents the integer t[0]+2^26 t[1]+2^51 t[2]+2^77
// t[3]+2^102 t[4]+...+2^230 t[9]. Bounds on each t[i] vary depending on
// context.
type fieldElement [10]int32

func feZero(fe *fieldElement) {
	for i := range fe {
		fe[i] = 0
	}
}

func feOne(fe *fieldElement) {
	feZero(fe)
	fe[0] = 1
}

func feAdd(dst, a, b *fieldElement) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func feSub(dst, a, b *fieldElement) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func feCopy(dst, src *fieldElement) {
	for i := range dst {
		dst[i] = src[i]
	}
}

// feCSwap replaces (f,g) with (g,f) if b == 1; replaces (f,g) with (f,g) if b == 0.
//
// Preconditions: b in {0,1}.
func feCSwap(f, g *fieldElement, b int32) {
	b = -b
	for i := range f {
		t := b & (f[i] ^ g[i])
		f[i] ^= t
		g[i] ^= t
	}
}

// load3 reads a 24-bit, little-endian value from in.
func load3(in []byte) int64 {
	var r int64
	r = int64(in[0])
	r |= int64(in[1]) << 8
	r |= int64(in[2]) << 16
	return r
}

// load4 reads a 32-bit, little-endian value from in.
func load4(in []byte) int64 {
	return int64(binary.LittleEndian.Uint32(in))
}

func feFromBytes(dst *fieldElement, src *[32]byte) {
	h0 := load4(src[:])
	h1 := load3(src[4:]) << 6
	h2 := load3(src[7:]) << 5
	h3 := load3(src[10:]) << 3
	h4 := load3(src[13:]) << 2
	h5 := load4(src[16:])
	h6 := load3(src[20:]) << 7
	h7 := load3(src[23:]) << 5
	h8 := load3(src[26:]) << 4
	h9 := (load3(src[29:]) & 0x7fffff) << 2

	var carry [10]int64
	carry[9] = (h9 + 1<<24) >> 25
	h0 += carry[9] * 19
	h9 -= carry[9] << 25
	carry[1] = (h1 + 1<<24) >> 25
	h2 += carry[1]
	h1 -= carry[1] << 25
	carry[3] = (h3 + 1<<24) >> 25
	h4 += carry[3]
	h3 -= carry[3] << 25
	carry[5] = (h5 + 1<<24) >> 25
	h6 += carry[5]
	h5 -= carry[5] << 25
	carry[7] = (h7 + 1<<24) >> 25
	h8 += carry[7]
	h7 -= carry[7] << 25

	carry[0] = (h0 + 1<<25) >> 26
	h1 += carry[0]
	h0 -= carry[0] << 26
	carry[2] = (h2 + 1<<25) >> 26
	h3 += carry[2]
	h2 -= carry[2] << 26
	carry[4] = (h4 + 1<<25) >> 26
	h5 += carry[4]
	h4 -= carry[4] << 26
	carry[6] = (h6 + 1<<25) >> 26
	h7 += carry[6]
	h6 -= carry[6] << 26
	carry[8] = (h8 + 1<<25) >> 26
	h9 += carry[8]
	h8 -= carry[8] << 26

	dst[0] = int32(h0)
	dst[1] = int32(h1)
	dst[2] = int32(h2)
	dst[3] = int32(h3)
	dst[4] = int32(h4)
	dst[5] = int32(h5)
	dst[6] = int32(h6)
	dst[7] = int32(h7)
	dst[8] = int32(h8)
	dst[9] = int32(h9)
}

// feToBytes marshals h to s.
// Preconditions:
//   |h| bounded by 1.1*2^25,1.1*2^24,1.1*2^25,1.1*2^24,etc.
//
// Write p=2^255-19; q=floor(h/p).
// Basic claim: q = floor(2^(-255)(h + 19 2^(-25)h9 + 2^(-1))).
//
// Proof:
//   Have |h|<=p so |q|<=1 so |19^2 2^(-255) q|<1/4.
//   Also have |h-2^230 h9|<2^230 so |19 2^(-255)(h-2^230 h9)|<1/4.
//
//   Write y=2^(-1)-19^2 2^(-255)q-19 2^(-255)(h-2^230 h9).
//   Then 0<y<1.
//
//   Write r=h-pq.
//   Have 0<=r<=p-1=2^255-20.
//   Thus 0<=r+19(2^-255)r<r+19(2^-255)2^255<=2^255-1.
//
//   Write x=r+19(2^-255)r+y.
//   Then 0<x<2^255 so floor(2^(-255)x) = 0 so floor(q+2^(-255)x) = q.
//
//   Have q+2^(-255)x = 2^(-255)(h + 19 2^(-25) h9 + 2^(-1))
//   so floor(2^(-255)(h + 19 2^(-25) h9 + 2^(-1))) = q.
func feToBytes(s *[32]byte, h *fieldElement) {
	var carry [10]int32

	q := (19*h[9] + (1 << 24)) >> 25
	q = (h[0] + q) >> 26
	q = (h[1] + q) >> 25
	q = (h[2] + q) >> 26
	q = (h[3] + q) >> 25
	q = (h[4] + q) >> 26
	q = (h[5] + q) >> 25
	q = (h[6] + q) >> 26
	q = (h[7] + q) >> 25
	q = (h[8] + q) >> 26
	q = (h[9] + q) >> 25

	// Goal: Output h-(2^255-19)q, which is between 0 and 2^255-20.
	h[0] += 19 * q
	// Goal: Output h-2^255 q, which is between 0 and 2^255-20.

	carry[0] = h[0] >> 26
	h[1] += carry[0]
	h[0] -= carry[0] << 26
	carry[1] = h[1] >> 25
	h[2] += carry[1]
	h[1] -= carry[1] << 25
	carry[2] = h[2] >> 26
	h[3] += carry[2]
	h[2] -= carry[2] << 26
	carry[3] = h[3] >> 25
	h[4] += carry[3]
	h[3] -= carry[3] << 25
	carry[4] = h[4] >> 26
	h[5] += carry[4]
	h[4] -= carry[4] << 26
	carry[5] = h[5] >> 25
	h[6] += carry[5]
	h[5] -= carry[5] << 25
	carry[6] = h[6] >> 26
	h[7] += carry[6]
	h[6] -= carry[6] << 26
	carry[7] = h[7] >> 25
	h[8] += carry[7]
	h[7] -= carry[7] << 25
	carry[8] = h[8] >> 26
	h[9] += carry[8]
	h[8] -= carry[8] << 26
	carry[9] = h[9] >> 25
	h[9] -= carry[9] << 25
	// h10 = carry9

	// Goal: Output h[0]+...+2^255 h10-2^255 q, which is between 0 and 2^255-20.
	// Have h[0]+...+2^230 h[9] between 0 and 2^255-1;
	// evidently 2^255 h10-2^255 q = 0.
	// Goal: Output h[0]+...+2^230 h[9].

	s[0] = byte(h[0] >> 0)
	s[1] = byte(h[0] >> 8)
	s[2] = byte(h[0] >> 16)
	s[3] = byte((h[0] >> 24) | (h[1] << 2))
	s[4] = byte(h[1] >> 6)
	s[5] = byte(h[1] >> 14)
	s[6] = byte((h[1] >> 22) | (h[2] << 3))
	s[7] = byte(h[2] >> 5)
	s[8] = byte(h[2] >> 13)
	s[9] = byte((h[2] >> 21) | (h[3] << 5))
	s[10] = byte(h[3] >> 3)
	s[11] = byte(h[3] >> 11)
	s[12] = byte((h[3] >> 19) | (h[4] << 6))
	s[13] = byte(h[4] >> 2)
	s[14] = byte(h[4] >> 10)
	s[15] = byte(h[4] >> 18)
	s[16] = byte(h[5] >> 0)
	s[17] = byte(h[5] >> 8)
	s[18] = byte(h[5] >> 16)
	s[19] = byte((h[5] >> 24) | (h[6] << 1))
	s[20] = byte(h[6] >> 7)
	s[21] = byte(h[6] >> 15)
	s[22] = byte((h[6] >> 23) | (h[7] << 3))
	s[23] = byte(h[7] >> 5)
	s[24] = byte(h[7] >> 13)
	s[25] = byte((h[7] >> 21) | (h[8] << 4))
	s[26] = byte(h[8] >> 4)
	s[27] = byte(h[8] >> 12)
	s[28] = byte((h[8] >> 20) | (h[9] << 6))
	s[29] = byte(h[9] >> 2)
	s[30] = byte(h[9] >> 10)
	s[31] = byte(h[9] >> 18)
}

// feMul calculates h = f * g
// Can overlap h with f or g.
//
// Preconditions:
//    |f| bounded by 1.1*2^26,1.1*2^25,1.1*2^26,1.1*2^25,etc.
//    |g| bounded by 1.1*2^26,1.1*2^25,1.1*2^26,1.1*2^25,etc.
//
// Postconditions:
//    |h| bounded by 1.1*2^25,1.1*2^24,1.1*2^25,1.1*2^24,etc.
//
// Notes on implementation strategy:
//
// Using schoolbook multiplication.
// Karatsuba would save a little in some cost models.
//
// Most multiplications by 2 and 19 are 32-bit precomputations;
// cheaper than 64-bit postcomputations.
//
// There is one remaining multiplication by 19 in the carry chain;
// one *19 precomputation can be merged into this,
// but the resulting data flow is considerably less clean.
//
// There are 12 carries below.
// 10 of them are 2-way parallelizable and vectorizable.
// Can get away with 11 carries, but then data flow is much deeper.
//
// With tighter constraints on inputs can squeeze carries into int32.
func feMul(h, f, g *fieldElement) {
	f0 := f[0]
	f1 := f[1]
	f2 := f[2]
	f3 := f[3]
	f4 := f[4]
	f5 := f[5]
	f6 := f[6]
	f7 := f[7]
	f8 := f[8]
	f9 := f[9]
	g0 := g[0]
	g1 := g[1]
	g2 := g[2]
	g3 := g[3]
	g4 := g[4]
	g5 := g[5]
	g6 := g[6]
	g7 := g[7]
	g8 := g[8]
	g9 := g[9]
	g1_19 := 19 * g1 // 1.4*2^29
	g2_19 := 19 * g2 // 1.4*2^30; still ok
	g3_19 := 19 * g3
	g4_19 := 19 * g4
	g5_19 := 19 * g5
	g6_19 := 19 * g6
	g7_19 := 19 * g7
	g8_19 := 19 * g8
	g9_19 := 19 * g9
	f1_2 := 2 * f1
	f3_2 := 2 * f3
	f5_2 := 2 * f5
	f7_2 := 2 * f7
	f9_2 := 2 * f9
	f0g0 := int64(f0) * int64(g0)
	f0g1 := int64(f0) * int64(g1)
	f0g2 := int64(f0) * int64(g2)
	f0g3 := int64(f0) * int64(g3)
	f0g4 := int64(f0) * int64(g4)
	f0g5 := int64(f0) * int64(g5)
	f0g6 := int64(f0) * int64(g6)
	f0g7 := int64(f0) * int64(g7)
	f0g8 := int64(f0) * int64(g8)
	f0g9 := int64(f0) * int64(g9)
	f1g0 := int64(f1) * int64(g0)
	f1g1_2 := int64(f1_2) * int64(g1)
	f1g2 := int64(f1) * int64(g2)
	f1g3_2 := int64(f1_2) * int64(g3)
	f1g4 := int64(f1) * int64(g4)
	f1g5_2 := int64(f1_2) * int64(g5)
	f1g6 := int64(f1) * int64(g6)
	f1g7_2 := int64(f1_2) * int64(g7)
	f1g8 := int64(f1) * int64(g8)
	f1g9_38 := int64(f1_2) * int64(g9_19)
	f2g0 := int64(f2) * int64(g0)
	f2g1 := int64(f2) * int64(g1)
	f2g2 := int64(f2) * int64(g2)
	f2g3 := int64(f2) * int64(g3)
	f2g4 := int64(f2) * int64(g4)
	f2g5 := int64(f2) * int64(g5)
	f2g6 := int64(f2) * int64(g6)
	f2g7 := int64(f2) * int64(g7)
	f2g8_19 := int64(f2) * int64(g8_19)
	f2g9_19 := int64(f2) * int64(g9_19)
	f3g0 := int64(f3) * int64(g0)
	f3g1_2 := int64(f3_2) * int64(g1)
	f3g2 := int64(f3) * int64(g2)
	f3g3_2 := int64(f3_2) * int64(g3)
	f3g4 := int64(f3) * int64(g4)
	f3g5_2 := int64(f3_2) * int64(g5)
	f3g6 := int64(f3) * int64(g6)
	f3g7_38 := int64(f3_2) * int64(g7_19)
	f3g8_19 := int64(f3) * int64(g8_19)
	f3g9_38 := int64(f3_2) * int64(g9_19)
	f4g0 := int64(f4) * int64(g0)
	f4g1 := int64(f4) * int64(g1)
	f4g2 := int64(f4) * int64(g2)
	f4g3 := int64(f4) * int64(g3)
	f4g4 := int64(f4) * int64(g4)
	f4g5 := int64(f4) * int64(g5)
	f4g6_19 := int64(f4) * int64(g6_19)
	f4g7_19 := int64(f4) * int64(g7_19)
	f4g8_19 := int64(f4) * int64(g8_19)
	f4g9_19 := int64(f4) * int64(g9_19)
	f5g0 := int64(f5) * int64(g0)
	f5g1_2 := int64(f5_2) * int64(g1)
	f5g2 := int64(f5) * int64(g2)
	f5g3_2 := int64(f5_2) * int64(g3)
	f5g4 := int64(f5) * int64(g4)
	f5g5_38 := int64(f5_2) * int64(g5_19)
	f5g6_19 := int64(f5) * int64(g6_19)
	f5g7_38 := int64(f5_2) * int64(g7_19)
	f5g8_19 := int64(f5) * int64(g8_19)
	f5g9_38 := int64(f5_2) * int64(g9_19)
	f6g0 := int64(f6) * int64(g0)
	f6g1 := int64(f6) * int64(g1)
	f6g2 := int64(f6) * int64(g2)
	f6g3 := int64(f6) * int64(g3)
	f6g4_19 := int64(f6) * int64(g4_19)
	f6g5_19 := int64(f6) * int64(g5_19)
	f6g6_19 := int64(f6) * int64(g6_19)
	f6g7_19 := int64(f6) * int64(g7_19)
	f6g8_19 := int64(f6) * int64(g8_19)
	f6g9_19 := int64(f6) * int64(g9_19)
	f7g0 := int64(f7) * int64(g0)
	f7g1_2 := int64(f7_2) * int64(g1)
	f7g2 := int64(f7) * int64(g2)
	f7g3_38 := int64(f7_2) * int64(g3_19)
	f7g4_19 := int64(f7) * int64(g4_19)
	f7g5_38 := int64(f7_2) * int64(g5_19)
	f7g6_19 := int64(f7) * int64(g6_19)
	f7g7_38 := int64(f7_2) * int64(g7_19)
	f7g8_19 := int64(f7) * int64(g8_19)
	f7g9_38 := int64(f7_2) * int64(g9_19)
	f8g0 := int64(f8) * int64(g0)
	f8g1 := int64(f8) * int64(g1)
	f8g2_19 := int64(f8) * int64(g2_19)
	f8g3_19 := int64(f8) * int64(g3_19)
	f8g4_19 := int64(f8) * int64(g4_19)
	f8g5_19 := int64(f8) * int64(g5_19)
	f8g6_19 := int64(f8) * int64(g6_19)
	f8g7_19 := int64(f8) * int64(g7_19)
	f8g8_19 := int64(f8) * int64(g8_19)
	f8g9_19 := int64(f8) * int64(g9_19)
	f9g0 := int64(f9) * int64(g0)
	f9g1_38 := int64(f9_2) * int64(g1_19)
	f9g2_19 := int64(f9) * int64(g2_19)
	f9g3_38 := int64(f9_2) * int64(g3_19)
	f9g4_19 := int64(f9) * int64(g4_19)
	f9g5_38 := int64(f9_2) * int64(g5_19)
	f9g6_19 := int64(f9) * int64(g6_19)
	f9g7_38 := int64(f9_2) * int64(g7_19)
	f9g8_19 := int64(f9) * int64(g8_19)
	f9g9_38 := int64(f9_2) * int64(g9_19)
	h0 := f0g0 + f1g9_38 + f2g8_19 + f3g7_38 + f4g6_19 + f5g5_38 + f6g4_19 + f7g3_38 + f8g2_19 + f9g1_38
	h1 := f0g1 + f1g0 + f2g9_19 + f3g8_19 + f4g7_19 + f5g6_19 + f6g5_19 + f7g4_19 + f8g3_19 + f9g2_19
	h2 := f0g2 + f1g1_2 + f2g0 + f3g9_38 + f4g8_19 + f5g7_38 + f6g6_19 + f7g5_38 + f8g4_19 + f9g3_38
	h3 := f0g3 + f1g2 + f2g1 + f3g0 + f4g9_19 + f5g8_19 + f6g7_19 + f7g6_19 + f8g5_19 + f9g4_19
	h4 := f0g4 + f1g3_2 + f2g2 + f3g1_2 + f4g0 + f5g9_38 + f6g8_19 + f7g7_38 + f8g6_19 + f9g5_38
	h5 := f0g5 + f1g4 + f2g3 + f3g2 + f4g1 + f5g0 + f6g9_19 + f7g8_19 + f8g7_19 + f9g6_19
	h6 := f0g6 + f1g5_2 + f2g4 + f3g3_2 + f4g2 + f5g1_2 + f6g0 + f7g9_38 + f8g8_19 + f9g7_38
	h7 := f0g7 + f1g6 + f2g5 + f3g4 + f4g3 + f5g2 + f6g1 + f7g0 + f8g9_19 + f9g8_19
	h8 := f0g8 + f1g7_2 + f2g6 + f3g5_2 + f4g4 + f5g3_2 + f6g2 + f7g1_2 + f8g0 + f9g9_38
	h9 := f0g9 + f1g8 + f2g7 + f3g6 + f4g5 + f5g4 + f6g3 + f7g2 + f8g1 + f9g0
	var carry [10]int64

	// |h0| <= (1.1*1.1*2^52*(1+19+19+19+19)+1.1*1.1*2^50*(38+38+38+38+38))
	//   i.e. |h0| <= 1.2*2^59; narrower ranges for h2, h4, h6, h8
	// |h1| <= (1.1*1.1*2^51*(1+1+19+19+19+19+19+19+19+19))
	//   i.e. |h1| <= 1.5*2^58; narrower ranges for h3, h5, h7, h9

	carry[0] = (h0 + (1 << 25)) >> 26
	h1 += carry[0]
	h0 -= carry[0] << 26
	carry[4] = (h4 + (1 << 25)) >> 26
	h5 += carry[4]
	h4 -= carry[4] << 26
	// |h0| <= 2^25
	// |h4| <= 2^25
	// |h1| <= 1.51*2^58
	// |h5| <= 1.51*2^58

	carry[1] = (h1 + (1 << 24)) >> 25
	h2 += carry[1]
	h1 -= carry[1] << 25
	carry[5] = (h5 + (1 << 24)) >> 25
	h6 += carry[5]
	h5 -= carry[5] << 25
	// |h1| <= 2^24; from now on fits into int32
	// |h5| <= 2^24; from now on fits into int32
	// |h2| <= 1.21*2^59
	// |h6| <= 1.21*2^59

	carry[2] = (h2 + (1 << 25)) >> 26
	h3 += carry[2]
	h2 -= carry[2] << 26
	carry[6] = (h6 + (1 << 25)) >> 26
	h7 += carry[6]
	h6 -= carry[6] << 26
	// |h2| <= 2^25; from now on fits into int32 unchanged
	// |h6| <= 2^25; from now on fits into int32 unchanged
	// |h3| <= 1.51*2^58
	// |h7| <= 1.51*2^58

	carry[3] = (h3 + (1 << 24)) >> 25
	h4 += carry[3]
	h3 -= carry[3] << 25
	carry[7] = (h7 + (1 << 24)) >> 25
	h8 += carry[7]
	h7 -= carry[7] << 25
	// |h3| <= 2^24; from now on fits into int32 unchanged
	// |h7| <= 2^24; from now on fits into int32 unchanged
	// |h4| <= 1.52*2^33
	// |h8| <= 1.52*2^33

	carry[4] = (h4 + (1 << 25)) >> 26
	h5 += carry[4]
	h4 -= carry[4] << 26
	carry[8] = (h8 + (1 << 25)) >> 26
	h9 += carry[8]
	h8 -= carry[8] << 26
	// |h4| <= 2^25; from now on fits into int32 unchanged
	// |h8| <= 2^25; from now on fits into int32 unchanged
	// |h5| <= 1.01*2^24
	// |h9| <= 1.51*2^58

	carry[9] = (h9 + (1 << 24)) >> 25
	h0 += carry[9] * 19
	h9 -= carry[9] << 25
	// |h9| <= 2^24; from now on fits into int32 unchanged
	// |h0| <= 1.8*2^37

	carry[0] = (h0 + (1 << 25)) >> 26
	h1 += carry[0]
	h0 -= carry[0] << 26
	// |h0| <= 2^25; from now on fits into int32 unchanged
	// |h1| <= 1.01*2^24

	h[0] = int32(h0)
	h[1] = int32(h1)
	h[2] = int32(h2)
	h[3] = int32(h3)
	h[4] = int32(h4)
	h[5] = int32(h5)
	h[6] = int32(h6)
	h[7] = int32(h7)
	h[8] = int32(h8)
	h[9] = int32(h9)
}

// feSquare calculates h = f*f. Can overlap h with f.
//
// Preconditions:
//    |f| bounded by 1.1*2^26,1.1*2^25,1.1*2^26,1.1*2^25,etc.
//
// Postconditions:
//    |h| bounded by 1.1*2^25,1.1*2^24,1.1*2^25,1.1*2^24,etc.
func feSquare(h, f *fieldElement) {
	f0 := f[0]
	f1 := f[1]
	f2 := f[2]
	f3 := f[3]
	f4 := f[4]
	f5 := f[5]
	f6 := f[6]
	f7 := f[7]
	f8 := f[8]
	f9 := f[9]
	f0_2 := 2 * f0
	f1_2 := 2 * f1
	f2_2 := 2 * f2
	f3_2 := 2 * f3
	f4_2 := 2 * f4
	f5_2 := 2 * f5
	f6_2 := 2 * f6
	f7_2 := 2 * f7
	f5_38 := 38 * f5 // 1.31*2^30
	f6_19 := 19 * f6 // 1.31*2^30
	f7_38 := 38 * f7 // 1.31*2^30
	f8_19 := 19 * f8 // 1.31*2^30
	f9_38 := 38 * f9 // 1.31*2^30
	f0f0 := int64(f0) * int64(f0)
	f0f1_2 := int64(f0_2) * int64(f1)
	f0f2_2 := int64(f0_2) * int64(f2)
	f0f3_2 := int64(f0_2) * int64(f3)
	f0f4_2 := int64(f0_2) * int64(f4)
	f0f5_2 := int64(f0_2) * int64(f5)
	f0f6_2 := int64(f0_2) * int64(f6)
	f0f7_2 := int64(f0_2) * int64(f7)
	f0f8_2 := int64(f0_2) * int64(f8)
	f0f9_2 := int64(f0_2) * int64(f9)
	f1f1_2 := int64(f1_2) * int64(f1)
	f1f2_2 := int64(f1_2) * int64(f2)
	f1f3_4 := int64(f1_2) * int64(f3_2)
	f1f4_2 := int64(f1_2) * int64(f4)
	f1f5_4 := int64(f1_2) * int64(f5_2)
	f1f6_2 := int64(f1_2) * int64(f6)
	f1f7_4 := int64(f1_2) * int64(f7_2)
	f1f8_2 := int64(f1_2) * int64(f8)
	f1f9_76 := int64(f1_2) * int64(f9_38)
	f2f2 := int64(f2) * int64(f2)
	f2f3_2 := int64(f2_2) * int64(f3)
	f2f4_2 := int64(f2_2) * int64(f4)
	f2f5_2 := int64(f2_2) * int64(f5)
	f2f6_2 := int64(f2_2) * int64(f6)
	f2f7_2 := int64(f2_2) * int64(f7)
	f2f8_38 := int64(f2_2) * int64(f8_19)
	f2f9_38 := int64(f2) * int64(f9_38)
	f3f3_2 := int64(f3_2) * int64(f3)
	f3f4_2 := int64(f3_2) * int64(f4)
	f3f5_4 := int64(f3_2) * int64(f5_2)
	f3f6_2 := int64(f3_2) * int64(f6)
	f3f7_76 := int64(f3_2) * int64(f7_38)
	f3f8_38 := int64(f3_2) * int64(f8_19)
	f3f9_76 := int64(f3_2) * int64(f9_38)
	f4f4 := int64(f4) * int64(f4)
	f4f5_2 := int64(f4_2) * int64(f5)
	f4f6_38 := int64(f4_2) * int64(f6_19)
	f4f7_38 := int64(f4) * int64(f7_38)
	f4f8_38 := int64(f4_2) * int64(f8_19)
	f4f9_38 := int64(f4) * int64(f9_38)
	f5f5_38 := int64(f5) * int64(f5_38)
	f5f6_38 := int64(f5_2) * int64(f6_19)
	f5f7_76 := int64(f5_2) * int64(f7_38)
	f5f8_38 := int64(f5_2) * int64(f8_19)
	f5f9_76 := int64(f5_2) * int64(f9_38)
	f6f6_19 := int64(f6) * int64(f6_19)
	f6f7_38 := int64(f6) * int64(f7_38)
	f6f8_38 := int64(f6_2) * int64(f8_19)
	f6f9_38 := int64(f6) * int64(f9_38)
	f7f7_38 := int64(f7) * int64(f7_38)
	f7f8_38 := int64(f7_2) * int64(f8_19)
	f7f9_76 := int64(f7_2) * int64(f9_38)
	f8f8_19 := int64(f8) * int64(f8_19)
	f8f9_38 := int64(f8) * int64(f9_38)
	f9f9_38 := int64(f9) * int64(f9_38)
	h0 := f0f0 + f1f9_76 + f2f8_38 + f3f7_76 + f4f6_38 + f5f5_38
	h1 := f0f1_2 + f2f9_38 + f3f8_38 + f4f7_38 + f5f6_38
	h2 := f0f2_2 + f1f1_2 + f3f9_76 + f4f8_38 + f5f7_76 + f6f6_19
	h3 := f0f3_2 + f1f2_2 + f4f9_38 + f5f8_38 + f6f7_38
	h4 := f0f4_2 + f1f3_4 + f2f2 + f5f9_76 + f6f8_38 + f7f7_38
	h5 := f0f5_2 + f1f4_2 + f2f3_2 + f6f9_38 + f7f8_38
	h6 := f0f6_2 + f1f5_4 + f2f4_2 + f3f3_2 + f7f9_76 + f8f8_19
	h7 := f0f7_2 + f1f6_2 + f2f5_2 + f3f4_2 + f8f9_38
	h8 := f0f8_2 + f1f7_4 + f2f6_2 + f3f5_4 + f4f4 + f9f9_38
	h9 := f0f9_2 + f1f8_2 + f2f7_2 + f3f6_2 + f4f5_2
	var carry [10]int64

	carry[0] = (h0 + (1 << 25)) >> 26
	h1 += carry[0]
	h0 -= carry[0] << 26
	carry[4] = (h4 + (1 << 25)) >> 26
	h5 += carry[4]
	h4 -= carry[4] << 26

	carry[1] = (h1 + (1 << 24)) >> 25
	h2 += carry[1]
	h1 -= carry[1] << 25
	carry[5] = (h5 + (1 << 24)) >> 25
	h6 += carry[5]
	h5 -= carry[5] << 25

	carry[2] = (h2 + (1 << 25)) >> 26
	h3 += carry[2]
	h2 -= carry[2] << 26
	carry[6] = (h6 + (1 << 25)) >> 26
	h7 += carry[6]
	h6 -= carry[6] << 26

	carry[3] = (h3 + (1 << 24)) >> 25
	h4 += carry[3]
	h3 -= carry[3] << 25
	carry[7] = (h7 + (1 << 24)) >> 25
	h8 += carry[7]
	h7 -= carry[7] << 25

	carry[4] = (h4 + (1 << 25)) >> 26
	h5 += carry[4]
	h4 -= carry[4] << 26
	carry[8] = (h8 + (1 << 25)) >> 26
	h9 += carry[8]
	h8 -= carry[8] << 26

	carry[9] = (h9 + (1 << 24)) >> 25
	h0 += carry[9] * 19
	h9 -= carry[9] << 25

	carry[0] = (h0 + (1 << 25)) >> 26
	h1 += carry[0]
	h0 -= carry[0] << 26

	h[0] = int32(h0)
	h[1] = int32(h1)
	h[2] = int32(h2)
	h[3] = int32(h3)
	h[4] = int32(h4)
	h[5] = int32(h5)
	h[6] = int32(h6)
	h[7] = int32(h7)
	h[8] = int32(h8)
	h[9] = int32(h9)
}

// feMul121666 calculates h = f * 121666. Can overlap h with f.
//
// Preconditions:
//    |f| bounded by 1.1*2^26,1.1*2^25,1.1*2^26,1.1*2^25,etc.
//
// Postconditions:
//    |h| bounded by 1.1*2^25,1.1*2^24,1.1*2^25,1.1*2^24,etc.
func feMul121666(h, f *fieldElement) {
	h0 := int64(f[0]) * 121666
	h1 := int64(f[1]) * 121666
	h2 := int64(f[2]) * 121666
	h3 := int64(f[3]) * 121666
	h4 := int64(f[4]) * 121666
	h5 := int64(f[5]) * 121666
	h6 := int64(f[6]) * 121666
	h7 := int64(f[7]) * 121666
	h8 := int64(f[8]) * 121666
	h9 := int64(f[9]) * 121666
	var carry [10]int64

	carry[9] = (h9 + (1 << 24)) >> 25
	h0 += carry[9] * 19
	h9 -= carry[9] << 25
	carry[1] = (h1 + (1 << 24)) >> 25
	h2 += carry[1]
	h1 -= carry[1] << 25
	carry[3] = (h3 + (1 << 24)) >> 25
	h4 += carry[3]
	h3 -= carry[3] << 25
	carry[5] = (h5 + (1 << 24)) >> 25
	h6 += carry[5]
	h5 -= carry[5] << 25
	carry[7] = (h7 + (1 << 24)) >> 25
	h8 += carry[7]
	h7 -= carry[7] << 25

	carry[0] = (h0 + (1 << 25)) >> 26
	h1 += carry[0]
	h0 -= carry[0] << 26
	carry[2] = (h2 + (1 << 25)) >> 26
	h3 += carry[2]
	h2 -= carry[2] << 26
	carry[4] = (h4 + (1 << 25)) >> 26
	h5 += carry[4]
	h4 -= carry[4] << 26
	carry[6] = (h6 + (1 << 25)) >> 26
	h7 += carry[6]
	h6 -= carry[6] << 26
	carry[8] = (h8 + (1 << 25)) >> 26
	h9 += carry[8]
	h8 -= carry[8] << 26

	h[0] = int32(h0)
	h[1] = int32(h1)
	h[2] = int32(h2)
	h[3] = int32(h3)
	h[4] = int32(h4)
	h[5] = int32(h5)
	h[6] = int32(h6)
	h[7] = int32(h7)
	h[8] = int32(h8)
	h[9] = int32(h9)
}

// feInvert sets out = z^-1.
func feInvert(out, z *fieldElement) {
	var t0, t1, t2, t3 fieldElement
	var i int

	feSquare(&t0, z)
	for i = 1; i < 1; i++ {
		feSquare(&t0, &t0)
	}
	feSquare(&t1, &t0)
	for i = 1; i < 2; i++ {
		feSquare(&t1, &t1)
	}
	feMul(&t1, z, &t1)
	feMul(&t0, &t0, &t1)
	feSquare(&t2, &t0)
	for i = 1; i < 1; i++ {
		feSquare(&t2, &t2)
	}
	feMul(&t1, &t1, &t2)
	feSquare(&t2, &t1)
	for i = 1; i < 5; i++ {
		feSquare(&t2, &t2)
	}
	feMul(&t1, &t2, &t1)
	feSquare(&t2, &t1)
	for i = 1; i < 10; i++ {
		feSquare(&t2, &t2)
	}
	feMul(&t2, &t2, &t1)
	feSquare(&t3, &t2)
	for i = 1; i < 20; i++ {
		feSquare(&t3, &t3)
	}
	feMul(&t2, &t3, &t2)
	feSquare(&t2, &t2)
	for i = 1; i < 10; i++ {
		feSquare(&t2, &t2)
	}
	feMul(&t1, &t2, &t1)
	feSquare(&t2, &t1)
	for i = 1; i < 50; i++ {
		feSquare(&t2, &t2)
	}
	feMul(&t2, &t2, &t1)
	feSquare(&t3, &t2)
	for i = 1; i < 100; i++ {
		feSquare(&t3, &t3)
	}
	feMul(&t2, &t3, &t2)
	feSquare(&t2, &t2)
	for i = 1; i < 50; i++ {
		feSquare(&t2, &t2)
	}
	feMul(&t1, &t2, &t1)
	feSquare(&t1, &t1)
	for i = 1; i < 5; i++ {
		feSquare(&t1, &t1)
	}
	feMul(out, &t1, &t0)
}

func scalarMult(out, in, base *[32]byte) {
	var e [32]byte

	copy(e[:], in[:])
	e[0] &= 248
	e[31] &= 127
	e[31] |= 64

	var x1, x2, z2, x3, z3, tmp0, tmp1 fieldElement
	feFromBytes(&x1, base)
	feOne(&x2)
	feCopy(&x3, &x1)
	feOne(&z3)

	swap := int32(0)
	for pos := 254; pos >= 0; pos-- {
		b := e[pos/8] >> uint(pos&7)
		b &= 1
		swap ^= int32(b)
		feCSwap(&x2, &x3, swap)
		feCSwap(&z2, &z3, swap)
		swap = int32(b)

		feSub(&tmp0, &x3, &z3)
		feSub(&tmp1, &x2, &z2)
		feAdd(&x2, &x2, &z2)
		feAdd(&z2, &x3, &z3)
		feMul(&z3, &tmp0, &x2)
		feMul(&z2, &z2, &tmp1)
		feSquare(&tmp0, &tmp1)
		feSquare(&tmp1, &x2)
		feAdd(&x3, &z3, &z2)
		feSub(&z2, &z3, &z2)
		feMul(&x2, &tmp1, &tmp0)
		feSub(&tmp1, &tmp1, &tmp0)
		feSquare(&z2, &z2)
		feMul121666(&z3, &tmp1)
		feSquare(&x3, &x3)
		feAdd(&tmp0, &tmp0, &z3)
		feMul(&z3, &x1, &z2)
		feMul(&z2, &tmp1, &tmp0)
	}

	feCSwap(&x2, &x3, swap)
	feCSwap(&z2, &z3, swap)

	feInvert(&z2, &z2)
	feMul(&x2, &x2, &z2)
	feToBytes(out, &x2)
}
