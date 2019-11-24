// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,!gccgo,!appengine,!purego

package curve25519

// These functions are implemented in the .s files. The names of the functions
// in the rest of the file are also taken from the SUPERCOP sources to help
// people following along.

//go:noescape

func cswap(inout *[5]uint64, v uint64)

//go:noescape

func ladderstep(inout *[5][5]uint64)

//go:noescape

func freeze(inout *[5]uint64)

//go:noescape

func mul(dest, a, b *[5]uint64)

//go:noescape

func square(out, in *[5]uint64)

// mladder uses a Montgomery ladder to calculate (xr/zr) *= s.
func mladder(xr, zr *[5]uint64, s *[32]byte) {
	var work [5][5]uint64

	work[0] = *xr
	setint(&work[1], 1)
	setint(&work[2], 0)
	work[3] = *xr
	setint(&work[4], 1)

	j := uint(6)
	var prevbit byte

	for i := 31; i >= 0; i-- {
		for j < 8 {
			bit := ((*s)[i] >> j) & 1
			swap := bit ^ prevbit
			prevbit = bit
			cswap(&work[1], uint64(swap))
			ladderstep(&work)
			j--
		}
		j = 7
	}

	*xr = work[1]
	*zr = work[2]
}

func scalarMult(out, in, base *[32]byte) {
	var e [32]byte
	copy(e[:], (*in)[:])
	e[0] &= 248
	e[31] &= 127
	e[31] |= 64

	var t, z [5]uint64
	unpack(&t, base)
	mladder(&t, &z, &e)
	invert(&z, &z)
	mul(&t, &t, &z)
	pack(out, &t)
}

func setint(r *[5]uint64, v uint64) {
	r[0] = v
	r[1] = 0
	r[2] = 0
	r[3] = 0
	r[4] = 0
}

// unpack sets r = x where r consists of 5, 51-bit limbs in little-endian
// order.
func unpack(r *[5]uint64, x *[32]byte) {
	r[0] = uint64(x[0]) |
		uint64(x[1])<<8 |
		uint64(x[2])<<16 |
		uint64(x[3])<<24 |
		uint64(x[4])<<32 |
		uint64(x[5])<<40 |
		uint64(x[6]&7)<<48

	r[1] = uint64(x[6])>>3 |
		uint64(x[7])<<5 |
		uint64(x[8])<<13 |
		uint64(x[9])<<21 |
		uint64(x[10])<<29 |
		uint64(x[11])<<37 |
		uint64(x[12]&63)<<45

	r[2] = uint64(x[12])>>6 |
		uint64(x[13])<<2 |
		uint64(x[14])<<10 |
		uint64(x[15])<<18 |
		uint64(x[16])<<26 |
		uint64(x[17])<<34 |
		uint64(x[18])<<42 |
		uint64(x[19]&1)<<50

	r[3] = uint64(x[19])>>1 |
		uint64(x[20])<<7 |
		uint64(x[21])<<15 |
		uint64(x[22])<<23 |
		uint64(x[23])<<31 |
		uint64(x[24])<<39 |
		uint64(x[25]&15)<<47

	r[4] = uint64(x[25])>>4 |
		uint64(x[26])<<4 |
		uint64(x[27])<<12 |
		uint64(x[28])<<20 |
		uint64(x[29])<<28 |
		uint64(x[30])<<36 |
		uint64(x[31]&127)<<44
}

// pack sets out = x where out is the usual, little-endian form of the 5,
// 51-bit limbs in x.
func pack(out *[32]byte, x *[5]uint64) {
	t := *x
	freeze(&t)

	out[0] = byte(t[0])
	out[1] = byte(t[0] >> 8)
	out[2] = byte(t[0] >> 16)
	out[3] = byte(t[0] >> 24)
	out[4] = byte(t[0] >> 32)
	out[5] = byte(t[0] >> 40)
	out[6] = byte(t[0] >> 48)

	out[6] ^= byte(t[1]<<3) & 0xf8
	out[7] = byte(t[1] >> 5)
	out[8] = byte(t[1] >> 13)
	out[9] = byte(t[1] >> 21)
	out[10] = byte(t[1] >> 29)
	out[11] = byte(t[1] >> 37)
	out[12] = byte(t[1] >> 45)

	out[12] ^= byte(t[2]<<6) & 0xc0
	out[13] = byte(t[2] >> 2)
	out[14] = byte(t[2] >> 10)
	out[15] = byte(t[2] >> 18)
	out[16] = byte(t[2] >> 26)
	out[17] = byte(t[2] >> 34)
	out[18] = byte(t[2] >> 42)
	out[19] = byte(t[2] >> 50)

	out[19] ^= byte(t[3]<<1) & 0xfe
	out[20] = byte(t[3] >> 7)
	out[21] = byte(t[3] >> 15)
	out[22] = byte(t[3] >> 23)
	out[23] = byte(t[3] >> 31)
	out[24] = byte(t[3] >> 39)
	out[25] = byte(t[3] >> 47)

	out[25] ^= byte(t[4]<<4) & 0xf0
	out[26] = byte(t[4] >> 4)
	out[27] = byte(t[4] >> 12)
	out[28] = byte(t[4] >> 20)
	out[29] = byte(t[4] >> 28)
	out[30] = byte(t[4] >> 36)
	out[31] = byte(t[4] >> 44)
}

// invert calculates r = x^-1 mod p using Fermat's little theorem.
func invert(r *[5]uint64, x *[5]uint64) {
	var z2, z9, z11, z2_5_0, z2_10_0, z2_20_0, z2_50_0, z2_100_0, t [5]uint64

	square(&z2, x)        /* 2 */
	square(&t, &z2)       /* 4 */
	square(&t, &t)        /* 8 */
	mul(&z9, &t, x)       /* 9 */
	mul(&z11, &z9, &z2)   /* 11 */
	square(&t, &z11)      /* 22 */
	mul(&z2_5_0, &t, &z9) /* 2^5 - 2^0 = 31 */

	square(&t, &z2_5_0)      /* 2^6 - 2^1 */
	for i := 1; i < 5; i++ { /* 2^20 - 2^10 */
		square(&t, &t)
	}
	mul(&z2_10_0, &t, &z2_5_0) /* 2^10 - 2^0 */

	square(&t, &z2_10_0)      /* 2^11 - 2^1 */
	for i := 1; i < 10; i++ { /* 2^20 - 2^10 */
		square(&t, &t)
	}
	mul(&z2_20_0, &t, &z2_10_0) /* 2^20 - 2^0 */

	square(&t, &z2_20_0)      /* 2^21 - 2^1 */
	for i := 1; i < 20; i++ { /* 2^40 - 2^20 */
		square(&t, &t)
	}
	mul(&t, &t, &z2_20_0) /* 2^40 - 2^0 */

	square(&t, &t)            /* 2^41 - 2^1 */
	for i := 1; i < 10; i++ { /* 2^50 - 2^10 */
		square(&t, &t)
	}
	mul(&z2_50_0, &t, &z2_10_0) /* 2^50 - 2^0 */

	square(&t, &z2_50_0)      /* 2^51 - 2^1 */
	for i := 1; i < 50; i++ { /* 2^100 - 2^50 */
		square(&t, &t)
	}
	mul(&z2_100_0, &t, &z2_50_0) /* 2^100 - 2^0 */

	square(&t, &z2_100_0)      /* 2^101 - 2^1 */
	for i := 1; i < 100; i++ { /* 2^200 - 2^100 */
		square(&t, &t)
	}
	mul(&t, &t, &z2_100_0) /* 2^200 - 2^0 */

	square(&t, &t)            /* 2^201 - 2^1 */
	for i := 1; i < 50; i++ { /* 2^250 - 2^50 */
		square(&t, &t)
	}
	mul(&t, &t, &z2_50_0) /* 2^250 - 2^0 */

	square(&t, &t) /* 2^251 - 2^1 */
	square(&t, &t) /* 2^252 - 2^2 */
	square(&t, &t) /* 2^253 - 2^3 */

	square(&t, &t) /* 2^254 - 2^4 */

	square(&t, &t)   /* 2^255 - 2^5 */
	mul(r, &t, &z11) /* 2^255 - 21 */
}
