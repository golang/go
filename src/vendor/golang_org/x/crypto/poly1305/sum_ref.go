// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64,!arm gccgo appengine nacl

package poly1305

// Based on original, public domain implementation from NaCl by D. J.
// Bernstein.

import "math"

const (
	alpham80 = 0.00000000558793544769287109375
	alpham48 = 24.0
	alpham16 = 103079215104.0
	alpha0   = 6755399441055744.0
	alpha18  = 1770887431076116955136.0
	alpha32  = 29014219670751100192948224.0
	alpha50  = 7605903601369376408980219232256.0
	alpha64  = 124615124604835863084731911901282304.0
	alpha82  = 32667107224410092492483962313449748299776.0
	alpha96  = 535217884764734955396857238543560676143529984.0
	alpha112 = 35076039295941670036888435985190792471742381031424.0
	alpha130 = 9194973245195333150150082162901855101712434733101613056.0
	scale    = 0.0000000000000000000000000000000000000036734198463196484624023016788195177431833298649127735047148490821200539357960224151611328125
	offset0  = 6755408030990331.0
	offset1  = 29014256564239239022116864.0
	offset2  = 124615283061160854719918951570079744.0
	offset3  = 535219245894202480694386063513315216128475136.0
)

// Sum generates an authenticator for m using a one-time key and puts the
// 16-byte result into out. Authenticating two different messages with the same
// key allows an attacker to forge messages at will.
func Sum(out *[16]byte, m []byte, key *[32]byte) {
	r := key
	s := key[16:]
	var (
		y7        float64
		y6        float64
		y1        float64
		y0        float64
		y5        float64
		y4        float64
		x7        float64
		x6        float64
		x1        float64
		x0        float64
		y3        float64
		y2        float64
		x5        float64
		r3lowx0   float64
		x4        float64
		r0lowx6   float64
		x3        float64
		r3highx0  float64
		x2        float64
		r0highx6  float64
		r0lowx0   float64
		sr1lowx6  float64
		r0highx0  float64
		sr1highx6 float64
		sr3low    float64
		r1lowx0   float64
		sr2lowx6  float64
		r1highx0  float64
		sr2highx6 float64
		r2lowx0   float64
		sr3lowx6  float64
		r2highx0  float64
		sr3highx6 float64
		r1highx4  float64
		r1lowx4   float64
		r0highx4  float64
		r0lowx4   float64
		sr3highx4 float64
		sr3lowx4  float64
		sr2highx4 float64
		sr2lowx4  float64
		r0lowx2   float64
		r0highx2  float64
		r1lowx2   float64
		r1highx2  float64
		r2lowx2   float64
		r2highx2  float64
		sr3lowx2  float64
		sr3highx2 float64
		z0        float64
		z1        float64
		z2        float64
		z3        float64
		m0        int64
		m1        int64
		m2        int64
		m3        int64
		m00       uint32
		m01       uint32
		m02       uint32
		m03       uint32
		m10       uint32
		m11       uint32
		m12       uint32
		m13       uint32
		m20       uint32
		m21       uint32
		m22       uint32
		m23       uint32
		m30       uint32
		m31       uint32
		m32       uint32
		m33       uint64
		lbelow2   int32
		lbelow3   int32
		lbelow4   int32
		lbelow5   int32
		lbelow6   int32
		lbelow7   int32
		lbelow8   int32
		lbelow9   int32
		lbelow10  int32
		lbelow11  int32
		lbelow12  int32
		lbelow13  int32
		lbelow14  int32
		lbelow15  int32
		s00       uint32
		s01       uint32
		s02       uint32
		s03       uint32
		s10       uint32
		s11       uint32
		s12       uint32
		s13       uint32
		s20       uint32
		s21       uint32
		s22       uint32
		s23       uint32
		s30       uint32
		s31       uint32
		s32       uint32
		s33       uint32
		bits32    uint64
		f         uint64
		f0        uint64
		f1        uint64
		f2        uint64
		f3        uint64
		f4        uint64
		g         uint64
		g0        uint64
		g1        uint64
		g2        uint64
		g3        uint64
		g4        uint64
	)

	var p int32

	l := int32(len(m))

	r00 := uint32(r[0])

	r01 := uint32(r[1])

	r02 := uint32(r[2])
	r0 := int64(2151)

	r03 := uint32(r[3])
	r03 &= 15
	r0 <<= 51

	r10 := uint32(r[4])
	r10 &= 252
	r01 <<= 8
	r0 += int64(r00)

	r11 := uint32(r[5])
	r02 <<= 16
	r0 += int64(r01)

	r12 := uint32(r[6])
	r03 <<= 24
	r0 += int64(r02)

	r13 := uint32(r[7])
	r13 &= 15
	r1 := int64(2215)
	r0 += int64(r03)

	d0 := r0
	r1 <<= 51
	r2 := int64(2279)

	r20 := uint32(r[8])
	r20 &= 252
	r11 <<= 8
	r1 += int64(r10)

	r21 := uint32(r[9])
	r12 <<= 16
	r1 += int64(r11)

	r22 := uint32(r[10])
	r13 <<= 24
	r1 += int64(r12)

	r23 := uint32(r[11])
	r23 &= 15
	r2 <<= 51
	r1 += int64(r13)

	d1 := r1
	r21 <<= 8
	r2 += int64(r20)

	r30 := uint32(r[12])
	r30 &= 252
	r22 <<= 16
	r2 += int64(r21)

	r31 := uint32(r[13])
	r23 <<= 24
	r2 += int64(r22)

	r32 := uint32(r[14])
	r2 += int64(r23)
	r3 := int64(2343)

	d2 := r2
	r3 <<= 51

	r33 := uint32(r[15])
	r33 &= 15
	r31 <<= 8
	r3 += int64(r30)

	r32 <<= 16
	r3 += int64(r31)

	r33 <<= 24
	r3 += int64(r32)

	r3 += int64(r33)
	h0 := alpha32 - alpha32

	d3 := r3
	h1 := alpha32 - alpha32

	h2 := alpha32 - alpha32

	h3 := alpha32 - alpha32

	h4 := alpha32 - alpha32

	r0low := math.Float64frombits(uint64(d0))
	h5 := alpha32 - alpha32

	r1low := math.Float64frombits(uint64(d1))
	h6 := alpha32 - alpha32

	r2low := math.Float64frombits(uint64(d2))
	h7 := alpha32 - alpha32

	r0low -= alpha0

	r1low -= alpha32

	r2low -= alpha64

	r0high := r0low + alpha18

	r3low := math.Float64frombits(uint64(d3))

	r1high := r1low + alpha50
	sr1low := scale * r1low

	r2high := r2low + alpha82
	sr2low := scale * r2low

	r0high -= alpha18
	r0high_stack := r0high

	r3low -= alpha96

	r1high -= alpha50
	r1high_stack := r1high

	sr1high := sr1low + alpham80

	r0low -= r0high

	r2high -= alpha82
	sr3low = scale * r3low

	sr2high := sr2low + alpham48

	r1low -= r1high
	r1low_stack := r1low

	sr1high -= alpham80
	sr1high_stack := sr1high

	r2low -= r2high
	r2low_stack := r2low

	sr2high -= alpham48
	sr2high_stack := sr2high

	r3high := r3low + alpha112
	r0low_stack := r0low

	sr1low -= sr1high
	sr1low_stack := sr1low

	sr3high := sr3low + alpham16
	r2high_stack := r2high

	sr2low -= sr2high
	sr2low_stack := sr2low

	r3high -= alpha112
	r3high_stack := r3high

	sr3high -= alpham16
	sr3high_stack := sr3high

	r3low -= r3high
	r3low_stack := r3low

	sr3low -= sr3high
	sr3low_stack := sr3low

	if l < 16 {
		goto addatmost15bytes
	}

	m00 = uint32(m[p+0])
	m0 = 2151

	m0 <<= 51
	m1 = 2215
	m01 = uint32(m[p+1])

	m1 <<= 51
	m2 = 2279
	m02 = uint32(m[p+2])

	m2 <<= 51
	m3 = 2343
	m03 = uint32(m[p+3])

	m10 = uint32(m[p+4])
	m01 <<= 8
	m0 += int64(m00)

	m11 = uint32(m[p+5])
	m02 <<= 16
	m0 += int64(m01)

	m12 = uint32(m[p+6])
	m03 <<= 24
	m0 += int64(m02)

	m13 = uint32(m[p+7])
	m3 <<= 51
	m0 += int64(m03)

	m20 = uint32(m[p+8])
	m11 <<= 8
	m1 += int64(m10)

	m21 = uint32(m[p+9])
	m12 <<= 16
	m1 += int64(m11)

	m22 = uint32(m[p+10])
	m13 <<= 24
	m1 += int64(m12)

	m23 = uint32(m[p+11])
	m1 += int64(m13)

	m30 = uint32(m[p+12])
	m21 <<= 8
	m2 += int64(m20)

	m31 = uint32(m[p+13])
	m22 <<= 16
	m2 += int64(m21)

	m32 = uint32(m[p+14])
	m23 <<= 24
	m2 += int64(m22)

	m33 = uint64(m[p+15])
	m2 += int64(m23)

	d0 = m0
	m31 <<= 8
	m3 += int64(m30)

	d1 = m1
	m32 <<= 16
	m3 += int64(m31)

	d2 = m2
	m33 += 256

	m33 <<= 24
	m3 += int64(m32)

	m3 += int64(m33)
	d3 = m3

	p += 16
	l -= 16

	z0 = math.Float64frombits(uint64(d0))

	z1 = math.Float64frombits(uint64(d1))

	z2 = math.Float64frombits(uint64(d2))

	z3 = math.Float64frombits(uint64(d3))

	z0 -= alpha0

	z1 -= alpha32

	z2 -= alpha64

	z3 -= alpha96

	h0 += z0

	h1 += z1

	h3 += z2

	h5 += z3

	if l < 16 {
		goto multiplyaddatmost15bytes
	}

multiplyaddatleast16bytes:

	m2 = 2279
	m20 = uint32(m[p+8])
	y7 = h7 + alpha130

	m2 <<= 51
	m3 = 2343
	m21 = uint32(m[p+9])
	y6 = h6 + alpha130

	m3 <<= 51
	m0 = 2151
	m22 = uint32(m[p+10])
	y1 = h1 + alpha32

	m0 <<= 51
	m1 = 2215
	m23 = uint32(m[p+11])
	y0 = h0 + alpha32

	m1 <<= 51
	m30 = uint32(m[p+12])
	y7 -= alpha130

	m21 <<= 8
	m2 += int64(m20)
	m31 = uint32(m[p+13])
	y6 -= alpha130

	m22 <<= 16
	m2 += int64(m21)
	m32 = uint32(m[p+14])
	y1 -= alpha32

	m23 <<= 24
	m2 += int64(m22)
	m33 = uint64(m[p+15])
	y0 -= alpha32

	m2 += int64(m23)
	m00 = uint32(m[p+0])
	y5 = h5 + alpha96

	m31 <<= 8
	m3 += int64(m30)
	m01 = uint32(m[p+1])
	y4 = h4 + alpha96

	m32 <<= 16
	m02 = uint32(m[p+2])
	x7 = h7 - y7
	y7 *= scale

	m33 += 256
	m03 = uint32(m[p+3])
	x6 = h6 - y6
	y6 *= scale

	m33 <<= 24
	m3 += int64(m31)
	m10 = uint32(m[p+4])
	x1 = h1 - y1

	m01 <<= 8
	m3 += int64(m32)
	m11 = uint32(m[p+5])
	x0 = h0 - y0

	m3 += int64(m33)
	m0 += int64(m00)
	m12 = uint32(m[p+6])
	y5 -= alpha96

	m02 <<= 16
	m0 += int64(m01)
	m13 = uint32(m[p+7])
	y4 -= alpha96

	m03 <<= 24
	m0 += int64(m02)
	d2 = m2
	x1 += y7

	m0 += int64(m03)
	d3 = m3
	x0 += y6

	m11 <<= 8
	m1 += int64(m10)
	d0 = m0
	x7 += y5

	m12 <<= 16
	m1 += int64(m11)
	x6 += y4

	m13 <<= 24
	m1 += int64(m12)
	y3 = h3 + alpha64

	m1 += int64(m13)
	d1 = m1
	y2 = h2 + alpha64

	x0 += x1

	x6 += x7

	y3 -= alpha64
	r3low = r3low_stack

	y2 -= alpha64
	r0low = r0low_stack

	x5 = h5 - y5
	r3lowx0 = r3low * x0
	r3high = r3high_stack

	x4 = h4 - y4
	r0lowx6 = r0low * x6
	r0high = r0high_stack

	x3 = h3 - y3
	r3highx0 = r3high * x0
	sr1low = sr1low_stack

	x2 = h2 - y2
	r0highx6 = r0high * x6
	sr1high = sr1high_stack

	x5 += y3
	r0lowx0 = r0low * x0
	r1low = r1low_stack

	h6 = r3lowx0 + r0lowx6
	sr1lowx6 = sr1low * x6
	r1high = r1high_stack

	x4 += y2
	r0highx0 = r0high * x0
	sr2low = sr2low_stack

	h7 = r3highx0 + r0highx6
	sr1highx6 = sr1high * x6
	sr2high = sr2high_stack

	x3 += y1
	r1lowx0 = r1low * x0
	r2low = r2low_stack

	h0 = r0lowx0 + sr1lowx6
	sr2lowx6 = sr2low * x6
	r2high = r2high_stack

	x2 += y0
	r1highx0 = r1high * x0
	sr3low = sr3low_stack

	h1 = r0highx0 + sr1highx6
	sr2highx6 = sr2high * x6
	sr3high = sr3high_stack

	x4 += x5
	r2lowx0 = r2low * x0
	z2 = math.Float64frombits(uint64(d2))

	h2 = r1lowx0 + sr2lowx6
	sr3lowx6 = sr3low * x6

	x2 += x3
	r2highx0 = r2high * x0
	z3 = math.Float64frombits(uint64(d3))

	h3 = r1highx0 + sr2highx6
	sr3highx6 = sr3high * x6

	r1highx4 = r1high * x4
	z2 -= alpha64

	h4 = r2lowx0 + sr3lowx6
	r1lowx4 = r1low * x4

	r0highx4 = r0high * x4
	z3 -= alpha96

	h5 = r2highx0 + sr3highx6
	r0lowx4 = r0low * x4

	h7 += r1highx4
	sr3highx4 = sr3high * x4

	h6 += r1lowx4
	sr3lowx4 = sr3low * x4

	h5 += r0highx4
	sr2highx4 = sr2high * x4

	h4 += r0lowx4
	sr2lowx4 = sr2low * x4

	h3 += sr3highx4
	r0lowx2 = r0low * x2

	h2 += sr3lowx4
	r0highx2 = r0high * x2

	h1 += sr2highx4
	r1lowx2 = r1low * x2

	h0 += sr2lowx4
	r1highx2 = r1high * x2

	h2 += r0lowx2
	r2lowx2 = r2low * x2

	h3 += r0highx2
	r2highx2 = r2high * x2

	h4 += r1lowx2
	sr3lowx2 = sr3low * x2

	h5 += r1highx2
	sr3highx2 = sr3high * x2

	p += 16
	l -= 16
	h6 += r2lowx2

	h7 += r2highx2

	z1 = math.Float64frombits(uint64(d1))
	h0 += sr3lowx2

	z0 = math.Float64frombits(uint64(d0))
	h1 += sr3highx2

	z1 -= alpha32

	z0 -= alpha0

	h5 += z3

	h3 += z2

	h1 += z1

	h0 += z0

	if l >= 16 {
		goto multiplyaddatleast16bytes
	}

multiplyaddatmost15bytes:

	y7 = h7 + alpha130

	y6 = h6 + alpha130

	y1 = h1 + alpha32

	y0 = h0 + alpha32

	y7 -= alpha130

	y6 -= alpha130

	y1 -= alpha32

	y0 -= alpha32

	y5 = h5 + alpha96

	y4 = h4 + alpha96

	x7 = h7 - y7
	y7 *= scale

	x6 = h6 - y6
	y6 *= scale

	x1 = h1 - y1

	x0 = h0 - y0

	y5 -= alpha96

	y4 -= alpha96

	x1 += y7

	x0 += y6

	x7 += y5

	x6 += y4

	y3 = h3 + alpha64

	y2 = h2 + alpha64

	x0 += x1

	x6 += x7

	y3 -= alpha64
	r3low = r3low_stack

	y2 -= alpha64
	r0low = r0low_stack

	x5 = h5 - y5
	r3lowx0 = r3low * x0
	r3high = r3high_stack

	x4 = h4 - y4
	r0lowx6 = r0low * x6
	r0high = r0high_stack

	x3 = h3 - y3
	r3highx0 = r3high * x0
	sr1low = sr1low_stack

	x2 = h2 - y2
	r0highx6 = r0high * x6
	sr1high = sr1high_stack

	x5 += y3
	r0lowx0 = r0low * x0
	r1low = r1low_stack

	h6 = r3lowx0 + r0lowx6
	sr1lowx6 = sr1low * x6
	r1high = r1high_stack

	x4 += y2
	r0highx0 = r0high * x0
	sr2low = sr2low_stack

	h7 = r3highx0 + r0highx6
	sr1highx6 = sr1high * x6
	sr2high = sr2high_stack

	x3 += y1
	r1lowx0 = r1low * x0
	r2low = r2low_stack

	h0 = r0lowx0 + sr1lowx6
	sr2lowx6 = sr2low * x6
	r2high = r2high_stack

	x2 += y0
	r1highx0 = r1high * x0
	sr3low = sr3low_stack

	h1 = r0highx0 + sr1highx6
	sr2highx6 = sr2high * x6
	sr3high = sr3high_stack

	x4 += x5
	r2lowx0 = r2low * x0

	h2 = r1lowx0 + sr2lowx6
	sr3lowx6 = sr3low * x6

	x2 += x3
	r2highx0 = r2high * x0

	h3 = r1highx0 + sr2highx6
	sr3highx6 = sr3high * x6

	r1highx4 = r1high * x4

	h4 = r2lowx0 + sr3lowx6
	r1lowx4 = r1low * x4

	r0highx4 = r0high * x4

	h5 = r2highx0 + sr3highx6
	r0lowx4 = r0low * x4

	h7 += r1highx4
	sr3highx4 = sr3high * x4

	h6 += r1lowx4
	sr3lowx4 = sr3low * x4

	h5 += r0highx4
	sr2highx4 = sr2high * x4

	h4 += r0lowx4
	sr2lowx4 = sr2low * x4

	h3 += sr3highx4
	r0lowx2 = r0low * x2

	h2 += sr3lowx4
	r0highx2 = r0high * x2

	h1 += sr2highx4
	r1lowx2 = r1low * x2

	h0 += sr2lowx4
	r1highx2 = r1high * x2

	h2 += r0lowx2
	r2lowx2 = r2low * x2

	h3 += r0highx2
	r2highx2 = r2high * x2

	h4 += r1lowx2
	sr3lowx2 = sr3low * x2

	h5 += r1highx2
	sr3highx2 = sr3high * x2

	h6 += r2lowx2

	h7 += r2highx2

	h0 += sr3lowx2

	h1 += sr3highx2

addatmost15bytes:

	if l == 0 {
		goto nomorebytes
	}

	lbelow2 = l - 2

	lbelow3 = l - 3

	lbelow2 >>= 31
	lbelow4 = l - 4

	m00 = uint32(m[p+0])
	lbelow3 >>= 31
	p += lbelow2

	m01 = uint32(m[p+1])
	lbelow4 >>= 31
	p += lbelow3

	m02 = uint32(m[p+2])
	p += lbelow4
	m0 = 2151

	m03 = uint32(m[p+3])
	m0 <<= 51
	m1 = 2215

	m0 += int64(m00)
	m01 &^= uint32(lbelow2)

	m02 &^= uint32(lbelow3)
	m01 -= uint32(lbelow2)

	m01 <<= 8
	m03 &^= uint32(lbelow4)

	m0 += int64(m01)
	lbelow2 -= lbelow3

	m02 += uint32(lbelow2)
	lbelow3 -= lbelow4

	m02 <<= 16
	m03 += uint32(lbelow3)

	m03 <<= 24
	m0 += int64(m02)

	m0 += int64(m03)
	lbelow5 = l - 5

	lbelow6 = l - 6
	lbelow7 = l - 7

	lbelow5 >>= 31
	lbelow8 = l - 8

	lbelow6 >>= 31
	p += lbelow5

	m10 = uint32(m[p+4])
	lbelow7 >>= 31
	p += lbelow6

	m11 = uint32(m[p+5])
	lbelow8 >>= 31
	p += lbelow7

	m12 = uint32(m[p+6])
	m1 <<= 51
	p += lbelow8

	m13 = uint32(m[p+7])
	m10 &^= uint32(lbelow5)
	lbelow4 -= lbelow5

	m10 += uint32(lbelow4)
	lbelow5 -= lbelow6

	m11 &^= uint32(lbelow6)
	m11 += uint32(lbelow5)

	m11 <<= 8
	m1 += int64(m10)

	m1 += int64(m11)
	m12 &^= uint32(lbelow7)

	lbelow6 -= lbelow7
	m13 &^= uint32(lbelow8)

	m12 += uint32(lbelow6)
	lbelow7 -= lbelow8

	m12 <<= 16
	m13 += uint32(lbelow7)

	m13 <<= 24
	m1 += int64(m12)

	m1 += int64(m13)
	m2 = 2279

	lbelow9 = l - 9
	m3 = 2343

	lbelow10 = l - 10
	lbelow11 = l - 11

	lbelow9 >>= 31
	lbelow12 = l - 12

	lbelow10 >>= 31
	p += lbelow9

	m20 = uint32(m[p+8])
	lbelow11 >>= 31
	p += lbelow10

	m21 = uint32(m[p+9])
	lbelow12 >>= 31
	p += lbelow11

	m22 = uint32(m[p+10])
	m2 <<= 51
	p += lbelow12

	m23 = uint32(m[p+11])
	m20 &^= uint32(lbelow9)
	lbelow8 -= lbelow9

	m20 += uint32(lbelow8)
	lbelow9 -= lbelow10

	m21 &^= uint32(lbelow10)
	m21 += uint32(lbelow9)

	m21 <<= 8
	m2 += int64(m20)

	m2 += int64(m21)
	m22 &^= uint32(lbelow11)

	lbelow10 -= lbelow11
	m23 &^= uint32(lbelow12)

	m22 += uint32(lbelow10)
	lbelow11 -= lbelow12

	m22 <<= 16
	m23 += uint32(lbelow11)

	m23 <<= 24
	m2 += int64(m22)

	m3 <<= 51
	lbelow13 = l - 13

	lbelow13 >>= 31
	lbelow14 = l - 14

	lbelow14 >>= 31
	p += lbelow13
	lbelow15 = l - 15

	m30 = uint32(m[p+12])
	lbelow15 >>= 31
	p += lbelow14

	m31 = uint32(m[p+13])
	p += lbelow15
	m2 += int64(m23)

	m32 = uint32(m[p+14])
	m30 &^= uint32(lbelow13)
	lbelow12 -= lbelow13

	m30 += uint32(lbelow12)
	lbelow13 -= lbelow14

	m3 += int64(m30)
	m31 &^= uint32(lbelow14)

	m31 += uint32(lbelow13)
	m32 &^= uint32(lbelow15)

	m31 <<= 8
	lbelow14 -= lbelow15

	m3 += int64(m31)
	m32 += uint32(lbelow14)
	d0 = m0

	m32 <<= 16
	m33 = uint64(lbelow15 + 1)
	d1 = m1

	m33 <<= 24
	m3 += int64(m32)
	d2 = m2

	m3 += int64(m33)
	d3 = m3

	z3 = math.Float64frombits(uint64(d3))

	z2 = math.Float64frombits(uint64(d2))

	z1 = math.Float64frombits(uint64(d1))

	z0 = math.Float64frombits(uint64(d0))

	z3 -= alpha96

	z2 -= alpha64

	z1 -= alpha32

	z0 -= alpha0

	h5 += z3

	h3 += z2

	h1 += z1

	h0 += z0

	y7 = h7 + alpha130

	y6 = h6 + alpha130

	y1 = h1 + alpha32

	y0 = h0 + alpha32

	y7 -= alpha130

	y6 -= alpha130

	y1 -= alpha32

	y0 -= alpha32

	y5 = h5 + alpha96

	y4 = h4 + alpha96

	x7 = h7 - y7
	y7 *= scale

	x6 = h6 - y6
	y6 *= scale

	x1 = h1 - y1

	x0 = h0 - y0

	y5 -= alpha96

	y4 -= alpha96

	x1 += y7

	x0 += y6

	x7 += y5

	x6 += y4

	y3 = h3 + alpha64

	y2 = h2 + alpha64

	x0 += x1

	x6 += x7

	y3 -= alpha64
	r3low = r3low_stack

	y2 -= alpha64
	r0low = r0low_stack

	x5 = h5 - y5
	r3lowx0 = r3low * x0
	r3high = r3high_stack

	x4 = h4 - y4
	r0lowx6 = r0low * x6
	r0high = r0high_stack

	x3 = h3 - y3
	r3highx0 = r3high * x0
	sr1low = sr1low_stack

	x2 = h2 - y2
	r0highx6 = r0high * x6
	sr1high = sr1high_stack

	x5 += y3
	r0lowx0 = r0low * x0
	r1low = r1low_stack

	h6 = r3lowx0 + r0lowx6
	sr1lowx6 = sr1low * x6
	r1high = r1high_stack

	x4 += y2
	r0highx0 = r0high * x0
	sr2low = sr2low_stack

	h7 = r3highx0 + r0highx6
	sr1highx6 = sr1high * x6
	sr2high = sr2high_stack

	x3 += y1
	r1lowx0 = r1low * x0
	r2low = r2low_stack

	h0 = r0lowx0 + sr1lowx6
	sr2lowx6 = sr2low * x6
	r2high = r2high_stack

	x2 += y0
	r1highx0 = r1high * x0
	sr3low = sr3low_stack

	h1 = r0highx0 + sr1highx6
	sr2highx6 = sr2high * x6
	sr3high = sr3high_stack

	x4 += x5
	r2lowx0 = r2low * x0

	h2 = r1lowx0 + sr2lowx6
	sr3lowx6 = sr3low * x6

	x2 += x3
	r2highx0 = r2high * x0

	h3 = r1highx0 + sr2highx6
	sr3highx6 = sr3high * x6

	r1highx4 = r1high * x4

	h4 = r2lowx0 + sr3lowx6
	r1lowx4 = r1low * x4

	r0highx4 = r0high * x4

	h5 = r2highx0 + sr3highx6
	r0lowx4 = r0low * x4

	h7 += r1highx4
	sr3highx4 = sr3high * x4

	h6 += r1lowx4
	sr3lowx4 = sr3low * x4

	h5 += r0highx4
	sr2highx4 = sr2high * x4

	h4 += r0lowx4
	sr2lowx4 = sr2low * x4

	h3 += sr3highx4
	r0lowx2 = r0low * x2

	h2 += sr3lowx4
	r0highx2 = r0high * x2

	h1 += sr2highx4
	r1lowx2 = r1low * x2

	h0 += sr2lowx4
	r1highx2 = r1high * x2

	h2 += r0lowx2
	r2lowx2 = r2low * x2

	h3 += r0highx2
	r2highx2 = r2high * x2

	h4 += r1lowx2
	sr3lowx2 = sr3low * x2

	h5 += r1highx2
	sr3highx2 = sr3high * x2

	h6 += r2lowx2

	h7 += r2highx2

	h0 += sr3lowx2

	h1 += sr3highx2

nomorebytes:

	y7 = h7 + alpha130

	y0 = h0 + alpha32

	y1 = h1 + alpha32

	y2 = h2 + alpha64

	y7 -= alpha130

	y3 = h3 + alpha64

	y4 = h4 + alpha96

	y5 = h5 + alpha96

	x7 = h7 - y7
	y7 *= scale

	y0 -= alpha32

	y1 -= alpha32

	y2 -= alpha64

	h6 += x7

	y3 -= alpha64

	y4 -= alpha96

	y5 -= alpha96

	y6 = h6 + alpha130

	x0 = h0 - y0

	x1 = h1 - y1

	x2 = h2 - y2

	y6 -= alpha130

	x0 += y7

	x3 = h3 - y3

	x4 = h4 - y4

	x5 = h5 - y5

	x6 = h6 - y6

	y6 *= scale

	x2 += y0

	x3 += y1

	x4 += y2

	x0 += y6

	x5 += y3

	x6 += y4

	x2 += x3

	x0 += x1

	x4 += x5

	x6 += y5

	x2 += offset1
	d1 = int64(math.Float64bits(x2))

	x0 += offset0
	d0 = int64(math.Float64bits(x0))

	x4 += offset2
	d2 = int64(math.Float64bits(x4))

	x6 += offset3
	d3 = int64(math.Float64bits(x6))

	f0 = uint64(d0)

	f1 = uint64(d1)
	bits32 = math.MaxUint64

	f2 = uint64(d2)
	bits32 >>= 32

	f3 = uint64(d3)
	f = f0 >> 32

	f0 &= bits32
	f &= 255

	f1 += f
	g0 = f0 + 5

	g = g0 >> 32
	g0 &= bits32

	f = f1 >> 32
	f1 &= bits32

	f &= 255
	g1 = f1 + g

	g = g1 >> 32
	f2 += f

	f = f2 >> 32
	g1 &= bits32

	f2 &= bits32
	f &= 255

	f3 += f
	g2 = f2 + g

	g = g2 >> 32
	g2 &= bits32

	f4 = f3 >> 32
	f3 &= bits32

	f4 &= 255
	g3 = f3 + g

	g = g3 >> 32
	g3 &= bits32

	g4 = f4 + g

	g4 = g4 - 4
	s00 = uint32(s[0])

	f = uint64(int64(g4) >> 63)
	s01 = uint32(s[1])

	f0 &= f
	g0 &^= f
	s02 = uint32(s[2])

	f1 &= f
	f0 |= g0
	s03 = uint32(s[3])

	g1 &^= f
	f2 &= f
	s10 = uint32(s[4])

	f3 &= f
	g2 &^= f
	s11 = uint32(s[5])

	g3 &^= f
	f1 |= g1
	s12 = uint32(s[6])

	f2 |= g2
	f3 |= g3
	s13 = uint32(s[7])

	s01 <<= 8
	f0 += uint64(s00)
	s20 = uint32(s[8])

	s02 <<= 16
	f0 += uint64(s01)
	s21 = uint32(s[9])

	s03 <<= 24
	f0 += uint64(s02)
	s22 = uint32(s[10])

	s11 <<= 8
	f1 += uint64(s10)
	s23 = uint32(s[11])

	s12 <<= 16
	f1 += uint64(s11)
	s30 = uint32(s[12])

	s13 <<= 24
	f1 += uint64(s12)
	s31 = uint32(s[13])

	f0 += uint64(s03)
	f1 += uint64(s13)
	s32 = uint32(s[14])

	s21 <<= 8
	f2 += uint64(s20)
	s33 = uint32(s[15])

	s22 <<= 16
	f2 += uint64(s21)

	s23 <<= 24
	f2 += uint64(s22)

	s31 <<= 8
	f3 += uint64(s30)

	s32 <<= 16
	f3 += uint64(s31)

	s33 <<= 24
	f3 += uint64(s32)

	f2 += uint64(s23)
	f3 += uint64(s33)

	out[0] = byte(f0)
	f0 >>= 8
	out[1] = byte(f0)
	f0 >>= 8
	out[2] = byte(f0)
	f0 >>= 8
	out[3] = byte(f0)
	f0 >>= 8
	f1 += f0

	out[4] = byte(f1)
	f1 >>= 8
	out[5] = byte(f1)
	f1 >>= 8
	out[6] = byte(f1)
	f1 >>= 8
	out[7] = byte(f1)
	f1 >>= 8
	f2 += f1

	out[8] = byte(f2)
	f2 >>= 8
	out[9] = byte(f2)
	f2 >>= 8
	out[10] = byte(f2)
	f2 >>= 8
	out[11] = byte(f2)
	f2 >>= 8
	f3 += f2

	out[12] = byte(f3)
	f3 >>= 8
	out[13] = byte(f3)
	f3 >>= 8
	out[14] = byte(f3)
	f3 >>= 8
	out[15] = byte(f3)
}
