// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

// 64-bit MD5 (does full MD5 but returns 64 bits only).
// Translation of ../../crypto/md5/md5*.go.

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

type MD5 struct {
	s   [4]uint32
	x   [64]uint8
	nx  int
	len uint64
}

const (
	_Chunk = 64
)

const (
	_Init0 = 0x67452301
	_Init1 = 0xEFCDAB89
	_Init2 = 0x98BADCFE
	_Init3 = 0x10325476
)

func md5reset(d *MD5) {
	d.s[0] = _Init0
	d.s[1] = _Init1
	d.s[2] = _Init2
	d.s[3] = _Init3
	d.nx = 0
	d.len = 0
}

func md5write(d *MD5, p []byte, nn int) {
	d.len += uint64(nn)
	if d.nx > 0 {
		n := nn
		if n > _Chunk-d.nx {
			n = _Chunk - d.nx
		}
		for i := 0; i < n; i++ {
			d.x[d.nx+i] = uint8(p[i])
		}
		d.nx += n
		if d.nx == _Chunk {
			md5block(d, d.x[:], _Chunk)
			d.nx = 0
		}

		p = p[n:]
		nn -= n
	}

	n := md5block(d, p, nn)
	p = p[n:]
	nn -= n
	if nn > 0 {
		for i := 0; i < nn; i++ {
			d.x[i] = uint8(p[i])
		}
		d.nx = nn
	}
}

func md5sum(d *MD5, hi *uint64) uint64 {
	// Padding.  Add a 1 bit and 0 bits until 56 bytes mod 64.
	len := d.len

	tmp := [64]uint8{}
	tmp[0] = 0x80
	if len%64 < 56 {
		md5write(d, tmp[:], int(56-len%64))
	} else {
		md5write(d, tmp[:], int(64+56-len%64))
	}

	// Length in bits.
	len <<= 3

	for i := 0; i < 8; i++ {
		tmp[i] = uint8(len >> uint(8*i))
	}
	md5write(d, tmp[:], 8)

	if d.nx != 0 {
		Fatal("md5sum")
	}

	if hi != nil {
		*hi = uint64(d.s[2]) | uint64(d.s[3])<<32
	}
	return uint64(d.s[0]) | uint64(d.s[1])<<32
}

// MD5 block step.
// In its own file so that a faster assembly or C version
// can be substituted easily.

// table[i] = int((1<<32) * abs(sin(i+1 radians))).
var table = [64]uint32{
	// round 1
	0xd76aa478,
	0xe8c7b756,
	0x242070db,
	0xc1bdceee,
	0xf57c0faf,
	0x4787c62a,
	0xa8304613,
	0xfd469501,
	0x698098d8,
	0x8b44f7af,
	0xffff5bb1,
	0x895cd7be,
	0x6b901122,
	0xfd987193,
	0xa679438e,
	0x49b40821,

	// round 2
	0xf61e2562,
	0xc040b340,
	0x265e5a51,
	0xe9b6c7aa,
	0xd62f105d,
	0x2441453,
	0xd8a1e681,
	0xe7d3fbc8,
	0x21e1cde6,
	0xc33707d6,
	0xf4d50d87,
	0x455a14ed,
	0xa9e3e905,
	0xfcefa3f8,
	0x676f02d9,
	0x8d2a4c8a,

	// round3
	0xfffa3942,
	0x8771f681,
	0x6d9d6122,
	0xfde5380c,
	0xa4beea44,
	0x4bdecfa9,
	0xf6bb4b60,
	0xbebfbc70,
	0x289b7ec6,
	0xeaa127fa,
	0xd4ef3085,
	0x4881d05,
	0xd9d4d039,
	0xe6db99e5,
	0x1fa27cf8,
	0xc4ac5665,

	// round 4
	0xf4292244,
	0x432aff97,
	0xab9423a7,
	0xfc93a039,
	0x655b59c3,
	0x8f0ccc92,
	0xffeff47d,
	0x85845dd1,
	0x6fa87e4f,
	0xfe2ce6e0,
	0xa3014314,
	0x4e0811a1,
	0xf7537e82,
	0xbd3af235,
	0x2ad7d2bb,
	0xeb86d391,
}

var shift1 = []uint32{7, 12, 17, 22}

var shift2 = []uint32{5, 9, 14, 20}

var shift3 = []uint32{4, 11, 16, 23}

var shift4 = []uint32{6, 10, 15, 21}

func md5block(dig *MD5, p []byte, nn int) int {
	var aa uint32
	var bb uint32
	var cc uint32
	var dd uint32
	var i int
	var j int
	var X [16]uint32

	a := dig.s[0]
	b := dig.s[1]
	c := dig.s[2]
	d := dig.s[3]
	n := 0

	for nn >= _Chunk {
		aa = a
		bb = b
		cc = c
		dd = d

		for i = 0; i < 16; i++ {
			j = i * 4
			X[i] = uint32(p[j]) | uint32(p[j+1])<<8 | uint32(p[j+2])<<16 | uint32(p[j+3])<<24
		}

		// Round 1.
		for i = 0; i < 16; i++ {
			x := uint32(i)
			t := uint32(i)
			s := shift1[i%4]
			f := ((c ^ d) & b) ^ d
			a += f + X[x] + table[t]
			a = a<<s | a>>(32-s)
			a += b

			t = d
			d = c
			c = b
			b = a
			a = t
		}

		// Round 2.
		for i = 0; i < 16; i++ {
			x := (1 + 5*uint32(i)) % 16
			t := 16 + uint32(i)
			s := shift2[i%4]
			g := ((b ^ c) & d) ^ c
			a += g + X[x] + table[t]
			a = a<<s | a>>(32-s)
			a += b

			t = d
			d = c
			c = b
			b = a
			a = t
		}

		// Round 3.
		for i = 0; i < 16; i++ {
			x := (5 + 3*uint32(i)) % 16
			t := 32 + uint32(i)
			s := shift3[i%4]
			h := b ^ c ^ d
			a += h + X[x] + table[t]
			a = a<<s | a>>(32-s)
			a += b

			t = d
			d = c
			c = b
			b = a
			a = t
		}

		// Round 4.
		for i = 0; i < 16; i++ {
			x := (7 * uint32(i)) % 16
			s := shift4[i%4]
			t := 48 + uint32(i)
			ii := c ^ (b | ^d)
			a += ii + X[x] + table[t]
			a = a<<s | a>>(32-s)
			a += b

			t = d
			d = c
			c = b
			b = a
			a = t
		}

		a += aa
		b += bb
		c += cc
		d += dd

		p = p[_Chunk:]
		n += _Chunk
		nn -= _Chunk
	}

	dig.s[0] = a
	dig.s[1] = b
	dig.s[2] = c
	dig.s[3] = d
	return n
}
