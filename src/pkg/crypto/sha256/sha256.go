// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sha256 implements the SHA224 and SHA256 hash algorithms as defined
// in FIPS 180-2.
package sha256

import (
	"crypto"
	"hash"
)

func init() {
	crypto.RegisterHash(crypto.SHA224, New224)
	crypto.RegisterHash(crypto.SHA256, New)
}

// The size of a SHA256 checksum in bytes.
const Size = 32

// The size of a SHA224 checksum in bytes.
const Size224 = 28

const (
	_Chunk     = 64
	_Init0     = 0x6A09E667
	_Init1     = 0xBB67AE85
	_Init2     = 0x3C6EF372
	_Init3     = 0xA54FF53A
	_Init4     = 0x510E527F
	_Init5     = 0x9B05688C
	_Init6     = 0x1F83D9AB
	_Init7     = 0x5BE0CD19
	_Init0_224 = 0xC1059ED8
	_Init1_224 = 0x367CD507
	_Init2_224 = 0x3070DD17
	_Init3_224 = 0xF70E5939
	_Init4_224 = 0xFFC00B31
	_Init5_224 = 0x68581511
	_Init6_224 = 0x64F98FA7
	_Init7_224 = 0xBEFA4FA4
)

// digest represents the partial evaluation of a checksum.
type digest struct {
	h     [8]uint32
	x     [_Chunk]byte
	nx    int
	len   uint64
	is224 bool // mark if this digest is SHA-224
}

func (d *digest) Reset() {
	if !d.is224 {
		d.h[0] = _Init0
		d.h[1] = _Init1
		d.h[2] = _Init2
		d.h[3] = _Init3
		d.h[4] = _Init4
		d.h[5] = _Init5
		d.h[6] = _Init6
		d.h[7] = _Init7
	} else {
		d.h[0] = _Init0_224
		d.h[1] = _Init1_224
		d.h[2] = _Init2_224
		d.h[3] = _Init3_224
		d.h[4] = _Init4_224
		d.h[5] = _Init5_224
		d.h[6] = _Init6_224
		d.h[7] = _Init7_224
	}
	d.nx = 0
	d.len = 0
}

// New returns a new hash.Hash computing the SHA256 checksum.
func New() hash.Hash {
	d := new(digest)
	d.Reset()
	return d
}

// New224 returns a new hash.Hash computing the SHA224 checksum.
func New224() hash.Hash {
	d := new(digest)
	d.is224 = true
	d.Reset()
	return d
}

func (d *digest) Size() int {
	if !d.is224 {
		return Size
	}
	return Size224
}

func (d *digest) Write(p []byte) (nn int, err error) {
	nn = len(p)
	d.len += uint64(nn)
	if d.nx > 0 {
		n := len(p)
		if n > _Chunk-d.nx {
			n = _Chunk - d.nx
		}
		for i := 0; i < n; i++ {
			d.x[d.nx+i] = p[i]
		}
		d.nx += n
		if d.nx == _Chunk {
			_Block(d, d.x[0:])
			d.nx = 0
		}
		p = p[n:]
	}
	n := _Block(d, p)
	p = p[n:]
	if len(p) > 0 {
		d.nx = copy(d.x[:], p)
	}
	return
}

func (d0 *digest) Sum(in []byte) []byte {
	// Make a copy of d0 so that caller can keep writing and summing.
	d := *d0

	// Padding.  Add a 1 bit and 0 bits until 56 bytes mod 64.
	len := d.len
	var tmp [64]byte
	tmp[0] = 0x80
	if len%64 < 56 {
		d.Write(tmp[0 : 56-len%64])
	} else {
		d.Write(tmp[0 : 64+56-len%64])
	}

	// Length in bits.
	len <<= 3
	for i := uint(0); i < 8; i++ {
		tmp[i] = byte(len >> (56 - 8*i))
	}
	d.Write(tmp[0:8])

	if d.nx != 0 {
		panic("d.nx != 0")
	}

	h := d.h[:]
	size := Size
	if d.is224 {
		h = d.h[:7]
		size = Size224
	}

	var digest [Size]byte
	for i, s := range h {
		digest[i*4] = byte(s >> 24)
		digest[i*4+1] = byte(s >> 16)
		digest[i*4+2] = byte(s >> 8)
		digest[i*4+3] = byte(s)
	}

	return append(in, digest[:size]...)
}
