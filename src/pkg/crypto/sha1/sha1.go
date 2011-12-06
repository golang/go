// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sha1 implements the SHA1 hash algorithm as defined in RFC 3174.
package sha1

import (
	"crypto"
	"hash"
)

func init() {
	crypto.RegisterHash(crypto.SHA1, New)
}

// The size of a SHA1 checksum in bytes.
const Size = 20

const (
	_Chunk = 64
	_Init0 = 0x67452301
	_Init1 = 0xEFCDAB89
	_Init2 = 0x98BADCFE
	_Init3 = 0x10325476
	_Init4 = 0xC3D2E1F0
)

// digest represents the partial evaluation of a checksum.
type digest struct {
	h   [5]uint32
	x   [_Chunk]byte
	nx  int
	len uint64
}

func (d *digest) Reset() {
	d.h[0] = _Init0
	d.h[1] = _Init1
	d.h[2] = _Init2
	d.h[3] = _Init3
	d.h[4] = _Init4
	d.nx = 0
	d.len = 0
}

// New returns a new hash.Hash computing the SHA1 checksum.
func New() hash.Hash {
	d := new(digest)
	d.Reset()
	return d
}

func (d *digest) Size() int { return Size }

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

	var digest [Size]byte
	for i, s := range d.h {
		digest[i*4] = byte(s >> 24)
		digest[i*4+1] = byte(s >> 16)
		digest[i*4+2] = byte(s >> 8)
		digest[i*4+3] = byte(s)
	}

	return append(in, digest[:]...)
}
