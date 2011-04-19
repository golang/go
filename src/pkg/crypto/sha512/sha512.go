// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sha512 implements the SHA384 and SHA512 hash algorithms as defined
// in FIPS 180-2.
package sha512

import (
	"crypto"
	"hash"
	"os"
)

func init() {
	crypto.RegisterHash(crypto.SHA384, New384)
	crypto.RegisterHash(crypto.SHA512, New)
}

// The size of a SHA512 checksum in bytes.
const Size = 64

// The size of a SHA384 checksum in bytes.
const Size384 = 48

const (
	_Chunk     = 128
	_Init0     = 0x6a09e667f3bcc908
	_Init1     = 0xbb67ae8584caa73b
	_Init2     = 0x3c6ef372fe94f82b
	_Init3     = 0xa54ff53a5f1d36f1
	_Init4     = 0x510e527fade682d1
	_Init5     = 0x9b05688c2b3e6c1f
	_Init6     = 0x1f83d9abfb41bd6b
	_Init7     = 0x5be0cd19137e2179
	_Init0_384 = 0xcbbb9d5dc1059ed8
	_Init1_384 = 0x629a292a367cd507
	_Init2_384 = 0x9159015a3070dd17
	_Init3_384 = 0x152fecd8f70e5939
	_Init4_384 = 0x67332667ffc00b31
	_Init5_384 = 0x8eb44a8768581511
	_Init6_384 = 0xdb0c2e0d64f98fa7
	_Init7_384 = 0x47b5481dbefa4fa4
)

// digest represents the partial evaluation of a checksum.
type digest struct {
	h     [8]uint64
	x     [_Chunk]byte
	nx    int
	len   uint64
	is384 bool // mark if this digest is SHA-384
}

func (d *digest) Reset() {
	if !d.is384 {
		d.h[0] = _Init0
		d.h[1] = _Init1
		d.h[2] = _Init2
		d.h[3] = _Init3
		d.h[4] = _Init4
		d.h[5] = _Init5
		d.h[6] = _Init6
		d.h[7] = _Init7
	} else {
		d.h[0] = _Init0_384
		d.h[1] = _Init1_384
		d.h[2] = _Init2_384
		d.h[3] = _Init3_384
		d.h[4] = _Init4_384
		d.h[5] = _Init5_384
		d.h[6] = _Init6_384
		d.h[7] = _Init7_384
	}
	d.nx = 0
	d.len = 0
}

// New returns a new hash.Hash computing the SHA512 checksum.
func New() hash.Hash {
	d := new(digest)
	d.Reset()
	return d
}

// New384 returns a new hash.Hash computing the SHA384 checksum.
func New384() hash.Hash {
	d := new(digest)
	d.is384 = true
	d.Reset()
	return d
}

func (d *digest) Size() int {
	if !d.is384 {
		return Size
	}
	return Size384
}

func (d *digest) Write(p []byte) (nn int, err os.Error) {
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

func (d0 *digest) Sum() []byte {
	// Make a copy of d0 so that caller can keep writing and summing.
	d := new(digest)
	*d = *d0

	// Padding.  Add a 1 bit and 0 bits until 112 bytes mod 128.
	len := d.len
	var tmp [128]byte
	tmp[0] = 0x80
	if len%128 < 112 {
		d.Write(tmp[0 : 112-len%128])
	} else {
		d.Write(tmp[0 : 128+112-len%128])
	}

	// Length in bits.
	len <<= 3
	for i := uint(0); i < 16; i++ {
		tmp[i] = byte(len >> (120 - 8*i))
	}
	d.Write(tmp[0:16])

	if d.nx != 0 {
		panic("d.nx != 0")
	}

	p := make([]byte, 64)
	j := 0
	for _, s := range d.h {
		p[j+0] = byte(s >> 56)
		p[j+1] = byte(s >> 48)
		p[j+2] = byte(s >> 40)
		p[j+3] = byte(s >> 32)
		p[j+4] = byte(s >> 24)
		p[j+5] = byte(s >> 16)
		p[j+6] = byte(s >> 8)
		p[j+7] = byte(s >> 0)
		j += 8
	}
	if d.is384 {
		return p[0:48]
	}
	return p
}
