// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements the SHA512 hash algorithm as defined in FIPS 180-2.
package sha512

import (
	"hash"
	"os"
)

// The size of a SHA512 checksum in bytes.
const Size = 64

const (
	_Chunk = 128
	_Init0 = 0x6a09e667f3bcc908
	_Init1 = 0xbb67ae8584caa73b
	_Init2 = 0x3c6ef372fe94f82b
	_Init3 = 0xa54ff53a5f1d36f1
	_Init4 = 0x510e527fade682d1
	_Init5 = 0x9b05688c2b3e6c1f
	_Init6 = 0x1f83d9abfb41bd6b
	_Init7 = 0x5be0cd19137e2179
)

// digest represents the partial evaluation of a checksum.
type digest struct {
	h   [8]uint64
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
	d.h[5] = _Init5
	d.h[6] = _Init6
	d.h[7] = _Init7
	d.nx = 0
	d.len = 0
}

// New returns a new hash.Hash computing the SHA512 checksum.
func New() hash.Hash {
	d := new(digest)
	d.Reset()
	return d
}

func (d *digest) Size() int { return Size }

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
			_Block(d, &d.x)
			d.nx = 0
		}
		p = p[n:]
	}
	n := _Block(d, p)
	p = p[n:]
	if len(p) > 0 {
		for i, x := range p {
			d.x[i] = x
		}
		d.nx = len(p)
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
		panicln("oops")
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
	return p
}
