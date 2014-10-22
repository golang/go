// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sha512 implements the SHA384 and SHA512 hash algorithms as defined
// in FIPS 180-2.
package sha512

import (
	"crypto"
	"hash"
)

func init() {
	crypto.RegisterHash(crypto.SHA384, New384)
	crypto.RegisterHash(crypto.SHA512, New)
}

// The size of a SHA512 checksum in bytes.
const Size = 64

// The size of a SHA384 checksum in bytes.
const Size384 = 48

// The blocksize of SHA512 and SHA384 in bytes.
const BlockSize = 128

const (
	chunk     = 128
	init0     = 0x6a09e667f3bcc908
	init1     = 0xbb67ae8584caa73b
	init2     = 0x3c6ef372fe94f82b
	init3     = 0xa54ff53a5f1d36f1
	init4     = 0x510e527fade682d1
	init5     = 0x9b05688c2b3e6c1f
	init6     = 0x1f83d9abfb41bd6b
	init7     = 0x5be0cd19137e2179
	init0_384 = 0xcbbb9d5dc1059ed8
	init1_384 = 0x629a292a367cd507
	init2_384 = 0x9159015a3070dd17
	init3_384 = 0x152fecd8f70e5939
	init4_384 = 0x67332667ffc00b31
	init5_384 = 0x8eb44a8768581511
	init6_384 = 0xdb0c2e0d64f98fa7
	init7_384 = 0x47b5481dbefa4fa4
)

// digest represents the partial evaluation of a checksum.
type digest struct {
	h     [8]uint64
	x     [chunk]byte
	nx    int
	len   uint64
	is384 bool // mark if this digest is SHA-384
}

func (d *digest) Reset() {
	if !d.is384 {
		d.h[0] = init0
		d.h[1] = init1
		d.h[2] = init2
		d.h[3] = init3
		d.h[4] = init4
		d.h[5] = init5
		d.h[6] = init6
		d.h[7] = init7
	} else {
		d.h[0] = init0_384
		d.h[1] = init1_384
		d.h[2] = init2_384
		d.h[3] = init3_384
		d.h[4] = init4_384
		d.h[5] = init5_384
		d.h[6] = init6_384
		d.h[7] = init7_384
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

func (d *digest) BlockSize() int { return BlockSize }

func (d *digest) Write(p []byte) (nn int, err error) {
	nn = len(p)
	d.len += uint64(nn)
	if d.nx > 0 {
		n := copy(d.x[d.nx:], p)
		d.nx += n
		if d.nx == chunk {
			block(d, d.x[:])
			d.nx = 0
		}
		p = p[n:]
	}
	if len(p) >= chunk {
		n := len(p) &^ (chunk - 1)
		block(d, p[:n])
		p = p[n:]
	}
	if len(p) > 0 {
		d.nx = copy(d.x[:], p)
	}
	return
}

func (d0 *digest) Sum(in []byte) []byte {
	// Make a copy of d0 so that caller can keep writing and summing.
	d := new(digest)
	*d = *d0
	hash := d.checkSum()
	if d.is384 {
		return append(in, hash[:Size384]...)
	}
	return append(in, hash[:]...)
}

func (d *digest) checkSum() [Size]byte {
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

	h := d.h[:]
	if d.is384 {
		h = d.h[:6]
	}

	var digest [Size]byte
	for i, s := range h {
		digest[i*8] = byte(s >> 56)
		digest[i*8+1] = byte(s >> 48)
		digest[i*8+2] = byte(s >> 40)
		digest[i*8+3] = byte(s >> 32)
		digest[i*8+4] = byte(s >> 24)
		digest[i*8+5] = byte(s >> 16)
		digest[i*8+6] = byte(s >> 8)
		digest[i*8+7] = byte(s)
	}

	return digest
}

// Sum512 returns the SHA512 checksum of the data.
func Sum512(data []byte) [Size]byte {
	var d digest
	d.Reset()
	d.Write(data)
	return d.checkSum()
}

// Sum384 returns the SHA384 checksum of the data.
func Sum384(data []byte) (sum384 [Size384]byte) {
	var d digest
	d.is384 = true
	d.Reset()
	d.Write(data)
	sum := d.checkSum()
	copy(sum384[:], sum[:Size384])
	return
}
