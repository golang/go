// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sha512 implements the SHA-384, SHA-512, SHA-512/224, and SHA-512/256
// hash algorithms as defined in FIPS 180-4.
package sha512

import (
	"crypto"
	"hash"
)

func init() {
	crypto.RegisterHash(crypto.SHA384, New384)
	crypto.RegisterHash(crypto.SHA512, New)
	crypto.RegisterHash(crypto.SHA512_224, New512_224)
	crypto.RegisterHash(crypto.SHA512_256, New512_256)
}

const (
	// Size is the size, in bytes, of a SHA-512 checksum.
	Size = 64

	// Size224 is the size, in bytes, of a SHA-512/224 checksum.
	Size224 = 28

	// Size256 is the size, in bytes, of a SHA-512/256 checksum.
	Size256 = 32

	// Size384 is the size, in bytes, of a SHA-384 checksum.
	Size384 = 48

	// BlockSize is the block size, in bytes, of the SHA-512/224,
	// SHA-512/256, SHA-384 and SHA-512 hash functions.
	BlockSize = 128
)

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
	init0_224 = 0x8c3d37c819544da2
	init1_224 = 0x73e1996689dcd4d6
	init2_224 = 0x1dfab7ae32ff9c82
	init3_224 = 0x679dd514582f9fcf
	init4_224 = 0x0f6d2b697bd44da8
	init5_224 = 0x77e36f7304c48942
	init6_224 = 0x3f9d85a86a1d36c8
	init7_224 = 0x1112e6ad91d692a1
	init0_256 = 0x22312194fc2bf72c
	init1_256 = 0x9f555fa3c84c64c2
	init2_256 = 0x2393b86b6f53b151
	init3_256 = 0x963877195940eabd
	init4_256 = 0x96283ee2a88effe3
	init5_256 = 0xbe5e1e2553863992
	init6_256 = 0x2b0199fc2c85b8aa
	init7_256 = 0x0eb72ddc81c52ca2
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
	h        [8]uint64
	x        [chunk]byte
	nx       int
	len      uint64
	function crypto.Hash
}

func (d *digest) Reset() {
	switch d.function {
	case crypto.SHA384:
		d.h[0] = init0_384
		d.h[1] = init1_384
		d.h[2] = init2_384
		d.h[3] = init3_384
		d.h[4] = init4_384
		d.h[5] = init5_384
		d.h[6] = init6_384
		d.h[7] = init7_384
	case crypto.SHA512_224:
		d.h[0] = init0_224
		d.h[1] = init1_224
		d.h[2] = init2_224
		d.h[3] = init3_224
		d.h[4] = init4_224
		d.h[5] = init5_224
		d.h[6] = init6_224
		d.h[7] = init7_224
	case crypto.SHA512_256:
		d.h[0] = init0_256
		d.h[1] = init1_256
		d.h[2] = init2_256
		d.h[3] = init3_256
		d.h[4] = init4_256
		d.h[5] = init5_256
		d.h[6] = init6_256
		d.h[7] = init7_256
	default:
		d.h[0] = init0
		d.h[1] = init1
		d.h[2] = init2
		d.h[3] = init3
		d.h[4] = init4
		d.h[5] = init5
		d.h[6] = init6
		d.h[7] = init7
	}
	d.nx = 0
	d.len = 0
}

// New returns a new hash.Hash computing the SHA-512 checksum.
func New() hash.Hash {
	d := &digest{function: crypto.SHA512}
	d.Reset()
	return d
}

// New512_224 returns a new hash.Hash computing the SHA-512/224 checksum.
func New512_224() hash.Hash {
	d := &digest{function: crypto.SHA512_224}
	d.Reset()
	return d
}

// New512_256 returns a new hash.Hash computing the SHA-512/256 checksum.
func New512_256() hash.Hash {
	d := &digest{function: crypto.SHA512_256}
	d.Reset()
	return d
}

// New384 returns a new hash.Hash computing the SHA-384 checksum.
func New384() hash.Hash {
	d := &digest{function: crypto.SHA384}
	d.Reset()
	return d
}

func (d *digest) Size() int {
	switch d.function {
	case crypto.SHA512_224:
		return Size224
	case crypto.SHA512_256:
		return Size256
	case crypto.SHA384:
		return Size384
	default:
		return Size
	}
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
	switch d.function {
	case crypto.SHA384:
		return append(in, hash[:Size384]...)
	case crypto.SHA512_224:
		return append(in, hash[:Size224]...)
	case crypto.SHA512_256:
		return append(in, hash[:Size256]...)
	default:
		return append(in, hash[:]...)
	}
}

func (d *digest) checkSum() [Size]byte {
	// Padding. Add a 1 bit and 0 bits until 112 bytes mod 128.
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
	if d.function == crypto.SHA384 {
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
	d := digest{function: crypto.SHA512}
	d.Reset()
	d.Write(data)
	return d.checkSum()
}

// Sum384 returns the SHA384 checksum of the data.
func Sum384(data []byte) (sum384 [Size384]byte) {
	d := digest{function: crypto.SHA384}
	d.Reset()
	d.Write(data)
	sum := d.checkSum()
	copy(sum384[:], sum[:Size384])
	return
}

// Sum512_224 returns the Sum512/224 checksum of the data.
func Sum512_224(data []byte) (sum224 [Size224]byte) {
	d := digest{function: crypto.SHA512_224}
	d.Reset()
	d.Write(data)
	sum := d.checkSum()
	copy(sum224[:], sum[:Size224])
	return
}

// Sum512_256 returns the Sum512/256 checksum of the data.
func Sum512_256(data []byte) (sum256 [Size256]byte) {
	d := digest{function: crypto.SHA512_256}
	d.Reset()
	d.Write(data)
	sum := d.checkSum()
	copy(sum256[:], sum[:Size256])
	return
}
