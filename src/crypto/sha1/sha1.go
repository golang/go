// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sha1 implements the SHA-1 hash algorithm as defined in RFC 3174.
//
// SHA-1 is cryptographically broken and should not be used for secure
// applications.
package sha1

import (
	"crypto"
	"crypto/internal/boring"
	"crypto/internal/fips140only"
	"errors"
	"hash"
	"internal/byteorder"
)

func init() {
	crypto.RegisterHash(crypto.SHA1, New)
}

// The size of a SHA-1 checksum in bytes.
const Size = 20

// The blocksize of SHA-1 in bytes.
const BlockSize = 64

const (
	chunk = 64
	init0 = 0x67452301
	init1 = 0xEFCDAB89
	init2 = 0x98BADCFE
	init3 = 0x10325476
	init4 = 0xC3D2E1F0
)

// digest represents the partial evaluation of a checksum.
type digest struct {
	h   [5]uint32
	x   [chunk]byte
	nx  int
	len uint64
}

const (
	magic         = "sha\x01"
	marshaledSize = len(magic) + 5*4 + chunk + 8
)

func (d *digest) MarshalBinary() ([]byte, error) {
	return d.AppendBinary(make([]byte, 0, marshaledSize))
}

func (d *digest) AppendBinary(b []byte) ([]byte, error) {
	b = append(b, magic...)
	b = byteorder.BEAppendUint32(b, d.h[0])
	b = byteorder.BEAppendUint32(b, d.h[1])
	b = byteorder.BEAppendUint32(b, d.h[2])
	b = byteorder.BEAppendUint32(b, d.h[3])
	b = byteorder.BEAppendUint32(b, d.h[4])
	b = append(b, d.x[:d.nx]...)
	b = append(b, make([]byte, len(d.x)-d.nx)...)
	b = byteorder.BEAppendUint64(b, d.len)
	return b, nil
}

func (d *digest) UnmarshalBinary(b []byte) error {
	if len(b) < len(magic) || string(b[:len(magic)]) != magic {
		return errors.New("crypto/sha1: invalid hash state identifier")
	}
	if len(b) != marshaledSize {
		return errors.New("crypto/sha1: invalid hash state size")
	}
	b = b[len(magic):]
	b, d.h[0] = consumeUint32(b)
	b, d.h[1] = consumeUint32(b)
	b, d.h[2] = consumeUint32(b)
	b, d.h[3] = consumeUint32(b)
	b, d.h[4] = consumeUint32(b)
	b = b[copy(d.x[:], b):]
	b, d.len = consumeUint64(b)
	d.nx = int(d.len % chunk)
	return nil
}

func consumeUint64(b []byte) ([]byte, uint64) {
	return b[8:], byteorder.BEUint64(b)
}

func consumeUint32(b []byte) ([]byte, uint32) {
	return b[4:], byteorder.BEUint32(b)
}

func (d *digest) Reset() {
	d.h[0] = init0
	d.h[1] = init1
	d.h[2] = init2
	d.h[3] = init3
	d.h[4] = init4
	d.nx = 0
	d.len = 0
}

// New returns a new [hash.Hash] computing the SHA1 checksum. The Hash
// also implements [encoding.BinaryMarshaler], [encoding.BinaryAppender] and
// [encoding.BinaryUnmarshaler] to marshal and unmarshal the internal
// state of the hash.
func New() hash.Hash {
	if boring.Enabled {
		return boring.NewSHA1()
	}
	if fips140only.Enabled {
		panic("crypto/sha1: use of weak SHA-1 is not allowed in FIPS 140-only mode")
	}
	d := new(digest)
	d.Reset()
	return d
}

func (d *digest) Size() int { return Size }

func (d *digest) BlockSize() int { return BlockSize }

func (d *digest) Write(p []byte) (nn int, err error) {
	boring.Unreachable()
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

func (d *digest) Sum(in []byte) []byte {
	boring.Unreachable()
	// Make a copy of d so that caller can keep writing and summing.
	d0 := *d
	hash := d0.checkSum()
	return append(in, hash[:]...)
}

func (d *digest) checkSum() [Size]byte {
	len := d.len
	// Padding.  Add a 1 bit and 0 bits until 56 bytes mod 64.
	var tmp [64 + 8]byte // padding + length buffer
	tmp[0] = 0x80
	var t uint64
	if len%64 < 56 {
		t = 56 - len%64
	} else {
		t = 64 + 56 - len%64
	}

	// Length in bits.
	len <<= 3
	padlen := tmp[:t+8]
	byteorder.BEPutUint64(padlen[t:], len)
	d.Write(padlen)

	if d.nx != 0 {
		panic("d.nx != 0")
	}

	var digest [Size]byte

	byteorder.BEPutUint32(digest[0:], d.h[0])
	byteorder.BEPutUint32(digest[4:], d.h[1])
	byteorder.BEPutUint32(digest[8:], d.h[2])
	byteorder.BEPutUint32(digest[12:], d.h[3])
	byteorder.BEPutUint32(digest[16:], d.h[4])

	return digest
}

// ConstantTimeSum computes the same result of [Sum] but in constant time
func (d *digest) ConstantTimeSum(in []byte) []byte {
	d0 := *d
	hash := d0.constSum()
	return append(in, hash[:]...)
}

func (d *digest) constSum() [Size]byte {
	var length [8]byte
	l := d.len << 3
	for i := uint(0); i < 8; i++ {
		length[i] = byte(l >> (56 - 8*i))
	}

	nx := byte(d.nx)
	t := nx - 56                 // if nx < 56 then the MSB of t is one
	mask1b := byte(int8(t) >> 7) // mask1b is 0xFF iff one block is enough

	separator := byte(0x80) // gets reset to 0x00 once used
	for i := byte(0); i < chunk; i++ {
		mask := byte(int8(i-nx) >> 7) // 0x00 after the end of data

		// if we reached the end of the data, replace with 0x80 or 0x00
		d.x[i] = (^mask & separator) | (mask & d.x[i])

		// zero the separator once used
		separator &= mask

		if i >= 56 {
			// we might have to write the length here if all fit in one block
			d.x[i] |= mask1b & length[i-56]
		}
	}

	// compress, and only keep the digest if all fit in one block
	block(d, d.x[:])

	var digest [Size]byte
	for i, s := range d.h {
		digest[i*4] = mask1b & byte(s>>24)
		digest[i*4+1] = mask1b & byte(s>>16)
		digest[i*4+2] = mask1b & byte(s>>8)
		digest[i*4+3] = mask1b & byte(s)
	}

	for i := byte(0); i < chunk; i++ {
		// second block, it's always past the end of data, might start with 0x80
		if i < 56 {
			d.x[i] = separator
			separator = 0
		} else {
			d.x[i] = length[i-56]
		}
	}

	// compress, and only keep the digest if we actually needed the second block
	block(d, d.x[:])

	for i, s := range d.h {
		digest[i*4] |= ^mask1b & byte(s>>24)
		digest[i*4+1] |= ^mask1b & byte(s>>16)
		digest[i*4+2] |= ^mask1b & byte(s>>8)
		digest[i*4+3] |= ^mask1b & byte(s)
	}

	return digest
}

// Sum returns the SHA-1 checksum of the data.
func Sum(data []byte) [Size]byte {
	if boring.Enabled {
		return boring.SHA1(data)
	}
	if fips140only.Enabled {
		panic("crypto/sha1: use of weak SHA-1 is not allowed in FIPS 140-only mode")
	}
	var d digest
	d.Reset()
	d.Write(data)
	return d.checkSum()
}
