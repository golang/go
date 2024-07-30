// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sha256 implements the SHA224 and SHA256 hash algorithms as defined
// in FIPS 180-4.
package sha256

import (
	"crypto"
	"crypto/internal/boring"
	"errors"
	"hash"
	"internal/byteorder"
)

func init() {
	crypto.RegisterHash(crypto.SHA224, New224)
	crypto.RegisterHash(crypto.SHA256, New)
}

// The size of a SHA256 checksum in bytes.
const Size = 32

// The size of a SHA224 checksum in bytes.
const Size224 = 28

// The blocksize of SHA256 and SHA224 in bytes.
const BlockSize = 64

const (
	chunk     = 64
	init0     = 0x6A09E667
	init1     = 0xBB67AE85
	init2     = 0x3C6EF372
	init3     = 0xA54FF53A
	init4     = 0x510E527F
	init5     = 0x9B05688C
	init6     = 0x1F83D9AB
	init7     = 0x5BE0CD19
	init0_224 = 0xC1059ED8
	init1_224 = 0x367CD507
	init2_224 = 0x3070DD17
	init3_224 = 0xF70E5939
	init4_224 = 0xFFC00B31
	init5_224 = 0x68581511
	init6_224 = 0x64F98FA7
	init7_224 = 0xBEFA4FA4
)

// digest represents the partial evaluation of a checksum.
type digest struct {
	h     [8]uint32
	x     [chunk]byte
	nx    int
	len   uint64
	is224 bool // mark if this digest is SHA-224
}

const (
	magic224      = "sha\x02"
	magic256      = "sha\x03"
	marshaledSize = len(magic256) + 8*4 + chunk + 8
)

func (d *digest) MarshalBinary() ([]byte, error) {
	return d.AppendBinary(make([]byte, 0, marshaledSize))
}

func (d *digest) AppendBinary(b []byte) ([]byte, error) {
	if d.is224 {
		b = append(b, magic224...)
	} else {
		b = append(b, magic256...)
	}
	b = byteorder.BeAppendUint32(b, d.h[0])
	b = byteorder.BeAppendUint32(b, d.h[1])
	b = byteorder.BeAppendUint32(b, d.h[2])
	b = byteorder.BeAppendUint32(b, d.h[3])
	b = byteorder.BeAppendUint32(b, d.h[4])
	b = byteorder.BeAppendUint32(b, d.h[5])
	b = byteorder.BeAppendUint32(b, d.h[6])
	b = byteorder.BeAppendUint32(b, d.h[7])
	b = append(b, d.x[:d.nx]...)
	b = append(b, make([]byte, len(d.x)-d.nx)...) // already zero
	b = byteorder.BeAppendUint64(b, d.len)
	return b, nil
}

func (d *digest) UnmarshalBinary(b []byte) error {
	if len(b) < len(magic224) || (d.is224 && string(b[:len(magic224)]) != magic224) || (!d.is224 && string(b[:len(magic256)]) != magic256) {
		return errors.New("crypto/sha256: invalid hash state identifier")
	}
	if len(b) != marshaledSize {
		return errors.New("crypto/sha256: invalid hash state size")
	}
	b = b[len(magic224):]
	b, d.h[0] = consumeUint32(b)
	b, d.h[1] = consumeUint32(b)
	b, d.h[2] = consumeUint32(b)
	b, d.h[3] = consumeUint32(b)
	b, d.h[4] = consumeUint32(b)
	b, d.h[5] = consumeUint32(b)
	b, d.h[6] = consumeUint32(b)
	b, d.h[7] = consumeUint32(b)
	b = b[copy(d.x[:], b):]
	b, d.len = consumeUint64(b)
	d.nx = int(d.len % chunk)
	return nil
}

func consumeUint64(b []byte) ([]byte, uint64) {
	return b[8:], byteorder.BeUint64(b)
}

func consumeUint32(b []byte) ([]byte, uint32) {
	return b[4:], byteorder.BeUint32(b)
}

func (d *digest) Reset() {
	if !d.is224 {
		d.h[0] = init0
		d.h[1] = init1
		d.h[2] = init2
		d.h[3] = init3
		d.h[4] = init4
		d.h[5] = init5
		d.h[6] = init6
		d.h[7] = init7
	} else {
		d.h[0] = init0_224
		d.h[1] = init1_224
		d.h[2] = init2_224
		d.h[3] = init3_224
		d.h[4] = init4_224
		d.h[5] = init5_224
		d.h[6] = init6_224
		d.h[7] = init7_224
	}
	d.nx = 0
	d.len = 0
}

// New returns a new [hash.Hash] computing the SHA256 checksum. The Hash
// also implements [encoding.BinaryMarshaler], [encoding.BinaryAppender] and
// [encoding.BinaryUnmarshaler] to marshal and unmarshal the internal
// state of the hash.
func New() hash.Hash {
	if boring.Enabled {
		return boring.NewSHA256()
	}
	d := new(digest)
	d.Reset()
	return d
}

// New224 returns a new [hash.Hash] computing the SHA224 checksum. The Hash
// also implements [encoding.BinaryMarshaler], [encoding.BinaryAppender] and
// [encoding.BinaryUnmarshaler] to marshal and unmarshal the internal
// state of the hash.
func New224() hash.Hash {
	if boring.Enabled {
		return boring.NewSHA224()
	}
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
	if d0.is224 {
		return append(in, hash[:Size224]...)
	}
	return append(in, hash[:]...)
}

func (d *digest) checkSum() [Size]byte {
	len := d.len
	// Padding. Add a 1 bit and 0 bits until 56 bytes mod 64.
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
	byteorder.BePutUint64(padlen[t+0:], len)
	d.Write(padlen)

	if d.nx != 0 {
		panic("d.nx != 0")
	}

	var digest [Size]byte

	byteorder.BePutUint32(digest[0:], d.h[0])
	byteorder.BePutUint32(digest[4:], d.h[1])
	byteorder.BePutUint32(digest[8:], d.h[2])
	byteorder.BePutUint32(digest[12:], d.h[3])
	byteorder.BePutUint32(digest[16:], d.h[4])
	byteorder.BePutUint32(digest[20:], d.h[5])
	byteorder.BePutUint32(digest[24:], d.h[6])
	if !d.is224 {
		byteorder.BePutUint32(digest[28:], d.h[7])
	}

	return digest
}

// Sum256 returns the SHA256 checksum of the data.
func Sum256(data []byte) [Size]byte {
	if boring.Enabled {
		return boring.SHA256(data)
	}
	var d digest
	d.Reset()
	d.Write(data)
	return d.checkSum()
}

// Sum224 returns the SHA224 checksum of the data.
func Sum224(data []byte) [Size224]byte {
	if boring.Enabled {
		return boring.SHA224(data)
	}
	var d digest
	d.is224 = true
	d.Reset()
	d.Write(data)
	sum := d.checkSum()
	ap := (*[Size224]byte)(sum[:])
	return *ap
}
