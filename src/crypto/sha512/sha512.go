// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sha512 implements the SHA-384, SHA-512, SHA-512/224, and SHA-512/256
// hash algorithms as defined in FIPS 180-4.
//
// All the hash.Hash implementations returned by this package also
// implement encoding.BinaryMarshaler and encoding.BinaryUnmarshaler to
// marshal and unmarshal the internal state of the hash.
package sha512

import (
	"crypto"
	"crypto/internal/boring"
	"errors"
	"hash"
	"internal/binary"
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

const (
	magic384      = "sha\x04"
	magic512_224  = "sha\x05"
	magic512_256  = "sha\x06"
	magic512      = "sha\x07"
	marshaledSize = len(magic512) + 8*8 + chunk + 8
)

func (d *digest) MarshalBinary() ([]byte, error) {
	b := make([]byte, 0, marshaledSize)
	switch d.function {
	case crypto.SHA384:
		b = append(b, magic384...)
	case crypto.SHA512_224:
		b = append(b, magic512_224...)
	case crypto.SHA512_256:
		b = append(b, magic512_256...)
	case crypto.SHA512:
		b = append(b, magic512...)
	default:
		return nil, errors.New("crypto/sha512: invalid hash function")
	}
	b = binary.BigEndian.AppendUint64(b, d.h[0])
	b = binary.BigEndian.AppendUint64(b, d.h[1])
	b = binary.BigEndian.AppendUint64(b, d.h[2])
	b = binary.BigEndian.AppendUint64(b, d.h[3])
	b = binary.BigEndian.AppendUint64(b, d.h[4])
	b = binary.BigEndian.AppendUint64(b, d.h[5])
	b = binary.BigEndian.AppendUint64(b, d.h[6])
	b = binary.BigEndian.AppendUint64(b, d.h[7])
	b = append(b, d.x[:d.nx]...)
	b = b[:len(b)+len(d.x)-d.nx] // already zero
	b = binary.BigEndian.AppendUint64(b, d.len)
	return b, nil
}

func (d *digest) UnmarshalBinary(b []byte) error {
	if len(b) < len(magic512) {
		return errors.New("crypto/sha512: invalid hash state identifier")
	}
	switch {
	case d.function == crypto.SHA384 && string(b[:len(magic384)]) == magic384:
	case d.function == crypto.SHA512_224 && string(b[:len(magic512_224)]) == magic512_224:
	case d.function == crypto.SHA512_256 && string(b[:len(magic512_256)]) == magic512_256:
	case d.function == crypto.SHA512 && string(b[:len(magic512)]) == magic512:
	default:
		return errors.New("crypto/sha512: invalid hash state identifier")
	}
	if len(b) != marshaledSize {
		return errors.New("crypto/sha512: invalid hash state size")
	}
	b = b[len(magic512):]
	b, d.h[0] = consumeUint64(b)
	b, d.h[1] = consumeUint64(b)
	b, d.h[2] = consumeUint64(b)
	b, d.h[3] = consumeUint64(b)
	b, d.h[4] = consumeUint64(b)
	b, d.h[5] = consumeUint64(b)
	b, d.h[6] = consumeUint64(b)
	b, d.h[7] = consumeUint64(b)
	b = b[copy(d.x[:], b):]
	b, d.len = consumeUint64(b)
	d.nx = int(d.len % chunk)
	return nil
}

func consumeUint64(b []byte) ([]byte, uint64) {
	_ = b[7]
	x := uint64(b[7]) | uint64(b[6])<<8 | uint64(b[5])<<16 | uint64(b[4])<<24 |
		uint64(b[3])<<32 | uint64(b[2])<<40 | uint64(b[1])<<48 | uint64(b[0])<<56
	return b[8:], x
}

// New returns a new hash.Hash computing the SHA-512 checksum.
func New() hash.Hash {
	if boring.Enabled {
		return boring.NewSHA512()
	}
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
	if boring.Enabled {
		return boring.NewSHA384()
	}
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
	if d.function != crypto.SHA512_224 && d.function != crypto.SHA512_256 {
		boring.Unreachable()
	}
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
	if d.function != crypto.SHA512_224 && d.function != crypto.SHA512_256 {
		boring.Unreachable()
	}
	// Make a copy of d so that caller can keep writing and summing.
	d0 := new(digest)
	*d0 = *d
	hash := d0.checkSum()
	switch d0.function {
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
	var tmp [128 + 16]byte // padding + length buffer
	tmp[0] = 0x80
	var t uint64
	if len%128 < 112 {
		t = 112 - len%128
	} else {
		t = 128 + 112 - len%128
	}

	// Length in bits.
	len <<= 3
	padlen := tmp[:t+16]
	// Upper 64 bits are always zero, because len variable has type uint64,
	// and tmp is already zeroed at that index, so we can skip updating it.
	// binary.BigEndian.PutUint64(padlen[t+0:], 0)
	binary.BigEndian.PutUint64(padlen[t+8:], len)
	d.Write(padlen)

	if d.nx != 0 {
		panic("d.nx != 0")
	}

	var digest [Size]byte
	binary.BigEndian.PutUint64(digest[0:], d.h[0])
	binary.BigEndian.PutUint64(digest[8:], d.h[1])
	binary.BigEndian.PutUint64(digest[16:], d.h[2])
	binary.BigEndian.PutUint64(digest[24:], d.h[3])
	binary.BigEndian.PutUint64(digest[32:], d.h[4])
	binary.BigEndian.PutUint64(digest[40:], d.h[5])
	if d.function != crypto.SHA384 {
		binary.BigEndian.PutUint64(digest[48:], d.h[6])
		binary.BigEndian.PutUint64(digest[56:], d.h[7])
	}

	return digest
}

// Sum512 returns the SHA512 checksum of the data.
func Sum512(data []byte) [Size]byte {
	if boring.Enabled {
		return boring.SHA512(data)
	}
	d := digest{function: crypto.SHA512}
	d.Reset()
	d.Write(data)
	return d.checkSum()
}

// Sum384 returns the SHA384 checksum of the data.
func Sum384(data []byte) [Size384]byte {
	if boring.Enabled {
		return boring.SHA384(data)
	}
	d := digest{function: crypto.SHA384}
	d.Reset()
	d.Write(data)
	sum := d.checkSum()
	ap := (*[Size384]byte)(sum[:])
	return *ap
}

// Sum512_224 returns the Sum512/224 checksum of the data.
func Sum512_224(data []byte) [Size224]byte {
	d := digest{function: crypto.SHA512_224}
	d.Reset()
	d.Write(data)
	sum := d.checkSum()
	ap := (*[Size224]byte)(sum[:])
	return *ap
}

// Sum512_256 returns the Sum512/256 checksum of the data.
func Sum512_256(data []byte) [Size256]byte {
	d := digest{function: crypto.SHA512_256}
	d.Reset()
	d.Write(data)
	sum := d.checkSum()
	ap := (*[Size256]byte)(sum[:])
	return *ap
}
