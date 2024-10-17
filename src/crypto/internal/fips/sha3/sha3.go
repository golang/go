// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sha3 implements the SHA-3 fixed-output-length hash functions and
// the SHAKE variable-output-length functions defined by [FIPS 202], as well as
// the cSHAKE extendable-output-length functions defined by [SP 800-185].
//
// [FIPS 202]: https://doi.org/10.6028/NIST.FIPS.202
// [SP 800-185]: https://doi.org/10.6028/NIST.SP.800-185
package sha3

import (
	"crypto/internal/fips"
	"crypto/internal/fips/subtle"
	"errors"
)

// spongeDirection indicates the direction bytes are flowing through the sponge.
type spongeDirection int

const (
	// spongeAbsorbing indicates that the sponge is absorbing input.
	spongeAbsorbing spongeDirection = iota
	// spongeSqueezing indicates that the sponge is being squeezed.
	spongeSqueezing
)

type Digest struct {
	a [1600 / 8]byte // main state of the hash

	// a[n:rate] is the buffer. If absorbing, it's the remaining space to XOR
	// into before running the permutation. If squeezing, it's the remaining
	// output to produce before running the permutation.
	n, rate int

	// dsbyte contains the "domain separation" bits and the first bit of
	// the padding. Sections 6.1 and 6.2 of [1] separate the outputs of the
	// SHA-3 and SHAKE functions by appending bitstrings to the message.
	// Using a little-endian bit-ordering convention, these are "01" for SHA-3
	// and "1111" for SHAKE, or 00000010b and 00001111b, respectively. Then the
	// padding rule from section 5.1 is applied to pad the message to a multiple
	// of the rate, which involves adding a "1" bit, zero or more "0" bits, and
	// a final "1" bit. We merge the first "1" bit from the padding into dsbyte,
	// giving 00000110b (0x06) and 00011111b (0x1f).
	// [1] http://csrc.nist.gov/publications/drafts/fips-202/fips_202_draft.pdf
	//     "Draft FIPS 202: SHA-3 Standard: Permutation-Based Hash and
	//      Extendable-Output Functions (May 2014)"
	dsbyte byte

	outputLen int             // the default output size in bytes
	state     spongeDirection // whether the sponge is absorbing or squeezing
}

// BlockSize returns the rate of sponge underlying this hash function.
func (d *Digest) BlockSize() int { return d.rate }

// Size returns the output size of the hash function in bytes.
func (d *Digest) Size() int { return d.outputLen }

// Reset resets the Digest to its initial state.
func (d *Digest) Reset() {
	// Zero the permutation's state.
	for i := range d.a {
		d.a[i] = 0
	}
	d.state = spongeAbsorbing
	d.n = 0
}

func (d *Digest) Clone() *Digest {
	ret := *d
	return &ret
}

// permute applies the KeccakF-1600 permutation.
func (d *Digest) permute() {
	keccakF1600(&d.a)
	d.n = 0
}

// padAndPermute appends the domain separation bits in dsbyte, applies
// the multi-bitrate 10..1 padding rule, and permutes the state.
func (d *Digest) padAndPermute() {
	// Pad with this instance's domain-separator bits. We know that there's
	// at least one byte of space in the sponge because, if it were full,
	// permute would have been called to empty it. dsbyte also contains the
	// first one bit for the padding. See the comment in the state struct.
	d.a[d.n] ^= d.dsbyte
	// This adds the final one bit for the padding. Because of the way that
	// bits are numbered from the LSB upwards, the final bit is the MSB of
	// the last byte.
	d.a[d.rate-1] ^= 0x80
	// Apply the permutation
	d.permute()
	d.state = spongeSqueezing
}

// Write absorbs more data into the hash's state.
func (d *Digest) Write(p []byte) (n int, err error) { return d.write(p) }
func (d *Digest) writeGeneric(p []byte) (n int, err error) {
	if d.state != spongeAbsorbing {
		panic("sha3: Write after Read")
	}

	n = len(p)

	for len(p) > 0 {
		x := subtle.XORBytes(d.a[d.n:d.rate], d.a[d.n:d.rate], p)
		d.n += x
		p = p[x:]

		// If the sponge is full, apply the permutation.
		if d.n == d.rate {
			d.permute()
		}
	}

	return
}

// read squeezes an arbitrary number of bytes from the sponge.
func (d *Digest) readGeneric(out []byte) (n int, err error) {
	// If we're still absorbing, pad and apply the permutation.
	if d.state == spongeAbsorbing {
		d.padAndPermute()
	}

	n = len(out)

	// Now, do the squeezing.
	for len(out) > 0 {
		// Apply the permutation if we've squeezed the sponge dry.
		if d.n == d.rate {
			d.permute()
		}

		x := copy(out, d.a[d.n:d.rate])
		d.n += x
		out = out[x:]
	}

	return
}

// Sum appends the current hash to b and returns the resulting slice.
// It does not change the underlying hash state.
func (d *Digest) Sum(b []byte) []byte {
	fips.RecordApproved()
	return d.sum(b)
}

func (d *Digest) sumGeneric(b []byte) []byte {
	if d.state != spongeAbsorbing {
		panic("sha3: Sum after Read")
	}

	// Make a copy of the original hash so that caller can keep writing
	// and summing.
	dup := d.Clone()
	hash := make([]byte, dup.outputLen, 64) // explicit cap to allow stack allocation
	dup.read(hash)
	return append(b, hash...)
}

const (
	magicSHA3   = "sha\x08"
	magicShake  = "sha\x09"
	magicCShake = "sha\x0a"
	magicKeccak = "sha\x0b"
	// magic || rate || main state || n || sponge direction
	marshaledSize = len(magicSHA3) + 1 + 200 + 1 + 1
)

func (d *Digest) MarshalBinary() ([]byte, error) {
	return d.AppendBinary(make([]byte, 0, marshaledSize))
}

func (d *Digest) AppendBinary(b []byte) ([]byte, error) {
	switch d.dsbyte {
	case dsbyteSHA3:
		b = append(b, magicSHA3...)
	case dsbyteShake:
		b = append(b, magicShake...)
	case dsbyteCShake:
		b = append(b, magicCShake...)
	case dsbyteKeccak:
		b = append(b, magicKeccak...)
	default:
		panic("unknown dsbyte")
	}
	// rate is at most 168, and n is at most rate.
	b = append(b, byte(d.rate))
	b = append(b, d.a[:]...)
	b = append(b, byte(d.n), byte(d.state))
	return b, nil
}

func (d *Digest) UnmarshalBinary(b []byte) error {
	if len(b) != marshaledSize {
		return errors.New("sha3: invalid hash state")
	}

	magic := string(b[:len(magicSHA3)])
	b = b[len(magicSHA3):]
	switch {
	case magic == magicSHA3 && d.dsbyte == dsbyteSHA3:
	case magic == magicShake && d.dsbyte == dsbyteShake:
	case magic == magicCShake && d.dsbyte == dsbyteCShake:
	case magic == magicKeccak && d.dsbyte == dsbyteKeccak:
	default:
		return errors.New("sha3: invalid hash state identifier")
	}

	rate := int(b[0])
	b = b[1:]
	if rate != d.rate {
		return errors.New("sha3: invalid hash state function")
	}

	copy(d.a[:], b)
	b = b[len(d.a):]

	n, state := int(b[0]), spongeDirection(b[1])
	if n > d.rate {
		return errors.New("sha3: invalid hash state")
	}
	d.n = n
	if state != spongeAbsorbing && state != spongeSqueezing {
		return errors.New("sha3: invalid hash state")
	}
	d.state = state

	return nil
}
