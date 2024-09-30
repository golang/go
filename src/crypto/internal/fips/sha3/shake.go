// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha3

// This file defines the ShakeHash interface, and provides
// functions for creating SHAKE and cSHAKE instances, as well as utility
// functions for hashing bytes to arbitrary-length output.
//
//
// SHAKE implementation is based on FIPS PUB 202 [1]
// cSHAKE implementations is based on NIST SP 800-185 [2]
//
// [1] https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
// [2] https://doi.org/10.6028/NIST.SP.800-185

import (
	"bytes"
	"crypto/internal/fips"
	"errors"
	"internal/byteorder"
	"io"
	"math/bits"
)

// ShakeHash defines the interface to hash functions that support
// arbitrary-length output. When used as a plain [hash.Hash], it
// produces minimum-length outputs that provide full-strength generic
// security.
type ShakeHash interface {
	fips.Hash

	// Read reads more output from the hash; reading affects the hash's
	// state. (ShakeHash.Read is thus very different from Hash.Sum)
	// It never returns an error, but subsequent calls to Write or Sum
	// will panic.
	io.Reader

	// Clone returns a copy of the ShakeHash in its current state.
	Clone() ShakeHash
}

// cSHAKE specific context
type cshakeState struct {
	*state // SHA-3 state context and Read/Write operations

	// initBlock is the cSHAKE specific initialization set of bytes. It is initialized
	// by newCShake function and stores concatenation of N followed by S, encoded
	// by the method specified in 3.3 of [1].
	// It is stored here in order for Reset() to be able to put context into
	// initial state.
	initBlock []byte
}

func bytepad(data []byte, rate int) []byte {
	out := make([]byte, 0, 9+len(data)+rate-1)
	out = append(out, leftEncode(uint64(rate))...)
	out = append(out, data...)
	if padlen := rate - len(out)%rate; padlen < rate {
		out = append(out, make([]byte, padlen)...)
	}
	return out
}

func leftEncode(x uint64) []byte {
	// Let n be the smallest positive integer for which 2^(8n) > x.
	n := (bits.Len64(x) + 7) / 8
	if n == 0 {
		n = 1
	}
	// Return n || x with n as a byte and x an n bytes in big-endian order.
	b := make([]byte, 9)
	byteorder.BePutUint64(b[1:], x)
	b = b[9-n-1:]
	b[0] = byte(n)
	return b
}

func newCShake(N, S []byte, rate, outputLen int, dsbyte byte) ShakeHash {
	c := cshakeState{state: &state{rate: rate, outputLen: outputLen, dsbyte: dsbyte}}
	c.initBlock = make([]byte, 0, 9+len(N)+9+len(S)) // leftEncode returns max 9 bytes
	c.initBlock = append(c.initBlock, leftEncode(uint64(len(N))*8)...)
	c.initBlock = append(c.initBlock, N...)
	c.initBlock = append(c.initBlock, leftEncode(uint64(len(S))*8)...)
	c.initBlock = append(c.initBlock, S...)
	c.Write(bytepad(c.initBlock, c.rate))
	return &c
}

// Reset resets the hash to initial state.
func (c *cshakeState) Reset() {
	c.state.Reset()
	c.Write(bytepad(c.initBlock, c.rate))
}

// Clone returns copy of a cSHAKE context within its current state.
func (c *cshakeState) Clone() ShakeHash {
	b := make([]byte, len(c.initBlock))
	copy(b, c.initBlock)
	return &cshakeState{state: c.clone(), initBlock: b}
}

// Clone returns copy of SHAKE context within its current state.
func (c *state) Clone() ShakeHash {
	return c.clone()
}

func (c *cshakeState) MarshalBinary() ([]byte, error) {
	return c.AppendBinary(make([]byte, 0, marshaledSize+len(c.initBlock)))
}

func (c *cshakeState) AppendBinary(b []byte) ([]byte, error) {
	b, err := c.state.AppendBinary(b)
	if err != nil {
		return nil, err
	}
	b = append(b, c.initBlock...)
	return b, nil
}

func (c *cshakeState) UnmarshalBinary(b []byte) error {
	if len(b) <= marshaledSize {
		return errors.New("sha3: invalid hash state")
	}
	if err := c.state.UnmarshalBinary(b[:marshaledSize]); err != nil {
		return err
	}
	c.initBlock = bytes.Clone(b[marshaledSize:])
	return nil
}

// NewShake128 creates a new SHAKE128 variable-output-length ShakeHash.
// Its generic security strength is 128 bits against all attacks if at
// least 32 bytes of its output are used.
func NewShake128() ShakeHash {
	return newShake128()
}

// NewShake256 creates a new SHAKE256 variable-output-length ShakeHash.
// Its generic security strength is 256 bits against all attacks if
// at least 64 bytes of its output are used.
func NewShake256() ShakeHash {
	return newShake256()
}

func newShake128Generic() *state {
	return &state{rate: rateK256, outputLen: 32, dsbyte: dsbyteShake}
}

func newShake256Generic() *state {
	return &state{rate: rateK512, outputLen: 64, dsbyte: dsbyteShake}
}

// NewCShake128 creates a new instance of cSHAKE128 variable-output-length ShakeHash,
// a customizable variant of SHAKE128.
// N is used to define functions based on cSHAKE, it can be empty when plain cSHAKE is
// desired. S is a customization byte string used for domain separation - two cSHAKE
// computations on same input with different S yield unrelated outputs.
// When N and S are both empty, this is equivalent to NewShake128.
func NewCShake128(N, S []byte) ShakeHash {
	if len(N) == 0 && len(S) == 0 {
		return NewShake128()
	}
	return newCShake(N, S, rateK256, 32, dsbyteCShake)
}

// NewCShake256 creates a new instance of cSHAKE256 variable-output-length ShakeHash,
// a customizable variant of SHAKE256.
// N is used to define functions based on cSHAKE, it can be empty when plain cSHAKE is
// desired. S is a customization byte string used for domain separation - two cSHAKE
// computations on same input with different S yield unrelated outputs.
// When N and S are both empty, this is equivalent to NewShake256.
func NewCShake256(N, S []byte) ShakeHash {
	if len(N) == 0 && len(S) == 0 {
		return NewShake256()
	}
	return newCShake(N, S, rateK512, 64, dsbyteCShake)
}

// ShakeSum128 writes an arbitrary-length digest of data into hash.
func ShakeSum128(hash, data []byte) {
	h := NewShake128()
	h.Write(data)
	h.Read(hash)
}

// ShakeSum256 writes an arbitrary-length digest of data into hash.
func ShakeSum256(hash, data []byte) {
	h := NewShake256()
	h.Write(data)
	h.Read(hash)
}
