// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha3

import (
	"bytes"
	"crypto/internal/fips"
	"errors"
	"internal/byteorder"
	"math/bits"
)

type SHAKE struct {
	d Digest // SHA-3 state context and Read/Write operations

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

func newCShake(N, S []byte, rate, outputLen int, dsbyte byte) *SHAKE {
	c := &SHAKE{d: Digest{rate: rate, outputLen: outputLen, dsbyte: dsbyte}}
	c.initBlock = make([]byte, 0, 9+len(N)+9+len(S)) // leftEncode returns max 9 bytes
	c.initBlock = append(c.initBlock, leftEncode(uint64(len(N))*8)...)
	c.initBlock = append(c.initBlock, N...)
	c.initBlock = append(c.initBlock, leftEncode(uint64(len(S))*8)...)
	c.initBlock = append(c.initBlock, S...)
	c.Write(bytepad(c.initBlock, c.d.rate))
	return c
}

func (s *SHAKE) BlockSize() int { return s.d.BlockSize() }
func (s *SHAKE) Size() int      { return s.d.Size() }

// Sum appends a portion of output to b and returns the resulting slice. The
// output length is selected to provide full-strength generic security: 32 bytes
// for SHAKE128 and 64 bytes for SHAKE256. It does not change the underlying
// state. It panics if any output has already been read.
func (s *SHAKE) Sum(in []byte) []byte { return s.d.Sum(in) }

// Write absorbs more data into the hash's state.
// It panics if any output has already been read.
func (s *SHAKE) Write(p []byte) (n int, err error) { return s.d.Write(p) }

func (s *SHAKE) Read(out []byte) (n int, err error) {
	fips.RecordApproved()
	// Note that read is not exposed on Digest since SHA-3 does not offer
	// variable output length. It is only used internally by Sum.
	return s.d.read(out)
}

// Reset resets the hash to initial state.
func (s *SHAKE) Reset() {
	s.d.Reset()
	if len(s.initBlock) != 0 {
		s.Write(bytepad(s.initBlock, s.d.rate))
	}
}

// Clone returns a copy of the SHAKE context in its current state.
func (s *SHAKE) Clone() *SHAKE {
	ret := *s
	return &ret
}

func (s *SHAKE) MarshalBinary() ([]byte, error) {
	return s.AppendBinary(make([]byte, 0, marshaledSize+len(s.initBlock)))
}

func (s *SHAKE) AppendBinary(b []byte) ([]byte, error) {
	b, err := s.d.AppendBinary(b)
	if err != nil {
		return nil, err
	}
	b = append(b, s.initBlock...)
	return b, nil
}

func (s *SHAKE) UnmarshalBinary(b []byte) error {
	if len(b) < marshaledSize {
		return errors.New("sha3: invalid hash state")
	}
	if err := s.d.UnmarshalBinary(b[:marshaledSize]); err != nil {
		return err
	}
	s.initBlock = bytes.Clone(b[marshaledSize:])
	return nil
}

// NewShake128 creates a new SHAKE128 XOF.
func NewShake128() *SHAKE {
	return &SHAKE{d: Digest{rate: rateK256, outputLen: 32, dsbyte: dsbyteShake}}
}

// NewShake256 creates a new SHAKE256 XOF.
func NewShake256() *SHAKE {
	return &SHAKE{d: Digest{rate: rateK512, outputLen: 64, dsbyte: dsbyteShake}}
}

// NewCShake128 creates a new cSHAKE128 XOF.
//
// N is used to define functions based on cSHAKE, it can be empty when plain
// cSHAKE is desired. S is a customization byte string used for domain
// separation. When N and S are both empty, this is equivalent to NewShake128.
func NewCShake128(N, S []byte) *SHAKE {
	if len(N) == 0 && len(S) == 0 {
		return NewShake128()
	}
	return newCShake(N, S, rateK256, 32, dsbyteCShake)
}

// NewCShake256 creates a new cSHAKE256 XOF.
//
// N is used to define functions based on cSHAKE, it can be empty when plain
// cSHAKE is desired. S is a customization byte string used for domain
// separation. When N and S are both empty, this is equivalent to NewShake256.
func NewCShake256(N, S []byte) *SHAKE {
	if len(N) == 0 && len(S) == 0 {
		return NewShake256()
	}
	return newCShake(N, S, rateK512, 64, dsbyteCShake)
}
