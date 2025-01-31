// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sha3 implements the SHA-3 hash algorithms and the SHAKE extendable
// output functions defined in FIPS 202.
package sha3

import (
	"crypto"
	"crypto/internal/fips140/sha3"
	"hash"
	_ "unsafe"
)

func init() {
	crypto.RegisterHash(crypto.SHA3_224, func() hash.Hash { return New224() })
	crypto.RegisterHash(crypto.SHA3_256, func() hash.Hash { return New256() })
	crypto.RegisterHash(crypto.SHA3_384, func() hash.Hash { return New384() })
	crypto.RegisterHash(crypto.SHA3_512, func() hash.Hash { return New512() })
}

// Sum224 returns the SHA3-224 hash of data.
func Sum224(data []byte) [28]byte {
	var out [28]byte
	h := sha3.New224()
	h.Write(data)
	h.Sum(out[:0])
	return out
}

// Sum256 returns the SHA3-256 hash of data.
func Sum256(data []byte) [32]byte {
	var out [32]byte
	h := sha3.New256()
	h.Write(data)
	h.Sum(out[:0])
	return out
}

// Sum384 returns the SHA3-384 hash of data.
func Sum384(data []byte) [48]byte {
	var out [48]byte
	h := sha3.New384()
	h.Write(data)
	h.Sum(out[:0])
	return out
}

// Sum512 returns the SHA3-512 hash of data.
func Sum512(data []byte) [64]byte {
	var out [64]byte
	h := sha3.New512()
	h.Write(data)
	h.Sum(out[:0])
	return out
}

// SumSHAKE128 applies the SHAKE128 extendable output function to data and
// returns an output of the given length in bytes.
func SumSHAKE128(data []byte, length int) []byte {
	// Outline the allocation for up to 256 bits of output to the caller's stack.
	out := make([]byte, 32)
	return sumSHAKE128(out, data, length)
}

func sumSHAKE128(out, data []byte, length int) []byte {
	if len(out) < length {
		out = make([]byte, length)
	} else {
		out = out[:length]
	}
	h := sha3.NewShake128()
	h.Write(data)
	h.Read(out)
	return out
}

// SumSHAKE256 applies the SHAKE256 extendable output function to data and
// returns an output of the given length in bytes.
func SumSHAKE256(data []byte, length int) []byte {
	// Outline the allocation for up to 512 bits of output to the caller's stack.
	out := make([]byte, 64)
	return sumSHAKE256(out, data, length)
}

func sumSHAKE256(out, data []byte, length int) []byte {
	if len(out) < length {
		out = make([]byte, length)
	} else {
		out = out[:length]
	}
	h := sha3.NewShake256()
	h.Write(data)
	h.Read(out)
	return out
}

// SHA3 is an instance of a SHA-3 hash. It implements [hash.Hash].
type SHA3 struct {
	s sha3.Digest
}

//go:linkname fips140hash_sha3Unwrap crypto/internal/fips140hash.sha3Unwrap
func fips140hash_sha3Unwrap(sha3 *SHA3) *sha3.Digest {
	return &sha3.s
}

// New224 creates a new SHA3-224 hash.
func New224() *SHA3 {
	return &SHA3{*sha3.New224()}
}

// New256 creates a new SHA3-256 hash.
func New256() *SHA3 {
	return &SHA3{*sha3.New256()}
}

// New384 creates a new SHA3-384 hash.
func New384() *SHA3 {
	return &SHA3{*sha3.New384()}
}

// New512 creates a new SHA3-512 hash.
func New512() *SHA3 {
	return &SHA3{*sha3.New512()}
}

// Write absorbs more data into the hash's state.
func (s *SHA3) Write(p []byte) (n int, err error) {
	return s.s.Write(p)
}

// Sum appends the current hash to b and returns the resulting slice.
func (s *SHA3) Sum(b []byte) []byte {
	return s.s.Sum(b)
}

// Reset resets the hash to its initial state.
func (s *SHA3) Reset() {
	s.s.Reset()
}

// Size returns the number of bytes Sum will produce.
func (s *SHA3) Size() int {
	return s.s.Size()
}

// BlockSize returns the hash's rate.
func (s *SHA3) BlockSize() int {
	return s.s.BlockSize()
}

// MarshalBinary implements [encoding.BinaryMarshaler].
func (s *SHA3) MarshalBinary() ([]byte, error) {
	return s.s.MarshalBinary()
}

// AppendBinary implements [encoding.BinaryAppender].
func (s *SHA3) AppendBinary(p []byte) ([]byte, error) {
	return s.s.AppendBinary(p)
}

// UnmarshalBinary implements [encoding.BinaryUnmarshaler].
func (s *SHA3) UnmarshalBinary(data []byte) error {
	return s.s.UnmarshalBinary(data)
}

// SHAKE is an instance of a SHAKE extendable output function.
type SHAKE struct {
	s sha3.SHAKE
}

// NewSHAKE128 creates a new SHAKE128 XOF.
func NewSHAKE128() *SHAKE {
	return &SHAKE{*sha3.NewShake128()}
}

// NewSHAKE256 creates a new SHAKE256 XOF.
func NewSHAKE256() *SHAKE {
	return &SHAKE{*sha3.NewShake256()}
}

// NewCSHAKE128 creates a new cSHAKE128 XOF.
//
// N is used to define functions based on cSHAKE, it can be empty when plain
// cSHAKE is desired. S is a customization byte string used for domain
// separation. When N and S are both empty, this is equivalent to NewSHAKE128.
func NewCSHAKE128(N, S []byte) *SHAKE {
	return &SHAKE{*sha3.NewCShake128(N, S)}
}

// NewCSHAKE256 creates a new cSHAKE256 XOF.
//
// N is used to define functions based on cSHAKE, it can be empty when plain
// cSHAKE is desired. S is a customization byte string used for domain
// separation. When N and S are both empty, this is equivalent to NewSHAKE256.
func NewCSHAKE256(N, S []byte) *SHAKE {
	return &SHAKE{*sha3.NewCShake256(N, S)}
}

// Write absorbs more data into the XOF's state.
//
// It panics if any output has already been read.
func (s *SHAKE) Write(p []byte) (n int, err error) {
	return s.s.Write(p)
}

// Read squeezes more output from the XOF.
//
// Any call to Write after a call to Read will panic.
func (s *SHAKE) Read(p []byte) (n int, err error) {
	return s.s.Read(p)
}

// Reset resets the XOF to its initial state.
func (s *SHAKE) Reset() {
	s.s.Reset()
}

// BlockSize returns the rate of the XOF.
func (s *SHAKE) BlockSize() int {
	return s.s.BlockSize()
}

// MarshalBinary implements [encoding.BinaryMarshaler].
func (s *SHAKE) MarshalBinary() ([]byte, error) {
	return s.s.MarshalBinary()
}

// AppendBinary implements [encoding.BinaryAppender].
func (s *SHAKE) AppendBinary(p []byte) ([]byte, error) {
	return s.s.AppendBinary(p)
}

// UnmarshalBinary implements [encoding.BinaryUnmarshaler].
func (s *SHAKE) UnmarshalBinary(data []byte) error {
	return s.s.UnmarshalBinary(data)
}
