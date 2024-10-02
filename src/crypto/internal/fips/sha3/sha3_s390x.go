// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc && !purego && ignore

package sha3

// This file contains code for using the 'compute intermediate
// message digest' (KIMD) and 'compute last message digest' (KLMD)
// instructions to compute SHA-3 and SHAKE hashes on IBM Z.

import "internal/cpu"

// codes represent 7-bit KIMD/KLMD function codes as defined in
// the Principles of Operation.
type code uint64

const (
	// function codes for KIMD/KLMD
	sha3_224  code = 32
	sha3_256       = 33
	sha3_384       = 34
	sha3_512       = 35
	shake_128      = 36
	shake_256      = 37
	nopad          = 0x100
)

// kimd is a wrapper for the 'compute intermediate message digest' instruction.
// src must be a multiple of the rate for the given function code.
//
//go:noescape
func kimd(function code, chain *[200]byte, src []byte)

// klmd is a wrapper for the 'compute last message digest' instruction.
// src padding is handled by the instruction.
//
//go:noescape
func klmd(function code, chain *[200]byte, dst, src []byte)

type asmState struct {
	a         [200]byte       // 1600 bit state
	buf       []byte          // care must be taken to ensure cap(buf) is a multiple of rate
	rate      int             // equivalent to block size
	storage   [3072]byte      // underlying storage for buf
	outputLen int             // output length for full security
	function  code            // KIMD/KLMD function code
	state     spongeDirection // whether the sponge is absorbing or squeezing
}

func newAsmState(function code) *asmState {
	var s asmState
	s.function = function
	switch function {
	case sha3_224:
		s.rate = 144
		s.outputLen = 28
	case sha3_256:
		s.rate = 136
		s.outputLen = 32
	case sha3_384:
		s.rate = 104
		s.outputLen = 48
	case sha3_512:
		s.rate = 72
		s.outputLen = 64
	case shake_128:
		s.rate = 168
		s.outputLen = 32
	case shake_256:
		s.rate = 136
		s.outputLen = 64
	default:
		panic("sha3: unrecognized function code")
	}

	// limit s.buf size to a multiple of s.rate
	s.resetBuf()
	return &s
}

func (s *asmState) clone() *asmState {
	c := *s
	c.buf = c.storage[:len(s.buf):cap(s.buf)]
	return &c
}

// copyIntoBuf copies b into buf. It will panic if there is not enough space to
// store all of b.
func (s *asmState) copyIntoBuf(b []byte) {
	bufLen := len(s.buf)
	s.buf = s.buf[:len(s.buf)+len(b)]
	copy(s.buf[bufLen:], b)
}

// resetBuf points buf at storage, sets the length to 0 and sets cap to be a
// multiple of the rate.
func (s *asmState) resetBuf() {
	max := (cap(s.storage) / s.rate) * s.rate
	s.buf = s.storage[:0:max]
}

// Write (via the embedded io.Writer interface) adds more data to the running hash.
// It never returns an error.
func (s *asmState) Write(b []byte) (int, error) {
	if s.state != spongeAbsorbing {
		panic("sha3: Write after Read")
	}
	length := len(b)
	for len(b) > 0 {
		if len(s.buf) == 0 && len(b) >= cap(s.buf) {
			// Hash the data directly and push any remaining bytes
			// into the buffer.
			remainder := len(b) % s.rate
			kimd(s.function, &s.a, b[:len(b)-remainder])
			if remainder != 0 {
				s.copyIntoBuf(b[len(b)-remainder:])
			}
			return length, nil
		}

		if len(s.buf) == cap(s.buf) {
			// flush the buffer
			kimd(s.function, &s.a, s.buf)
			s.buf = s.buf[:0]
		}

		// copy as much as we can into the buffer
		n := len(b)
		if len(b) > cap(s.buf)-len(s.buf) {
			n = cap(s.buf) - len(s.buf)
		}
		s.copyIntoBuf(b[:n])
		b = b[n:]
	}
	return length, nil
}

// Read squeezes an arbitrary number of bytes from the sponge.
func (s *asmState) Read(out []byte) (n int, err error) {
	// The 'compute last message digest' instruction only stores the digest
	// at the first operand (dst) for SHAKE functions.
	if s.function != shake_128 && s.function != shake_256 {
		panic("sha3: can only call Read for SHAKE functions")
	}

	n = len(out)

	// need to pad if we were absorbing
	if s.state == spongeAbsorbing {
		s.state = spongeSqueezing

		// write hash directly into out if possible
		if len(out)%s.rate == 0 {
			klmd(s.function, &s.a, out, s.buf) // len(out) may be 0
			s.buf = s.buf[:0]
			return
		}

		// write hash into buffer
		max := cap(s.buf)
		if max > len(out) {
			max = (len(out)/s.rate)*s.rate + s.rate
		}
		klmd(s.function, &s.a, s.buf[:max], s.buf)
		s.buf = s.buf[:max]
	}

	for len(out) > 0 {
		// flush the buffer
		if len(s.buf) != 0 {
			c := copy(out, s.buf)
			out = out[c:]
			s.buf = s.buf[c:]
			continue
		}

		// write hash directly into out if possible
		if len(out)%s.rate == 0 {
			klmd(s.function|nopad, &s.a, out, nil)
			return
		}

		// write hash into buffer
		s.resetBuf()
		if cap(s.buf) > len(out) {
			s.buf = s.buf[:(len(out)/s.rate)*s.rate+s.rate]
		}
		klmd(s.function|nopad, &s.a, s.buf, nil)
	}
	return
}

// Sum appends the current hash to b and returns the resulting slice.
// It does not change the underlying hash state.
func (s *asmState) Sum(b []byte) []byte {
	if s.state != spongeAbsorbing {
		panic("sha3: Sum after Read")
	}

	// Copy the state to preserve the original.
	a := s.a

	// Hash the buffer. Note that we don't clear it because we
	// aren't updating the state.
	switch s.function {
	case sha3_224, sha3_256, sha3_384, sha3_512:
		klmd(s.function, &a, nil, s.buf)
		return append(b, a[:s.outputLen]...)
	case shake_128, shake_256:
		d := make([]byte, s.outputLen, 64)
		klmd(s.function, &a, d, s.buf)
		return append(b, d[:s.outputLen]...)
	default:
		panic("sha3: unknown function")
	}
}

// Reset resets the Hash to its initial state.
func (s *asmState) Reset() {
	for i := range s.a {
		s.a[i] = 0
	}
	s.resetBuf()
	s.state = spongeAbsorbing
}

// Size returns the number of bytes Sum will return.
func (s *asmState) Size() int {
	return s.outputLen
}

// BlockSize returns the hash's underlying block size.
// The Write method must be able to accept any amount
// of data, but it may operate more efficiently if all writes
// are a multiple of the block size.
func (s *asmState) BlockSize() int {
	return s.rate
}

// Clone returns a copy of the ShakeHash in its current state.
func (s *asmState) Clone() ShakeHash {
	return s.clone()
}

// new224 returns an assembly implementation of SHA3-224 if available,
// otherwise it returns a generic implementation.
func new224() *Digest {
	if cpu.S390X.HasSHA3 {
		return newAsmState(sha3_224)
	}
	return new224Generic()
}

// new256 returns an assembly implementation of SHA3-256 if available,
// otherwise it returns a generic implementation.
func new256() *Digest {
	if cpu.S390X.HasSHA3 {
		return newAsmState(sha3_256)
	}
	return new256Generic()
}

// new384 returns an assembly implementation of SHA3-384 if available,
// otherwise it returns a generic implementation.
func new384() *Digest {
	if cpu.S390X.HasSHA3 {
		return newAsmState(sha3_384)
	}
	return new384Generic()
}

// new512 returns an assembly implementation of SHA3-512 if available,
// otherwise it returns a generic implementation.
func new512() *Digest {
	if cpu.S390X.HasSHA3 {
		return newAsmState(sha3_512)
	}
	return new512Generic()
}

// newShake128 returns an assembly implementation of SHAKE-128 if available,
// otherwise it returns a generic implementation.
func newShake128() ShakeHash {
	if cpu.S390X.HasSHA3 {
		return newAsmState(shake_128)
	}
	return newShake128Generic()
}

// newShake256 returns an assembly implementation of SHAKE-256 if available,
// otherwise it returns a generic implementation.
func newShake256() ShakeHash {
	if cpu.S390X.HasSHA3 {
		return newAsmState(shake_256)
	}
	return newShake256Generic()
}
