// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ChaCha20 implements the core ChaCha20 function as specified
// in https://tools.ietf.org/html/rfc7539#section-2.3.
package chacha20

import (
	"crypto/cipher"
	"encoding/binary"

	"golang.org/x/crypto/internal/subtle"
)

// assert that *Cipher implements cipher.Stream
var _ cipher.Stream = (*Cipher)(nil)

// Cipher is a stateful instance of ChaCha20 using a particular key
// and nonce. A *Cipher implements the cipher.Stream interface.
type Cipher struct {
	key     [8]uint32
	counter uint32 // incremented after each block
	nonce   [3]uint32
	buf     [bufSize]byte // buffer for unused keystream bytes
	len     int           // number of unused keystream bytes at end of buf
}

// New creates a new ChaCha20 stream cipher with the given key and nonce.
// The initial counter value is set to 0.
func New(key [8]uint32, nonce [3]uint32) *Cipher {
	return &Cipher{key: key, nonce: nonce}
}

// ChaCha20 constants spelling "expand 32-byte k"
const (
	j0 uint32 = 0x61707865
	j1 uint32 = 0x3320646e
	j2 uint32 = 0x79622d32
	j3 uint32 = 0x6b206574
)

func quarterRound(a, b, c, d uint32) (uint32, uint32, uint32, uint32) {
	a += b
	d ^= a
	d = (d << 16) | (d >> 16)
	c += d
	b ^= c
	b = (b << 12) | (b >> 20)
	a += b
	d ^= a
	d = (d << 8) | (d >> 24)
	c += d
	b ^= c
	b = (b << 7) | (b >> 25)
	return a, b, c, d
}

// XORKeyStream XORs each byte in the given slice with a byte from the
// cipher's key stream. Dst and src must overlap entirely or not at all.
//
// If len(dst) < len(src), XORKeyStream will panic. It is acceptable
// to pass a dst bigger than src, and in that case, XORKeyStream will
// only update dst[:len(src)] and will not touch the rest of dst.
//
// Multiple calls to XORKeyStream behave as if the concatenation of
// the src buffers was passed in a single run. That is, Cipher
// maintains state and does not reset at each XORKeyStream call.
func (s *Cipher) XORKeyStream(dst, src []byte) {
	if len(dst) < len(src) {
		panic("chacha20: output smaller than input")
	}
	if subtle.InexactOverlap(dst[:len(src)], src) {
		panic("chacha20: invalid buffer overlap")
	}

	// xor src with buffered keystream first
	if s.len != 0 {
		buf := s.buf[len(s.buf)-s.len:]
		if len(src) < len(buf) {
			buf = buf[:len(src)]
		}
		td, ts := dst[:len(buf)], src[:len(buf)] // BCE hint
		for i, b := range buf {
			td[i] = ts[i] ^ b
		}
		s.len -= len(buf)
		if s.len != 0 {
			return
		}
		s.buf = [len(s.buf)]byte{} // zero the empty buffer
		src = src[len(buf):]
		dst = dst[len(buf):]
	}

	if len(src) == 0 {
		return
	}
	if haveAsm {
		if uint64(len(src))+uint64(s.counter)*64 > (1<<38)-64 {
			panic("chacha20: counter overflow")
		}
		s.xorKeyStreamAsm(dst, src)
		return
	}

	// set up a 64-byte buffer to pad out the final block if needed
	// (hoisted out of the main loop to avoid spills)
	rem := len(src) % 64  // length of final block
	fin := len(src) - rem // index of final block
	if rem > 0 {
		copy(s.buf[len(s.buf)-64:], src[fin:])
	}

	// pre-calculate most of the first round
	s1, s5, s9, s13 := quarterRound(j1, s.key[1], s.key[5], s.nonce[0])
	s2, s6, s10, s14 := quarterRound(j2, s.key[2], s.key[6], s.nonce[1])
	s3, s7, s11, s15 := quarterRound(j3, s.key[3], s.key[7], s.nonce[2])

	n := len(src)
	src, dst = src[:n:n], dst[:n:n] // BCE hint
	for i := 0; i < n; i += 64 {
		// calculate the remainder of the first round
		s0, s4, s8, s12 := quarterRound(j0, s.key[0], s.key[4], s.counter)

		// execute the second round
		x0, x5, x10, x15 := quarterRound(s0, s5, s10, s15)
		x1, x6, x11, x12 := quarterRound(s1, s6, s11, s12)
		x2, x7, x8, x13 := quarterRound(s2, s7, s8, s13)
		x3, x4, x9, x14 := quarterRound(s3, s4, s9, s14)

		// execute the remaining 18 rounds
		for i := 0; i < 9; i++ {
			x0, x4, x8, x12 = quarterRound(x0, x4, x8, x12)
			x1, x5, x9, x13 = quarterRound(x1, x5, x9, x13)
			x2, x6, x10, x14 = quarterRound(x2, x6, x10, x14)
			x3, x7, x11, x15 = quarterRound(x3, x7, x11, x15)

			x0, x5, x10, x15 = quarterRound(x0, x5, x10, x15)
			x1, x6, x11, x12 = quarterRound(x1, x6, x11, x12)
			x2, x7, x8, x13 = quarterRound(x2, x7, x8, x13)
			x3, x4, x9, x14 = quarterRound(x3, x4, x9, x14)
		}

		x0 += j0
		x1 += j1
		x2 += j2
		x3 += j3

		x4 += s.key[0]
		x5 += s.key[1]
		x6 += s.key[2]
		x7 += s.key[3]
		x8 += s.key[4]
		x9 += s.key[5]
		x10 += s.key[6]
		x11 += s.key[7]

		x12 += s.counter
		x13 += s.nonce[0]
		x14 += s.nonce[1]
		x15 += s.nonce[2]

		// increment the counter
		s.counter += 1
		if s.counter == 0 {
			panic("chacha20: counter overflow")
		}

		// pad to 64 bytes if needed
		in, out := src[i:], dst[i:]
		if i == fin {
			// src[fin:] has already been copied into s.buf before
			// the main loop
			in, out = s.buf[len(s.buf)-64:], s.buf[len(s.buf)-64:]
		}
		in, out = in[:64], out[:64] // BCE hint

		// XOR the key stream with the source and write out the result
		xor(out[0:], in[0:], x0)
		xor(out[4:], in[4:], x1)
		xor(out[8:], in[8:], x2)
		xor(out[12:], in[12:], x3)
		xor(out[16:], in[16:], x4)
		xor(out[20:], in[20:], x5)
		xor(out[24:], in[24:], x6)
		xor(out[28:], in[28:], x7)
		xor(out[32:], in[32:], x8)
		xor(out[36:], in[36:], x9)
		xor(out[40:], in[40:], x10)
		xor(out[44:], in[44:], x11)
		xor(out[48:], in[48:], x12)
		xor(out[52:], in[52:], x13)
		xor(out[56:], in[56:], x14)
		xor(out[60:], in[60:], x15)
	}
	// copy any trailing bytes out of the buffer and into dst
	if rem != 0 {
		s.len = 64 - rem
		copy(dst[fin:], s.buf[len(s.buf)-64:])
	}
}

// Advance discards bytes in the key stream until the next 64 byte block
// boundary is reached and updates the counter accordingly. If the key
// stream is already at a block boundary no bytes will be discarded and
// the counter will be unchanged.
func (s *Cipher) Advance() {
	s.len -= s.len % 64
	if s.len == 0 {
		s.buf = [len(s.buf)]byte{}
	}
}

// XORKeyStream crypts bytes from in to out using the given key and counters.
// In and out must overlap entirely or not at all. Counter contains the raw
// ChaCha20 counter bytes (i.e. block counter followed by nonce).
func XORKeyStream(out, in []byte, counter *[16]byte, key *[32]byte) {
	s := Cipher{
		key: [8]uint32{
			binary.LittleEndian.Uint32(key[0:4]),
			binary.LittleEndian.Uint32(key[4:8]),
			binary.LittleEndian.Uint32(key[8:12]),
			binary.LittleEndian.Uint32(key[12:16]),
			binary.LittleEndian.Uint32(key[16:20]),
			binary.LittleEndian.Uint32(key[20:24]),
			binary.LittleEndian.Uint32(key[24:28]),
			binary.LittleEndian.Uint32(key[28:32]),
		},
		nonce: [3]uint32{
			binary.LittleEndian.Uint32(counter[4:8]),
			binary.LittleEndian.Uint32(counter[8:12]),
			binary.LittleEndian.Uint32(counter[12:16]),
		},
		counter: binary.LittleEndian.Uint32(counter[0:4]),
	}
	s.XORKeyStream(out, in)
}

// HChaCha20 uses the ChaCha20 core to generate a derived key from a key and a
// nonce. It should only be used as part of the XChaCha20 construction.
func HChaCha20(key *[8]uint32, nonce *[4]uint32) [8]uint32 {
	x0, x1, x2, x3 := j0, j1, j2, j3
	x4, x5, x6, x7 := key[0], key[1], key[2], key[3]
	x8, x9, x10, x11 := key[4], key[5], key[6], key[7]
	x12, x13, x14, x15 := nonce[0], nonce[1], nonce[2], nonce[3]

	for i := 0; i < 10; i++ {
		x0, x4, x8, x12 = quarterRound(x0, x4, x8, x12)
		x1, x5, x9, x13 = quarterRound(x1, x5, x9, x13)
		x2, x6, x10, x14 = quarterRound(x2, x6, x10, x14)
		x3, x7, x11, x15 = quarterRound(x3, x7, x11, x15)

		x0, x5, x10, x15 = quarterRound(x0, x5, x10, x15)
		x1, x6, x11, x12 = quarterRound(x1, x6, x11, x12)
		x2, x7, x8, x13 = quarterRound(x2, x7, x8, x13)
		x3, x4, x9, x14 = quarterRound(x3, x4, x9, x14)
	}

	var out [8]uint32
	out[0], out[1], out[2], out[3] = x0, x1, x2, x3
	out[4], out[5], out[6], out[7] = x12, x13, x14, x15
	return out
}
