// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ChaCha20 implements the core ChaCha20 function as specified
// in https://tools.ietf.org/html/rfc7539#section-2.3.
package chacha20

import (
	"crypto/cipher"
	"encoding/binary"
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

	// qr calculates a quarter round
	qr := func(a, b, c, d uint32) (uint32, uint32, uint32, uint32) {
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

	// ChaCha20 constants
	const (
		j0 = 0x61707865
		j1 = 0x3320646e
		j2 = 0x79622d32
		j3 = 0x6b206574
	)

	// pre-calculate most of the first round
	s1, s5, s9, s13 := qr(j1, s.key[1], s.key[5], s.nonce[0])
	s2, s6, s10, s14 := qr(j2, s.key[2], s.key[6], s.nonce[1])
	s3, s7, s11, s15 := qr(j3, s.key[3], s.key[7], s.nonce[2])

	n := len(src)
	src, dst = src[:n:n], dst[:n:n] // BCE hint
	for i := 0; i < n; i += 64 {
		// calculate the remainder of the first round
		s0, s4, s8, s12 := qr(j0, s.key[0], s.key[4], s.counter)

		// execute the second round
		x0, x5, x10, x15 := qr(s0, s5, s10, s15)
		x1, x6, x11, x12 := qr(s1, s6, s11, s12)
		x2, x7, x8, x13 := qr(s2, s7, s8, s13)
		x3, x4, x9, x14 := qr(s3, s4, s9, s14)

		// execute the remaining 18 rounds
		for i := 0; i < 9; i++ {
			x0, x4, x8, x12 = qr(x0, x4, x8, x12)
			x1, x5, x9, x13 = qr(x1, x5, x9, x13)
			x2, x6, x10, x14 = qr(x2, x6, x10, x14)
			x3, x7, x11, x15 = qr(x3, x7, x11, x15)

			x0, x5, x10, x15 = qr(x0, x5, x10, x15)
			x1, x6, x11, x12 = qr(x1, x6, x11, x12)
			x2, x7, x8, x13 = qr(x2, x7, x8, x13)
			x3, x4, x9, x14 = qr(x3, x4, x9, x14)
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
