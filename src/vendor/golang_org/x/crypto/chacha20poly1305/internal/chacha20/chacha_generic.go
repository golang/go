// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ChaCha20 implements the core ChaCha20 function as specified in https://tools.ietf.org/html/rfc7539#section-2.3.
package chacha20

import "encoding/binary"

const rounds = 20

// core applies the ChaCha20 core function to 16-byte input in, 32-byte key k,
// and 16-byte constant c, and puts the result into 64-byte array out.
func core(out *[64]byte, in *[16]byte, k *[32]byte) {
	j0 := uint32(0x61707865)
	j1 := uint32(0x3320646e)
	j2 := uint32(0x79622d32)
	j3 := uint32(0x6b206574)
	j4 := binary.LittleEndian.Uint32(k[0:4])
	j5 := binary.LittleEndian.Uint32(k[4:8])
	j6 := binary.LittleEndian.Uint32(k[8:12])
	j7 := binary.LittleEndian.Uint32(k[12:16])
	j8 := binary.LittleEndian.Uint32(k[16:20])
	j9 := binary.LittleEndian.Uint32(k[20:24])
	j10 := binary.LittleEndian.Uint32(k[24:28])
	j11 := binary.LittleEndian.Uint32(k[28:32])
	j12 := binary.LittleEndian.Uint32(in[0:4])
	j13 := binary.LittleEndian.Uint32(in[4:8])
	j14 := binary.LittleEndian.Uint32(in[8:12])
	j15 := binary.LittleEndian.Uint32(in[12:16])

	x0, x1, x2, x3, x4, x5, x6, x7 := j0, j1, j2, j3, j4, j5, j6, j7
	x8, x9, x10, x11, x12, x13, x14, x15 := j8, j9, j10, j11, j12, j13, j14, j15

	for i := 0; i < rounds; i += 2 {
		x0 += x4
		x12 ^= x0
		x12 = (x12 << 16) | (x12 >> (16))
		x8 += x12
		x4 ^= x8
		x4 = (x4 << 12) | (x4 >> (20))
		x0 += x4
		x12 ^= x0
		x12 = (x12 << 8) | (x12 >> (24))
		x8 += x12
		x4 ^= x8
		x4 = (x4 << 7) | (x4 >> (25))
		x1 += x5
		x13 ^= x1
		x13 = (x13 << 16) | (x13 >> 16)
		x9 += x13
		x5 ^= x9
		x5 = (x5 << 12) | (x5 >> 20)
		x1 += x5
		x13 ^= x1
		x13 = (x13 << 8) | (x13 >> 24)
		x9 += x13
		x5 ^= x9
		x5 = (x5 << 7) | (x5 >> 25)
		x2 += x6
		x14 ^= x2
		x14 = (x14 << 16) | (x14 >> 16)
		x10 += x14
		x6 ^= x10
		x6 = (x6 << 12) | (x6 >> 20)
		x2 += x6
		x14 ^= x2
		x14 = (x14 << 8) | (x14 >> 24)
		x10 += x14
		x6 ^= x10
		x6 = (x6 << 7) | (x6 >> 25)
		x3 += x7
		x15 ^= x3
		x15 = (x15 << 16) | (x15 >> 16)
		x11 += x15
		x7 ^= x11
		x7 = (x7 << 12) | (x7 >> 20)
		x3 += x7
		x15 ^= x3
		x15 = (x15 << 8) | (x15 >> 24)
		x11 += x15
		x7 ^= x11
		x7 = (x7 << 7) | (x7 >> 25)
		x0 += x5
		x15 ^= x0
		x15 = (x15 << 16) | (x15 >> 16)
		x10 += x15
		x5 ^= x10
		x5 = (x5 << 12) | (x5 >> 20)
		x0 += x5
		x15 ^= x0
		x15 = (x15 << 8) | (x15 >> 24)
		x10 += x15
		x5 ^= x10
		x5 = (x5 << 7) | (x5 >> 25)
		x1 += x6
		x12 ^= x1
		x12 = (x12 << 16) | (x12 >> 16)
		x11 += x12
		x6 ^= x11
		x6 = (x6 << 12) | (x6 >> 20)
		x1 += x6
		x12 ^= x1
		x12 = (x12 << 8) | (x12 >> 24)
		x11 += x12
		x6 ^= x11
		x6 = (x6 << 7) | (x6 >> 25)
		x2 += x7
		x13 ^= x2
		x13 = (x13 << 16) | (x13 >> 16)
		x8 += x13
		x7 ^= x8
		x7 = (x7 << 12) | (x7 >> 20)
		x2 += x7
		x13 ^= x2
		x13 = (x13 << 8) | (x13 >> 24)
		x8 += x13
		x7 ^= x8
		x7 = (x7 << 7) | (x7 >> 25)
		x3 += x4
		x14 ^= x3
		x14 = (x14 << 16) | (x14 >> 16)
		x9 += x14
		x4 ^= x9
		x4 = (x4 << 12) | (x4 >> 20)
		x3 += x4
		x14 ^= x3
		x14 = (x14 << 8) | (x14 >> 24)
		x9 += x14
		x4 ^= x9
		x4 = (x4 << 7) | (x4 >> 25)
	}

	x0 += j0
	x1 += j1
	x2 += j2
	x3 += j3
	x4 += j4
	x5 += j5
	x6 += j6
	x7 += j7
	x8 += j8
	x9 += j9
	x10 += j10
	x11 += j11
	x12 += j12
	x13 += j13
	x14 += j14
	x15 += j15

	binary.LittleEndian.PutUint32(out[0:4], x0)
	binary.LittleEndian.PutUint32(out[4:8], x1)
	binary.LittleEndian.PutUint32(out[8:12], x2)
	binary.LittleEndian.PutUint32(out[12:16], x3)
	binary.LittleEndian.PutUint32(out[16:20], x4)
	binary.LittleEndian.PutUint32(out[20:24], x5)
	binary.LittleEndian.PutUint32(out[24:28], x6)
	binary.LittleEndian.PutUint32(out[28:32], x7)
	binary.LittleEndian.PutUint32(out[32:36], x8)
	binary.LittleEndian.PutUint32(out[36:40], x9)
	binary.LittleEndian.PutUint32(out[40:44], x10)
	binary.LittleEndian.PutUint32(out[44:48], x11)
	binary.LittleEndian.PutUint32(out[48:52], x12)
	binary.LittleEndian.PutUint32(out[52:56], x13)
	binary.LittleEndian.PutUint32(out[56:60], x14)
	binary.LittleEndian.PutUint32(out[60:64], x15)
}

// XORKeyStream crypts bytes from in to out using the given key and counters.
// In and out may be the same slice but otherwise should not overlap. Counter
// contains the raw ChaCha20 counter bytes (i.e. block counter followed by
// nonce).
func XORKeyStream(out, in []byte, counter *[16]byte, key *[32]byte) {
	var block [64]byte
	var counterCopy [16]byte
	copy(counterCopy[:], counter[:])

	for len(in) >= 64 {
		core(&block, &counterCopy, key)
		for i, x := range block {
			out[i] = in[i] ^ x
		}
		u := uint32(1)
		for i := 0; i < 4; i++ {
			u += uint32(counterCopy[i])
			counterCopy[i] = byte(u)
			u >>= 8
		}
		in = in[64:]
		out = out[64:]
	}

	if len(in) > 0 {
		core(&block, &counterCopy, key)
		for i, v := range in {
			out[i] = v ^ block[i]
		}
	}
}
