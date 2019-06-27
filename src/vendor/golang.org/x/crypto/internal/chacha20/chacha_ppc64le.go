// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64le,!gccgo,!appengine

package chacha20

import "encoding/binary"

const (
	bufSize = 256
	haveAsm = true
)

//go:noescape
func chaCha20_ctr32_vmx(out, inp *byte, len int, key *[8]uint32, counter *uint32)

func (c *Cipher) xorKeyStreamAsm(dst, src []byte) {
	if len(src) >= bufSize {
		chaCha20_ctr32_vmx(&dst[0], &src[0], len(src)-len(src)%bufSize, &c.key, &c.counter)
	}
	if len(src)%bufSize != 0 {
		chaCha20_ctr32_vmx(&c.buf[0], &c.buf[0], bufSize, &c.key, &c.counter)
		start := len(src) - len(src)%bufSize
		ts, td, tb := src[start:], dst[start:], c.buf[:]
		// Unroll loop to XOR 32 bytes per iteration.
		for i := 0; i < len(ts)-32; i += 32 {
			td, tb = td[:len(ts)], tb[:len(ts)] // bounds check elimination
			s0 := binary.LittleEndian.Uint64(ts[0:8])
			s1 := binary.LittleEndian.Uint64(ts[8:16])
			s2 := binary.LittleEndian.Uint64(ts[16:24])
			s3 := binary.LittleEndian.Uint64(ts[24:32])
			b0 := binary.LittleEndian.Uint64(tb[0:8])
			b1 := binary.LittleEndian.Uint64(tb[8:16])
			b2 := binary.LittleEndian.Uint64(tb[16:24])
			b3 := binary.LittleEndian.Uint64(tb[24:32])
			binary.LittleEndian.PutUint64(td[0:8], s0^b0)
			binary.LittleEndian.PutUint64(td[8:16], s1^b1)
			binary.LittleEndian.PutUint64(td[16:24], s2^b2)
			binary.LittleEndian.PutUint64(td[24:32], s3^b3)
			ts, td, tb = ts[32:], td[32:], tb[32:]
		}
		td, tb = td[:len(ts)], tb[:len(ts)] // bounds check elimination
		for i, v := range ts {
			td[i] = tb[i] ^ v
		}
		c.len = bufSize - (len(src) % bufSize)

	}

}
