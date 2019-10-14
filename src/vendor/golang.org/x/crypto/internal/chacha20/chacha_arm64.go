// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.11
// +build !gccgo

package chacha20

const (
	haveAsm = true
	bufSize = 256
)

//go:noescape
func xorKeyStreamVX(dst, src []byte, key *[8]uint32, nonce *[3]uint32, counter *uint32)

func (c *Cipher) xorKeyStreamAsm(dst, src []byte) {

	if len(src) >= bufSize {
		xorKeyStreamVX(dst, src, &c.key, &c.nonce, &c.counter)
	}

	if len(src)%bufSize != 0 {
		i := len(src) - len(src)%bufSize
		c.buf = [bufSize]byte{}
		copy(c.buf[:], src[i:])
		xorKeyStreamVX(c.buf[:], c.buf[:], &c.key, &c.nonce, &c.counter)
		c.len = bufSize - copy(dst[i:], c.buf[:len(src)%bufSize])
	}
}
