// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build s390x,!gccgo,!appengine

package chacha20

var haveAsm = hasVectorFacility()

const bufSize = 256

// hasVectorFacility reports whether the machine supports the vector
// facility (vx).
// Implementation in asm_s390x.s.
func hasVectorFacility() bool

// xorKeyStreamVX is an assembly implementation of XORKeyStream. It must only
// be called when the vector facility is available.
// Implementation in asm_s390x.s.
//go:noescape
func xorKeyStreamVX(dst, src []byte, key *[8]uint32, nonce *[3]uint32, counter *uint32, buf *[256]byte, len *int)

func (c *Cipher) xorKeyStreamAsm(dst, src []byte) {
	xorKeyStreamVX(dst, src, &c.key, &c.nonce, &c.counter, &c.buf, &c.len)
}

// EXRL targets, DO NOT CALL!
func mvcSrcToBuf()
func mvcBufToDst()
