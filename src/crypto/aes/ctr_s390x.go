// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aes

import (
	"crypto/cipher"
	"unsafe"
)

// Assert that aesCipherAsm implements the ctrAble interface.
var _ ctrAble = (*aesCipherAsm)(nil)

// xorBytes xors the contents of a and b and places the resulting values into
// dst. If a and b are not the same length then the number of bytes processed
// will be equal to the length of shorter of the two. Returns the number
// of bytes processed.
//go:noescape
func xorBytes(dst, a, b []byte) int

// streamBufferSize is the number of bytes of encrypted counter values to cache.
const streamBufferSize = 32 * BlockSize

type aesctr struct {
	block   *aesCipherAsm          // block cipher
	ctr     [2]uint64              // next value of the counter (big endian)
	buffer  []byte                 // buffer for the encrypted counter values
	storage [streamBufferSize]byte // array backing buffer slice
}

// NewCTR returns a Stream which encrypts/decrypts using the AES block
// cipher in counter mode. The length of iv must be the same as BlockSize.
func (c *aesCipherAsm) NewCTR(iv []byte) cipher.Stream {
	if len(iv) != BlockSize {
		panic("cipher.NewCTR: IV length must equal block size")
	}
	var ac aesctr
	ac.block = c
	ac.ctr[0] = *(*uint64)(unsafe.Pointer((&iv[0]))) // high bits
	ac.ctr[1] = *(*uint64)(unsafe.Pointer((&iv[8]))) // low bits
	ac.buffer = ac.storage[:0]
	return &ac
}

func (c *aesctr) refill() {
	// Fill up the buffer with an incrementing count.
	c.buffer = c.storage[:streamBufferSize]
	c0, c1 := c.ctr[0], c.ctr[1]
	for i := 0; i < streamBufferSize; i += BlockSize {
		b0 := (*uint64)(unsafe.Pointer(&c.buffer[i]))
		b1 := (*uint64)(unsafe.Pointer(&c.buffer[i+BlockSize/2]))
		*b0, *b1 = c0, c1
		// Increment in big endian: c0 is high, c1 is low.
		c1++
		if c1 == 0 {
			// add carry
			c0++
		}
	}
	c.ctr[0], c.ctr[1] = c0, c1
	// Encrypt the buffer using AES in ECB mode.
	cryptBlocks(c.block.function, &c.block.key[0], &c.buffer[0], &c.buffer[0], streamBufferSize)
}

func (c *aesctr) XORKeyStream(dst, src []byte) {
	for len(src) > 0 {
		if len(c.buffer) == 0 {
			c.refill()
		}
		n := xorBytes(dst, src, c.buffer)
		c.buffer = c.buffer[n:]
		src = src[n:]
		dst = dst[n:]
	}
}
