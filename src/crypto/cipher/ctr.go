// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Counter (CTR) mode.

// CTR converts a block cipher into a stream cipher by
// repeatedly encrypting an incrementing counter and
// xoring the resulting stream of data with the input.

// See NIST SP 800-38A, pp 13-15

package cipher

import (
	"bytes"
	"crypto/internal/alias"
	"crypto/subtle"
)

type ctr struct {
	b         Block
	ctr       []byte
	out       []byte
	outUsed   int
	offsetPos int
}

const streamBufferSize = 512

// ctrAble is an interface implemented by ciphers that have a specific optimized
// implementation of CTR, like crypto/aes. NewCTR will check for this interface
// and return the specific Stream if found.
type ctrAble interface {
	NewCTR(iv []byte) Stream
}

// NewCTR returns a Stream which encrypts/decrypts using the given Block in
// counter mode. The length of iv must be the same as the Block's block size.

func NewCTR(block Block, iv []byte) Stream {
	return NewCTRWithOffset(block, iv, 0)
}

func NewCTRWithOffset(block Block, iv []byte, offsetPos int) Stream {
	if ctr, ok := block.(ctrAble); ok {
		return ctr.NewCTR(iv)
	}
	if len(iv) != block.BlockSize() {
		panic("cipher.NewCTR: IV length must equal block size")
	}
	bufSize := streamBufferSize
	if bufSize < block.BlockSize() {
		bufSize = block.BlockSize()
	}
	return &ctr{
		offsetPos: offsetPos,
		b:         block,
		ctr:       IncreaseCtr(offsetPos, block.BlockSize(), iv),
		out:       make([]byte, 0, bufSize),
		outUsed:   0,
	}
}

func IncreaseCtr(offsetPos, BlockSize int, iv []byte) []byte {

	iv = bytes.Clone(iv)
	
	if len(iv) == 0 || offsetPos <= 0 || BlockSize <= 0 {
		return iv
	}


	needAdd := offsetPos / BlockSize
	index := 0

	for {
		add := needAdd & 0xff

		tmpIv := iv[:len(iv)-index]
		for i := len(tmpIv) - 1; i >= 0; i-- {
			if i == len(tmpIv)-1 && int(tmpIv[i])+add > 255 {
				tmpIv[i] = byte((int(tmpIv[i]) + add) % 256)
				add = 1
				continue
			}
			tmpIv[i] += byte(add)
			if tmpIv[i] != 0 {
				break
			}
		}

		index++
		if index >= len(iv) {
			break
		}

		needAdd >>= 8
		if needAdd <= 0 {
			break
		}
	}

	return iv
}

func (x *ctr) refill() {
	remain := len(x.out) - x.outUsed
	copy(x.out, x.out[x.outUsed:])
	x.out = x.out[:cap(x.out)]
	bs := x.b.BlockSize()
	for remain <= len(x.out)-bs {
		x.b.Encrypt(x.out[remain:], x.ctr)
		remain += bs

		// Increment counter
		for i := len(x.ctr) - 1; i >= 0; i-- {
			x.ctr[i]++
			if x.ctr[i] != 0 {
				break
			}
		}
	}
	x.out = x.out[:remain]
	x.outUsed = 0

	if x.offsetPos > 0 {
		offset := x.offsetPos % x.b.BlockSize()
		x.out = x.out[offset:]
		x.offsetPos = 0
	}
}

func (x *ctr) XORKeyStream(dst, src []byte) {
	if len(dst) < len(src) {
		panic("crypto/cipher: output smaller than input")
	}
	if alias.InexactOverlap(dst[:len(src)], src) {
		panic("crypto/cipher: invalid buffer overlap")
	}
	for len(src) > 0 {
		if x.outUsed >= len(x.out)-x.b.BlockSize() {
			x.refill()
		}
		n := subtle.XORBytes(dst, src, x.out[x.outUsed:])
		dst = dst[n:]
		src = src[n:]
		x.outUsed += n
	}
}
