// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CFB (Cipher Feedback) Mode.

package cipher

type cfb struct {
	b       Block
	out     []byte
	outUsed int
	decrypt bool
}

// NewCFBEncrypter returns a Stream which encrypts with cipher feedback mode,
// using the given Block. The iv must be the same length as the Block's block
// size.
func NewCFBEncrypter(block Block, iv []byte) Stream {
	return newCFB(block, iv, false)
}

// NewCFBDecrypter returns a Stream which decrypts with cipher feedback mode,
// using the given Block. The iv must be the same length as the Block's block
// size.
func NewCFBDecrypter(block Block, iv []byte) Stream {
	return newCFB(block, iv, true)
}

func newCFB(block Block, iv []byte, decrypt bool) Stream {
	blockSize := block.BlockSize()
	if len(iv) != blockSize {
		return nil
	}

	x := &cfb{
		b:       block,
		out:     make([]byte, blockSize),
		outUsed: 0,
		decrypt: decrypt,
	}
	block.Encrypt(x.out, iv)

	return x
}

func (x *cfb) XORKeyStream(dst, src []byte) {
	for i := 0; i < len(src); i++ {
		if x.outUsed == len(x.out) {
			x.b.Encrypt(x.out, x.out)
			x.outUsed = 0
		}

		if x.decrypt {
			t := src[i]
			dst[i] = src[i] ^ x.out[x.outUsed]
			x.out[x.outUsed] = t
		} else {
			x.out[x.outUsed] ^= src[i]
			dst[i] = x.out[x.outUsed]
		}
		x.outUsed++
	}
}
