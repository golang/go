// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// OFB (Output Feedback) Mode.

package cipher

type ofb struct {
	b       Block
	out     []byte
	outUsed int
}

// NewOFB returns a Stream that encrypts or decrypts using the block cipher b
// in output feedback mode. The initialization vector iv's length must be equal
// to b's block size.
func NewOFB(b Block, iv []byte) Stream {
	blockSize := b.BlockSize()
	if len(iv) != blockSize {
		return nil
	}

	x := &ofb{
		b:       b,
		out:     make([]byte, blockSize),
		outUsed: 0,
	}
	b.Encrypt(x.out, iv)

	return x
}

func (x *ofb) XORKeyStream(dst, src []byte) {
	for i, s := range src {
		if x.outUsed == len(x.out) {
			x.b.Encrypt(x.out, x.out)
			x.outUsed = 0
		}

		dst[i] = s ^ x.out[x.outUsed]
		x.outUsed++
	}
}
