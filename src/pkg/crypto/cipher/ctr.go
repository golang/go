// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Counter (CTR) mode.

// CTR converts a block cipher into a stream cipher by
// repeatedly encrypting an incrementing counter and
// xoring the resulting stream of data with the input.

// See NIST SP 800-38A, pp 13-15

package cipher

type ctr struct {
	b       Block
	ctr     []byte
	out     []byte
	outUsed int
}

// NewCTR returns a Stream which encrypts/decrypts using the given Block in
// counter mode. The length of iv must be the same as the Block's block size.
func NewCTR(block Block, iv []byte) Stream {
	if len(iv) != block.BlockSize() {
		panic("cipher.NewCTR: IV length must equal block size")
	}

	return &ctr{
		b:       block,
		ctr:     dup(iv),
		out:     make([]byte, len(iv)),
		outUsed: len(iv),
	}
}

func (x *ctr) XORKeyStream(dst, src []byte) {
	for i := 0; i < len(src); i++ {
		if x.outUsed == len(x.ctr) {
			x.b.Encrypt(x.out, x.ctr)
			x.outUsed = 0

			// Increment counter
			for i := len(x.ctr) - 1; i >= 0; i-- {
				x.ctr[i]++
				if x.ctr[i] != 0 {
					break
				}
			}
		}

		dst[i] = src[i] ^ x.out[x.outUsed]
		x.outUsed++
	}
}
