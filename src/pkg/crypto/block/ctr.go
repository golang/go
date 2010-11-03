// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Counter (CTR) mode.

// CTR converts a block cipher into a stream cipher by
// repeatedly encrypting an incrementing counter and
// xoring the resulting stream of data with the input.

// See NIST SP 800-38A, pp 13-15

package block

import (
	"io"
)

type ctrStream struct {
	c   Cipher
	ctr []byte
	out []byte
}

func newCTRStream(c Cipher, ctr []byte) *ctrStream {
	x := new(ctrStream)
	x.c = c
	x.ctr = dup(ctr)
	x.out = make([]byte, len(ctr))
	return x
}

func (x *ctrStream) Next() []byte {
	// Next block is encryption of counter.
	x.c.Encrypt(x.out, x.ctr)

	// Increment counter
	for i := len(x.ctr) - 1; i >= 0; i-- {
		x.ctr[i]++
		if x.ctr[i] != 0 {
			break
		}
	}

	return x.out
}

// NewCTRReader returns a reader that reads data from r, decrypts (or encrypts)
// it using c in counter (CTR) mode with the initialization vector iv.
// The returned Reader does not buffer and has no block size.
// In CTR mode, encryption and decryption are the same operation:
// a CTR reader applied to an encrypted stream produces a decrypted
// stream and vice versa.
func NewCTRReader(c Cipher, iv []byte, r io.Reader) io.Reader {
	return newXorReader(newCTRStream(c, iv), r)
}

// NewCTRWriter returns a writer that encrypts (or decrypts) data using c
// in counter (CTR) mode with the initialization vector iv
// and writes the encrypted data to w.
// The returned Writer does not buffer and has no block size.
// In CTR mode, encryption and decryption are the same operation:
// a CTR writer applied to an decrypted stream produces an encrypted
// stream and vice versa.
func NewCTRWriter(c Cipher, iv []byte, w io.Writer) io.Writer {
	return newXorWriter(newCTRStream(c, iv), w)
}
