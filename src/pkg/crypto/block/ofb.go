// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Output feedback (OFB) mode.

// OFB converts a block cipher into a stream cipher by
// repeatedly encrypting an initialization vector and
// xoring the resulting stream of data with the input.

// See NIST SP 800-38A, pp 13-15

package block

import (
	"fmt"
	"io"
)

type ofbStream struct {
	c  Cipher
	iv []byte
}

func newOFBStream(c Cipher, iv []byte) *ofbStream {
	x := new(ofbStream)
	x.c = c
	n := len(iv)
	if n != c.BlockSize() {
		panic(fmt.Sprintln("crypto/block: newOFBStream: invalid iv size", n, "!=", c.BlockSize()))
	}
	x.iv = dup(iv)
	return x
}

func (x *ofbStream) Next() []byte {
	x.c.Encrypt(x.iv, x.iv)
	return x.iv
}

// NewOFBReader returns a reader that reads data from r, decrypts (or encrypts)
// it using c in output feedback (OFB) mode with the initialization vector iv.
// The returned Reader does not buffer and has no block size.
// In OFB mode, encryption and decryption are the same operation:
// an OFB reader applied to an encrypted stream produces a decrypted
// stream and vice versa.
func NewOFBReader(c Cipher, iv []byte, r io.Reader) io.Reader {
	return newXorReader(newOFBStream(c, iv), r)
}

// NewOFBWriter returns a writer that encrypts (or decrypts) data using c
// in cipher feedback (OFB) mode with the initialization vector iv
// and writes the encrypted data to w.
// The returned Writer does not buffer and has no block size.
// In OFB mode, encryption and decryption are the same operation:
// an OFB writer applied to an decrypted stream produces an encrypted
// stream and vice versa.
func NewOFBWriter(c Cipher, iv []byte, w io.Writer) io.Writer {
	return newXorWriter(newOFBStream(c, iv), w)
}
