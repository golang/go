// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64 !go1.7 gccgo appengine

package chacha20poly1305

func (c *chacha20poly1305) seal(dst, nonce, plaintext, additionalData []byte) []byte {
	return c.sealGeneric(dst, nonce, plaintext, additionalData)
}

func (c *chacha20poly1305) open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	return c.openGeneric(dst, nonce, ciphertext, additionalData)
}
