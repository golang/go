// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package poly1305 implements Poly1305 one-time message authentication code as
specified in https://cr.yp.to/mac/poly1305-20050329.pdf.

Poly1305 is a fast, one-time authentication function. It is infeasible for an
attacker to generate an authenticator for a message without the key. However, a
key must only be used for a single message. Authenticating two different
messages with the same key allows an attacker to forge authenticators for other
messages with the same key.

Poly1305 was originally coupled with AES in order to make Poly1305-AES. AES was
used with a fixed key in order to generate one-time keys from an nonce.
However, in this package AES isn't used and the one-time key is specified
directly.
*/
package poly1305 // import "golang.org/x/crypto/poly1305"

import "crypto/subtle"

// TagSize is the size, in bytes, of a poly1305 authenticator.
const TagSize = 16

// Verify returns true if mac is a valid authenticator for m with the given
// key.
func Verify(mac *[16]byte, m []byte, key *[32]byte) bool {
	var tmp [16]byte
	Sum(&tmp, m, key)
	return subtle.ConstantTimeCompare(tmp[:], mac[:]) == 1
}
