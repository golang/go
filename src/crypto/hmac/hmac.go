// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package hmac implements the Keyed-Hash Message Authentication Code (HMAC) as
defined in U.S. Federal Information Processing Standards Publication 198.
An HMAC is a cryptographic hash that uses a key to sign a message.
The receiver verifies the hash by recomputing it using the same key.

Receivers should be careful to use Equal to compare MACs in order to avoid
timing side-channels:

	// ValidMAC reports whether messageMAC is a valid HMAC tag for message.
	func ValidMAC(message, messageMAC, key []byte) bool {
		mac := hmac.New(sha256.New, key)
		mac.Write(message)
		expectedMAC := mac.Sum(nil)
		return hmac.Equal(messageMAC, expectedMAC)
	}
*/
package hmac

import (
	"crypto/internal/boring"
	"crypto/internal/fips140/hmac"
	"crypto/subtle"
	"hash"
)

// New returns a new HMAC hash using the given [hash.Hash] type and key.
// New functions like [crypto/sha256.New] can be used as h.
// h must return a new Hash every time it is called.
// Note that unlike other hash implementations in the standard library,
// the returned Hash does not implement [encoding.BinaryMarshaler]
// or [encoding.BinaryUnmarshaler].
func New(h func() hash.Hash, key []byte) hash.Hash {
	if boring.Enabled {
		hm := boring.NewHMAC(h, key)
		if hm != nil {
			return hm
		}
		// BoringCrypto did not recognize h, so fall through to standard Go code.
	}
	return hmac.New(h, key)
}

// Equal compares two MACs for equality without leaking timing information.
func Equal(mac1, mac2 []byte) bool {
	// We don't have to be constant time if the lengths of the MACs are
	// different as that suggests that a completely different hash function
	// was used.
	return subtle.ConstantTimeCompare(mac1, mac2) == 1
}
