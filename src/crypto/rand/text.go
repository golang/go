// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

const base32alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"

// Text returns a cryptographically random string using the standard RFC 4648 base32 alphabet
// for use when a secret string, token, password, or other text is needed.
// The result contains at least 128 bits of randomness, enough to prevent brute force
// guessing attacks and to make the likelihood of collisions vanishingly small.
// A future version may return longer texts as needed to maintain those properties.
func Text() string {
	// ⌈log₃₂ 2¹²⁸⌉ = 26 chars
	src := make([]byte, 26)
	fillText(src)
	return string(src)
}

// fillText is outlined to make Text inlineable, so that callers who don't
// escape the returned string can avoid an allocation.
//
//go:noinline
func fillText(b []byte) {
	Read(b)
	for i := range b {
		b[i] = base32alphabet[b[i]%32]
	}
}
