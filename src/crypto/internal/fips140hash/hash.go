// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140hash

import "hash"

// sha3Unwrap is registered by crypto/sha3, so that this package doesn't need to
// import it, linking it into every binary that uses this package.
var sha3Unwrap func(hash.Hash) (hash.Hash, bool)

// RegisterSHA3Unwrap is called by crypto/sha3 at init time with a function
// that unwraps a *crypto/sha3.SHA3 into a *crypto/internal/fips140/sha3.Digest.
func RegisterSHA3Unwrap(f func(hash.Hash) (hash.Hash, bool)) {
	sha3Unwrap = f
}

// Unwrap returns h, or a crypto/internal/fips140 inner implementation of h.
//
// The return value can be type asserted to one of
// [crypto/internal/fips140/sha256.Digest],
// [crypto/internal/fips140/sha512.Digest], or
// [crypto/internal/fips140/sha3.Digest] if it is a FIPS 140-3 approved hash.
func Unwrap(h hash.Hash) hash.Hash {
	if sha3Unwrap != nil {
		if d, ok := sha3Unwrap(h); ok {
			return d
		}
	}
	return h
}

// UnwrapNew returns a function that calls newHash and applies [Unwrap] to the
// return value.
func UnwrapNew[Hash hash.Hash](newHash func() Hash) func() hash.Hash {
	return func() hash.Hash { return Unwrap(newHash()) }
}
