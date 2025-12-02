// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140only

import (
	"crypto/fips140"
	"crypto/internal/fips140/drbg"
	"crypto/internal/fips140/sha256"
	"crypto/internal/fips140/sha3"
	"crypto/internal/fips140/sha512"
	"hash"
	"io"
)

// Enforced reports whether FIPS 140-only mode is enabled and enforced, in which non-approved
// cryptography returns an error or panics.
func Enforced() bool {
	return fips140.Enforced()
}

func ApprovedHash(h hash.Hash) bool {
	switch h.(type) {
	case *sha256.Digest, *sha512.Digest, *sha3.Digest:
		return true
	default:
		return false
	}
}

func ApprovedRandomReader(r io.Reader) bool {
	_, ok := r.(drbg.DefaultReader)
	return ok
}
