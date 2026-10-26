// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build fips140v1.0 || fips140v1.26

package fips140only

import (
	"crypto/internal/fips140/sha256"
	"crypto/internal/fips140/sha3"
	"crypto/internal/fips140/sha512"
	"crypto/internal/fips140hash"
	"hash"
)

// ApprovedHash reports whether h, after [fips140hash.Unwrap], is an approved
// hash implementation from the FIPS 140-3 module.
func ApprovedHash(h hash.Hash) bool {
	switch fips140hash.Unwrap(h).(type) {
	case *sha256.Digest, *sha512.Digest, *sha3.Digest:
		return true
	default:
		return false
	}
}
