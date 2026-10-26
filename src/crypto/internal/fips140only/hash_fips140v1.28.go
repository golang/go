// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(fips140v1.0 || fips140v1.26)

package fips140only

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140hash"
	"hash"
)

// ApprovedHash reports whether h, after [fips140hash.Unwrap], is an approved
// hash implementation from the FIPS 140-3 module.
func ApprovedHash(h hash.Hash) bool {
	return fips140.ApprovedHash(fips140hash.Unwrap(h))
}
