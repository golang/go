// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140

import "hash"

var approvedHashes []func(hash.Hash) bool

// RegisterApprovedHash registers a function that reports whether a [hash.Hash]
// is an approved hash implementation, for use by [ApprovedHash].
//
// It must only be called from the init functions of the module's hash
// packages. Having them register themselves, rather than having ApprovedHash
// refer to their types, avoids linking every hash implementation into every
// binary that uses ApprovedHash.
func RegisterApprovedHash(f func(hash.Hash) bool) {
	approvedHashes = append(approvedHashes, f)
}

// ApprovedHash reports whether h is an approved hash implementation from the
// module, such as SHA-256, SHA-512, or SHA3-256.
func ApprovedHash(h hash.Hash) bool {
	for _, f := range approvedHashes {
		if f(h) {
			return true
		}
	}
	return false
}
