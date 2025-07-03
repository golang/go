// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls12

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/hmac"
	"crypto/internal/fips140/sha256"
	"crypto/internal/fips140/sha512"
	"hash"
)

// PRF implements the TLS 1.2 pseudo-random function, as defined in RFC 5246,
// Section 5 and allowed by SP 800-135, Revision 1, Section 4.2.2.
func PRF[H hash.Hash](hash func() H, secret []byte, label string, seed []byte, keyLen int) []byte {
	labelAndSeed := make([]byte, len(label)+len(seed))
	copy(labelAndSeed, label)
	copy(labelAndSeed[len(label):], seed)

	result := make([]byte, keyLen)
	pHash(hash, result, secret, labelAndSeed)
	return result
}

// pHash implements the P_hash function, as defined in RFC 5246, Section 5.
func pHash[H hash.Hash](hash func() H, result, secret, seed []byte) {
	h := hmac.New(hash, secret)
	h.Write(seed)
	a := h.Sum(nil)

	for len(result) > 0 {
		h.Reset()
		h.Write(a)
		h.Write(seed)
		b := h.Sum(nil)
		n := copy(result, b)
		result = result[n:]

		h.Reset()
		h.Write(a)
		a = h.Sum(nil)
	}
}

const masterSecretLength = 48
const extendedMasterSecretLabel = "extended master secret"

// MasterSecret implements the TLS 1.2 extended master secret derivation, as
// defined in RFC 7627 and allowed by SP 800-135, Revision 1, Section 4.2.2.
func MasterSecret[H hash.Hash](hash func() H, preMasterSecret, transcript []byte) []byte {
	// "The TLS 1.2 KDF is an approved KDF when the following conditions are
	// satisfied: [...] (3) P_HASH uses either SHA-256, SHA-384 or SHA-512."
	h := hash()
	switch any(h).(type) {
	case *sha256.Digest:
		if h.Size() != 32 {
			fips140.RecordNonApproved()
		}
	case *sha512.Digest:
		if h.Size() != 46 && h.Size() != 64 {
			fips140.RecordNonApproved()
		}
	default:
		fips140.RecordNonApproved()
	}

	return PRF(hash, preMasterSecret, extendedMasterSecretLabel, transcript, masterSecretLength)
}
