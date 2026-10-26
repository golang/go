// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(fips140v1.0 || fips140v1.26)

package fipstest

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/hmac"
	"crypto/internal/fips140/sha256"
	"crypto/internal/fips140/sha3"
	"crypto/internal/fips140/sha512"
	"hash"
	"testing"
)

func TestApprovedHash(t *testing.T) {
	type wrapper struct{ *sha256.Digest }

	tests := []struct {
		name     string
		h        func() hash.Hash
		approved bool
	}{
		{"SHA2-224", func() hash.Hash { return sha256.New224() }, true},
		{"SHA2-256", func() hash.Hash { return sha256.New() }, true},
		{"SHA2-384", func() hash.Hash { return sha512.New384() }, true},
		{"SHA2-512", func() hash.Hash { return sha512.New() }, true},
		{"SHA2-512/224", func() hash.Hash { return sha512.New512_224() }, true},
		{"SHA2-512/256", func() hash.Hash { return sha512.New512_256() }, true},
		{"SHA3-224", func() hash.Hash { return sha3.New224() }, true},
		{"SHA3-256", func() hash.Hash { return sha3.New256() }, true},
		{"SHA3-384", func() hash.Hash { return sha3.New384() }, true},
		{"SHA3-512", func() hash.Hash { return sha3.New512() }, true},
		{"Keccak-256", func() hash.Hash { return sha3.NewLegacyKeccak256() }, false},
		{"Keccak-512", func() hash.Hash { return sha3.NewLegacyKeccak512() }, false},
		{"wrapped SHA2-256", func() hash.Hash { return wrapper{sha256.New()} }, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := fips140.ApprovedHash(tt.h()); got != tt.approved {
				t.Errorf("ApprovedHash = %v, want %v", got, tt.approved)
			}

			fips140.ResetServiceIndicator()
			h := hmac.New(tt.h, make([]byte, 32))
			h.Write([]byte("input"))
			h.Sum(nil)
			if got := fips140.ServiceIndicator(); got != tt.approved {
				t.Errorf("HMAC service indicator = %v, want %v", got, tt.approved)
			}
		})
	}
}
