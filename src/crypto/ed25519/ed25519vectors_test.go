// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ed25519_test

import (
	"crypto/ed25519"
	"crypto/internal/cryptotest"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestEd25519Vectors runs a very large set of test vectors that exercise all
// combinations of low-order points, low-order components, and non-canonical
// encodings. These vectors lock in unspecified and spec-divergent behaviors in
// edge cases that are not security relevant in most contexts, but that can
// cause issues in consensus applications if changed.
//
// Our behavior matches the "classic" unwritten verification rules of the
// "ref10" reference implementation.
//
// Note that although we test for these edge cases, they are not covered by the
// Go 1 Compatibility Promise. Applications that need stable verification rules
// should use github.com/hdevalence/ed25519consensus.
//
// See https://hdevalence.ca/blog/2020-10-04-its-25519am for more details.
func TestEd25519Vectors(t *testing.T) {
	jsonVectors := downloadEd25519Vectors(t)
	var vectors []struct {
		A, R, S, M string
		Flags      []string
	}
	if err := json.Unmarshal(jsonVectors, &vectors); err != nil {
		t.Fatal(err)
	}
	for i, v := range vectors {
		expectedToVerify := true
		for _, f := range v.Flags {
			switch f {
			// We use the simplified verification formula that doesn't multiply
			// by the cofactor, so any low order residue will cause the
			// signature not to verify.
			//
			// This is allowed, but not required, by RFC 8032.
			case "LowOrderResidue":
				expectedToVerify = false
			// Our point decoding allows non-canonical encodings (in violation
			// of RFC 8032) but R is not decoded: instead, R is recomputed and
			// compared bytewise against the canonical encoding.
			case "NonCanonicalR":
				expectedToVerify = false
			}
		}

		publicKey := decodeHex(t, v.A)
		signature := append(decodeHex(t, v.R), decodeHex(t, v.S)...)
		message := []byte(v.M)

		didVerify := ed25519.Verify(publicKey, message, signature)
		if didVerify && !expectedToVerify {
			t.Errorf("#%d: vector with flags %s unexpectedly verified", i, v.Flags)
		}
		if !didVerify && expectedToVerify {
			t.Errorf("#%d: vector with flags %s unexpectedly rejected", i, v.Flags)
		}
	}
}

func downloadEd25519Vectors(t *testing.T) []byte {
	// Download the JSON test file from the GOPROXY with `go mod download`,
	// pinning the version so test and module caching works as expected.
	path := "filippo.io/mostly-harmless/ed25519vectors"
	version := "v0.0.0-20210322192420-30a2d7243a94"
	dir := cryptotest.FetchModule(t, path, version)

	jsonVectors, err := os.ReadFile(filepath.Join(dir, "ed25519vectors.json"))
	if err != nil {
		t.Fatalf("failed to read ed25519vectors.json: %v", err)
	}
	return jsonVectors
}

func decodeHex(t *testing.T, s string) []byte {
	t.Helper()
	b, err := hex.DecodeString(s)
	if err != nil {
		t.Errorf("invalid hex: %v", err)
	}
	return b
}
