// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package hkdf_test

import (
	"bytes"
	"crypto/hkdf"
	"crypto/internal/cryptotest/wycheproof"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/sha512"
	"hash"
	"testing"
)

func TestHKDFWycheproof(t *testing.T) {
	filesToHash := map[string]func() hash.Hash{
		"hkdf_sha1_test.json":   sha1.New,
		"hkdf_sha256_test.json": sha256.New,
		"hkdf_sha384_test.json": sha512.New384,
		"hkdf_sha512_test.json": sha512.New,
	}

	for file, h := range filesToHash {
		var testdata wycheproof.HkdfTestSchemaV1Json
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()

					ikm := wycheproof.MustDecodeHex(tv.Ikm)
					salt := wycheproof.MustDecodeHex(tv.Salt)
					info := string(wycheproof.MustDecodeHex(tv.Info))
					wantPass := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)

					okm, err := hkdf.Key(h, ikm, salt, info, tv.Size)
					if !wantPass {
						if err == nil {
							t.Errorf("Key succeeded for invalid vector")
						}
						return
					}
					if err != nil {
						t.Fatalf("Key: %v", err)
					}
					if expectedOkm := wycheproof.MustDecodeHex(tv.Okm); !bytes.Equal(okm, expectedOkm) {
						t.Errorf("output key mismatch: got %x, want %x", okm, expectedOkm)
					}
				})
			}
		}
	}
}
