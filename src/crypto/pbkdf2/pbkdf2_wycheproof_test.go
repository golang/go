// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbkdf2_test

import (
	"bytes"
	"crypto/internal/cryptotest/wycheproof"
	"crypto/pbkdf2"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/sha512"
	"hash"
	"testing"
)

func TestWycheproof(t *testing.T) {
	filesToHash := map[string]func() hash.Hash{
		"pbkdf2_hmacsha1_test.json":   sha1.New,
		"pbkdf2_hmacsha224_test.json": sha256.New224,
		"pbkdf2_hmacsha256_test.json": sha256.New,
		"pbkdf2_hmacsha384_test.json": sha512.New384,
		"pbkdf2_hmacsha512_test.json": sha512.New,
	}

	for file, h := range filesToHash {
		var testdata wycheproof.PbkdfTestSchemaJson
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()

					password := wycheproof.MustDecodeHex(tv.Password)
					salt := wycheproof.MustDecodeHex(tv.Salt)
					expectedDk := wycheproof.MustDecodeHex(tv.Dk)
					wantPass := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)

					dk, err := pbkdf2.Key(h, string(password), salt, tv.IterationCount, tv.DkLen)
					if err != nil {
						if wantPass {
							t.Fatalf("Key: %v", err)
						}
						return
					}
					if !wantPass {
						t.Fatalf("Key unexpectedly succeeded")
						return
					}
					if !bytes.Equal(dk, expectedDk) {
						t.Errorf("derived key mismatch: got %x, want %x", dk, expectedDk)
					}
				})
			}
		}
	}
}
