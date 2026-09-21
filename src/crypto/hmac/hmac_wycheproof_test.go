// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package hmac_test

import (
	"bytes"
	"crypto/hmac"
	"crypto/internal/cryptotest/wycheproof"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/sha3"
	"crypto/sha512"
	"hash"
	"testing"
)

func TestHMACWycheproof(t *testing.T) {
	filesToHash := map[string]func() hash.Hash{
		"hmac_sha1_test.json":       sha1.New,
		"hmac_sha224_test.json":     sha256.New224,
		"hmac_sha256_test.json":     sha256.New,
		"hmac_sha3_224_test.json":   func() hash.Hash { return sha3.New224() },
		"hmac_sha3_256_test.json":   func() hash.Hash { return sha3.New256() },
		"hmac_sha3_384_test.json":   func() hash.Hash { return sha3.New384() },
		"hmac_sha3_512_test.json":   func() hash.Hash { return sha3.New512() },
		"hmac_sha384_test.json":     sha512.New384,
		"hmac_sha512_test.json":     sha512.New,
		"hmac_sha512_224_test.json": sha512.New512_224,
		"hmac_sha512_256_test.json": sha512.New512_256,
	}

	for file, h := range filesToHash {
		var testdata wycheproof.MacTestSchemaV1Json
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			tagSize := tg.TagSize / 8

			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()

					hm := hmac.New(h, wycheproof.MustDecodeHex(tv.Key))
					hm.Write(wycheproof.MustDecodeHex(tv.Msg))
					// Truncate the computed tag to match the expected tag size.
					computedTag := hm.Sum(nil)[:tagSize]
					got := bytes.Equal(wycheproof.MustDecodeHex(tv.Tag), computedTag)
					want := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)
					if want != got {
						t.Errorf("unexpected result")
					}
				})
			}
		}
	}
}
