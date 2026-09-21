// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

import (
	"bytes"
	"crypto/internal/cryptotest/wycheproof"
	"crypto/internal/fips140/aes"
	"crypto/internal/fips140/aes/gcm"
	"testing"
)

func TestCMACWycheproof(t *testing.T) {
	const file = "aes_cmac_test.json"
	var testdata wycheproof.MacTestSchemaV1Json
	wycheproof.LoadVectorFile(t, file, &testdata)

	for _, tg := range testdata.TestGroups {
		// Skip test groups with invalid AES key sizes.
		// The CMAC API takes an already-constructed aes.Block,
		// so invalid key sizes are rejected at AES key creation time.
		switch tg.KeySize {
		case 128, 192, 256:
			// Valid AES key sizes.
		default:
			continue
		}

		if tg.TagSize != 128 {
			continue
		}

		for _, tv := range tg.Tests {
			t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
				t.Parallel()

				key := wycheproof.MustDecodeHex(tv.Key)
				msg := wycheproof.MustDecodeHex(tv.Msg)
				expectedTag := wycheproof.MustDecodeHex(tv.Tag)
				wantPass := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)

				b, err := aes.New(key)
				if err != nil {
					t.Fatalf("aes.New: %v", err)
				}
				c := gcm.NewCMAC(b)
				tag := c.MAC(msg)

				if bytes.Equal(tag[:], expectedTag) {
					if !wantPass {
						t.Errorf("expected failure but tag matched")
					}
				} else {
					if wantPass {
						t.Errorf("tag mismatch: got %x, want %x", tag[:], expectedTag)
					}
				}
			})
		}
	}
}
