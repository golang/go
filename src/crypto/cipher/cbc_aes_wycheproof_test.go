// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package cipher_test

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/internal/cryptotest"
	"crypto/internal/cryptotest/wycheproof"
	"encoding/hex"
	"testing"
)

func TestCBCAESWycheproof(t *testing.T) {
	cryptotest.TestAllImplementations(t, "aes", func(t *testing.T) {
		file := "aes_cbc_pkcs5_test.json"
		var testdata wycheproof.IndCpaTestSchemaV1Json
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()

					block, err := aes.NewCipher(wycheproof.MustDecodeHex(tv.Key))
					if err != nil {
						t.Fatalf("NewCipher: %v", err)
					}
					mode := cipher.NewCBCDecrypter(block, wycheproof.MustDecodeHex(tv.Iv))
					ct := wycheproof.MustDecodeHex(tv.Ct)
					if len(ct)%aes.BlockSize != 0 {
						t.Fatalf("ciphertext is not a multiple of the block size")
					}
					mode.CryptBlocks(ct, ct) // decrypt the block in place

					// Test cases with bad/missing padding are expected to fail,
					// but cipher.CBCDecrypter doesn't validate padding. Skip these.
					// Fail loudly if there's an invalid test for any other reason,
					// so we can evaluate what to do with it.
					for _, flag := range tv.Flags {
						if flag == "BadPadding" || flag == "NoPadding" {
							return
						}
					}
					if !wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil) {
						t.Fatalf("unexpected invalid test (not BadPadding/NoPadding)")
					}

					// Remove the PKCS#5 padding from the given ciphertext to validate it
					padding := ct[len(ct)-1]
					paddingNum := int(padding)
					for i := paddingNum; i > 0; i-- {
						if ct[len(ct)-i] != padding { // panic if the padding is unexpectedly bad
							t.Fatalf("bad padding at index=%d of %v", i, ct)
						}
					}
					ct = ct[:len(ct)-paddingNum]

					if got, want := hex.EncodeToString(ct), tv.Msg; got != want {
						t.Errorf("decoded ciphertext not equal: %s, want %s", got, want)
					}
				})
			}
		}
	})
}
