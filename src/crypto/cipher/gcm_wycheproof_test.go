// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package cipher_test

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/internal/cryptotest"
	"crypto/internal/cryptotest/wycheproof"
	"testing"
)

func TestGCMWycheproof(t *testing.T) {
	cryptotest.TestAllImplementations(t, "gcm", func(t *testing.T) {
		file := "aes_gcm_test.json"
		var testdata wycheproof.AeadTestSchemaV1Json
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()

					aesCipher, err := aes.NewCipher(wycheproof.MustDecodeHex(tv.Key))
					if err != nil {
						t.Fatalf("failed to construct cipher: %s", err)
					}
					aead, err := cipher.NewGCM(aesCipher)
					if err != nil {
						t.Fatalf("failed to construct cipher: %s", err)
					}

					iv := wycheproof.MustDecodeHex(tv.Iv)
					tag := wycheproof.MustDecodeHex(tv.Tag)
					ct := wycheproof.MustDecodeHex(tv.Ct)
					msg := wycheproof.MustDecodeHex(tv.Msg)
					aad := wycheproof.MustDecodeHex(tv.Aad)

					// The Go implementation panics on invalid nonce sizes rather
					// than returning an error. Verify this behavior.
					if len(iv) != aead.NonceSize() {
						ctWithTag := append(ct, tag...)
						wycheproof.MustPanic(t, "Seal", func() { aead.Seal(nil, iv, msg, aad) })
						wycheproof.MustPanic(t, "Open", func() { aead.Open(nil, iv, ctWithTag, aad) })
						return
					}

					genCT := aead.Seal(nil, iv, msg, aad)
					genMsg, err := aead.Open(nil, iv, genCT, aad)
					if err != nil {
						t.Errorf("failed to decrypt generated ciphertext: %s", err)
					}
					if !bytes.Equal(genMsg, msg) {
						t.Errorf("unexpected roundtripped plaintext: got %x, want %x", genMsg, msg)
					}

					ctWithTag := append(ct, tag...)
					msg2, err := aead.Open(nil, iv, ctWithTag, aad)
					wantPass := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)
					if !wantPass && err == nil {
						t.Error("decryption succeeded when it should've failed")
					} else if wantPass {
						if err != nil {
							t.Fatalf("decryption failed: %s", err)
						}
						if !bytes.Equal(genCT, ctWithTag) {
							t.Errorf("generated ciphertext doesn't match expected: got %x, want %x", genCT, ctWithTag)
						}
						if !bytes.Equal(msg, msg2) {
							t.Errorf("decrypted ciphertext doesn't match expected: got %x, want %x", msg2, msg)
						}
					}
				})
			}
		}
	})
}
