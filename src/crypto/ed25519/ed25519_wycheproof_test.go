// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package ed25519_test

import (
	"crypto/ed25519"
	"crypto/internal/cryptotest/wycheproof"
	"crypto/x509"
	"fmt"
	"testing"
)

func TestEd25519Wycheproof(t *testing.T) {
	file := "ed25519_test.json"
	var testdata wycheproof.EddsaVerifySchemaV1Json
	wycheproof.LoadVectorFile(t, file, &testdata)

	parseSPKIPub := func(p []byte) (ed25519.PublicKey, error) {
		pubKeyAny, err := x509.ParsePKIXPublicKey(p)
		if err != nil {
			return nil, err
		}
		pub, ok := pubKeyAny.(ed25519.PublicKey)
		if !ok {
			return nil, fmt.Errorf("unexpected key type %T", pubKeyAny)
		}
		return pub, nil
	}

	for tgIdx, tg := range testdata.TestGroups {
		pubkey, err := parseSPKIPub(wycheproof.MustDecodeHex(tg.PublicKeyDer))
		if err != nil {
			t.Fatalf("test group %d invalid DER encoded public key: %v", tgIdx+1, err)
		}

		for _, tv := range tg.Tests {
			t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
				t.Parallel()

				got := ed25519.Verify(
					pubkey, wycheproof.MustDecodeHex(tv.Msg), wycheproof.MustDecodeHex(tv.Sig))
				want := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)
				if got != want {
					t.Errorf("Verify wanted success: %t", want)
				}
			})
		}
	}
}
