// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package ecdsa_test

import (
	"crypto/ecdsa"
	"crypto/internal/cryptotest/wycheproof"
	"crypto/x509"
	"fmt"
	"testing"
)

func TestECDSAWycheproof(t *testing.T) {
	// A map of supported curves to the list of hashes that Wycheproof has
	// test vector coverage for.
	curveAndHashes := map[string][]string{
		"secp224r1": {
			"sha224",
			"sha256",
			"sha512",
			"sha3_224",
			"sha3_256",
			"sha3_512",
		},
		"secp256r1": {
			"sha256",
			"sha512",
			"sha3_256",
			"sha3_512",
		},
		"secp384r1": {
			"sha256",
			"sha384",
			"sha512",
			"sha3_384",
			"sha3_512",
		},
		"secp521r1": {
			"sha512",
			"sha3_512",
		},
	}

	var files []string
	for c, hashes := range curveAndHashes {
		for _, h := range hashes {
			files = append(files, fmt.Sprintf("ecdsa_%s_%s_test.json", c, h))
		}
	}

	parseSPKIPub := func(p []byte) (*ecdsa.PublicKey, error) {
		pubKeyAny, err := x509.ParsePKIXPublicKey(p)
		if err != nil {
			return nil, err
		}
		ecdsaPub, ok := pubKeyAny.(*ecdsa.PublicKey)
		if !ok {
			return nil, fmt.Errorf("unexpected key type %T", pubKeyAny)
		}
		return ecdsaPub, nil
	}

	for _, file := range files {
		var testdata wycheproof.EcdsaVerifySchemaV1Json
		wycheproof.LoadVectorFile(t, file, &testdata)

		for tgIdx, tg := range testdata.TestGroups {
			pubkey, err := parseSPKIPub(wycheproof.MustDecodeHex(tg.PublicKeyDer))
			if err != nil {
				t.Fatalf("test group %d invalid DER encoded public key: %v", tgIdx+1, err)
			}

			h := wycheproof.ParseHash(tg.Sha)

			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()

					h := h.New()
					h.Write(wycheproof.MustDecodeHex(tv.Msg))
					got := ecdsa.VerifyASN1(pubkey, h.Sum(nil), wycheproof.MustDecodeHex(tv.Sig))

					if want := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil); got != want {
						t.Errorf("VerifyASN1 wanted success: %t", want)
					}
				})
			}
		}
	}
}
