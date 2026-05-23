// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package dsa_test

import (
	"crypto/dsa"
	"crypto/internal/cryptotest/wycheproof"
	"crypto/x509"
	"math/big"
	"testing"

	"golang.org/x/crypto/cryptobyte"
	"golang.org/x/crypto/cryptobyte/asn1"
)

func TestDSAWycheproof(t *testing.T) {
	flagsShouldPass := map[string]bool{
		// An encoded ASN.1 integer missing a leading zero is invalid,
		// but accepted by some implementations.
		"MissingZero": false,
	}

	for _, file := range []string{
		"dsa_2048_224_sha224_test.json",
		"dsa_2048_224_sha256_test.json",
		"dsa_2048_256_sha256_test.json",
		"dsa_3072_256_sha256_test.json",
	} {
		var testdata wycheproof.DsaVerifySchemaV1Json
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			rawPub, err := x509.ParsePKIXPublicKey(wycheproof.MustDecodeHex(tg.PublicKeyDer))
			if err != nil {
				t.Fatalf("failed to parse DER encoded public key: %v", err)
			}

			pub := rawPub.(*dsa.PublicKey)
			h := wycheproof.ParseHash(tg.Sha)

			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()

					h := h.New()
					h.Write(wycheproof.MustDecodeHex(tv.Msg))
					hashed := h.Sum(nil)
					// Truncate to the byte-length of the subgroup (Q)
					hashed = hashed[:pub.Q.BitLen()/8]
					got := verifyASN1(pub, hashed, wycheproof.MustDecodeHex(tv.Sig))
					if want := wycheproof.ShouldPass(t, tv.Result, tv.Flags, flagsShouldPass); got != want {
						t.Errorf("wanted success: %t", want)
					}
				})
			}
		}
	}
}

func verifyASN1(pub *dsa.PublicKey, hash, sig []byte) bool {
	var (
		r, s  = &big.Int{}, &big.Int{}
		inner cryptobyte.String
	)
	input := cryptobyte.String(sig)
	if !input.ReadASN1(&inner, asn1.SEQUENCE) ||
		!input.Empty() ||
		!inner.ReadASN1Integer(r) ||
		!inner.ReadASN1Integer(s) ||
		!inner.Empty() {
		return false
	}
	return dsa.Verify(pub, hash, r, s)
}
