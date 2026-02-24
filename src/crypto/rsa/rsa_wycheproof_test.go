// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package rsa_test

import (
	"bytes"
	"crypto/internal/cryptotest/wycheproof"
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"slices"
	"testing"
)

func TestRSAOAEPDecryptWycheproof(t *testing.T) {
	flagsShouldPass := map[string]bool{
		"Constructed":         true,
		"EncryptionWithLabel": true,
		// rsa.DecryptOAEP happily supports small key sizes
		"SmallIntegerCiphertext": true,
	}

	// TODO(XXX): support test files with different hashes for MGF/label
	for _, file := range []string{
		"rsa_oaep_2048_sha1_mgf1sha1_test.json",
		"rsa_oaep_2048_sha224_mgf1sha224_test.json",
		"rsa_oaep_2048_sha256_mgf1sha256_test.json",
		"rsa_oaep_2048_sha384_mgf1sha384_test.json",
		"rsa_oaep_2048_sha512_mgf1sha512_test.json",
		"rsa_oaep_3072_sha256_mgf1sha256_test.json",
		"rsa_oaep_3072_sha512_mgf1sha512_test.json",
		"rsa_oaep_4096_sha256_mgf1sha256_test.json",
		"rsa_oaep_4096_sha512_mgf1sha512_test.json",
		"rsa_oaep_misc_test.json",
	} {
		var testdata wycheproof.RsaesOaepDecryptSchemaV1Json
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			// TODO(XXX): support rsa_oaep_misc_test test cases with different hashes for MGF/label
			if tg.MgfSha != tg.Sha {
				t.Skip("test cases with different hashes for MGF/label not yet supported")
			}

			rawPriv, err := x509.ParsePKCS8PrivateKey(wycheproof.MustDecodeHex(tg.PrivateKeyPkcs8))
			if err != nil {
				t.Fatalf("%s failed to parse PKCS #8 private key: %s", file, err)
			}
			priv := rawPriv.(*rsa.PrivateKey)
			hash := wycheproof.ParseHash(tg.Sha)

			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()

					ct := wycheproof.MustDecodeHex(tv.Ct)
					label := wycheproof.MustDecodeHex(tv.Label)
					wantPass := wycheproof.ShouldPass(t, tv.Result, tv.Flags, flagsShouldPass)
					plaintext, err := rsa.DecryptOAEP(hash.New(), nil, priv, ct, label)
					if wantPass {
						if err != nil {
							t.Fatalf("expected success: %s", err)
						}
						if !bytes.Equal(plaintext, wycheproof.MustDecodeHex(tv.Msg)) {
							t.Errorf("unexpected plaintext: got %x, want %s", plaintext, tv.Msg)
						}
					} else if err == nil {
						t.Errorf("expected failure")
					}
				})
			}
		}
	}
}

func TestRSAPKCS1SignaturesWycheproof(t *testing.T) {
	// A map of supported modulus sizes to the list of hashes that Wycheproof has
	// test vector coverage for.
	modsAndHashes := map[int][]string{
		2048: {
			"sha224",
			"sha256",
			"sha384",
			"sha512",
			"sha512_224",
			"sha512_256",
			"sha3_224",
			"sha3_256",
			"sha3_384",
			"sha3_512",
		},
		3072: {
			"sha256",
			"sha384",
			"sha512",
			"sha512_256",
			"sha3_256",
			"sha3_384",
			"sha3_512",
		},
		4096: {
			"sha256",
			"sha384",
			"sha512",
			"sha512_256",
		},
		8192: {
			"sha256",
			"sha384",
			"sha512",
		},
	}

	var files []string
	for m, hashes := range modsAndHashes {
		for _, h := range hashes {
			files = append(files, fmt.Sprintf("rsa_signature_%d_%s_test.json", m, h))
		}
	}

	flagsShouldPass := map[string]bool{
		// Omitting the parameter field in an ASN encoded integer is a legacy behavior.
		"MissingNull": false,
	}

	for _, file := range files {
		var testdata wycheproof.RsassaPkcs1VerifySchemaV1Json
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			hash := wycheproof.ParseHash(tg.Sha)

			pub, err := x509.ParsePKCS1PublicKey(wycheproof.MustDecodeHex(tg.PublicKeyAsn))
			if err != nil {
				t.Fatalf("failed to decode pubkey: %v", err)
			}

			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()

					sig := wycheproof.MustDecodeHex(tv.Sig)
					h := hash.New()
					h.Write(wycheproof.MustDecodeHex(tv.Msg))
					err := rsa.VerifyPKCS1v15(pub, hash, h.Sum(nil), sig)
					want := wycheproof.ShouldPass(t, tv.Result, tv.Flags, flagsShouldPass)
					if (err == nil) != want {
						t.Errorf("wanted success: %t err: %v", want, err)
					}
				})
			}
		}
	}
}

func TestRSAPSSSignaturesWycheproof(t *testing.T) {
	// filesOverrideToPassZeroSLen is a map of all test files
	// and which TcIds that should be overridden to pass if the
	// rsa.PSSOptions.SaltLength is zero.
	// These tests expect a failure with a PSSOptions.SaltLength: 0
	// and a signature that uses a different salt length. However,
	// a salt length of 0 is defined as rsa.PSSSaltLengthAuto which
	// works deterministically to auto-detect the length when
	// verifying, so these tests actually pass as they should.
	filesOverrideToPassZeroSLen := map[string][]int{
		"rsa_pss_2048_sha1_mgf1_20_test.json":   {46, 47, 48, 49, 50, 51},
		"rsa_pss_2048_sha256_mgf1_0_test.json":  {67, 68, 69, 70},
		"rsa_pss_2048_sha256_mgf1_32_test.json": {67, 68, 69, 70, 71, 72},
		"rsa_pss_3072_sha256_mgf1_32_test.json": {67, 68, 69, 70, 71, 72},
		"rsa_pss_4096_sha256_mgf1_32_test.json": {67, 68, 69, 70, 71, 72},
		"rsa_pss_4096_sha512_mgf1_32_test.json": {136, 137, 138, 139, 140, 141},
		// "rsa_pss_misc_test.json": nil,  // TODO: This ones seems to be broken right now, but can enable later on.
	}

	for file, overrideIDs := range filesOverrideToPassZeroSLen {
		var testdata wycheproof.RsassaPssVerifySchemaV1Json
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			hash := wycheproof.ParseHash(tg.Sha)

			pub, err := x509.ParsePKCS1PublicKey(wycheproof.MustDecodeHex(tg.PublicKeyAsn))
			if err != nil {
				t.Fatalf("failed to decode pubkey: %v", err)
			}

			// Run all the tests twice: the first time with the salt length
			// as PSSSaltLengthAuto, and the second time with the salt length
			// explicitly set to tg.SLen.
			for i := 0; i < 2; i++ {
				saltLabel := "autoSalt"
				if i == 1 {
					saltLabel = "vecSalt"
				}
				opts := &rsa.PSSOptions{
					Hash:       hash,
					SaltLength: rsa.PSSSaltLengthAuto,
				}

				for _, tv := range tg.Tests {
					t.Run(wycheproof.TestName(file, tv)+" "+saltLabel, func(t *testing.T) {
						h := hash.New()
						h.Write(wycheproof.MustDecodeHex(tv.Msg))
						sig := wycheproof.MustDecodeHex(tv.Sig)
						err = rsa.VerifyPSS(pub, hash, h.Sum(nil), sig, opts)
						want := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)
						if opts.SaltLength == 0 && slices.Contains(overrideIDs, tv.TcId) {
							want = true
						}
						if (err == nil) != want {
							t.Errorf("wanted success: %t err: %v", want, err)
						}
					})
				}

				// Update opts.SaltLength for the second run of the tests.
				opts.SaltLength = tg.SLen
			}
		}
	}
}
