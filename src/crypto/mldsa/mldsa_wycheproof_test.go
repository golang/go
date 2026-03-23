// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !fips140v1.0

package mldsa_test

import (
	"bytes"
	"crypto"
	"crypto/internal/cryptotest/wycheproof"
	internalmldsa "crypto/internal/fips140/mldsa"
	"crypto/mldsa"
	"encoding/json"
	"slices"
	"testing"
)

// TestVerifyWycheproof test signature verification using the public
// mldsa API.
func TestVerifyWycheproof(t *testing.T) {
	for _, file := range []string{
		"mldsa_44_verify_test.json",
		"mldsa_65_verify_test.json",
		"mldsa_87_verify_test.json",
	} {
		var testdata wycheproof.MldsaVerifySchemaJson
		wycheproof.LoadVectorFile(t, file, &testdata)

		params := paramsForAlg(testdata.Algorithm)

		for _, tg := range testdata.TestGroups {
			publicKey := wycheproof.MustDecodeHex(tg.PublicKey)

			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()

					shouldPass := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)

					pub, err := mldsa.NewPublicKey(params, publicKey)
					if err != nil {
						if shouldPass {
							t.Fatalf("NewPublicKey: %v", err)
						}
						return
					}

					if !bytes.Equal(pub.Bytes(), publicKey) {
						t.Errorf("public key roundtrip mismatch")
					}

					msg := wycheproof.MustDecodeHex(tv.Msg)
					sig := wycheproof.MustDecodeHex(tv.Sig)
					opts := new(mldsa.Options)
					if tv.Ctx != nil {
						opts.Context = string(wycheproof.MustDecodeHex(*tv.Ctx))
					}

					err = mldsa.Verify(pub, msg, sig, opts)
					if shouldPass && err != nil {
						t.Errorf("Verify: %v", err)
					}
					if !shouldPass && err == nil {
						t.Errorf("Verify should have failed")
					}
				})
			}
		}
	}
}

// TestSignSeedWycheproof tests key generation and signature creation using
// the public mldsa API for seed private key inputs.
//
// It covers deterministic signature creation with and without pre-hashed mu.
func TestSignSeedWycheproof(t *testing.T) {
	// We don't include the mldsa_*_sign_noseed_test.json test vector files.
	// Semi-expanded keys are not supported with the public API.
	for _, file := range []string{
		"mldsa_44_sign_seed_test.json",
		"mldsa_65_sign_seed_test.json",
		"mldsa_87_sign_seed_test.json",
	} {
		var testdata wycheproof.MldsaSignSeedSchemaJson
		wycheproof.LoadVectorFile(t, file, &testdata)

		params := paramsForAlg(testdata.Algorithm)

		for _, tg := range testdata.TestGroups {
			seed := wycheproof.MustDecodeHex(tg.PrivateSeed)
			var expectedPublicKey []byte
			if pk, ok := tg.PublicKey.(string); ok {
				expectedPublicKey = wycheproof.MustDecodeHex(pk)
			}

			for _, raw := range tg.Tests {
				tv := decodeMLDSASignTestVector(t, raw)

				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()

					shouldPass := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)

					priv, err := mldsa.NewPrivateKey(params, seed)
					if err != nil {
						if shouldPass {
							t.Fatalf("NewPrivateKey: %v", err)
						}
						return
					}
					// By checking the derived public key is equal to the vector's
					// provided public key the 'sign' vectors double as key
					// generation vectors.
					if expectedPublicKey != nil && !bytes.Equal(priv.PublicKey().Bytes(), expectedPublicKey) {
						t.Fatalf("public key mismatch")
					}

					if slices.Contains(tv.Flags, "Randomized") {
						t.Skipf("randomized signatures not supported with public API")
					}

					runSignTest(t, priv, tv, shouldPass)
				})
			}
		}
	}
}

func runSignTest(t *testing.T, priv *mldsa.PrivateKey, tv mldsaSignTestVector, shouldPass bool) {
	t.Helper()

	var msg, μ []byte
	opts := new(mldsa.Options)
	if tv.Msg != nil {
		msg = wycheproof.MustDecodeHex(*tv.Msg)
		if tv.Ctx != nil {
			opts.Context = string(wycheproof.MustDecodeHex(*tv.Ctx))
		}
	}
	if tv.Mu != nil && *tv.Mu != "" {
		μ = wycheproof.MustDecodeHex(*tv.Mu)
	}
	if msg == nil && μ == nil {
		t.Fatalf("test vector has neither msg nor mu")
	}

	var sigMsg, sigMu []byte
	var errMsg, errMu error
	if msg != nil {
		sigMsg, errMsg = priv.SignDeterministic(msg, opts)
	}
	if μ != nil {
		sigMu, errMu = priv.SignDeterministic(μ, crypto.MLDSAMu)
	}

	for _, e := range []error{errMsg, errMu} {
		if e != nil {
			if shouldPass {
				t.Fatalf("Sign: %v", e)
			}
			return
		}
	}
	if !shouldPass {
		t.Errorf("Sign unexpectedly succeeded")
		return
	}

	expectedSig := wycheproof.MustDecodeHex(tv.Sig)
	sig := sigMsg
	if sig == nil {
		sig = sigMu
	}
	if sigMsg != nil && sigMu != nil && !bytes.Equal(sigMsg, sigMu) {
		t.Errorf("Sign(msg, ctx) and SignExternalMu(mu) disagree")
	}
	if !bytes.Equal(sig, expectedSig) {
		t.Errorf("signature mismatch")
	}

	pub := priv.PublicKey()
	if msg != nil {
		if err := mldsa.Verify(pub, msg, sig, opts); err != nil {
			t.Errorf("Verify of own signature failed: %v", err)
		}
	}
	// note: we can't round-trip verify external-mu signatures with the public API.
	//  but if that capability were exposed in the future we could check here for
	//  mu != nil.
}

// TestMLDSASignSeedRandomizedWycheproof tests randomized signing with the
// internal testing-only API.
//
// It covers randomized signature creation with and without pre-hashed mu.
func TestMLDSASignSeedRandomizedWycheproof(t *testing.T) {
	for _, file := range []string{
		"mldsa_44_sign_seed_test.json",
		"mldsa_65_sign_seed_test.json",
		"mldsa_87_sign_seed_test.json",
	} {
		var testdata wycheproof.MldsaSignSeedSchemaJson
		wycheproof.LoadVectorFile(t, file, &testdata)

		newPriv := newPrivateKeyFromSeedFn(t, testdata.Algorithm)

		for _, tg := range testdata.TestGroups {
			seed := wycheproof.MustDecodeHex(tg.PrivateSeed)
			var expectedPublicKey []byte
			if pk, ok := tg.PublicKey.(string); ok {
				expectedPublicKey = wycheproof.MustDecodeHex(pk)
			}

			for _, raw := range tg.Tests {
				tv := decodeMLDSASignTestVector(t, raw)

				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()

					shouldPass := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)
					priv, err := newPriv(seed)
					if err != nil {
						if shouldPass {
							t.Fatalf("NewPrivateKey: %v", err)
						}
						return
					}

					// By checking the derived public key is equal to the vector's
					// provided public key the 'sign' vectors double as key
					// generation vectors.
					if expectedPublicKey != nil && !bytes.Equal(priv.PublicKey().Bytes(), expectedPublicKey) {
						t.Fatalf("public key mismatch")
					}

					runRandomizedSignTest(t, priv, tv, shouldPass)
				})
			}
		}
	}
}

func runRandomizedSignTest(t *testing.T, priv *internalmldsa.PrivateKey, tv mldsaSignTestVector, shouldPass bool) {
	t.Helper()

	var msg, μ []byte
	var ctx string
	rnd := make([]byte, 32)

	if tv.Msg != nil {
		msg = wycheproof.MustDecodeHex(*tv.Msg)
		if tv.Ctx != nil {
			ctx = string(wycheproof.MustDecodeHex(*tv.Ctx))
		}
	}
	if tv.Mu != nil && *tv.Mu != "" {
		μ = wycheproof.MustDecodeHex(*tv.Mu)
	}
	if tv.Rnd != nil && *tv.Rnd != "" {
		rnd = wycheproof.MustDecodeHex(*tv.Rnd)
	}

	if msg == nil && μ == nil {
		t.Fatalf("test vector has neither msg nor mu")
	}

	var sigMsg, sigMu []byte
	var errMsg, errMu error
	if msg != nil {
		sigMsg, errMsg = internalmldsa.TestingOnlySignWithRandom(priv, msg, ctx, rnd)
	}
	if μ != nil {
		sigMu, errMu = internalmldsa.TestingOnlySignExternalMuWithRandom(priv, μ, rnd)
	}

	for _, e := range []error{errMsg, errMu} {
		if e != nil {
			if shouldPass {
				t.Fatalf("Sign: %v", e)
			}
			return
		}
	}
	if !shouldPass {
		t.Errorf("Sign unexpectedly succeeded")
		return
	}

	expectedSig := wycheproof.MustDecodeHex(tv.Sig)
	sig := sigMsg
	if sig == nil {
		sig = sigMu
	}
	if sigMsg != nil && sigMu != nil && !bytes.Equal(sigMsg, sigMu) {
		t.Errorf("Sign(msg, ctx) and SignExternalMu(mu) disagree")
	}
	if !bytes.Equal(sig, expectedSig) {
		t.Errorf("signature mismatch")
	}

	pub := priv.PublicKey()
	if msg != nil {
		if err := internalmldsa.Verify(pub, msg, sig, ctx); err != nil {
			t.Errorf("Verify of own signature failed: %v", err)
		}
	}
	if μ != nil {
		if err := internalmldsa.VerifyExternalMu(pub, μ, sig); err != nil {
			t.Errorf("VerifyExternalMu of own signature failed: %v", err)
		}
	}
}

// TestMLDSANoSeedWycheproof tests semi-expanded private key inputs
// derive the correct public key using the internal testing-only API.
//
// We don't perform further signature signing operations as this is covered
// by the seed-form TestSignSeedWycheproof.
func TestMLDSANoSeedWycheproof(t *testing.T) {
	for _, file := range []string{
		"mldsa_44_sign_noseed_test.json",
		"mldsa_65_sign_noseed_test.json",
		"mldsa_87_sign_noseed_test.json",
	} {
		var testdata wycheproof.MldsaSignNoseedSchemaJson
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			privateKey := wycheproof.MustDecodeHex(tg.PrivateKey)
			var expectedPublicKey []byte
			if pk, ok := tg.PublicKey.(string); ok {
				expectedPublicKey = wycheproof.MustDecodeHex(pk)
			}

			for _, raw := range tg.Tests {
				tv := decodeMLDSASignTestVector(t, raw)
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()

					shouldPass := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)
					priv, err := internalmldsa.TestingOnlyNewPrivateKeyFromSemiExpanded(privateKey)
					if err != nil {
						if shouldPass {
							t.Fatalf("TestingOnlyNewPrivateKeyFromSemiExpanded: %v", err)
						}
						return
					}

					if expectedPublicKey != nil && !bytes.Equal(priv.PublicKey().Bytes(), expectedPublicKey) {
						t.Fatalf("public key mismatch")
					}
				})
			}
		}
	}
}

func paramsForAlg(algorithm string) mldsa.Parameters {
	switch algorithm {
	case "ML-DSA-44":
		return mldsa.MLDSA44()
	case "ML-DSA-65":
		return mldsa.MLDSA65()
	case "ML-DSA-87":
		return mldsa.MLDSA87()
	}
	panic("unknown algorithm: " + algorithm)
}

func newPrivateKeyFromSeedFn(t *testing.T, algorithm string) func([]byte) (*internalmldsa.PrivateKey, error) {
	switch algorithm {
	case "ML-DSA-44":
		return internalmldsa.NewPrivateKey44
	case "ML-DSA-65":
		return internalmldsa.NewPrivateKey65
	case "ML-DSA-87":
		return internalmldsa.NewPrivateKey87
	}
	t.Fatalf("unknown algorithm: %s", algorithm)
	return nil
}

// mldsaSignTestVector is a typed view of wycheproof.MlDsaSignTestVector,
// which the schema generator emits as interface{} because of the schema's
// conditional clauses.
type mldsaSignTestVector struct {
	TcId    int               `json:"tcId"`
	Comment string            `json:"comment"`
	Msg     *string           `json:"msg,omitempty"`
	Ctx     *string           `json:"ctx,omitempty"`
	Mu      *string           `json:"mu,omitempty"`
	Rnd     *string           `json:"rnd,omitempty"`
	Sig     string            `json:"sig"`
	Result  wycheproof.Result `json:"result"`
	Flags   []string          `json:"flags"`
}

// decodeMLDSASignTestVector roundtrips an interface{} typed raw
// MlDsaSignTestVector to produce a typed MLDSASignTestVector.
// This is a workaround for a limitation of the schema generator.
func decodeMLDSASignTestVector(t *testing.T, raw wycheproof.MlDsaSignTestVector) mldsaSignTestVector {
	t.Helper()
	b, err := json.Marshal(raw)
	if err != nil {
		t.Fatalf("re-marshal sign test vector: %v", err)
	}
	var tv mldsaSignTestVector
	if err := json.Unmarshal(b, &tv); err != nil {
		t.Fatalf("decode sign test vector: %v", err)
	}
	return tv
}
