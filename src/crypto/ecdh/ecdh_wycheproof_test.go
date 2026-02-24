// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package ecdh_test

import (
	"bytes"
	"crypto/ecdh"
	"crypto/ecdsa"
	"crypto/internal/cryptotest/wycheproof"
	"crypto/x509"
	"fmt"
	"testing"
)

func TestSecpECDHWycheproof(t *testing.T) {
	flagsShouldPass := map[string]bool{
		// We don't support compressed points or public keys.
		"CompressedPoint":  false,
		"CompressedPublic": false,
	}

	curveToCurve := map[string]ecdh.Curve{
		"secp256r1": ecdh.P256(),
		"secp384r1": ecdh.P384(),
		"secp521r1": ecdh.P521(),
	}

	curveToKeySize := map[string]int{
		"secp256r1": 32,
		"secp384r1": 48,
		"secp521r1": 66,
	}

	for _, file := range []string{
		"ecdh_secp256r1_ecpoint_test.json",
		"ecdh_secp384r1_ecpoint_test.json",
		"ecdh_secp521r1_ecpoint_test.json",
	} {
		var testdata wycheproof.EcdhEcpointTestSchemaV1Json
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			if _, ok := curveToCurve[tg.Curve]; !ok {
				continue
			}
			curve := curveToCurve[tg.Curve]
			keySize := curveToKeySize[tg.Curve]

			for _, tv := range tg.Tests {
				testName := wycheproof.TestName(file, tv)
				tv := ecdhWycheproofTV{
					tcID:    tv.TcId,
					comment: tv.Comment,
					flags:   tv.Flags,
					result:  tv.Result,
					public:  tv.Public,
					private: tv.Private,
					shared:  tv.Shared,
				}
				t.Run(testName, func(t *testing.T) {
					t.Parallel()
					runECDHWycheproofTest(t, curve, keySize, flagsShouldPass, tv, curve.NewPublicKey)
				})
			}
		}
	}
}

func TestSecpECDHSPKIWycheproof(t *testing.T) {
	flagsShouldPass := map[string]bool{
		"CompressedPublic":         false,
		"CompressedPoint":          false,
		"UnnamedCurve":             false,
		"WrongOrder":               false,
		"UnusedParam":              false,
		"ModifiedGenerator":        false,
		"ModifiedCofactor":         false,
		"ModifiedCurveParameter":   false,
		"NoCofactor":               false,
		"Modified curve parameter": false,
		"InvalidAsn":               false,
	}

	parseSPKIPub := func(p []byte) (*ecdh.PublicKey, error) {
		pubKeyAny, err := x509.ParsePKIXPublicKey(p)
		if err != nil {
			return nil, err
		}
		ecdsaPub, ok := pubKeyAny.(*ecdsa.PublicKey)
		if !ok {
			return nil, fmt.Errorf("unexpected key type %T", pubKeyAny)
		}
		return ecdsaPub.ECDH()
	}

	curveToCurve := map[string]ecdh.Curve{
		"secp256r1": ecdh.P256(),
		"secp384r1": ecdh.P384(),
		"secp521r1": ecdh.P521(),
	}

	curveToKeySize := map[string]int{
		"secp256r1": 32,
		"secp384r1": 48,
		"secp521r1": 66,
	}

	for _, file := range []string{
		"ecdh_secp256r1_test.json",
		"ecdh_secp384r1_test.json",
		"ecdh_secp521r1_test.json",
	} {
		var testdata wycheproof.EcdhTestSchemaV1Json
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			if _, ok := curveToCurve[tg.Curve]; !ok {
				continue
			}
			curve := curveToCurve[tg.Curve]
			keySize := curveToKeySize[tg.Curve]

			for _, tv := range tg.Tests {
				testName := wycheproof.TestName(file, tv)
				tv := ecdhWycheproofTV{
					tcID:    tv.TcId,
					comment: tv.Comment,
					flags:   tv.Flags,
					result:  tv.Result,
					public:  tv.Public,
					private: tv.Private,
					shared:  tv.Shared,
				}
				t.Run(testName, func(t *testing.T) {
					t.Parallel()
					runECDHWycheproofTest(t, curve, keySize, flagsShouldPass, tv, parseSPKIPub)
				})
			}
		}
	}
}

func TestX25519ECDHWycheproof(t *testing.T) {
	flagsShouldPass := map[string]bool{
		"Twist":                  true,
		"SmallPublicKey":         false,
		"LowOrderPublic":         false,
		"ZeroSharedSecret":       false,
		"NonCanonicalPublic":     true,
		"SpecialPublicKey":       true,
		"EdgeCaseMultiplication": true,
		"EdgeCaseShared":         true,
		"Ktv":                    true,
	}

	file := "x25519_test.json"
	var testdata wycheproof.XdhCompSchemaV1Json
	wycheproof.LoadVectorFile(t, file, &testdata)

	for _, tg := range testdata.TestGroups {
		if tg.Curve != "curve25519" {
			continue
		}

		for _, tv := range tg.Tests {
			testName := wycheproof.TestName(file, tv)
			tv := ecdhWycheproofTV{
				tcID:    tv.TcId,
				comment: tv.Comment,
				flags:   tv.Flags,
				result:  tv.Result,
				public:  tv.Public,
				private: tv.Private,
				shared:  tv.Shared,
			}
			t.Run(testName, func(t *testing.T) {
				t.Parallel()
				runECDHWycheproofTest(t, ecdh.X25519(), 32, flagsShouldPass, tv, ecdh.X25519().NewPublicKey)
			})
		}
	}
}

// ecdhWycheproofTV is a representation common to the three different schemas
// we process in this test: wycheproof.XdhCompSchemaV1Json,
// wycheproof.EcdhTestSchemaV1Json and wycheproof.EcdhEcpointTestSchemaV1Json
type ecdhWycheproofTV struct {
	tcID    int
	comment string
	flags   []string
	result  wycheproof.Result
	public  string
	private string
	shared  string
}

// runECDHWycheproofTest runs test logic common to the three ECDH test schemas
// we process in this file.
func runECDHWycheproofTest(
	t *testing.T,
	curve ecdh.Curve,
	expectedKeySize int,
	flagsShouldPass map[string]bool,
	tv ecdhWycheproofTV,
	parsePub func([]byte) (*ecdh.PublicKey, error)) {
	t.Helper()

	shouldPass := wycheproof.ShouldPass(t, tv.result, tv.flags, flagsShouldPass)

	pub, err := parsePub(wycheproof.MustDecodeHex(tv.public))
	if err != nil {
		if shouldPass {
			t.Errorf("parsePub: %v", err)
		}
		return
	}

	privBytes := wycheproof.MustDecodeHex(tv.private)
	priv, err := curve.NewPrivateKey(privBytes)
	if err != nil {
		if shouldPass && len(privBytes) == expectedKeySize {
			t.Errorf("NewPrivateKey: %v", err)
		}
		return
	}

	x, err := priv.ECDH(pub)
	if err != nil {
		if shouldPass {
			t.Fatalf("ECDH: %v", err)
		}
		return
	}

	shared := wycheproof.MustDecodeHex(tv.shared)
	if shouldPass {
		if !bytes.Equal(shared, x) {
			t.Errorf("ECDH = %x, want %x", x, shared)
		}
	} else if tv.result == "invalid" {
		// For invalid inputs, a correct ECDH result is a test failure.
		if bytes.Equal(shared, x) {
			t.Errorf("ECDH = %x, want anything else", x)
		}
	}
}
