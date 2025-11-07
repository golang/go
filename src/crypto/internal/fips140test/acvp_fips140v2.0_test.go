// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !fips140v1.0

package fipstest

import (
	"crypto/internal/fips140/mldsa"
	_ "embed"
	"fmt"
)

//go:embed acvp_capabilities_fips140v2.0.json
var capabilitiesJson []byte

var testConfigFile = "acvp_test_fips140v2.0.config.json"

func init() {
	commands["ML-DSA-44/keyGen"] = cmdMlDsaKeyGenAft(mldsa.NewPrivateKey44)
	commands["ML-DSA-65/keyGen"] = cmdMlDsaKeyGenAft(mldsa.NewPrivateKey65)
	commands["ML-DSA-87/keyGen"] = cmdMlDsaKeyGenAft(mldsa.NewPrivateKey87)
	commands["ML-DSA-44/sigGen"] = cmdMlDsaSigGenAft()
	commands["ML-DSA-65/sigGen"] = cmdMlDsaSigGenAft()
	commands["ML-DSA-87/sigGen"] = cmdMlDsaSigGenAft()
	commands["ML-DSA-44/sigVer"] = cmdMlDsaSigVerAft(mldsa.NewPublicKey44)
	commands["ML-DSA-65/sigVer"] = cmdMlDsaSigVerAft(mldsa.NewPublicKey65)
	commands["ML-DSA-87/sigVer"] = cmdMlDsaSigVerAft(mldsa.NewPublicKey87)
}

func cmdMlDsaKeyGenAft(keyGen func([]byte) (*mldsa.PrivateKey, error)) command {
	return command{
		requiredArgs: 1, // Seed
		handler: func(args [][]byte) ([][]byte, error) {
			seed := args[0]

			sk, err := keyGen(seed)
			if err != nil {
				return nil, fmt.Errorf("generating ML-DSA 44 private key: %w", err)
			}

			// Important: we must return the full encoding of sk, not the seed.
			return [][]byte{sk.PublicKey().Bytes(), mldsa.TestingOnlyPrivateKeySemiExpandedBytes(sk)}, nil
		},
	}
}

func cmdMlDsaSigGenAft() command {
	return command{
		requiredArgs: 5, // secret key, message, randomizer, mu, context
		handler: func(args [][]byte) ([][]byte, error) {
			skSmiExpanded := args[0]
			message := args[1]         // Optional, exclusive with mu
			randomizer := args[2]      // Optional
			context := string(args[3]) // Optional
			mu := args[4]              // Optional, exclusive with message

			sk, err := mldsa.TestingOnlyNewPrivateKeyFromSemiExpanded(skSmiExpanded)
			if err != nil {
				return nil, fmt.Errorf("making ML-DSA private key from semi-expanded form: %w", err)
			}

			haveMessage := len(message) != 0
			haveRandomizer := len(randomizer) != 0
			haveMu := len(mu) != 0

			var sig []byte
			if haveMessage && !haveRandomizer && !haveMu {
				sig, err = mldsa.SignDeterministic(sk, message, context)
			} else if haveMessage && haveRandomizer && !haveMu {
				sig, err = mldsa.TestingOnlySignWithRandom(sk, message, context, randomizer)
			} else if !haveMessage && !haveRandomizer && haveMu {
				sig, err = mldsa.SignExternalMuDeterministic(sk, mu)
			} else if !haveMessage && haveRandomizer && haveMu {
				sig, err = mldsa.TestingOnlySignExternalMuWithRandom(sk, mu, randomizer)
			} else {
				return nil, fmt.Errorf(
					"unsupported ML-DSA sigGen args: have message=%v have randomizer=%v haveMu=%v haveContext=%v",
					haveMessage, haveRandomizer, haveMu, len(context) != 0)
			}

			if err != nil {
				return nil, fmt.Errorf("creating deterministic ML-DSA signature: %w", err)
			}

			return [][]byte{sig}, nil
		},
	}
}

func cmdMlDsaSigVerAft(pubKey func([]byte) (*mldsa.PublicKey, error)) command {
	return command{
		requiredArgs: 5, // public key, message, signature, context, mu
		handler: func(args [][]byte) ([][]byte, error) {
			pkRaw := args[0]
			message := args[1] // Optional, exclusive with mu
			signature := args[2]
			context := string(args[3]) // Optional
			mu := args[4]              // Optional, exclusive with message

			pk, err := pubKey(pkRaw)
			if err != nil {
				return nil, fmt.Errorf("loading ML-DSA public key: %w", err)
			}

			haveMessage := len(message) != 0
			haveMu := len(mu) != 0
			if haveMessage && !haveMu {
				err = mldsa.Verify(pk, message, signature, context)
			} else if !haveMessage && haveMu {
				err = mldsa.VerifyExternalMu(pk, mu, signature)
			} else {
				return nil, fmt.Errorf(
					"unsupported ML-DSA sigVer args: have message=%v haveMu=%v haveContext=%v",
					haveMessage, haveMu, len(context) != 0)
			}

			if err != nil {
				return [][]byte{{0}}, nil
			}

			return [][]byte{{1}}, nil
		},
	}
}
