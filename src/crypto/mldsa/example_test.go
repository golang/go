// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !fips140v1.0

package mldsa_test

import (
	"crypto/mldsa"
	"fmt"
	"log"
)

func Example() {
	// The signer generates a new ML-DSA-44 key pair.
	sk, err := mldsa.GenerateKey(mldsa.MLDSA44())
	if err != nil {
		log.Fatal(err)
	}

	// The signer publishes the public key encoding.
	publicKey := sk.PublicKey().Bytes()
	fmt.Printf("public key: %d bytes\n", len(publicKey))

	// The signer signs a message and publishes the signature.
	msg := []byte("hello, world")
	sig, err := sk.Sign(nil, msg, &mldsa.Options{Context: "example"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("signature: %d bytes\n", len(sig))

	// The verifier reconstructs the public key and checks the signature.
	// The context string must match the one used by the signer.
	pk, err := mldsa.NewPublicKey(mldsa.MLDSA44(), publicKey)
	if err != nil {
		log.Fatal(err)
	}
	if err := mldsa.Verify(pk, msg, sig, &mldsa.Options{Context: "example"}); err != nil {
		log.Fatal("invalid signature: ", err)
	}

	// Output:
	// public key: 1312 bytes
	// signature: 2420 bytes
}
