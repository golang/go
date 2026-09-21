// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !fips140v1.0

package fipstest

import (
	"crypto/internal/fips140/mldsa"
	"testing"
)

func fips140v126Conditionals() {
	// ML-DSA sign and verify PCT
	kMLDSA := mldsa.GenerateKey44()
	// ML-DSA-44
	mldsa.SignDeterministic(kMLDSA, make([]byte, 32), "")
}

func testFIPS140v126(t *testing.T, plaintext []byte) {
	t.Run("ML-DSA KeyGen, SigGen, SigVer", func(t *testing.T) {
		ensureServiceIndicator(t)
		k := mldsa.GenerateKey44()

		sig, err := mldsa.SignDeterministic(k, plaintext, "")
		fatalIfErr(t, err)
		t.Logf("ML-DSA signature: %x", sig)

		err = mldsa.Verify(k.PublicKey(), plaintext, sig, "")
		fatalIfErr(t, err)
	})
}
