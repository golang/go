// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ed25519

import (
	"bytes"
	"crypto/internal/fips"
	_ "crypto/internal/fips/check"
	"errors"
	"sync"
)

func fipsPCT(k *PrivateKey) error {
	return fips.PCT("Ed25519 sign and verify PCT", func() error {
		return pairwiseTest(k)
	})
}

// pairwiseTest needs to be a top-level function declaration to let the calls
// inline and their allocations not escape.
func pairwiseTest(k *PrivateKey) error {
	msg := []byte("PCT")
	sig := Sign(k, msg)
	// Note that this runs pub.a.SetBytes. If we wanted to make key generation
	// in FIPS mode faster, we could reuse A from GenerateKey. But another thing
	// that could make it faster is just _not doing a useless self-test_.
	pub, err := NewPublicKey(k.PublicKey())
	if err != nil {
		return err
	}
	return Verify(pub, msg, sig)
}

func signWithoutSelfTest(priv *PrivateKey, message []byte) []byte {
	signature := make([]byte, signatureSize)
	return signWithDom(signature, priv, message, domPrefixPure, "")
}

func verifyWithoutSelfTest(pub *PublicKey, message, sig []byte) error {
	return verifyWithDom(pub, message, sig, domPrefixPure, "")
}

var fipsSelfTest = sync.OnceFunc(func() {
	fips.CAST("Ed25519 sign and verify", func() error {
		seed := [32]byte{
			0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
			0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
			0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
			0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
		}
		msg := []byte("CAST")
		want := []byte{
			0xbd, 0xe7, 0xa5, 0xf3, 0x40, 0x73, 0xb9, 0x5a,
			0x2e, 0x6d, 0x63, 0x20, 0x0a, 0xd5, 0x92, 0x9b,
			0xa2, 0x3d, 0x00, 0x44, 0xb4, 0xc5, 0xfd, 0x62,
			0x1d, 0x5e, 0x33, 0x2f, 0xe4, 0x61, 0x42, 0x31,
			0x5b, 0x10, 0x53, 0x13, 0x4d, 0xcb, 0xd1, 0x1b,
			0x2a, 0xf6, 0xcd, 0x0e, 0xdb, 0x9a, 0xd3, 0x1e,
			0x35, 0xdb, 0x0b, 0xcf, 0x58, 0x90, 0x4f, 0xd7,
			0x69, 0x38, 0xed, 0x30, 0x51, 0x0f, 0xaa, 0x03,
		}
		k := &PrivateKey{seed: seed}
		precomputePrivateKey(k)
		pub, err := NewPublicKey(k.PublicKey())
		if err != nil {
			return err
		}
		sig := signWithoutSelfTest(k, msg)
		if !bytes.Equal(sig, want) {
			return errors.New("unexpected result")
		}
		return verifyWithoutSelfTest(pub, msg, sig)
	})
})
