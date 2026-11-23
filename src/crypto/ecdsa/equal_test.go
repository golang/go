// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdsa_test

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/internal/cryptotest"
	"crypto/rand"
	"crypto/x509"
	"strings"
	"testing"
)

func TestKeys(t *testing.T) {
	testAllCurves(t, testKeys)
}

func testKeys(t *testing.T, curve elliptic.Curve) {
	if strings.HasSuffix(t.Name(), "/Generic") {
		t.Skip("these methods don't support generic curves")
	}
	k1, err := ecdsa.GenerateKey(curve, rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	k2, err := ecdsa.GenerateKey(curve, rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	newPrivateKey := func(b []byte) (*ecdsa.PrivateKey, error) {
		return ecdsa.ParseRawPrivateKey(curve, b)
	}
	newPublicKey := func(b []byte) (*ecdsa.PublicKey, error) {
		return ecdsa.ParseUncompressedPublicKey(curve, b)
	}
	t.Run("PrivateKey", func(t *testing.T) {
		cryptotest.TestPrivateKey(t, k1, k2, newPrivateKey, (*ecdsa.PrivateKey).Bytes)
	})
	t.Run("PublicKey", func(t *testing.T) {
		cryptotest.TestPublicKey(t, &k1.PublicKey, &k2.PublicKey, newPublicKey, (*ecdsa.PublicKey).Bytes)
	})

	if curve == elliptic.P224() {
		return // not supported by crypto/ecdh
	}
	// The crypto/ecdh keys use the same encodings as the Bytes methods.
	t.Run("PrivateKey.ECDH", func(t *testing.T) {
		cryptotest.TestKeyEncoding(t, k1, newPrivateKey, func(k *ecdsa.PrivateKey) ([]byte, error) {
			kk, err := k.ECDH()
			if err != nil {
				return nil, err
			}
			return kk.Bytes(), nil
		})
	})
	t.Run("PublicKey.ECDH", func(t *testing.T) {
		cryptotest.TestKeyEncoding(t, &k1.PublicKey, newPublicKey, func(k *ecdsa.PublicKey) ([]byte, error) {
			kk, err := k.ECDH()
			if err != nil {
				return nil, err
			}
			return kk.Bytes(), nil
		})
	})
}

func testEqual(t *testing.T, c elliptic.Curve) {
	private, _ := ecdsa.GenerateKey(c, rand.Reader)
	public := &private.PublicKey

	if !public.Equal(public) {
		t.Errorf("public key is not equal to itself: %v", public)
	}
	if !public.Equal(crypto.Signer(private).Public().(*ecdsa.PublicKey)) {
		t.Errorf("private.Public() is not Equal to public: %q", public)
	}
	if !private.Equal(private) {
		t.Errorf("private key is not equal to itself: %v", private)
	}

	enc, err := x509.MarshalPKCS8PrivateKey(private)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := x509.ParsePKCS8PrivateKey(enc)
	if err != nil {
		t.Fatal(err)
	}
	if !public.Equal(decoded.(crypto.Signer).Public()) {
		t.Errorf("public key is not equal to itself after decoding: %v", public)
	}
	if !private.Equal(decoded) {
		t.Errorf("private key is not equal to itself after decoding: %v", private)
	}

	// Ensure that keys with the same coordinates but on different curves
	// aren't considered Equal.
	differentCurve := &ecdsa.PublicKey{}
	*differentCurve = *public // make a copy of the public key
	if differentCurve.Curve == elliptic.P256() {
		differentCurve.Curve = elliptic.P224()
	} else {
		differentCurve.Curve = elliptic.P256()
	}
	if public.Equal(differentCurve) {
		t.Errorf("public keys with different curves are Equal")
	}
}

func TestEqual(t *testing.T) {
	t.Run("P224", func(t *testing.T) { testEqual(t, elliptic.P224()) })
	if testing.Short() {
		return
	}
	t.Run("P256", func(t *testing.T) { testEqual(t, elliptic.P256()) })
	t.Run("P384", func(t *testing.T) { testEqual(t, elliptic.P384()) })
	t.Run("P521", func(t *testing.T) { testEqual(t, elliptic.P521()) })
}
