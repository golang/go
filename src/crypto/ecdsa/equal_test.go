// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdsa_test

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"testing"
)

func testEqual(t *testing.T, c elliptic.Curve) {
	private, _ := ecdsa.GenerateKey(c, rand.Reader)
	public := &private.PublicKey

	if !public.Equal(public) {
		t.Errorf("public key is not equal to itself: %v", public)
	}
	if !public.Equal(crypto.Signer(private).Public().(*ecdsa.PublicKey)) {
		t.Errorf("private.Public() is not Equal to public: %q", public)
	}

	enc, err := x509.MarshalPKIXPublicKey(public)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := x509.ParsePKIXPublicKey(enc)
	if err != nil {
		t.Fatal(err)
	}
	if !public.Equal(decoded) {
		t.Errorf("public key is not equal to itself after decoding: %v", public)
	}

	other, _ := ecdsa.GenerateKey(c, rand.Reader)
	if public.Equal(other) {
		t.Errorf("different public keys are Equal")
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
