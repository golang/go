// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa_test

import (
	"crypto"
	"crypto/internal/cryptotest"
	"crypto/rsa"
	"crypto/x509"
	"testing"
)

func TestKeys(t *testing.T) {
	t.Setenv("GODEBUG", "rsa1024min=0")

	t.Run("PrivateKey", func(t *testing.T) {
		cryptotest.TestPrivateKey(t, test512Key, test512KeyTwo,
			x509.ParsePKCS1PrivateKey, x509.MarshalPKCS1PrivateKey)
	})
	t.Run("PublicKey", func(t *testing.T) {
		cryptotest.TestPublicKey(t, &test512Key.PublicKey, &test512KeyTwo.PublicKey,
			x509.ParsePKCS1PublicKey, x509.MarshalPKCS1PublicKey)
	})
}

func TestEqual(t *testing.T) {
	t.Setenv("GODEBUG", "rsa1024min=0")

	private := test512Key
	public := &private.PublicKey

	if !public.Equal(public) {
		t.Errorf("public key is not equal to itself: %v", public)
	}
	if !public.Equal(crypto.Signer(private).Public().(*rsa.PublicKey)) {
		t.Errorf("private.Public() is not Equal to public: %v", public)
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

	noPrecomp := *private
	noPrecomp.Precomputed = rsa.PrecomputedValues{}
	if !private.Equal(&noPrecomp) {
		t.Errorf("private key with no precomputation is not equal to itself: %v", private)
	}
}
