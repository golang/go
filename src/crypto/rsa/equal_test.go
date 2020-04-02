// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa_test

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"testing"
)

func TestEqual(t *testing.T) {
	private, _ := rsa.GenerateKey(rand.Reader, 512)
	public := &private.PublicKey

	if !public.Equal(public) {
		t.Errorf("public key is not equal to itself: %v", public)
	}
	if !public.Equal(crypto.Signer(private).Public().(*rsa.PublicKey)) {
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

	other, _ := rsa.GenerateKey(rand.Reader, 512)
	if public.Equal(other) {
		t.Errorf("different public keys are Equal")
	}
}
