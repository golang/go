// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdsa_test

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"testing"
)

func TestMarshalText(t *testing.T) {
	curves := []struct {
		name  string
		curve elliptic.Curve
	}{
		{"P224", elliptic.P224()},
		{"P256", elliptic.P256()},
		{"P384", elliptic.P384()},
		{"P521", elliptic.P521()},
	}
	for _, test := range curves {
		t.Run(test.name, func(t *testing.T) {
			private, err := ecdsa.GenerateKey(test.curve, rand.Reader)
			if err != nil {
				t.Fatal(err)
			}

			privateText, err := private.MarshalText()
			if err != nil {
				t.Fatalf("PrivateKey.MarshalText: %v", err)
			}
			privateBlock, rest := pem.Decode(privateText)
			if privateBlock == nil || privateBlock.Type != "PRIVATE KEY" || len(rest) != 0 {
				t.Fatalf("PrivateKey.MarshalText did not produce a single PRIVATE KEY PEM block")
			}
			var private2 ecdsa.PrivateKey
			if err := private2.UnmarshalText(privateText); err != nil {
				t.Fatalf("PrivateKey.UnmarshalText: %v", err)
			}
			if !private.Equal(&private2) {
				t.Fatalf("private key changed after round trip")
			}
			parsedPrivate, err := x509.ParsePKCS8PrivateKey(privateBlock.Bytes)
			if err != nil {
				t.Fatalf("x509.ParsePKCS8PrivateKey: %v", err)
			}
			parsedPrivateKey, ok := parsedPrivate.(*ecdsa.PrivateKey)
			if !ok || !private.Equal(parsedPrivateKey) {
				t.Fatalf("x509 parsed a different private key: %T", parsedPrivate)
			}

			publicText, err := private.PublicKey.MarshalText()
			if err != nil {
				t.Fatalf("PublicKey.MarshalText: %v", err)
			}
			publicBlock, rest := pem.Decode(publicText)
			if publicBlock == nil || publicBlock.Type != "PUBLIC KEY" || len(rest) != 0 {
				t.Fatalf("PublicKey.MarshalText did not produce a single PUBLIC KEY PEM block")
			}
			var public2 ecdsa.PublicKey
			if err := public2.UnmarshalText(publicText); err != nil {
				t.Fatalf("PublicKey.UnmarshalText: %v", err)
			}
			if !private.PublicKey.Equal(&public2) {
				t.Fatalf("public key changed after round trip")
			}
			parsedPublic, err := x509.ParsePKIXPublicKey(publicBlock.Bytes)
			if err != nil {
				t.Fatalf("x509.ParsePKIXPublicKey: %v", err)
			}
			parsedPublicKey, ok := parsedPublic.(*ecdsa.PublicKey)
			if !ok || !private.PublicKey.Equal(parsedPublicKey) {
				t.Fatalf("x509 parsed a different public key: %T", parsedPublic)
			}

			encoded, err := json.Marshal(private)
			if err != nil {
				t.Fatalf("json.Marshal: %v", err)
			}
			var private3 ecdsa.PrivateKey
			if err := json.Unmarshal(encoded, &private3); err != nil {
				t.Fatalf("json.Unmarshal: %v", err)
			}
			if !private.Equal(&private3) {
				t.Fatalf("private key changed after JSON round trip")
			}
		})
	}
}
