// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build fips140v1.0

package mldsa_test

import (
	"crypto"
	. "crypto/mldsa"
	"testing"
)

var _ crypto.Signer = (*PrivateKey)(nil)

func TestUnavailable(t *testing.T) {
	for _, params := range []Parameters{MLDSA44(), MLDSA65(), MLDSA87()} {
		t.Run(params.String(), func(t *testing.T) {
			if _, err := GenerateKey(params); err == nil {
				t.Errorf("GenerateKey: want error, got nil")
			}
			if _, err := NewPrivateKey(params, make([]byte, PrivateKeySize)); err == nil {
				t.Errorf("NewPrivateKey: want error, got nil")
			}
			if _, err := NewPublicKey(params, make([]byte, params.PublicKeySize())); err == nil {
				t.Errorf("NewPublicKey: want error, got nil")
			}
			if err := Verify(&PublicKey{}, nil, nil, nil); err == nil {
				t.Errorf("Verify: want error, got nil")
			}
		})
	}
}

func TestMethodsPanic(t *testing.T) {
	// All PrivateKey and PublicKey methods are unreachable in the v1.0 stub
	// (since there is no way to construct a non-zero key) and panic if invoked
	// on the zero value.
	sk := &PrivateKey{}
	pk := &PublicKey{}
	cases := []struct {
		name string
		fn   func()
	}{
		{"PrivateKey.Public", func() { sk.Public() }},
		{"PrivateKey.Equal", func() { sk.Equal(sk) }},
		{"PrivateKey.PublicKey", func() { sk.PublicKey() }},
		{"PrivateKey.Bytes", func() { sk.Bytes() }},
		{"PrivateKey.Sign", func() { sk.Sign(nil, nil, nil) }},
		{"PrivateKey.SignDeterministic", func() { sk.SignDeterministic(nil, nil) }},
		{"PublicKey.Bytes", func() { pk.Bytes() }},
		{"PublicKey.Equal", func() { pk.Equal(pk) }},
		{"PublicKey.Parameters", func() { pk.Parameters() }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("%s: did not panic", tc.name)
				}
			}()
			tc.fn()
		})
	}
}

func TestParametersAvailable(t *testing.T) {
	// The Parameters value type and its methods must remain usable even under
	// the v1.0 stub, so callers can introspect parameter sets without invoking
	// the unavailable key APIs.
	cases := []struct {
		params  Parameters
		name    string
		pkSize  int
		sigSize int
	}{
		{MLDSA44(), "ML-DSA-44", MLDSA44PublicKeySize, MLDSA44SignatureSize},
		{MLDSA65(), "ML-DSA-65", MLDSA65PublicKeySize, MLDSA65SignatureSize},
		{MLDSA87(), "ML-DSA-87", MLDSA87PublicKeySize, MLDSA87SignatureSize},
	}
	for _, tc := range cases {
		if got := tc.params.String(); got != tc.name {
			t.Errorf("String() = %q, want %q", got, tc.name)
		}
		if got := tc.params.PublicKeySize(); got != tc.pkSize {
			t.Errorf("%s PublicKeySize() = %d, want %d", tc.name, got, tc.pkSize)
		}
		if got := tc.params.SignatureSize(); got != tc.sigSize {
			t.Errorf("%s SignatureSize() = %d, want %d", tc.name, got, tc.sigSize)
		}
	}
}
