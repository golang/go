// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509_test

import (
	"crypto/tls"
	"crypto/x509"
	"internal/testenv"
	"testing"
	"time"
)

func TestPlatformVerifierLegacy(t *testing.T) {
	// TODO(#52108): This can be removed once the synthetic test root is deployed on
	// builders.
	if !testenv.HasExternalNetwork() {
		t.Skip()
	}

	getChain := func(host string) []*x509.Certificate {
		t.Helper()
		c, err := tls.Dial("tcp", host+":443", &tls.Config{InsecureSkipVerify: true})
		if err != nil {
			t.Fatalf("tls connection failed: %s", err)
		}
		return c.ConnectionState().PeerCertificates
	}

	tests := []struct {
		name        string
		host        string
		verifyName  string
		verifyTime  time.Time
		verifyEKU   []x509.ExtKeyUsage
		expectedErr string
		skip        string
	}{
		{
			// whatever google.com serves should, hopefully, be trusted
			name: "valid chain",
			host: "google.com",
		},
		{
			name:        "expired leaf",
			host:        "expired.badssl.com",
			expectedErr: "x509: certificate has expired or is not yet valid: “*.badssl.com” certificate is expired",
		},
		{
			name:        "wrong host for leaf",
			host:        "wrong.host.badssl.com",
			verifyName:  "wrong.host.badssl.com",
			expectedErr: "x509: certificate is valid for *.badssl.com, badssl.com, not wrong.host.badssl.com",
		},
		{
			name:        "self-signed leaf",
			host:        "self-signed.badssl.com",
			expectedErr: "x509: certificate signed by unknown authority",
		},
		{
			name:        "untrusted root",
			host:        "untrusted-root.badssl.com",
			expectedErr: "x509: certificate signed by unknown authority",
		},
		{
			name:        "revoked leaf",
			host:        "revoked.badssl.com",
			expectedErr: "x509: “revoked.badssl.com” certificate is revoked",
			skip:        "skipping; broken on recent versions of macOS. See issue 57428.",
		},
		{
			name:        "leaf missing SCTs",
			host:        "no-sct.badssl.com",
			expectedErr: "x509: “no-sct.badssl.com” certificate is not standards compliant",
			skip:        "skipping; broken on recent versions of macOS. See issue 57428.",
		},
		{
			name:        "expired leaf (custom time)",
			host:        "google.com",
			verifyTime:  time.Time{}.Add(time.Hour),
			expectedErr: "x509: certificate has expired or is not yet valid: “*.google.com” certificate is expired",
		},
		{
			name:       "valid chain (custom time)",
			host:       "google.com",
			verifyTime: time.Now(),
		},
		{
			name:        "leaf doesn't have acceptable ExtKeyUsage",
			host:        "google.com",
			expectedErr: "x509: certificate specifies an incompatible key usage",
			verifyEKU:   []x509.ExtKeyUsage{x509.ExtKeyUsageEmailProtection},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.skip != "" {
				t.Skip(tc.skip)
			}

			chain := getChain(tc.host)
			var opts x509.VerifyOptions
			if len(chain) > 1 {
				opts.Intermediates = x509.NewCertPool()
				for _, c := range chain[1:] {
					opts.Intermediates.AddCert(c)
				}
			}
			if tc.verifyName != "" {
				opts.DNSName = tc.verifyName
			}
			if !tc.verifyTime.IsZero() {
				opts.CurrentTime = tc.verifyTime
			}
			if len(tc.verifyEKU) > 0 {
				opts.KeyUsages = tc.verifyEKU
			}

			_, err := chain[0].Verify(opts)
			if err != nil && tc.expectedErr == "" {
				t.Errorf("unexpected verification error: %s", err)
			} else if err != nil && err.Error() != tc.expectedErr {
				t.Errorf("unexpected verification error: got %q, want %q", err.Error(), tc.expectedErr)
			} else if err == nil && tc.expectedErr != "" {
				t.Errorf("unexpected verification success: want %q", tc.expectedErr)
			}
		})
	}
}
