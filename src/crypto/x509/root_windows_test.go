// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509_test

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"internal/testenv"
	"net"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestPlatformVerifierLegacy(t *testing.T) {
	// TODO(#52108): This can be removed once the synthetic test root is deployed on
	// builders.
	if !testenv.HasExternalNetwork() {
		t.Skip()
	}

	getChain := func(t *testing.T, host string) []*x509.Certificate {
		t.Helper()
		c, err := tls.Dial("tcp", host+":443", &tls.Config{InsecureSkipVerify: true})
		if err != nil {
			// From https://docs.microsoft.com/en-us/windows/win32/winsock/windows-sockets-error-codes-2,
			// matching the error string observed in https://go.dev/issue/52094.
			const WSATRY_AGAIN syscall.Errno = 11002
			var errDNS *net.DNSError
			if strings.HasSuffix(host, ".badssl.com") && errors.As(err, &errDNS) && strings.HasSuffix(errDNS.Err, WSATRY_AGAIN.Error()) {
				t.Log(err)
				testenv.SkipFlaky(t, 52094)
			}

			t.Fatalf("tls connection failed: %s", err)
		}
		return c.ConnectionState().PeerCertificates
	}

	tests := []struct {
		name        string
		host        string
		verifyName  string
		verifyTime  time.Time
		expectedErr string
	}{
		{
			// whatever google.com serves should, hopefully, be trusted
			name: "valid chain",
			host: "google.com",
		},
		{
			name:       "valid chain (dns check)",
			host:       "google.com",
			verifyName: "google.com",
		},
		{
			name:       "valid chain (fqdn dns check)",
			host:       "google.com.",
			verifyName: "google.com.",
		},
		{
			name:        "expired leaf",
			host:        "expired.badssl.com",
			expectedErr: "x509: certificate has expired or is not yet valid: ",
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
			name:        "expired leaf (custom time)",
			host:        "google.com",
			verifyTime:  time.Time{}.Add(time.Hour),
			expectedErr: "x509: certificate has expired or is not yet valid: ",
		},
		{
			name:       "valid chain (custom time)",
			host:       "google.com",
			verifyTime: time.Now(),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			chain := getChain(t, tc.host)
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
