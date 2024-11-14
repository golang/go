// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"encoding/pem"
	"math/big"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

// In order to run this test suite locally, you need to insert the test root, at
// the path below, into your trust store. This root is constrained such that it
// should not be dangerous to local developers to trust, but care should be
// taken when inserting it into the trust store not to give it increased
// permissions.
//
// On macOS the certificate can be further constrained to only be valid for
// 'SSL' in the certificate properties pane of the 'Keychain Access' program.
//
// On Windows the certificate can also be constrained to only server
// authentication in the properties pane of the certificate in the
// "Certificates" snap-in of mmc.exe.

const (
	rootCertPath = "platform_root_cert.pem"
	rootKeyPath  = "platform_root_key.pem"
)

func TestPlatformVerifier(t *testing.T) {
	if runtime.GOOS != "windows" && runtime.GOOS != "darwin" {
		t.Skip("only tested on windows and darwin")
	}

	der, err := os.ReadFile(rootCertPath)
	if err != nil {
		t.Fatalf("failed to read test root: %s", err)
	}
	b, _ := pem.Decode(der)
	testRoot, err := ParseCertificate(b.Bytes)
	if err != nil {
		t.Fatalf("failed to parse test root: %s", err)
	}

	der, err = os.ReadFile(rootKeyPath)
	if err != nil {
		t.Fatalf("failed to read test key: %s", err)
	}
	b, _ = pem.Decode(der)
	testRootKey, err := ParseECPrivateKey(b.Bytes)
	if err != nil {
		t.Fatalf("failed to parse test key: %s", err)
	}

	if _, err := testRoot.Verify(VerifyOptions{}); err != nil {
		t.Skipf("test root is not in trust store, skipping (err: %q)", err)
	}

	now := time.Now()

	tests := []struct {
		name       string
		cert       *Certificate
		selfSigned bool
		dnsName    string
		time       time.Time
		eku        []ExtKeyUsage

		expectedErr string
		windowsErr  string
		macosErr    string
	}{
		{
			name: "valid",
			cert: &Certificate{
				SerialNumber: big.NewInt(1),
				DNSNames:     []string{"valid.testing.golang.invalid"},
				NotBefore:    now.Add(-time.Hour),
				NotAfter:     now.Add(time.Hour),
				ExtKeyUsage:  []ExtKeyUsage{ExtKeyUsageServerAuth},
			},
		},
		{
			name: "valid (with name)",
			cert: &Certificate{
				SerialNumber: big.NewInt(1),
				DNSNames:     []string{"valid.testing.golang.invalid"},
				NotBefore:    now.Add(-time.Hour),
				NotAfter:     now.Add(time.Hour),
				ExtKeyUsage:  []ExtKeyUsage{ExtKeyUsageServerAuth},
			},
			dnsName: "valid.testing.golang.invalid",
		},
		{
			name: "valid (with time)",
			cert: &Certificate{
				SerialNumber: big.NewInt(1),
				DNSNames:     []string{"valid.testing.golang.invalid"},
				NotBefore:    now.Add(-time.Hour),
				NotAfter:     now.Add(time.Hour),
				ExtKeyUsage:  []ExtKeyUsage{ExtKeyUsageServerAuth},
			},
			time: now.Add(time.Minute * 30),
		},
		{
			name: "valid (with eku)",
			cert: &Certificate{
				SerialNumber: big.NewInt(1),
				DNSNames:     []string{"valid.testing.golang.invalid"},
				NotBefore:    now.Add(-time.Hour),
				NotAfter:     now.Add(time.Hour),
				ExtKeyUsage:  []ExtKeyUsage{ExtKeyUsageServerAuth},
			},
			eku: []ExtKeyUsage{ExtKeyUsageServerAuth},
		},
		{
			name: "wrong name",
			cert: &Certificate{
				SerialNumber: big.NewInt(1),
				DNSNames:     []string{"valid.testing.golang.invalid"},
				NotBefore:    now.Add(-time.Hour),
				NotAfter:     now.Add(time.Hour),
				ExtKeyUsage:  []ExtKeyUsage{ExtKeyUsageServerAuth},
			},
			dnsName:     "invalid.testing.golang.invalid",
			expectedErr: "x509: certificate is valid for valid.testing.golang.invalid, not invalid.testing.golang.invalid",
		},
		{
			name: "expired (future)",
			cert: &Certificate{
				SerialNumber: big.NewInt(1),
				DNSNames:     []string{"valid.testing.golang.invalid"},
				NotBefore:    now.Add(-time.Hour),
				NotAfter:     now.Add(time.Hour),
				ExtKeyUsage:  []ExtKeyUsage{ExtKeyUsageServerAuth},
			},
			time:        now.Add(time.Hour * 2),
			expectedErr: "x509: certificate has expired or is not yet valid",
		},
		{
			name: "expired (past)",
			cert: &Certificate{
				SerialNumber: big.NewInt(1),
				DNSNames:     []string{"valid.testing.golang.invalid"},
				NotBefore:    now.Add(-time.Hour),
				NotAfter:     now.Add(time.Hour),
				ExtKeyUsage:  []ExtKeyUsage{ExtKeyUsageServerAuth},
			},
			time:        now.Add(time.Hour * 2),
			expectedErr: "x509: certificate has expired or is not yet valid",
		},
		{
			name: "self-signed",
			cert: &Certificate{
				SerialNumber: big.NewInt(1),
				DNSNames:     []string{"valid.testing.golang.invalid"},
				NotBefore:    now.Add(-time.Hour),
				NotAfter:     now.Add(time.Hour),
				ExtKeyUsage:  []ExtKeyUsage{ExtKeyUsageServerAuth},
			},
			selfSigned: true,
			macosErr:   "x509: “valid.testing.golang.invalid” certificate is not trusted",
			windowsErr: "x509: certificate signed by unknown authority",
		},
		{
			name: "non-specified KU",
			cert: &Certificate{
				SerialNumber: big.NewInt(1),
				DNSNames:     []string{"valid.testing.golang.invalid"},
				NotBefore:    now.Add(-time.Hour),
				NotAfter:     now.Add(time.Hour),
				ExtKeyUsage:  []ExtKeyUsage{ExtKeyUsageServerAuth},
			},
			eku:         []ExtKeyUsage{ExtKeyUsageEmailProtection},
			expectedErr: "x509: certificate specifies an incompatible key usage",
		},
		{
			name: "non-nested KU",
			cert: &Certificate{
				SerialNumber: big.NewInt(1),
				DNSNames:     []string{"valid.testing.golang.invalid"},
				NotBefore:    now.Add(-time.Hour),
				NotAfter:     now.Add(time.Hour),
				ExtKeyUsage:  []ExtKeyUsage{ExtKeyUsageEmailProtection},
			},
			macosErr:   "x509: “valid.testing.golang.invalid” certificate is not permitted for this usage",
			windowsErr: "x509: certificate specifies an incompatible key usage",
		},
	}

	leafKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("ecdsa.GenerateKey failed: %s", err)
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func { t ->
			t.Parallel()
			parent := testRoot
			if tc.selfSigned {
				parent = tc.cert
			}
			certDER, err := CreateCertificate(rand.Reader, tc.cert, parent, leafKey.Public(), testRootKey)
			if err != nil {
				t.Fatalf("CreateCertificate failed: %s", err)
			}
			cert, err := ParseCertificate(certDER)
			if err != nil {
				t.Fatalf("ParseCertificate failed: %s", err)
			}

			var opts VerifyOptions
			if tc.dnsName != "" {
				opts.DNSName = tc.dnsName
			}
			if !tc.time.IsZero() {
				opts.CurrentTime = tc.time
			}
			if len(tc.eku) > 0 {
				opts.KeyUsages = tc.eku
			}

			expectedErr := tc.expectedErr
			if runtime.GOOS == "darwin" && tc.macosErr != "" {
				expectedErr = tc.macosErr
			} else if runtime.GOOS == "windows" && tc.windowsErr != "" {
				expectedErr = tc.windowsErr
			}

			_, err = cert.Verify(opts)
			if err != nil && expectedErr == "" {
				t.Errorf("unexpected verification error: %s", err)
			} else if err != nil && !strings.HasPrefix(err.Error(), expectedErr) {
				t.Errorf("unexpected verification error: got %q, want %q", err.Error(), expectedErr)
			} else if err == nil && expectedErr != "" {
				t.Errorf("unexpected verification success: want %q", expectedErr)
			}
		})
	}
}
