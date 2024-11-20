// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/pem"
	"errors"
	"fmt"
	"internal/testenv"
	"math/big"
	"os/exec"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

type verifyTest struct {
	name          string
	leaf          string
	intermediates []string
	roots         []string
	currentTime   int64
	dnsName       string
	systemSkip    bool
	systemLax     bool
	keyUsages     []ExtKeyUsage

	errorCallback  func(*testing.T, error)
	expectedChains [][]string
}

var verifyTests = []verifyTest{
	{
		name:          "Valid",
		leaf:          googleLeaf,
		intermediates: []string{gtsIntermediate},
		roots:         []string{gtsRoot},
		currentTime:   1677615892,
		dnsName:       "www.google.com",

		expectedChains: [][]string{
			{"www.google.com", "GTS CA 1C3", "GTS Root R1"},
		},
	},
	{
		name:          "Valid (fqdn)",
		leaf:          googleLeaf,
		intermediates: []string{gtsIntermediate},
		roots:         []string{gtsRoot},
		currentTime:   1677615892,
		dnsName:       "www.google.com.",

		expectedChains: [][]string{
			{"www.google.com", "GTS CA 1C3", "GTS Root R1"},
		},
	},
	{
		name:          "MixedCase",
		leaf:          googleLeaf,
		intermediates: []string{gtsIntermediate},
		roots:         []string{gtsRoot},
		currentTime:   1677615892,
		dnsName:       "WwW.GooGLE.coM",

		expectedChains: [][]string{
			{"www.google.com", "GTS CA 1C3", "GTS Root R1"},
		},
	},
	{
		name:          "HostnameMismatch",
		leaf:          googleLeaf,
		intermediates: []string{gtsIntermediate},
		roots:         []string{gtsRoot},
		currentTime:   1677615892,
		dnsName:       "www.example.com",

		errorCallback: expectHostnameError("certificate is valid for"),
	},
	{
		name:          "IPMissing",
		leaf:          googleLeaf,
		intermediates: []string{gtsIntermediate},
		roots:         []string{gtsRoot},
		currentTime:   1677615892,
		dnsName:       "1.2.3.4",

		errorCallback: expectHostnameError("doesn't contain any IP SANs"),
	},
	{
		name:          "Expired",
		leaf:          googleLeaf,
		intermediates: []string{gtsIntermediate},
		roots:         []string{gtsRoot},
		currentTime:   1,
		dnsName:       "www.example.com",

		errorCallback: expectExpired,
	},
	{
		name:        "MissingIntermediate",
		leaf:        googleLeaf,
		roots:       []string{gtsRoot},
		currentTime: 1677615892,
		dnsName:     "www.google.com",

		// Skip when using systemVerify, since Windows
		// *will* find the missing intermediate cert.
		systemSkip:    true,
		errorCallback: expectAuthorityUnknown,
	},
	{
		name:          "RootInIntermediates",
		leaf:          googleLeaf,
		intermediates: []string{gtsRoot, gtsIntermediate},
		roots:         []string{gtsRoot},
		currentTime:   1677615892,
		dnsName:       "www.google.com",

		expectedChains: [][]string{
			{"www.google.com", "GTS CA 1C3", "GTS Root R1"},
		},
		// CAPI doesn't build the chain with the duplicated GeoTrust
		// entry so the results don't match.
		systemLax: true,
	},
	{
		name:          "dnssec-exp",
		leaf:          dnssecExpLeaf,
		intermediates: []string{startComIntermediate},
		roots:         []string{startComRoot},
		currentTime:   1302726541,

		// The StartCom root is not trusted by Windows when the default
		// ServerAuth EKU is requested.
		systemSkip: true,

		expectedChains: [][]string{
			{"dnssec-exp", "StartCom Class 1", "StartCom Certification Authority"},
		},
	},
	{
		name:          "dnssec-exp/AnyEKU",
		leaf:          dnssecExpLeaf,
		intermediates: []string{startComIntermediate},
		roots:         []string{startComRoot},
		currentTime:   1302726541,
		keyUsages:     []ExtKeyUsage{ExtKeyUsageAny},

		expectedChains: [][]string{
			{"dnssec-exp", "StartCom Class 1", "StartCom Certification Authority"},
		},
	},
	{
		name:          "dnssec-exp/RootInIntermediates",
		leaf:          dnssecExpLeaf,
		intermediates: []string{startComIntermediate, startComRoot},
		roots:         []string{startComRoot},
		currentTime:   1302726541,
		systemSkip:    true, // see dnssec-exp test

		expectedChains: [][]string{
			{"dnssec-exp", "StartCom Class 1", "StartCom Certification Authority"},
		},
	},
	{
		name:          "InvalidHash",
		leaf:          googleLeafWithInvalidHash,
		intermediates: []string{gtsIntermediate},
		roots:         []string{gtsRoot},
		currentTime:   1677615892,
		dnsName:       "www.google.com",

		// The specific error message may not occur when using system
		// verification.
		systemLax:     true,
		errorCallback: expectHashError,
	},
	// EKULeaf tests use an unconstrained chain leading to a leaf certificate
	// with an E-mail Protection EKU but not a Server Auth one, checking that
	// the EKUs on the leaf are enforced.
	{
		name:          "EKULeaf",
		leaf:          smimeLeaf,
		intermediates: []string{smimeIntermediate},
		roots:         []string{smimeRoot},
		currentTime:   1594673418,

		errorCallback: expectUsageError,
	},
	{
		name:          "EKULeafExplicit",
		leaf:          smimeLeaf,
		intermediates: []string{smimeIntermediate},
		roots:         []string{smimeRoot},
		currentTime:   1594673418,
		keyUsages:     []ExtKeyUsage{ExtKeyUsageServerAuth},

		errorCallback: expectUsageError,
	},
	{
		name:          "EKULeafValid",
		leaf:          smimeLeaf,
		intermediates: []string{smimeIntermediate},
		roots:         []string{smimeRoot},
		currentTime:   1594673418,
		keyUsages:     []ExtKeyUsage{ExtKeyUsageEmailProtection},

		expectedChains: [][]string{
			{"CORPORATIVO FICTICIO ACTIVO", "EAEko Herri Administrazioen CA - CA AAPP Vascas (2)", "IZENPE S.A."},
		},
	},
	{
		// Check that a name constrained intermediate works even when
		// it lists multiple constraints.
		name:          "MultipleConstraints",
		leaf:          nameConstraintsLeaf,
		intermediates: []string{nameConstraintsIntermediate1, nameConstraintsIntermediate2},
		roots:         []string{globalSignRoot},
		currentTime:   1382387896,
		dnsName:       "secure.iddl.vt.edu",

		expectedChains: [][]string{
			{
				"Technology-enhanced Learning and Online Strategies",
				"Virginia Tech Global Qualified Server CA",
				"Trusted Root CA G2",
				"GlobalSign Root CA",
			},
		},
	},
	{
		// Check that SHA-384 intermediates (which are popping up)
		// work.
		name:          "SHA-384",
		leaf:          trustAsiaLeaf,
		intermediates: []string{trustAsiaSHA384Intermediate},
		roots:         []string{digicertRoot},
		currentTime:   1558051200,
		dnsName:       "tm.cn",

		// CryptoAPI can find alternative validation paths.
		systemLax: true,

		expectedChains: [][]string{
			{
				"tm.cn",
				"TrustAsia ECC OV TLS Pro CA",
				"DigiCert Global Root CA",
			},
		},
	},
	{
		// Putting a certificate as a root directly should work as a
		// way of saying “exactly this”.
		name:        "LeafInRoots",
		leaf:        selfSigned,
		roots:       []string{selfSigned},
		currentTime: 1471624472,
		dnsName:     "foo.example",
		systemSkip:  true, // does not chain to a system root

		expectedChains: [][]string{
			{"Acme Co"},
		},
	},
	{
		// Putting a certificate as a root directly should not skip
		// other checks however.
		name:        "LeafInRootsInvalid",
		leaf:        selfSigned,
		roots:       []string{selfSigned},
		currentTime: 1471624472,
		dnsName:     "notfoo.example",
		systemSkip:  true, // does not chain to a system root

		errorCallback: expectHostnameError("certificate is valid for"),
	},
	{
		// An X.509 v1 certificate should not be accepted as an
		// intermediate.
		name:          "X509v1Intermediate",
		leaf:          x509v1TestLeaf,
		intermediates: []string{x509v1TestIntermediate},
		roots:         []string{x509v1TestRoot},
		currentTime:   1481753183,
		systemSkip:    true, // does not chain to a system root

		errorCallback: expectNotAuthorizedError,
	},
	{
		name:        "IgnoreCNWithSANs",
		leaf:        ignoreCNWithSANLeaf,
		dnsName:     "foo.example.com",
		roots:       []string{ignoreCNWithSANRoot},
		currentTime: 1486684488,
		systemSkip:  true, // does not chain to a system root

		errorCallback: expectHostnameError("certificate is not valid for any names"),
	},
	{
		// Test that excluded names are respected.
		name:          "ExcludedNames",
		leaf:          excludedNamesLeaf,
		dnsName:       "bender.local",
		intermediates: []string{excludedNamesIntermediate},
		roots:         []string{excludedNamesRoot},
		currentTime:   1486684488,
		systemSkip:    true, // does not chain to a system root

		errorCallback: expectNameConstraintsError,
	},
	{
		// Test that unknown critical extensions in a leaf cause a
		// verify error.
		name:          "CriticalExtLeaf",
		leaf:          criticalExtLeafWithExt,
		intermediates: []string{criticalExtIntermediate},
		roots:         []string{criticalExtRoot},
		currentTime:   1486684488,
		systemSkip:    true, // does not chain to a system root

		errorCallback: expectUnhandledCriticalExtension,
	},
	{
		// Test that unknown critical extensions in an intermediate
		// cause a verify error.
		name:          "CriticalExtIntermediate",
		leaf:          criticalExtLeaf,
		intermediates: []string{criticalExtIntermediateWithExt},
		roots:         []string{criticalExtRoot},
		currentTime:   1486684488,
		systemSkip:    true, // does not chain to a system root

		errorCallback: expectUnhandledCriticalExtension,
	},
	{
		name:        "ValidCN",
		leaf:        validCNWithoutSAN,
		dnsName:     "foo.example.com",
		roots:       []string{invalidCNRoot},
		currentTime: 1540000000,
		systemSkip:  true, // does not chain to a system root

		errorCallback: expectHostnameError("certificate relies on legacy Common Name field"),
	},
	{
		// A certificate with an AKID should still chain to a parent without SKID.
		// See Issue 30079.
		name:        "AKIDNoSKID",
		leaf:        leafWithAKID,
		roots:       []string{rootWithoutSKID},
		currentTime: 1550000000,
		dnsName:     "example",
		systemSkip:  true, // does not chain to a system root

		expectedChains: [][]string{
			{"Acme LLC", "Acme Co"},
		},
	},
	{
		// When there are two parents, one with an incorrect subject but matching SKID
		// and one with a correct subject but missing SKID, the latter should be
		// considered as a possible parent.
		leaf:        leafMatchingAKIDMatchingIssuer,
		roots:       []string{rootMatchingSKIDMismatchingSubject, rootMismatchingSKIDMatchingSubject},
		currentTime: 1550000000,
		dnsName:     "example",
		systemSkip:  true,

		expectedChains: [][]string{
			{"Leaf", "Root B"},
		},
	},
}

func expectHostnameError(msg string) func(*testing.T, error) {
	return func(t *testing.T, err error) {
		if _, ok := err.(HostnameError); !ok {
			t.Fatalf("error was not a HostnameError: %v", err)
		}
		if !strings.Contains(err.Error(), msg) {
			t.Fatalf("HostnameError did not contain %q: %v", msg, err)
		}
	}
}

func expectExpired(t *testing.T, err error) {
	if inval, ok := err.(CertificateInvalidError); !ok || inval.Reason != Expired {
		t.Fatalf("error was not Expired: %v", err)
	}
}

func expectUsageError(t *testing.T, err error) {
	if inval, ok := err.(CertificateInvalidError); !ok || inval.Reason != IncompatibleUsage {
		t.Fatalf("error was not IncompatibleUsage: %v", err)
	}
}

func expectAuthorityUnknown(t *testing.T, err error) {
	e, ok := err.(UnknownAuthorityError)
	if !ok {
		t.Fatalf("error was not UnknownAuthorityError: %v", err)
	}
	if e.Cert == nil {
		t.Fatalf("error was UnknownAuthorityError, but missing Cert: %v", err)
	}
}

func expectHashError(t *testing.T, err error) {
	if err == nil {
		t.Fatalf("no error resulted from invalid hash")
	}
	if expected := "algorithm unimplemented"; !strings.Contains(err.Error(), expected) {
		t.Fatalf("error resulting from invalid hash didn't contain '%s', rather it was: %v", expected, err)
	}
}

func expectNameConstraintsError(t *testing.T, err error) {
	if inval, ok := err.(CertificateInvalidError); !ok || inval.Reason != CANotAuthorizedForThisName {
		t.Fatalf("error was not a CANotAuthorizedForThisName: %v", err)
	}
}

func expectNotAuthorizedError(t *testing.T, err error) {
	if inval, ok := err.(CertificateInvalidError); !ok || inval.Reason != NotAuthorizedToSign {
		t.Fatalf("error was not a NotAuthorizedToSign: %v", err)
	}
}

func expectUnhandledCriticalExtension(t *testing.T, err error) {
	if _, ok := err.(UnhandledCriticalExtension); !ok {
		t.Fatalf("error was not an UnhandledCriticalExtension: %v", err)
	}
}

func certificateFromPEM(pemBytes string) (*Certificate, error) {
	block, _ := pem.Decode([]byte(pemBytes))
	if block == nil {
		return nil, errors.New("failed to decode PEM")
	}
	return ParseCertificate(block.Bytes)
}

func testVerify(t *testing.T, test verifyTest, useSystemRoots bool) {
	opts := VerifyOptions{
		Intermediates: NewCertPool(),
		DNSName:       test.dnsName,
		CurrentTime:   time.Unix(test.currentTime, 0),
		KeyUsages:     test.keyUsages,
	}

	if !useSystemRoots {
		opts.Roots = NewCertPool()
		for j, root := range test.roots {
			ok := opts.Roots.AppendCertsFromPEM([]byte(root))
			if !ok {
				t.Fatalf("failed to parse root #%d", j)
			}
		}
	}

	for j, intermediate := range test.intermediates {
		ok := opts.Intermediates.AppendCertsFromPEM([]byte(intermediate))
		if !ok {
			t.Fatalf("failed to parse intermediate #%d", j)
		}
	}

	leaf, err := certificateFromPEM(test.leaf)
	if err != nil {
		t.Fatalf("failed to parse leaf: %v", err)
	}

	chains, err := leaf.Verify(opts)

	if test.errorCallback == nil && err != nil {
		if runtime.GOOS == "windows" && strings.HasSuffix(testenv.Builder(), "-2008") && err.Error() == "x509: certificate signed by unknown authority" {
			testenv.SkipFlaky(t, 19564)
		}
		t.Fatalf("unexpected error: %v", err)
	}
	if test.errorCallback != nil {
		if useSystemRoots && test.systemLax {
			if err == nil {
				t.Fatalf("expected error")
			}
		} else {
			test.errorCallback(t, err)
		}
	}

	doesMatch := func(expectedChain []string, chain []*Certificate) bool {
		if len(chain) != len(expectedChain) {
			return false
		}

		for k, cert := range chain {
			if !strings.Contains(nameToKey(&cert.Subject), expectedChain[k]) {
				return false
			}
		}
		return true
	}

	// Every expected chain should match one (or more) returned chain. We tolerate multiple
	// matches, as due to root store semantics it is plausible that (at least on the system
	// verifiers) multiple identical (looking) chains may be returned when two roots with the
	// same subject are present.
	for _, expectedChain := range test.expectedChains {
		var match bool
		for _, chain := range chains {
			if doesMatch(expectedChain, chain) {
				match = true
				break
			}
		}

		if !match {
			t.Errorf("No match found for %v", expectedChain)
		}
	}

	// Every returned chain should match 1 expected chain (or <2 if testing against the system)
	for _, chain := range chains {
		nMatched := 0
		for _, expectedChain := range test.expectedChains {
			if doesMatch(expectedChain, chain) {
				nMatched++
			}
		}
		// Allow additional unknown chains if systemLax is set
		if nMatched == 0 && test.systemLax == false || nMatched > 1 {
			t.Errorf("Got %v matches for chain %v", nMatched, chainToDebugString(chain))
			for _, expectedChain := range test.expectedChains {
				if doesMatch(expectedChain, chain) {
					t.Errorf("\t matched %v", expectedChain)
				}
			}
		}
	}
}

func TestGoVerify(t *testing.T) {
	// Temporarily enable SHA-1 verification since a number of test chains
	// require it. TODO(filippo): regenerate test chains.
	t.Setenv("GODEBUG", "x509sha1=1")

	for _, test := range verifyTests {
		t.Run(test.name, func(t *testing.T) {
			testVerify(t, test, false)
		})
	}
}

func TestSystemVerify(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skipf("skipping verify test using system APIs on %q", runtime.GOOS)
	}

	for _, test := range verifyTests {
		t.Run(test.name, func(t *testing.T) {
			if test.systemSkip {
				t.SkipNow()
			}
			testVerify(t, test, true)
		})
	}
}

func chainToDebugString(chain []*Certificate) string {
	var chainStr string
	for _, cert := range chain {
		if len(chainStr) > 0 {
			chainStr += " -> "
		}
		chainStr += nameToKey(&cert.Subject)
	}
	return chainStr
}

func nameToKey(name *pkix.Name) string {
	return strings.Join(name.Country, ",") + "/" + strings.Join(name.Organization, ",") + "/" + strings.Join(name.OrganizationalUnit, ",") + "/" + name.CommonName
}

const gtsIntermediate = `-----BEGIN CERTIFICATE-----
MIIFljCCA36gAwIBAgINAgO8U1lrNMcY9QFQZjANBgkqhkiG9w0BAQsFADBHMQsw
CQYDVQQGEwJVUzEiMCAGA1UEChMZR29vZ2xlIFRydXN0IFNlcnZpY2VzIExMQzEU
MBIGA1UEAxMLR1RTIFJvb3QgUjEwHhcNMjAwODEzMDAwMDQyWhcNMjcwOTMwMDAw
MDQyWjBGMQswCQYDVQQGEwJVUzEiMCAGA1UEChMZR29vZ2xlIFRydXN0IFNlcnZp
Y2VzIExMQzETMBEGA1UEAxMKR1RTIENBIDFDMzCCASIwDQYJKoZIhvcNAQEBBQAD
ggEPADCCAQoCggEBAPWI3+dijB43+DdCkH9sh9D7ZYIl/ejLa6T/belaI+KZ9hzp
kgOZE3wJCor6QtZeViSqejOEH9Hpabu5dOxXTGZok3c3VVP+ORBNtzS7XyV3NzsX
lOo85Z3VvMO0Q+sup0fvsEQRY9i0QYXdQTBIkxu/t/bgRQIh4JZCF8/ZK2VWNAcm
BA2o/X3KLu/qSHw3TT8An4Pf73WELnlXXPxXbhqW//yMmqaZviXZf5YsBvcRKgKA
gOtjGDxQSYflispfGStZloEAoPtR28p3CwvJlk/vcEnHXG0g/Zm0tOLKLnf9LdwL
tmsTDIwZKxeWmLnwi/agJ7u2441Rj72ux5uxiZ0CAwEAAaOCAYAwggF8MA4GA1Ud
DwEB/wQEAwIBhjAdBgNVHSUEFjAUBggrBgEFBQcDAQYIKwYBBQUHAwIwEgYDVR0T
AQH/BAgwBgEB/wIBADAdBgNVHQ4EFgQUinR/r4XN7pXNPZzQ4kYU83E1HScwHwYD
VR0jBBgwFoAU5K8rJnEaK0gnhS9SZizv8IkTcT4waAYIKwYBBQUHAQEEXDBaMCYG
CCsGAQUFBzABhhpodHRwOi8vb2NzcC5wa2kuZ29vZy9ndHNyMTAwBggrBgEFBQcw
AoYkaHR0cDovL3BraS5nb29nL3JlcG8vY2VydHMvZ3RzcjEuZGVyMDQGA1UdHwQt
MCswKaAnoCWGI2h0dHA6Ly9jcmwucGtpLmdvb2cvZ3RzcjEvZ3RzcjEuY3JsMFcG
A1UdIARQME4wOAYKKwYBBAHWeQIFAzAqMCgGCCsGAQUFBwIBFhxodHRwczovL3Br
aS5nb29nL3JlcG9zaXRvcnkvMAgGBmeBDAECATAIBgZngQwBAgIwDQYJKoZIhvcN
AQELBQADggIBAIl9rCBcDDy+mqhXlRu0rvqrpXJxtDaV/d9AEQNMwkYUuxQkq/BQ
cSLbrcRuf8/xam/IgxvYzolfh2yHuKkMo5uhYpSTld9brmYZCwKWnvy15xBpPnrL
RklfRuFBsdeYTWU0AIAaP0+fbH9JAIFTQaSSIYKCGvGjRFsqUBITTcFTNvNCCK9U
+o53UxtkOCcXCb1YyRt8OS1b887U7ZfbFAO/CVMkH8IMBHmYJvJh8VNS/UKMG2Yr
PxWhu//2m+OBmgEGcYk1KCTd4b3rGS3hSMs9WYNRtHTGnXzGsYZbr8w0xNPM1IER
lQCh9BIiAfq0g3GvjLeMcySsN1PCAJA/Ef5c7TaUEDu9Ka7ixzpiO2xj2YC/WXGs
Yye5TBeg2vZzFb8q3o/zpWwygTMD0IZRcZk0upONXbVRWPeyk+gB9lm+cZv9TSjO
z23HFtz30dZGm6fKa+l3D/2gthsjgx0QGtkJAITgRNOidSOzNIb2ILCkXhAd4FJG
AJ2xDx8hcFH1mt0G/FX0Kw4zd8NLQsLxdxP8c4CU6x+7Nz/OAipmsHMdMqUybDKw
juDEI/9bfU1lcKwrmz3O2+BtjjKAvpafkmO8l7tdufThcV4q5O8DIrGKZTqPwJNl
1IXNDw9bg1kWRxYtnCQ6yICmJhSFm/Y3m6xv+cXDBlHz4n/FsRC6UfTd
-----END CERTIFICATE-----`

const gtsRoot = `-----BEGIN CERTIFICATE-----
MIIFVzCCAz+gAwIBAgINAgPlk28xsBNJiGuiFzANBgkqhkiG9w0BAQwFADBHMQsw
CQYDVQQGEwJVUzEiMCAGA1UEChMZR29vZ2xlIFRydXN0IFNlcnZpY2VzIExMQzEU
MBIGA1UEAxMLR1RTIFJvb3QgUjEwHhcNMTYwNjIyMDAwMDAwWhcNMzYwNjIyMDAw
MDAwWjBHMQswCQYDVQQGEwJVUzEiMCAGA1UEChMZR29vZ2xlIFRydXN0IFNlcnZp
Y2VzIExMQzEUMBIGA1UEAxMLR1RTIFJvb3QgUjEwggIiMA0GCSqGSIb3DQEBAQUA
A4ICDwAwggIKAoICAQC2EQKLHuOhd5s73L+UPreVp0A8of2C+X0yBoJx9vaMf/vo
27xqLpeXo4xL+Sv2sfnOhB2x+cWX3u+58qPpvBKJXqeqUqv4IyfLpLGcY9vXmX7w
Cl7raKb0xlpHDU0QM+NOsROjyBhsS+z8CZDfnWQpJSMHobTSPS5g4M/SCYe7zUjw
TcLCeoiKu7rPWRnWr4+wB7CeMfGCwcDfLqZtbBkOtdh+JhpFAz2weaSUKK0Pfybl
qAj+lug8aJRT7oM6iCsVlgmy4HqMLnXWnOunVmSPlk9orj2XwoSPwLxAwAtcvfaH
szVsrBhQf4TgTM2S0yDpM7xSma8ytSmzJSq0SPly4cpk9+aCEI3oncKKiPo4Zor8
Y/kB+Xj9e1x3+naH+uzfsQ55lVe0vSbv1gHR6xYKu44LtcXFilWr06zqkUspzBmk
MiVOKvFlRNACzqrOSbTqn3yDsEB750Orp2yjj32JgfpMpf/VjsPOS+C12LOORc92
wO1AK/1TD7Cn1TsNsYqiA94xrcx36m97PtbfkSIS5r762DL8EGMUUXLeXdYWk70p
aDPvOmbsB4om3xPXV2V4J95eSRQAogB/mqghtqmxlbCluQ0WEdrHbEg8QOB+DVrN
VjzRlwW5y0vtOUucxD/SVRNuJLDWcfr0wbrM7Rv1/oFB2ACYPTrIrnqYNxgFlQID
AQABo0IwQDAOBgNVHQ8BAf8EBAMCAYYwDwYDVR0TAQH/BAUwAwEB/zAdBgNVHQ4E
FgQU5K8rJnEaK0gnhS9SZizv8IkTcT4wDQYJKoZIhvcNAQEMBQADggIBAJ+qQibb
C5u+/x6Wki4+omVKapi6Ist9wTrYggoGxval3sBOh2Z5ofmmWJyq+bXmYOfg6LEe
QkEzCzc9zolwFcq1JKjPa7XSQCGYzyI0zzvFIoTgxQ6KfF2I5DUkzps+GlQebtuy
h6f88/qBVRRiClmpIgUxPoLW7ttXNLwzldMXG+gnoot7TiYaelpkttGsN/H9oPM4
7HLwEXWdyzRSjeZ2axfG34arJ45JK3VmgRAhpuo+9K4l/3wV3s6MJT/KYnAK9y8J
ZgfIPxz88NtFMN9iiMG1D53Dn0reWVlHxYciNuaCp+0KueIHoI17eko8cdLiA6Ef
MgfdG+RCzgwARWGAtQsgWSl4vflVy2PFPEz0tv/bal8xa5meLMFrUKTX5hgUvYU/
Z6tGn6D/Qqc6f1zLXbBwHSs09dR2CQzreExZBfMzQsNhFRAbd03OIozUhfJFfbdT
6u9AWpQKXCBfTkBdYiJ23//OYb2MI3jSNwLgjt7RETeJ9r/tSQdirpLsQBqvFAnZ
0E6yove+7u7Y/9waLd64NnHi/Hm3lCXRSHNboTXns5lndcEZOitHTtNCjv0xyBZm
2tIMPNuzjsmhDYAPexZ3FL//2wmUspO8IFgV6dtxQ/PeEMMA3KgqlbbC1j+Qa3bb
bP6MvPJwNQzcmRk13NfIRmPVNnGuV/u3gm3c
-----END CERTIFICATE-----`

const googleLeaf = `-----BEGIN CERTIFICATE-----
MIIFUjCCBDqgAwIBAgIQERmRWTzVoz0SMeozw2RM3DANBgkqhkiG9w0BAQsFADBG
MQswCQYDVQQGEwJVUzEiMCAGA1UEChMZR29vZ2xlIFRydXN0IFNlcnZpY2VzIExM
QzETMBEGA1UEAxMKR1RTIENBIDFDMzAeFw0yMzAxMDIwODE5MTlaFw0yMzAzMjcw
ODE5MThaMBkxFzAVBgNVBAMTDnd3dy5nb29nbGUuY29tMIIBIjANBgkqhkiG9w0B
AQEFAAOCAQ8AMIIBCgKCAQEAq30odrKMT54TJikMKL8S+lwoCMT5geP0u9pWjk6a
wdB6i3kO+UE4ijCAmhbcZKeKaLnGJ38weZNwB1ayabCYyX7hDiC/nRcZU49LX5+o
55kDVaNn14YKkg2kCeX25HDxSwaOsNAIXKPTqiQL5LPvc4Twhl8HY51hhNWQrTEr
N775eYbixEULvyVLq5BLbCOpPo8n0/MTjQ32ku1jQq3GIYMJC/Rf2VW5doF6t9zs
KleflAN8OdKp0ME9OHg0T1P3yyb67T7n0SpisHbeG06AmQcKJF9g/9VPJtRf4l1Q
WRPDC+6JUqzXCxAGmIRGZ7TNMxPMBW/7DRX6w8oLKVNb0wIDAQABo4ICZzCCAmMw
DgYDVR0PAQH/BAQDAgWgMBMGA1UdJQQMMAoGCCsGAQUFBwMBMAwGA1UdEwEB/wQC
MAAwHQYDVR0OBBYEFBnboj3lf9+Xat4oEgo6ZtIMr8ZuMB8GA1UdIwQYMBaAFIp0
f6+Fze6VzT2c0OJGFPNxNR0nMGoGCCsGAQUFBwEBBF4wXDAnBggrBgEFBQcwAYYb
aHR0cDovL29jc3AucGtpLmdvb2cvZ3RzMWMzMDEGCCsGAQUFBzAChiVodHRwOi8v
cGtpLmdvb2cvcmVwby9jZXJ0cy9ndHMxYzMuZGVyMBkGA1UdEQQSMBCCDnd3dy5n
b29nbGUuY29tMCEGA1UdIAQaMBgwCAYGZ4EMAQIBMAwGCisGAQQB1nkCBQMwPAYD
VR0fBDUwMzAxoC+gLYYraHR0cDovL2NybHMucGtpLmdvb2cvZ3RzMWMzL1FPdkow
TjFzVDJBLmNybDCCAQQGCisGAQQB1nkCBAIEgfUEgfIA8AB2AHoyjFTYty22IOo4
4FIe6YQWcDIThU070ivBOlejUutSAAABhXHHOiUAAAQDAEcwRQIgBUkikUIXdo+S
3T8PP0/cvokhUlumRE3GRWGL4WRMLpcCIQDY+bwK384mZxyXGZ5lwNRTAPNzT8Fx
1+//nbaGK3BQMAB2AOg+0No+9QY1MudXKLyJa8kD08vREWvs62nhd31tBr1uAAAB
hXHHOfQAAAQDAEcwRQIgLoVydNfMFKV9IoZR+M0UuJ2zOqbxIRum7Sn9RMPOBGMC
IQD1/BgzCSDTvYvco6kpB6ifKSbg5gcb5KTnYxQYwRW14TANBgkqhkiG9w0BAQsF
AAOCAQEA2bQQu30e3OFu0bmvQHmcqYvXBu6tF6e5b5b+hj4O+Rn7BXTTmaYX3M6p
MsfRH4YVJJMB/dc3PROR2VtnKFC6gAZX+RKM6nXnZhIlOdmQnonS1ecOL19PliUd
VXbwKjXqAO0Ljd9y9oXaXnyPyHmUJNI5YXAcxE+XXiOZhcZuMYyWmoEKJQ/XlSga
zWfTn1IcKhA3IC7A1n/5bkkWD1Xi1mdWFQ6DQDMp//667zz7pKOgFMlB93aPDjvI
c78zEqNswn6xGKXpWF5xVwdFcsx9HKhJ6UAi2bQ/KQ1yb7LPUOR6wXXWrG1cLnNP
i8eNLnKL9PXQ+5SwJFCzfEhcIZuhzg==
-----END CERTIFICATE-----`

// googleLeafWithInvalidHash is the same as googleLeaf, but the signature
// algorithm in the certificate contains a nonsense OID.
const googleLeafWithInvalidHash = `-----BEGIN CERTIFICATE-----
MIIFUjCCBDqgAwIBAgIQERmRWTzVoz0SMeozw2RM3DANBgkqhkiG9w0BAQ4FADBG
MQswCQYDVQQGEwJVUzEiMCAGA1UEChMZR29vZ2xlIFRydXN0IFNlcnZpY2VzIExM
QzETMBEGA1UEAxMKR1RTIENBIDFDMzAeFw0yMzAxMDIwODE5MTlaFw0yMzAzMjcw
ODE5MThaMBkxFzAVBgNVBAMTDnd3dy5nb29nbGUuY29tMIIBIjANBgkqhkiG9w0B
AQEFAAOCAQ8AMIIBCgKCAQEAq30odrKMT54TJikMKL8S+lwoCMT5geP0u9pWjk6a
wdB6i3kO+UE4ijCAmhbcZKeKaLnGJ38weZNwB1ayabCYyX7hDiC/nRcZU49LX5+o
55kDVaNn14YKkg2kCeX25HDxSwaOsNAIXKPTqiQL5LPvc4Twhl8HY51hhNWQrTEr
N775eYbixEULvyVLq5BLbCOpPo8n0/MTjQ32ku1jQq3GIYMJC/Rf2VW5doF6t9zs
KleflAN8OdKp0ME9OHg0T1P3yyb67T7n0SpisHbeG06AmQcKJF9g/9VPJtRf4l1Q
WRPDC+6JUqzXCxAGmIRGZ7TNMxPMBW/7DRX6w8oLKVNb0wIDAQABo4ICZzCCAmMw
DgYDVR0PAQH/BAQDAgWgMBMGA1UdJQQMMAoGCCsGAQUFBwMBMAwGA1UdEwEB/wQC
MAAwHQYDVR0OBBYEFBnboj3lf9+Xat4oEgo6ZtIMr8ZuMB8GA1UdIwQYMBaAFIp0
f6+Fze6VzT2c0OJGFPNxNR0nMGoGCCsGAQUFBwEBBF4wXDAnBggrBgEFBQcwAYYb
aHR0cDovL29jc3AucGtpLmdvb2cvZ3RzMWMzMDEGCCsGAQUFBzAChiVodHRwOi8v
cGtpLmdvb2cvcmVwby9jZXJ0cy9ndHMxYzMuZGVyMBkGA1UdEQQSMBCCDnd3dy5n
b29nbGUuY29tMCEGA1UdIAQaMBgwCAYGZ4EMAQIBMAwGCisGAQQB1nkCBQMwPAYD
VR0fBDUwMzAxoC+gLYYraHR0cDovL2NybHMucGtpLmdvb2cvZ3RzMWMzL1FPdkow
TjFzVDJBLmNybDCCAQQGCisGAQQB1nkCBAIEgfUEgfIA8AB2AHoyjFTYty22IOo4
4FIe6YQWcDIThU070ivBOlejUutSAAABhXHHOiUAAAQDAEcwRQIgBUkikUIXdo+S
3T8PP0/cvokhUlumRE3GRWGL4WRMLpcCIQDY+bwK384mZxyXGZ5lwNRTAPNzT8Fx
1+//nbaGK3BQMAB2AOg+0No+9QY1MudXKLyJa8kD08vREWvs62nhd31tBr1uAAAB
hXHHOfQAAAQDAEcwRQIgLoVydNfMFKV9IoZR+M0UuJ2zOqbxIRum7Sn9RMPOBGMC
IQD1/BgzCSDTvYvco6kpB6ifKSbg5gcb5KTnYxQYwRW14TANBgkqhkiG9w0BAQ4F
AAOCAQEA2bQQu30e3OFu0bmvQHmcqYvXBu6tF6e5b5b+hj4O+Rn7BXTTmaYX3M6p
MsfRH4YVJJMB/dc3PROR2VtnKFC6gAZX+RKM6nXnZhIlOdmQnonS1ecOL19PliUd
VXbwKjXqAO0Ljd9y9oXaXnyPyHmUJNI5YXAcxE+XXiOZhcZuMYyWmoEKJQ/XlSga
zWfTn1IcKhA3IC7A1n/5bkkWD1Xi1mdWFQ6DQDMp//667zz7pKOgFMlB93aPDjvI
c78zEqNswn6xGKXpWF5xVwdFcsx9HKhJ6UAi2bQ/KQ1yb7LPUOR6wXXWrG1cLnNP
i8eNLnKL9PXQ+5SwJFCzfEhcIZuhzg==
-----END CERTIFICATE-----`

const dnssecExpLeaf = `-----BEGIN CERTIFICATE-----
MIIGzTCCBbWgAwIBAgIDAdD6MA0GCSqGSIb3DQEBBQUAMIGMMQswCQYDVQQGEwJJ
TDEWMBQGA1UEChMNU3RhcnRDb20gTHRkLjErMCkGA1UECxMiU2VjdXJlIERpZ2l0
YWwgQ2VydGlmaWNhdGUgU2lnbmluZzE4MDYGA1UEAxMvU3RhcnRDb20gQ2xhc3Mg
MSBQcmltYXJ5IEludGVybWVkaWF0ZSBTZXJ2ZXIgQ0EwHhcNMTAwNzA0MTQ1MjQ1
WhcNMTEwNzA1MTA1NzA0WjCBwTEgMB4GA1UEDRMXMjIxMTM3LWxpOWE5dHhJRzZM
NnNyVFMxCzAJBgNVBAYTAlVTMR4wHAYDVQQKExVQZXJzb25hIE5vdCBWYWxpZGF0
ZWQxKTAnBgNVBAsTIFN0YXJ0Q29tIEZyZWUgQ2VydGlmaWNhdGUgTWVtYmVyMRsw
GQYDVQQDExJ3d3cuZG5zc2VjLWV4cC5vcmcxKDAmBgkqhkiG9w0BCQEWGWhvc3Rt
YXN0ZXJAZG5zc2VjLWV4cC5vcmcwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEK
AoIBAQDEdF/22vaxrPbqpgVYMWi+alfpzBctpbfLBdPGuqOazJdCT0NbWcK8/+B4
X6OlSOURNIlwLzhkmwVsWdVv6dVSaN7d4yI/fJkvgfDB9+au+iBJb6Pcz8ULBfe6
D8HVvqKdORp6INzHz71z0sghxrQ0EAEkoWAZLh+kcn2ZHdcmZaBNUfjmGbyU6PRt
RjdqoP+owIaC1aktBN7zl4uO7cRjlYFdusINrh2kPP02KAx2W84xjxX1uyj6oS6e
7eBfvcwe8czW/N1rbE0CoR7h9+HnIrjnVG9RhBiZEiw3mUmF++Up26+4KTdRKbu3
+BL4yMpfd66z0+zzqu+HkvyLpFn5AgMBAAGjggL/MIIC+zAJBgNVHRMEAjAAMAsG
A1UdDwQEAwIDqDATBgNVHSUEDDAKBggrBgEFBQcDATAdBgNVHQ4EFgQUy04I5guM
drzfh2JQaXhgV86+4jUwHwYDVR0jBBgwFoAU60I00Jiwq5/0G2sI98xkLu8OLEUw
LQYDVR0RBCYwJIISd3d3LmRuc3NlYy1leHAub3Jngg5kbnNzZWMtZXhwLm9yZzCC
AUIGA1UdIASCATkwggE1MIIBMQYLKwYBBAGBtTcBAgIwggEgMC4GCCsGAQUFBwIB
FiJodHRwOi8vd3d3LnN0YXJ0c3NsLmNvbS9wb2xpY3kucGRmMDQGCCsGAQUFBwIB
FihodHRwOi8vd3d3LnN0YXJ0c3NsLmNvbS9pbnRlcm1lZGlhdGUucGRmMIG3Bggr
BgEFBQcCAjCBqjAUFg1TdGFydENvbSBMdGQuMAMCAQEagZFMaW1pdGVkIExpYWJp
bGl0eSwgc2VlIHNlY3Rpb24gKkxlZ2FsIExpbWl0YXRpb25zKiBvZiB0aGUgU3Rh
cnRDb20gQ2VydGlmaWNhdGlvbiBBdXRob3JpdHkgUG9saWN5IGF2YWlsYWJsZSBh
dCBodHRwOi8vd3d3LnN0YXJ0c3NsLmNvbS9wb2xpY3kucGRmMGEGA1UdHwRaMFgw
KqAooCaGJGh0dHA6Ly93d3cuc3RhcnRzc2wuY29tL2NydDEtY3JsLmNybDAqoCig
JoYkaHR0cDovL2NybC5zdGFydHNzbC5jb20vY3J0MS1jcmwuY3JsMIGOBggrBgEF
BQcBAQSBgTB/MDkGCCsGAQUFBzABhi1odHRwOi8vb2NzcC5zdGFydHNzbC5jb20v
c3ViL2NsYXNzMS9zZXJ2ZXIvY2EwQgYIKwYBBQUHMAKGNmh0dHA6Ly93d3cuc3Rh
cnRzc2wuY29tL2NlcnRzL3N1Yi5jbGFzczEuc2VydmVyLmNhLmNydDAjBgNVHRIE
HDAahhhodHRwOi8vd3d3LnN0YXJ0c3NsLmNvbS8wDQYJKoZIhvcNAQEFBQADggEB
ACXj6SB59KRJPenn6gUdGEqcta97U769SATyiQ87i9er64qLwvIGLMa3o2Rcgl2Y
kghUeyLdN/EXyFBYA8L8uvZREPoc7EZukpT/ZDLXy9i2S0jkOxvF2fD/XLbcjGjM
iEYG1/6ASw0ri9C0k4oDDoJLCoeH9++yqF7SFCCMcDkJqiAGXNb4euDpa8vCCtEQ
CSS+ObZbfkreRt3cNCf5LfCXe9OsTnCfc8Cuq81c0oLaG+SmaLUQNBuToq8e9/Zm
+b+/a3RVjxmkV5OCcGVBxsXNDn54Q6wsdw0TBMcjwoEndzpLS7yWgFbbkq5ZiGpw
Qibb2+CfKuQ+WFV1GkVQmVA=
-----END CERTIFICATE-----`

const startComIntermediate = `-----BEGIN CERTIFICATE-----
MIIGNDCCBBygAwIBAgIBGDANBgkqhkiG9w0BAQUFADB9MQswCQYDVQQGEwJJTDEW
MBQGA1UEChMNU3RhcnRDb20gTHRkLjErMCkGA1UECxMiU2VjdXJlIERpZ2l0YWwg
Q2VydGlmaWNhdGUgU2lnbmluZzEpMCcGA1UEAxMgU3RhcnRDb20gQ2VydGlmaWNh
dGlvbiBBdXRob3JpdHkwHhcNMDcxMDI0MjA1NDE3WhcNMTcxMDI0MjA1NDE3WjCB
jDELMAkGA1UEBhMCSUwxFjAUBgNVBAoTDVN0YXJ0Q29tIEx0ZC4xKzApBgNVBAsT
IlNlY3VyZSBEaWdpdGFsIENlcnRpZmljYXRlIFNpZ25pbmcxODA2BgNVBAMTL1N0
YXJ0Q29tIENsYXNzIDEgUHJpbWFyeSBJbnRlcm1lZGlhdGUgU2VydmVyIENBMIIB
IjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAtonGrO8JUngHrJJj0PREGBiE
gFYfka7hh/oyULTTRwbw5gdfcA4Q9x3AzhA2NIVaD5Ksg8asWFI/ujjo/OenJOJA
pgh2wJJuniptTT9uYSAK21ne0n1jsz5G/vohURjXzTCm7QduO3CHtPn66+6CPAVv
kvek3AowHpNz/gfK11+AnSJYUq4G2ouHI2mw5CrY6oPSvfNx23BaKA+vWjhwRRI/
ME3NO68X5Q/LoKldSKqxYVDLNM08XMML6BDAjJvwAwNi/rJsPnIO7hxDKslIDlc5
xDEhyBDBLIf+VJVSH1I8MRKbf+fAoKVZ1eKPPvDVqOHXcDGpxLPPr21TLwb0pwID
AQABo4IBrTCCAakwDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8EBAMCAQYwHQYD
VR0OBBYEFOtCNNCYsKuf9BtrCPfMZC7vDixFMB8GA1UdIwQYMBaAFE4L7xqkQFul
F2mHMMo0aEPQQa7yMGYGCCsGAQUFBwEBBFowWDAnBggrBgEFBQcwAYYbaHR0cDov
L29jc3Auc3RhcnRzc2wuY29tL2NhMC0GCCsGAQUFBzAChiFodHRwOi8vd3d3LnN0
YXJ0c3NsLmNvbS9zZnNjYS5jcnQwWwYDVR0fBFQwUjAnoCWgI4YhaHR0cDovL3d3
dy5zdGFydHNzbC5jb20vc2ZzY2EuY3JsMCegJaAjhiFodHRwOi8vY3JsLnN0YXJ0
c3NsLmNvbS9zZnNjYS5jcmwwgYAGA1UdIAR5MHcwdQYLKwYBBAGBtTcBAgEwZjAu
BggrBgEFBQcCARYiaHR0cDovL3d3dy5zdGFydHNzbC5jb20vcG9saWN5LnBkZjA0
BggrBgEFBQcCARYoaHR0cDovL3d3dy5zdGFydHNzbC5jb20vaW50ZXJtZWRpYXRl
LnBkZjANBgkqhkiG9w0BAQUFAAOCAgEAIQlJPqWIbuALi0jaMU2P91ZXouHTYlfp
tVbzhUV1O+VQHwSL5qBaPucAroXQ+/8gA2TLrQLhxpFy+KNN1t7ozD+hiqLjfDen
xk+PNdb01m4Ge90h2c9W/8swIkn+iQTzheWq8ecf6HWQTd35RvdCNPdFWAwRDYSw
xtpdPvkBnufh2lWVvnQce/xNFE+sflVHfXv0pQ1JHpXo9xLBzP92piVH0PN1Nb6X
t1gW66pceG/sUzCv6gRNzKkC4/C2BBL2MLERPZBOVmTX3DxDX3M570uvh+v2/miI
RHLq0gfGabDBoYvvF0nXYbFFSF87ICHpW7LM9NfpMfULFWE7epTj69m8f5SuauNi
YpaoZHy4h/OZMn6SolK+u/hlz8nyMPyLwcKmltdfieFcNID1j0cHL7SRv7Gifl9L
WtBbnySGBVFaaQNlQ0lxxeBvlDRr9hvYqbBMflPrj0jfyjO1SPo2ShpTpjMM0InN
SRXNiTE8kMBy12VLUjWKRhFEuT2OKGWmPnmeXAhEKa2wNREuIU640ucQPl2Eg7PD
wuTSxv0JS3QJ3fGz0xk+gA2iCxnwOOfFwq/iI9th4p1cbiCJSS4jarJiwUW0n6+L
p/EiO/h94pDQehn7Skzj0n1fSoMD7SfWI55rjbRZotnvbIIp3XUZPD9MEI3vu3Un
0q6Dp6jOW6c=
-----END CERTIFICATE-----`

const startComRoot = `-----BEGIN CERTIFICATE-----
MIIHyTCCBbGgAwIBAgIBATANBgkqhkiG9w0BAQUFADB9MQswCQYDVQQGEwJJTDEW
MBQGA1UEChMNU3RhcnRDb20gTHRkLjErMCkGA1UECxMiU2VjdXJlIERpZ2l0YWwg
Q2VydGlmaWNhdGUgU2lnbmluZzEpMCcGA1UEAxMgU3RhcnRDb20gQ2VydGlmaWNh
dGlvbiBBdXRob3JpdHkwHhcNMDYwOTE3MTk0NjM2WhcNMzYwOTE3MTk0NjM2WjB9
MQswCQYDVQQGEwJJTDEWMBQGA1UEChMNU3RhcnRDb20gTHRkLjErMCkGA1UECxMi
U2VjdXJlIERpZ2l0YWwgQ2VydGlmaWNhdGUgU2lnbmluZzEpMCcGA1UEAxMgU3Rh
cnRDb20gQ2VydGlmaWNhdGlvbiBBdXRob3JpdHkwggIiMA0GCSqGSIb3DQEBAQUA
A4ICDwAwggIKAoICAQDBiNsJvGxGfHiflXu1M5DycmLWwTYgIiRezul38kMKogZk
pMyONvg45iPwbm2xPN1yo4UcodM9tDMr0y+v/uqwQVlntsQGfQqedIXWeUyAN3rf
OQVSWff0G0ZDpNKFhdLDcfN1YjS6LIp/Ho/u7TTQEceWzVI9ujPW3U3eCztKS5/C
Ji/6tRYccjV3yjxd5srhJosaNnZcAdt0FCX+7bWgiA/deMotHweXMAEtcnn6RtYT
Kqi5pquDSR3l8u/d5AGOGAqPY1MWhWKpDhk6zLVmpsJrdAfkK+F2PrRt2PZE4XNi
HzvEvqBTViVsUQn3qqvKv3b9bZvzndu/PWa8DFaqr5hIlTpL36dYUNk4dalb6kMM
Av+Z6+hsTXBbKWWc3apdzK8BMewM69KN6Oqce+Zu9ydmDBpI125C4z/eIT574Q1w
+2OqqGwaVLRcJXrJosmLFqa7LH4XXgVNWG4SHQHuEhANxjJ/GP/89PrNbpHoNkm+
Gkhpi8KWTRoSsmkXwQqQ1vp5Iki/untp+HDH+no32NgN0nZPV/+Qt+OR0t3vwmC3
Zzrd/qqc8NSLf3Iizsafl7b4r4qgEKjZ+xjGtrVcUjyJthkqcwEKDwOzEmDyei+B
26Nu/yYwl/WL3YlXtq09s68rxbd2AvCl1iuahhQqcvbjM4xdCUsT37uMdBNSSwID
AQABo4ICUjCCAk4wDAYDVR0TBAUwAwEB/zALBgNVHQ8EBAMCAa4wHQYDVR0OBBYE
FE4L7xqkQFulF2mHMMo0aEPQQa7yMGQGA1UdHwRdMFswLKAqoCiGJmh0dHA6Ly9j
ZXJ0LnN0YXJ0Y29tLm9yZy9zZnNjYS1jcmwuY3JsMCugKaAnhiVodHRwOi8vY3Js
LnN0YXJ0Y29tLm9yZy9zZnNjYS1jcmwuY3JsMIIBXQYDVR0gBIIBVDCCAVAwggFM
BgsrBgEEAYG1NwEBATCCATswLwYIKwYBBQUHAgEWI2h0dHA6Ly9jZXJ0LnN0YXJ0
Y29tLm9yZy9wb2xpY3kucGRmMDUGCCsGAQUFBwIBFilodHRwOi8vY2VydC5zdGFy
dGNvbS5vcmcvaW50ZXJtZWRpYXRlLnBkZjCB0AYIKwYBBQUHAgIwgcMwJxYgU3Rh
cnQgQ29tbWVyY2lhbCAoU3RhcnRDb20pIEx0ZC4wAwIBARqBl0xpbWl0ZWQgTGlh
YmlsaXR5LCByZWFkIHRoZSBzZWN0aW9uICpMZWdhbCBMaW1pdGF0aW9ucyogb2Yg
dGhlIFN0YXJ0Q29tIENlcnRpZmljYXRpb24gQXV0aG9yaXR5IFBvbGljeSBhdmFp
bGFibGUgYXQgaHR0cDovL2NlcnQuc3RhcnRjb20ub3JnL3BvbGljeS5wZGYwEQYJ
YIZIAYb4QgEBBAQDAgAHMDgGCWCGSAGG+EIBDQQrFilTdGFydENvbSBGcmVlIFNT
TCBDZXJ0aWZpY2F0aW9uIEF1dGhvcml0eTANBgkqhkiG9w0BAQUFAAOCAgEAFmyZ
9GYMNPXQhV59CuzaEE44HF7fpiUFS5Eyweg78T3dRAlbB0mKKctmArexmvclmAk8
jhvh3TaHK0u7aNM5Zj2gJsfyOZEdUauCe37Vzlrk4gNXcGmXCPleWKYK34wGmkUW
FjgKXlf2Ysd6AgXmvB618p70qSmD+LIU424oh0TDkBreOKk8rENNZEXO3SipXPJz
ewT4F+irsfMuXGRuczE6Eri8sxHkfY+BUZo7jYn0TZNmezwD7dOaHZrzZVD1oNB1
ny+v8OqCQ5j4aZyJecRDjkZy42Q2Eq/3JR44iZB3fsNrarnDy0RLrHiQi+fHLB5L
EUTINFInzQpdn4XBidUaePKVEFMy3YCEZnXZtWgo+2EuvoSoOMCZEoalHmdkrQYu
L6lwhceWD3yJZfWOQ1QOq92lgDmUYMA0yZZwLKMS9R9Ie70cfmu3nZD0Ijuu+Pwq
yvqCUqDvr0tVk+vBtfAii6w0TiYiBKGHLHVKt+V9E9e4DGTANtLJL4YSjCMJwRuC
O3NJo2pXh5Tl1njFmUNj403gdy3hZZlyaQQaRwnmDwFWJPsfvw55qVguucQJAX6V
um0ABj6y6koQOdjQK/W/7HW/lwLFCRsI3FU34oH7N4RDYiDK51ZLZer+bMEkkySh
NOsF/5oirpt9P/FlUQqmMGqz9IgcgA38corog14=
-----END CERTIFICATE-----`

const smimeLeaf = `-----BEGIN CERTIFICATE-----
MIIIPDCCBiSgAwIBAgIQaMDxFS0pOMxZZeOBxoTJtjANBgkqhkiG9w0BAQsFADCB
nTELMAkGA1UEBhMCRVMxFDASBgNVBAoMC0laRU5QRSBTLkEuMTowOAYDVQQLDDFB
WlogWml1cnRhZ2lyaSBwdWJsaWtvYSAtIENlcnRpZmljYWRvIHB1YmxpY28gU0NB
MTwwOgYDVQQDDDNFQUVrbyBIZXJyaSBBZG1pbmlzdHJhemlvZW4gQ0EgLSBDQSBB
QVBQIFZhc2NhcyAoMikwHhcNMTcwNzEyMDg1MzIxWhcNMjEwNzEyMDg1MzIxWjCC
AQwxDzANBgNVBAoMBklaRU5QRTE4MDYGA1UECwwvWml1cnRhZ2lyaSBrb3Jwb3Jh
dGlib2EtQ2VydGlmaWNhZG8gY29ycG9yYXRpdm8xQzBBBgNVBAsMOkNvbmRpY2lv
bmVzIGRlIHVzbyBlbiB3d3cuaXplbnBlLmNvbSBub2xhIGVyYWJpbGkgamFraXRl
a28xFzAVBgNVBC4TDi1kbmkgOTk5OTk5ODlaMSQwIgYDVQQDDBtDT1JQT1JBVElW
TyBGSUNUSUNJTyBBQ1RJVk8xFDASBgNVBCoMC0NPUlBPUkFUSVZPMREwDwYDVQQE
DAhGSUNUSUNJTzESMBAGA1UEBRMJOTk5OTk5ODlaMIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEAwVOMwUDfBtsH0XuxYnb+v/L774jMH8valX7RPH8cl2Lb
SiqSo0RchW2RGA2d1yuYHlpChC9jGmt0X/g66/E/+q2hUJlfJtqVDJFwtFYV4u2S
yzA3J36V4PRkPQrKxAsbzZriFXAF10XgiHQz9aVeMMJ9GBhmh9+DK8Tm4cMF6i8l
+AuC35KdngPF1x0ealTYrYZplpEJFO7CiW42aLi6vQkDR2R7nmZA4AT69teqBWsK
0DZ93/f0G/3+vnWwNTBF0lB6dIXoaz8OMSyHLqGnmmAtMrzbjAr/O/WWgbB/BqhR
qjJQ7Ui16cuDldXaWQ/rkMzsxmsAox0UF+zdQNvXUQIDAQABo4IDBDCCAwAwgccG
A1UdEgSBvzCBvIYVaHR0cDovL3d3dy5pemVucGUuY29tgQ9pbmZvQGl6ZW5wZS5j
b22kgZEwgY4xRzBFBgNVBAoMPklaRU5QRSBTLkEuIC0gQ0lGIEEwMTMzNzI2MC1S
TWVyYy5WaXRvcmlhLUdhc3RlaXogVDEwNTUgRjYyIFM4MUMwQQYDVQQJDDpBdmRh
IGRlbCBNZWRpdGVycmFuZW8gRXRvcmJpZGVhIDE0IC0gMDEwMTAgVml0b3JpYS1H
YXN0ZWl6MB4GA1UdEQQXMBWBE2ZpY3RpY2lvQGl6ZW5wZS5ldXMwDgYDVR0PAQH/
BAQDAgXgMCkGA1UdJQQiMCAGCCsGAQUFBwMCBggrBgEFBQcDBAYKKwYBBAGCNxQC
AjAdBgNVHQ4EFgQUyeoOD4cgcljKY0JvrNuX2waFQLAwHwYDVR0jBBgwFoAUwKlK
90clh/+8taaJzoLSRqiJ66MwggEnBgNVHSAEggEeMIIBGjCCARYGCisGAQQB8zkB
AQEwggEGMDMGCCsGAQUFBwIBFidodHRwOi8vd3d3Lml6ZW5wZS5jb20vcnBhc2Nh
Y29ycG9yYXRpdm8wgc4GCCsGAQUFBwICMIHBGoG+Wml1cnRhZ2lyaWEgRXVza2Fs
IEF1dG9ub21pYSBFcmtpZGVnb2tvIHNla3RvcmUgcHVibGlrb2tvIGVyYWt1bmRl
ZW4gYmFybmUtc2FyZWV0YW4gYmFrYXJyaWsgZXJhYmlsIGRhaXRla2UuIFVzbyBy
ZXN0cmluZ2lkbyBhbCBhbWJpdG8gZGUgcmVkZXMgaW50ZXJuYXMgZGUgRW50aWRh
ZGVzIGRlbCBTZWN0b3IgUHVibGljbyBWYXNjbzAyBggrBgEFBQcBAQQmMCQwIgYI
KwYBBQUHMAGGFmh0dHA6Ly9vY3NwLml6ZW5wZS5jb20wOgYDVR0fBDMwMTAvoC2g
K4YpaHR0cDovL2NybC5pemVucGUuY29tL2NnaS1iaW4vY3JsaW50ZXJuYTIwDQYJ
KoZIhvcNAQELBQADggIBAIy5PQ+UZlCRq6ig43vpHwlwuD9daAYeejV0Q+ZbgWAE
GtO0kT/ytw95ZEJMNiMw3fYfPRlh27ThqiT0VDXZJDlzmn7JZd6QFcdXkCsiuv4+
ZoXAg/QwnA3SGUUO9aVaXyuOIIuvOfb9MzoGp9xk23SMV3eiLAaLMLqwB5DTfBdt
BGI7L1MnGJBv8RfP/TL67aJ5bgq2ri4S8vGHtXSjcZ0+rCEOLJtmDNMnTZxancg3
/H5edeNd+n6Z48LO+JHRxQufbC4mVNxVLMIP9EkGUejlq4E4w6zb5NwCQczJbSWL
i31rk2orsNsDlyaLGsWZp3JSNX6RmodU4KAUPor4jUJuUhrrm3Spb73gKlV/gcIw
bCE7mML1Kss3x1ySaXsis6SZtLpGWKkW2iguPWPs0ydV6RPhmsCxieMwPPIJ87vS
5IejfgyBae7RSuAIHyNFy4uI5xwvwUFf6OZ7az8qtW7ImFOgng3Ds+W9k1S2CNTx
d0cnKTfA6IpjGo8EeHcxnIXT8NPImWaRj0qqonvYady7ci6U4m3lkNSdXNn1afgw
mYust+gxVtOZs1gk2MUCgJ1V1X+g7r/Cg7viIn6TLkLrpS1kS1hvMqkl9M+7XqPo
Qd95nJKOkusQpy99X4dF/lfbYAQnnjnqh3DLD2gvYObXFaAYFaiBKTiMTV2X72F+
-----END CERTIFICATE-----`

const smimeIntermediate = `-----BEGIN CERTIFICATE-----
MIIHNzCCBSGgAwIBAgIQJMXIqlZvjuhMvqcFXOFkpDALBgkqhkiG9w0BAQswODEL
MAkGA1UEBhMCRVMxFDASBgNVBAoMC0laRU5QRSBTLkEuMRMwEQYDVQQDDApJemVu
cGUuY29tMB4XDTEwMTAyMDA4MjMzM1oXDTM3MTIxMjIzMDAwMFowgZ0xCzAJBgNV
BAYTAkVTMRQwEgYDVQQKDAtJWkVOUEUgUy5BLjE6MDgGA1UECwwxQVpaIFppdXJ0
YWdpcmkgcHVibGlrb2EgLSBDZXJ0aWZpY2FkbyBwdWJsaWNvIFNDQTE8MDoGA1UE
AwwzRUFFa28gSGVycmkgQWRtaW5pc3RyYXppb2VuIENBIC0gQ0EgQUFQUCBWYXNj
YXMgKDIpMIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAoIM7nEdI0N1h
rR5T4xuV/usKDoMIasaiKvfLhbwxaNtTt+a7W/6wV5bv3svQFIy3sUXjjdzV1nG2
To2wo/YSPQiOt8exWvOapvL21ogiof+kelWnXFjWaKJI/vThHYLgIYEMj/y4HdtU
ojI646rZwqsb4YGAopwgmkDfUh5jOhV2IcYE3TgJAYWVkj6jku9PLaIsHiarAHjD
PY8dig8a4SRv0gm5Yk7FXLmW1d14oxQBDeHZ7zOEXfpafxdEDO2SNaRJjpkh8XRr
PGqkg2y1Q3gT6b4537jz+StyDIJ3omylmlJsGCwqT7p8mEqjGJ5kC5I2VnjXKuNn
soShc72khWZVUJiJo5SGuAkNE2ZXqltBVm5Jv6QweQKsX6bkcMc4IZok4a+hx8FM
8IBpGf/I94pU6HzGXqCyc1d46drJgDY9mXa+6YDAJFl3xeXOOW2iGCfwXqhiCrKL
MYvyMZzqF3QH5q4nb3ZnehYvraeMFXJXDn+Utqp8vd2r7ShfQJz01KtM4hgKdgSg
jtW+shkVVN5ng/fPN85ovfAH2BHXFfHmQn4zKsYnLitpwYM/7S1HxlT61cdQ7Nnk
3LZTYEgAoOmEmdheklT40WAYakksXGM5VrzG7x9S7s1Tm+Vb5LSThdHC8bxxwyTb
KsDRDNJ84N9fPDO6qHnzaL2upQ43PycCAwEAAaOCAdkwggHVMIHHBgNVHREEgb8w
gbyGFWh0dHA6Ly93d3cuaXplbnBlLmNvbYEPaW5mb0BpemVucGUuY29tpIGRMIGO
MUcwRQYDVQQKDD5JWkVOUEUgUy5BLiAtIENJRiBBMDEzMzcyNjAtUk1lcmMuVml0
b3JpYS1HYXN0ZWl6IFQxMDU1IEY2MiBTODFDMEEGA1UECQw6QXZkYSBkZWwgTWVk
aXRlcnJhbmVvIEV0b3JiaWRlYSAxNCAtIDAxMDEwIFZpdG9yaWEtR2FzdGVpejAP
BgNVHRMBAf8EBTADAQH/MA4GA1UdDwEB/wQEAwIBBjAdBgNVHQ4EFgQUwKlK90cl
h/+8taaJzoLSRqiJ66MwHwYDVR0jBBgwFoAUHRxlDqjyJXu0kc/ksbHmvVV0bAUw
OgYDVR0gBDMwMTAvBgRVHSAAMCcwJQYIKwYBBQUHAgEWGWh0dHA6Ly93d3cuaXpl
bnBlLmNvbS9jcHMwNwYIKwYBBQUHAQEEKzApMCcGCCsGAQUFBzABhhtodHRwOi8v
b2NzcC5pemVucGUuY29tOjgwOTQwMwYDVR0fBCwwKjAooCagJIYiaHR0cDovL2Ny
bC5pemVucGUuY29tL2NnaS1iaW4vYXJsMjALBgkqhkiG9w0BAQsDggIBAMbjc3HM
3DG9ubWPkzsF0QsktukpujbTTcGk4h20G7SPRy1DiiTxrRzdAMWGjZioOP3/fKCS
M539qH0M+gsySNie+iKlbSZJUyE635T1tKw+G7bDUapjlH1xyv55NC5I6wCXGC6E
3TEP5B/E7dZD0s9E4lS511ubVZivFgOzMYo1DO96diny/N/V1enaTCpRl1qH1OyL
xUYTijV4ph2gL6exwuG7pxfRcVNHYlrRaXWfTz3F6NBKyULxrI3P/y6JAtN1GqT4
VF/+vMygx22n0DufGepBwTQz6/rr1ulSZ+eMnuJiTXgh/BzQnkUsXTb8mHII25iR
0oYF2qAsk6ecWbLiDpkHKIDHmML21MZE13MS8NSvTHoqJO4LyAmDe6SaeNHtrPlK
b6mzE1BN2ug+ZaX8wLA5IMPFaf0jKhb/Cxu8INsxjt00brsErCc9ip1VNaH0M4bi
1tGxfiew2436FaeyUxW7Pl6G5GgkNbuUc7QIoRy06DdU/U38BxW3uyJMY60zwHvS
FlKAn0OvYp4niKhAJwaBVN3kowmJuOU5Rid+TUnfyxbJ9cttSgzaF3hP/N4zgMEM
5tikXUskeckt8LUK96EH0QyssavAMECUEb/xrupyRdYWwjQGvNLq6T5+fViDGyOw
k+lzD44wofy8paAy9uC9Owae0zMEzhcsyRm7
-----END CERTIFICATE-----`

const smimeRoot = `-----BEGIN CERTIFICATE-----
MIIF8TCCA9mgAwIBAgIQALC3WhZIX7/hy/WL1xnmfTANBgkqhkiG9w0BAQsFADA4
MQswCQYDVQQGEwJFUzEUMBIGA1UECgwLSVpFTlBFIFMuQS4xEzARBgNVBAMMCkl6
ZW5wZS5jb20wHhcNMDcxMjEzMTMwODI4WhcNMzcxMjEzMDgyNzI1WjA4MQswCQYD
VQQGEwJFUzEUMBIGA1UECgwLSVpFTlBFIFMuQS4xEzARBgNVBAMMCkl6ZW5wZS5j
b20wggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIKAoICAQDJ03rKDx6sp4boFmVq
scIbRTJxldn+EFvMr+eleQGPicPK8lVx93e+d5TzcqQsRNiekpsUOqHnJJAKClaO
xdgmlOHZSOEtPtoKct2jmRXagaKH9HtuJneJWK3W6wyyQXpzbm3benhB6QiIEn6H
LmYRY2xU+zydcsC8Lv/Ct90NduM61/e0aL6i9eOBbsFGb12N4E3GVFWJGjMxCrFX
uaOKmMPsOzTFlUFpfnXCPCDFYbpRR6AgkJOhkEvzTnyFRVSa0QUmQbC1TR0zvsQD
yCV8wXDbO/QJLVQnSKwv4cSsPsjLkkxTOTcj7NMB+eAJRE1NZMDhDVqHIrytG6P+
JrUV86f8hBnp7KGItERphIPzidF0BqnMC9bC3ieFUCbKF7jJeodWLBoBHmy+E60Q
rLUk9TiRodZL2vG70t5HtfG8gfZZa88ZU+mNFctKy6lvROUbQc/hhqfK0GqfvEyN
BjNaooXlkDWgYlwWTvDjovoDGrQscbNYLN57C9saD+veIR8GdwYDsMnvmfzAuU8L
hij+0rnq49qlw0dpEuDb8PYZi+17cNcC1u2HGCgsBCRMd+RIihrGO5rUD8r6ddIB
QFqNeb+Lz0vPqhbBleStTIo+F5HUsWLlguWABKQDfo2/2n+iD5dPDNMN+9fR5XJ+
HMh3/1uaD7euBUbl8agW7EekFwIDAQABo4H2MIHzMIGwBgNVHREEgagwgaWBD2lu
Zm9AaXplbnBlLmNvbaSBkTCBjjFHMEUGA1UECgw+SVpFTlBFIFMuQS4gLSBDSUYg
QTAxMzM3MjYwLVJNZXJjLlZpdG9yaWEtR2FzdGVpeiBUMTA1NSBGNjIgUzgxQzBB
BgNVBAkMOkF2ZGEgZGVsIE1lZGl0ZXJyYW5lbyBFdG9yYmlkZWEgMTQgLSAwMTAx
MCBWaXRvcmlhLUdhc3RlaXowDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8EBAMC
AQYwHQYDVR0OBBYEFB0cZQ6o8iV7tJHP5LGx5r1VdGwFMA0GCSqGSIb3DQEBCwUA
A4ICAQB4pgwWSp9MiDrAyw6lFn2fuUhfGI8NYjb2zRlrrKvV9pF9rnHzP7MOeIWb
laQnIUdCSnxIOvVFfLMMjlF4rJUT3sb9fbgakEyrkgPH7UIBzg/YsfqikuFgba56
awmqxinuaElnMIAkejEWOVt+8Rwu3WwJrfIxwYJOubv5vr8qhT/AQKM6WfxZSzwo
JNu0FXWuDYi6LnPAvViH5ULy617uHjAimcs30cQhbIHsvm0m5hzkQiCeR7Csg1lw
LDXWrzY0tM07+DKo7+N4ifuNRSzanLh+QBxh5z6ikixL8s36mLYp//Pye6kfLqCT
VyvehQP5aTfLnnhqBbTFMXiJ7HqnheG5ezzevh55hM6fcA5ZwjUukCox2eRFekGk
LhObNA5me0mrZJfQRsN5nXJQY6aYWwa9SG3YOYNw6DXwBdGqvOPbyALqfP2C2sJb
UjWumDqtujWTI6cfSN01RpiyEGjkpTHCClguGYEQyVB1/OpaFs4R1+7vUIgtYf8/
QnMFlEPVjjxOAToZpR9GTnfQXeWBIiGH/pR9hNiTrdZoQ0iy2+tzJOeRf1SktoA+
naM8THLCV8Sg1Mw4J87VBp6iSNnpn86CcDaTmjvfliHjWbcM2pE38P1ZWrOZyGls
QyYBNWNgVYkDOnXYukrZVP/u3oDYLdE41V4tC5h9Pmzb/CaIxw==
-----END CERTIFICATE-----`

var nameConstraintsLeaf = `-----BEGIN CERTIFICATE-----
MIIHMTCCBRmgAwIBAgIIIZaV/3ezOJkwDQYJKoZIhvcNAQEFBQAwgcsxCzAJBgNV
BAYTAlVTMREwDwYDVQQIEwhWaXJnaW5pYTETMBEGA1UEBxMKQmxhY2tzYnVyZzEj
MCEGA1UECxMaR2xvYmFsIFF1YWxpZmllZCBTZXJ2ZXIgQ0ExPDA6BgNVBAoTM1Zp
cmdpbmlhIFBvbHl0ZWNobmljIEluc3RpdHV0ZSBhbmQgU3RhdGUgVW5pdmVyc2l0
eTExMC8GA1UEAxMoVmlyZ2luaWEgVGVjaCBHbG9iYWwgUXVhbGlmaWVkIFNlcnZl
ciBDQTAeFw0xMzA5MTkxNDM2NTVaFw0xNTA5MTkxNDM2NTVaMIHNMQswCQYDVQQG
EwJVUzERMA8GA1UECAwIVmlyZ2luaWExEzARBgNVBAcMCkJsYWNrc2J1cmcxPDA6
BgNVBAoMM1ZpcmdpbmlhIFBvbHl0ZWNobmljIEluc3RpdHV0ZSBhbmQgU3RhdGUg
VW5pdmVyc2l0eTE7MDkGA1UECwwyVGVjaG5vbG9neS1lbmhhbmNlZCBMZWFybmlu
ZyBhbmQgT25saW5lIFN0cmF0ZWdpZXMxGzAZBgNVBAMMEnNlY3VyZS5pZGRsLnZ0
LmVkdTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAKkOyPpsOK/6IuPG
WnIBlVwlHzeYf+cUlggqkLq0b0+vZbiTXgio9/VCuNQ8opSoss7J7o3ygV9to+9Y
YwJKVC5WDT/y5JWpQey0CWILymViJnpNSwnxBc8A+Q8w5NUGDd/UhtPx/U8/hqbd
WPDYj2hbOqyq8UlRhfS5pwtnv6BbCTaY11I6FhCLK7zttISyTuWCf9p9o/ggiipP
ii/5oh4dkl+r5SfuSp5GPNHlYO8lWqys5NAPoDD4fc/kuflcK7Exx7XJ+Oqu0W0/
psjEY/tES1ZgDWU/ParcxxFpFmKHbD5DXsfPOObzkVWXIY6tGMutSlE1Froy/Nn0
OZsAOrcCAwEAAaOCAhMwggIPMIG4BggrBgEFBQcBAQSBqzCBqDBYBggrBgEFBQcw
AoZMaHR0cDovL3d3dy5wa2kudnQuZWR1L2dsb2JhbHF1YWxpZmllZHNlcnZlci9j
YWNlcnQvZ2xvYmFscXVhbGlmaWVkc2VydmVyLmNydDBMBggrBgEFBQcwAYZAaHR0
cDovL3Z0Y2EtcC5lcHJvdi5zZXRpLnZ0LmVkdTo4MDgwL2VqYmNhL3B1YmxpY3dl
Yi9zdGF0dXMvb2NzcDAdBgNVHQ4EFgQUp7xbO6iHkvtZbPE4jmndmnAbSEcwDAYD
VR0TAQH/BAIwADAfBgNVHSMEGDAWgBS8YmAn1eM1SBfpS6tFatDIqHdxjDBqBgNV
HSAEYzBhMA4GDCsGAQQBtGgFAgICATAOBgwrBgEEAbRoBQICAQEwPwYMKwYBBAG0
aAUCAgMBMC8wLQYIKwYBBQUHAgEWIWh0dHA6Ly93d3cucGtpLnZ0LmVkdS9nbG9i
YWwvY3BzLzBKBgNVHR8EQzBBMD+gPaA7hjlodHRwOi8vd3d3LnBraS52dC5lZHUv
Z2xvYmFscXVhbGlmaWVkc2VydmVyL2NybC9jYWNybC5jcmwwDgYDVR0PAQH/BAQD
AgTwMB0GA1UdJQQWMBQGCCsGAQUFBwMBBggrBgEFBQcDAjAdBgNVHREEFjAUghJz
ZWN1cmUuaWRkbC52dC5lZHUwDQYJKoZIhvcNAQEFBQADggIBAEgoYo4aUtatY3gI
OyyKp7QlIOaLbTJZywESHqy+L5EGDdJW2DJV+mcE0LDGvqa2/1Lo+AR1ntsZwfOi
Y718JwgVVaX/RCd5+QKP25c5/x72xI8hb/L1bgS0ED9b0YAhd7Qm1K1ot82+6mqX
DW6WiGeDr8Z07MQ3143qQe2rBlq+QI69DYzm2GOqAIAnUIWv7tCyLUm31b4DwmrJ
TeudVreTKUbBNB1TWRFHEPkWhjjXKZnNGRO11wHXcyBu6YekIvVZ+vmx8ePee4jJ
3GFOi7lMuWOeq57jTVL7KOKaKLVXBb6gqo5aq+Wwt8RUD5MakrCAEeQZj7DKaFmZ
oQCO0Pxrsl3InCGvxnGzT+bFVO9nJ/BAMj7hknFdm9Jr6Bg5q33Z+gnf909AD9QF
ESqUSykaHu2LVdJx2MaCH1CyKnRgMw5tEwE15EXpUjCm24m8FMOYC+rNtf18pgrz
5D8Jhh+oxK9PjcBYqXNtnioIxiMCYcV0q5d4w4BYFEh71tk7/bYB0R55CsBUVPmp
timWNOdRd57Tfpk3USaVsumWZAf9MP3wPiC7gb4d5tYEEAG5BuDT8ruFw838wU8G
1VvAVutSiYBg7k3NYO7AUqZ+Ax4klQX3aM9lgonmJ78Qt94UPtbptrfZ4/lSqEf8
GBUwDrQNTb+gsXsDkjd5lcYxNx6l
-----END CERTIFICATE-----`

var nameConstraintsIntermediate1 = `-----BEGIN CERTIFICATE-----
MIINLjCCDBagAwIBAgIRIqpyf/YoGgvHc8HiDAxAI8owDQYJKoZIhvcNAQEFBQAw
XDELMAkGA1UEBhMCQkUxFTATBgNVBAsTDFRydXN0ZWQgUm9vdDEZMBcGA1UEChMQ
R2xvYmFsU2lnbiBudi1zYTEbMBkGA1UEAxMSVHJ1c3RlZCBSb290IENBIEcyMB4X
DTEyMTIxMzAwMDAwMFoXDTE3MTIxMzAwMDAwMFowgcsxCzAJBgNVBAYTAlVTMREw
DwYDVQQIEwhWaXJnaW5pYTETMBEGA1UEBxMKQmxhY2tzYnVyZzEjMCEGA1UECxMa
R2xvYmFsIFF1YWxpZmllZCBTZXJ2ZXIgQ0ExPDA6BgNVBAoTM1ZpcmdpbmlhIFBv
bHl0ZWNobmljIEluc3RpdHV0ZSBhbmQgU3RhdGUgVW5pdmVyc2l0eTExMC8GA1UE
AxMoVmlyZ2luaWEgVGVjaCBHbG9iYWwgUXVhbGlmaWVkIFNlcnZlciBDQTCCAiIw
DQYJKoZIhvcNAQEBBQADggIPADCCAgoCggIBALgIZhEaptBWADBqdJ45ueFGzMXa
GHnzNxoxR1fQIaaRQNdCg4cw3A4dWKMeEgYLtsp65ai3Xfw62Qaus0+KJ3RhgV+r
ihqK81NUzkls78fJlADVDI4fCTlothsrE1CTOMiy97jKHai5mVTiWxmcxpmjv7fm
5Nhc+uHgh2hIz6npryq495mD51ZrUTIaqAQN6Pw/VHfAmR524vgriTOjtp1t4lA9
pXGWjF/vkhAKFFheOQSQ00rngo2wHgCqMla64UTN0oz70AsCYNZ3jDLx0kOP0YmM
R3Ih91VA63kLqPXA0R6yxmmhhxLZ5bcyAy1SLjr1N302MIxLM/pSy6aquEnbELhz
qyp9yGgRyGJay96QH7c4RJY6gtcoPDbldDcHI9nXngdAL4DrZkJ9OkDkJLyqG66W
ZTF5q4EIs6yMdrywz0x7QP+OXPJrjYpbeFs6tGZCFnWPFfmHCRJF8/unofYrheq+
9J7Jx3U55S/k57NXbAM1RAJOuMTlfn9Etf9Dpoac9poI4Liav6rBoUQk3N3JWqnV
HNx/NdCyJ1/6UbKMJUZsStAVglsi6lVPo289HHOE4f7iwl3SyekizVOp01wUin3y
cnbZB/rXmZbwapSxTTSBf0EIOr9i4EGfnnhCAVA9U5uLrI5OEB69IY8PNX0071s3
Z2a2fio5c8m3JkdrAgMBAAGjggh5MIIIdTAOBgNVHQ8BAf8EBAMCAQYwTAYDVR0g
BEUwQzBBBgkrBgEEAaAyATwwNDAyBggrBgEFBQcCARYmaHR0cHM6Ly93d3cuZ2xv
YmFsc2lnbi5jb20vcmVwb3NpdG9yeS8wEgYDVR0TAQH/BAgwBgEB/wIBADCCBtAG
A1UdHgSCBscwggbDoIIGvzASghAzZGJsYWNrc2J1cmcub3JnMBiCFmFjY2VsZXJh
dGV2aXJnaW5pYS5jb20wGIIWYWNjZWxlcmF0ZXZpcmdpbmlhLm9yZzALgglhY3Zj
cC5vcmcwCYIHYmV2Lm5ldDAJggdiZXYub3JnMAuCCWNsaWdzLm9yZzAMggpjbWl3
ZWIub3JnMBeCFWVhc3Rlcm5icm9va3Ryb3V0Lm5ldDAXghVlYXN0ZXJuYnJvb2t0
cm91dC5vcmcwEYIPZWNvcnJpZG9ycy5pbmZvMBOCEWVkZ2FycmVzZWFyY2gub3Jn
MBKCEGdldC1lZHVjYXRlZC5jb20wE4IRZ2V0LWVkdWNhdGVkLmluZm8wEYIPZ2V0
ZWR1Y2F0ZWQubmV0MBKCEGdldC1lZHVjYXRlZC5uZXQwEYIPZ2V0ZWR1Y2F0ZWQu
b3JnMBKCEGdldC1lZHVjYXRlZC5vcmcwD4INaG9raWVjbHViLmNvbTAQgg5ob2tp
ZXBob3RvLmNvbTAPgg1ob2tpZXNob3AuY29tMBGCD2hva2llc3BvcnRzLmNvbTAS
ghBob2tpZXRpY2tldHMuY29tMBKCEGhvdGVscm9hbm9rZS5jb20wE4IRaHVtYW53
aWxkbGlmZS5vcmcwF4IVaW5uYXR2aXJnaW5pYXRlY2guY29tMA+CDWlzY2hwMjAx
MS5vcmcwD4INbGFuZHJlaGFiLm9yZzAggh5uYXRpb25hbHRpcmVyZXNlYXJjaGNl
bnRlci5jb20wFYITbmV0d29ya3ZpcmdpbmlhLm5ldDAMggpwZHJjdnQuY29tMBiC
FnBldGVkeWVyaXZlcmNvdXJzZS5jb20wDYILcmFkaW9pcS5vcmcwFYITcml2ZXJj
b3Vyc2Vnb2xmLmNvbTALgglzZGltaS5vcmcwEIIOc292YW1vdGlvbi5jb20wHoIc
c3VzdGFpbmFibGUtYmlvbWF0ZXJpYWxzLmNvbTAeghxzdXN0YWluYWJsZS1iaW9t
YXRlcmlhbHMub3JnMBWCE3RoaXNpc3RoZWZ1dHVyZS5jb20wGIIWdGhpcy1pcy10
aGUtZnV0dXJlLmNvbTAVghN0aGlzaXN0aGVmdXR1cmUubmV0MBiCFnRoaXMtaXMt
dGhlLWZ1dHVyZS5uZXQwCoIIdmFkcy5vcmcwDIIKdmFsZWFmLm9yZzANggt2YXRl
Y2guaW5mbzANggt2YXRlY2gubW9iaTAcghp2YXRlY2hsaWZlbG9uZ2xlYXJuaW5n
LmNvbTAcghp2YXRlY2hsaWZlbG9uZ2xlYXJuaW5nLm5ldDAcghp2YXRlY2hsaWZl
bG9uZ2xlYXJuaW5nLm9yZzAKggh2Y29tLmVkdTASghB2aXJnaW5pYXZpZXcubmV0
MDSCMnZpcmdpbmlhcG9seXRlY2huaWNpbnN0aXR1dGVhbmRzdGF0ZXVuaXZlcnNp
dHkuY29tMDWCM3ZpcmdpbmlhcG9seXRlY2huaWNpbnN0aXR1dGVhbmRzdGF0ZXVu
aXZlcnNpdHkuaW5mbzA0gjJ2aXJnaW5pYXBvbHl0ZWNobmljaW5zdGl0dXRlYW5k
c3RhdGV1bml2ZXJzaXR5Lm5ldDA0gjJ2aXJnaW5pYXBvbHl0ZWNobmljaW5zdGl0
dXRlYW5kc3RhdGV1bml2ZXJzaXR5Lm9yZzAZghd2aXJnaW5pYXB1YmxpY3JhZGlv
Lm9yZzASghB2aXJnaW5pYXRlY2guZWR1MBOCEXZpcmdpbmlhdGVjaC5tb2JpMByC
GnZpcmdpbmlhdGVjaGZvdW5kYXRpb24ub3JnMAiCBnZ0LmVkdTALggl2dGFyYy5v
cmcwDIIKdnQtYXJjLm9yZzALggl2dGNyYy5jb20wCoIIdnRpcC5vcmcwDIIKdnRs
ZWFuLm9yZzAWghR2dGtub3dsZWRnZXdvcmtzLmNvbTAYghZ2dGxpZmVsb25nbGVh
cm5pbmcuY29tMBiCFnZ0bGlmZWxvbmdsZWFybmluZy5uZXQwGIIWdnRsaWZlbG9u
Z2xlYXJuaW5nLm9yZzATghF2dHNwb3J0c21lZGlhLmNvbTALggl2dHdlaS5jb20w
D4INd2l3YXR3ZXJjLmNvbTAKggh3dnRmLm9yZzAIgQZ2dC5lZHUwd6R1MHMxCzAJ
BgNVBAYTAlVTMREwDwYDVQQIEwhWaXJnaW5pYTETMBEGA1UEBxMKQmxhY2tzYnVy
ZzE8MDoGA1UEChMzVmlyZ2luaWEgUG9seXRlY2huaWMgSW5zdGl0dXRlIGFuZCBT
dGF0ZSBVbml2ZXJzaXR5MCcGA1UdJQQgMB4GCCsGAQUFBwMCBggrBgEFBQcDAQYI
KwYBBQUHAwkwPQYDVR0fBDYwNDAyoDCgLoYsaHR0cDovL2NybC5nbG9iYWxzaWdu
LmNvbS9ncy90cnVzdHJvb3RnMi5jcmwwgYQGCCsGAQUFBwEBBHgwdjAzBggrBgEF
BQcwAYYnaHR0cDovL29jc3AyLmdsb2JhbHNpZ24uY29tL3RydXN0cm9vdGcyMD8G
CCsGAQUFBzAChjNodHRwOi8vc2VjdXJlLmdsb2JhbHNpZ24uY29tL2NhY2VydC90
cnVzdHJvb3RnMi5jcnQwHQYDVR0OBBYEFLxiYCfV4zVIF+lLq0Vq0Miod3GMMB8G
A1UdIwQYMBaAFBT25YsxtkWASkxt/MKHico2w5BiMA0GCSqGSIb3DQEBBQUAA4IB
AQAyJm/lOB2Er4tHXhc/+fSufSzgjohJgYfMkvG4LknkvnZ1BjliefR8tTXX49d2
SCDFWfGjqyJZwavavkl/4p3oXPG/nAMDMvxh4YAT+CfEK9HH+6ICV087kD4BLegi
+aFJMj8MMdReWCzn5sLnSR1rdse2mo2arX3Uod14SW+PGrbUmTuWNyvRbz3fVmxp
UdbGmj3laknO9YPsBGgHfv73pVVsTJkW4ZfY/7KdD/yaVv6ophpOB3coXfjl2+kd
Z4ypn2zK+cx9IL/LSewqd/7W9cD55PCUy4X9OTbEmAccwiz3LB66mQoUGfdHdkoB
jUY+v9vLQXmaVwI0AYL7g9LN
-----END CERTIFICATE-----`

var nameConstraintsIntermediate2 = `-----BEGIN CERTIFICATE-----
MIIEXTCCA0WgAwIBAgILBAAAAAABNuk6OrMwDQYJKoZIhvcNAQEFBQAwVzELMAkG
A1UEBhMCQkUxGTAXBgNVBAoTEEdsb2JhbFNpZ24gbnYtc2ExEDAOBgNVBAsTB1Jv
b3QgQ0ExGzAZBgNVBAMTEkdsb2JhbFNpZ24gUm9vdCBDQTAeFw0xMjA0MjUxMTAw
MDBaFw0yNzA0MjUxMTAwMDBaMFwxCzAJBgNVBAYTAkJFMRUwEwYDVQQLEwxUcnVz
dGVkIFJvb3QxGTAXBgNVBAoTEEdsb2JhbFNpZ24gbnYtc2ExGzAZBgNVBAMTElRy
dXN0ZWQgUm9vdCBDQSBHMjCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEB
AKyuvqrtcMr7g7EuNbu4sKwxM127UsCmx1RxbxxgcArGS7rjiefpBH/w4LYrymjf
vcw1ueyMNoqLo9nJMz/ORXupb35NNfE667prQYHa+tTjl1IiKpB7QUwt3wXPuTMF
Ja1tXtjKzkqJyuJlNuPKT76HcjgNqgV1s9qG44MD5I2JvI12du8zI1bgdQ+l/KsX
kTfbGjUvhOLOlVNWVQDpL+YMIrGqgBYxy5TUNgrAcRtwpNdS2KkF5otSmMweVb5k
hoUVv3u8UxQH/WWbNhHq1RrIlg/0rBUfi/ziShYFSB7U+aLx5DxPphTFBiDquQGp
tB+FC4JvnukDStFihZCZ1R8CAwEAAaOCASMwggEfMA4GA1UdDwEB/wQEAwIBBjAP
BgNVHRMBAf8EBTADAQH/MEcGA1UdIARAMD4wPAYEVR0gADA0MDIGCCsGAQUFBwIB
FiZodHRwczovL3d3dy5nbG9iYWxzaWduLmNvbS9yZXBvc2l0b3J5LzAdBgNVHQ4E
FgQUFPblizG2RYBKTG38woeJyjbDkGIwMwYDVR0fBCwwKjAooCagJIYiaHR0cDov
L2NybC5nbG9iYWxzaWduLm5ldC9yb290LmNybDA+BggrBgEFBQcBAQQyMDAwLgYI
KwYBBQUHMAGGImh0dHA6Ly9vY3NwMi5nbG9iYWxzaWduLmNvbS9yb290cjEwHwYD
VR0jBBgwFoAUYHtmGkUNl8qJUC99BM00qP/8/UswDQYJKoZIhvcNAQEFBQADggEB
AL7IG0l+k4LkcpI+a/kvZsSRwSM4uA6zGX34e78A2oytr8RG8bJwVb8+AHMUD+Xe
2kYdh/Uj/waQXfqR0OgxQXL9Ct4ZM+JlR1avsNKXWL5AwYXAXCOB3J5PW2XOck7H
Zw0vRbGQhjWjQx+B4KOUFg1b3ov/z6Xkr3yaCfRQhXh7KC0Bc0RXPPG5Nv5lCW+z
tbbg0zMm3kyfQITRusMSg6IBsDJqOnjaiaKQRcXiD0Sk43ZXb2bUKMxC7+Td3QL4
RyHcWJbQ7YylLTS/x+jxWIcOQ0oO5/54t5PTQ14neYhOz9x4gUk2AYAW6d1vePwb
hcC8roQwkHT7HvfYBoc74FM=
-----END CERTIFICATE-----`

var globalSignRoot = `-----BEGIN CERTIFICATE-----
MIIDdTCCAl2gAwIBAgILBAAAAAABFUtaw5QwDQYJKoZIhvcNAQEFBQAwVzELMAkG
A1UEBhMCQkUxGTAXBgNVBAoTEEdsb2JhbFNpZ24gbnYtc2ExEDAOBgNVBAsTB1Jv
b3QgQ0ExGzAZBgNVBAMTEkdsb2JhbFNpZ24gUm9vdCBDQTAeFw05ODA5MDExMjAw
MDBaFw0yODAxMjgxMjAwMDBaMFcxCzAJBgNVBAYTAkJFMRkwFwYDVQQKExBHbG9i
YWxTaWduIG52LXNhMRAwDgYDVQQLEwdSb290IENBMRswGQYDVQQDExJHbG9iYWxT
aWduIFJvb3QgQ0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDaDuaZ
jc6j40+Kfvvxi4Mla+pIH/EqsLmVEQS98GPR4mdmzxzdzxtIK+6NiY6arymAZavp
xy0Sy6scTHAHoT0KMM0VjU/43dSMUBUc71DuxC73/OlS8pF94G3VNTCOXkNz8kHp
1Wrjsok6Vjk4bwY8iGlbKk3Fp1S4bInMm/k8yuX9ifUSPJJ4ltbcdG6TRGHRjcdG
snUOhugZitVtbNV4FpWi6cgKOOvyJBNPc1STE4U6G7weNLWLBYy5d4ux2x8gkasJ
U26Qzns3dLlwR5EiUWMWea6xrkEmCMgZK9FGqkjWZCrXgzT/LCrBbBlDSgeF59N8
9iFo7+ryUp9/k5DPAgMBAAGjQjBAMA4GA1UdDwEB/wQEAwIBBjAPBgNVHRMBAf8E
BTADAQH/MB0GA1UdDgQWBBRge2YaRQ2XyolQL30EzTSo//z9SzANBgkqhkiG9w0B
AQUFAAOCAQEA1nPnfE920I2/7LqivjTFKDK1fPxsnCwrvQmeU79rXqoRSLblCKOz
yj1hTdNGCbM+w6DjY1Ub8rrvrTnhQ7k4o+YviiY776BQVvnGCv04zcQLcFGUl5gE
38NflNUVyRRBnMRddWQVDf9VMOyGj/8N7yy5Y0b2qvzfvGn9LhJIZJrglfCm7ymP
AbEVtQwdpf5pLGkkeB6zpxxxYu7KyJesF12KwvhHhm4qxFYxldBniYUr+WymXUad
DKqC5JlR3XC321Y9YeRq4VzW9v493kHMB65jUr9TU/Qr6cf9tveCX4XSQRjbgbME
HMUfpIBvFSDJ3gyICh3WZlXi/EjJKSZp4A==
-----END CERTIFICATE-----`

const digicertRoot = `-----BEGIN CERTIFICATE-----
MIIDrzCCApegAwIBAgIQCDvgVpBCRrGhdWrJWZHHSjANBgkqhkiG9w0BAQUFADBh
MQswCQYDVQQGEwJVUzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMRkwFwYDVQQLExB3
d3cuZGlnaWNlcnQuY29tMSAwHgYDVQQDExdEaWdpQ2VydCBHbG9iYWwgUm9vdCBD
QTAeFw0wNjExMTAwMDAwMDBaFw0zMTExMTAwMDAwMDBaMGExCzAJBgNVBAYTAlVT
MRUwEwYDVQQKEwxEaWdpQ2VydCBJbmMxGTAXBgNVBAsTEHd3dy5kaWdpY2VydC5j
b20xIDAeBgNVBAMTF0RpZ2lDZXJ0IEdsb2JhbCBSb290IENBMIIBIjANBgkqhkiG
9w0BAQEFAAOCAQ8AMIIBCgKCAQEA4jvhEXLeqKTTo1eqUKKPC3eQyaKl7hLOllsB
CSDMAZOnTjC3U/dDxGkAV53ijSLdhwZAAIEJzs4bg7/fzTtxRuLWZscFs3YnFo97
nh6Vfe63SKMI2tavegw5BmV/Sl0fvBf4q77uKNd0f3p4mVmFaG5cIzJLv07A6Fpt
43C/dxC//AH2hdmoRBBYMql1GNXRor5H4idq9Joz+EkIYIvUX7Q6hL+hqkpMfT7P
T19sdl6gSzeRntwi5m3OFBqOasv+zbMUZBfHWymeMr/y7vrTC0LUq7dBMtoM1O/4
gdW7jVg/tRvoSSiicNoxBN33shbyTApOB6jtSj1etX+jkMOvJwIDAQABo2MwYTAO
BgNVHQ8BAf8EBAMCAYYwDwYDVR0TAQH/BAUwAwEB/zAdBgNVHQ4EFgQUA95QNVbR
TLtm8KPiGxvDl7I90VUwHwYDVR0jBBgwFoAUA95QNVbRTLtm8KPiGxvDl7I90VUw
DQYJKoZIhvcNAQEFBQADggEBAMucN6pIExIK+t1EnE9SsPTfrgT1eXkIoyQY/Esr
hMAtudXH/vTBH1jLuG2cenTnmCmrEbXjcKChzUyImZOMkXDiqw8cvpOp/2PV5Adg
06O/nVsJ8dWO41P0jmP6P6fbtGbfYmbW0W5BjfIttep3Sp+dWOIrWcBAI+0tKIJF
PnlUkiaY4IBIqDfv8NZ5YBberOgOzW6sRBc4L0na4UU+Krk2U886UAb3LujEV0ls
YSEY1QSteDwsOoBrp+uvFRTp2InBuThs4pFsiv9kuXclVzDAGySj4dzp30d8tbQk
CAUw7C29C79Fv1C5qfPrmAESrciIxpg0X40KPMbp1ZWVbd4=
-----END CERTIFICATE-----`

const trustAsiaSHA384Intermediate = `-----BEGIN CERTIFICATE-----
MIID9zCCAt+gAwIBAgIQC965p4OR4AKrGlsyW0XrDzANBgkqhkiG9w0BAQwFADBh
MQswCQYDVQQGEwJVUzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMRkwFwYDVQQLExB3
d3cuZGlnaWNlcnQuY29tMSAwHgYDVQQDExdEaWdpQ2VydCBHbG9iYWwgUm9vdCBD
QTAeFw0xODA0MjcxMjQyNTlaFw0yODA0MjcxMjQyNTlaMFoxCzAJBgNVBAYTAkNO
MSUwIwYDVQQKExxUcnVzdEFzaWEgVGVjaG5vbG9naWVzLCBJbmMuMSQwIgYDVQQD
ExtUcnVzdEFzaWEgRUNDIE9WIFRMUyBQcm8gQ0EwdjAQBgcqhkjOPQIBBgUrgQQA
IgNiAAQPIUn75M5BCQLKoPsSU2KTr3mDMh13usnAQ38XfKOzjXiyQ+W0inA7meYR
xS+XMQgvnbCigEsKj3ErPIzO68uC9V/KdqMaXWBJp85Ws9A4KL92NB4Okbn5dp6v
Qzy08PajggFeMIIBWjAdBgNVHQ4EFgQULdRyBx6HyIH/+LOvuexyH5p/3PwwHwYD
VR0jBBgwFoAUA95QNVbRTLtm8KPiGxvDl7I90VUwDgYDVR0PAQH/BAQDAgGGMB0G
A1UdJQQWMBQGCCsGAQUFBwMBBggrBgEFBQcDAjASBgNVHRMBAf8ECDAGAQH/AgEA
MDcGCCsGAQUFBwEBBCswKTAnBggrBgEFBQcwAYYbaHR0cDovL29jc3AuZGlnaWNl
cnQtY24uY29tMEQGA1UdHwQ9MDswOaA3oDWGM2h0dHA6Ly9jcmwuZGlnaWNlcnQt
Y24uY29tL0RpZ2lDZXJ0R2xvYmFsUm9vdENBLmNybDBWBgNVHSAETzBNMDcGCWCG
SAGG/WwBATAqMCgGCCsGAQUFBwIBFhxodHRwczovL3d3dy5kaWdpY2VydC5jb20v
Q1BTMAgGBmeBDAECAjAIBgZngQwBAgMwDQYJKoZIhvcNAQEMBQADggEBACVRufYd
j81xUqngFCO+Pk8EYXie0pxHKsBZnOPygAyXKx+awUasKBAnHjmhoFPXaDGAP2oV
OeZTWgwnURVr6wUCuTkz2/8Tgl1egC7OrVcHSa0fIIhaVo9/zRA/hr31xMG7LFBk
GNd7jd06Up4f/UOGbcJsqJexc5QRcUeSwe1MiUDcTNiyCjZk74QCPdcfdFYM4xsa
SlUpboB5vyT7jFePZ2v95CKjcr0EhiQ0gwxpdgoipZdfYTiMFGxCLsk6v8pUv7Tq
PT/qadOGyC+PfLuZh1PtLp20mF06K+MzheCiv+w1NT5ofhmcObvukc68wvbvRFL6
rRzZxAYN36q1SX8=
-----END CERTIFICATE-----`

const trustAsiaLeaf = `-----BEGIN CERTIFICATE-----
MIIEwTCCBEegAwIBAgIQBOjomZfHfhgz2bVYZVuf2DAKBggqhkjOPQQDAzBaMQsw
CQYDVQQGEwJDTjElMCMGA1UEChMcVHJ1c3RBc2lhIFRlY2hub2xvZ2llcywgSW5j
LjEkMCIGA1UEAxMbVHJ1c3RBc2lhIEVDQyBPViBUTFMgUHJvIENBMB4XDTE5MDUx
NzAwMDAwMFoXDTIwMDcyODEyMDAwMFowgY0xCzAJBgNVBAYTAkNOMRIwEAYDVQQI
DAnnpo/lu7rnnIExEjAQBgNVBAcMCeWOpumXqOW4gjEqMCgGA1UECgwh5Y6m6Zeo
5Y+B546W5Y+B56eR5oqA5pyJ6ZmQ5YWs5Y+4MRgwFgYDVQQLDA/nn6Xor4bkuqfm
nYPpg6gxEDAOBgNVBAMMByoudG0uY24wWTATBgcqhkjOPQIBBggqhkjOPQMBBwNC
AARx/MDQ0oGnCLagQIzjIz57iqFYFmz4/W6gaU6N+GHBkzyvQU8aX02QkdlTTNYL
TCoGFJxHB0XlZVSxrqoIPlNKo4ICuTCCArUwHwYDVR0jBBgwFoAULdRyBx6HyIH/
+LOvuexyH5p/3PwwHQYDVR0OBBYEFGTyf5adc5smW8NvDZyummJwZRLEMBkGA1Ud
EQQSMBCCByoudG0uY26CBXRtLmNuMA4GA1UdDwEB/wQEAwIHgDAdBgNVHSUEFjAU
BggrBgEFBQcDAQYIKwYBBQUHAwIwRgYDVR0fBD8wPTA7oDmgN4Y1aHR0cDovL2Ny
bC5kaWdpY2VydC1jbi5jb20vVHJ1c3RBc2lhRUNDT1ZUTFNQcm9DQS5jcmwwTAYD
VR0gBEUwQzA3BglghkgBhv1sAQEwKjAoBggrBgEFBQcCARYcaHR0cHM6Ly93d3cu
ZGlnaWNlcnQuY29tL0NQUzAIBgZngQwBAgIwfgYIKwYBBQUHAQEEcjBwMCcGCCsG
AQUFBzABhhtodHRwOi8vb2NzcC5kaWdpY2VydC1jbi5jb20wRQYIKwYBBQUHMAKG
OWh0dHA6Ly9jYWNlcnRzLmRpZ2ljZXJ0LWNuLmNvbS9UcnVzdEFzaWFFQ0NPVlRM
U1Byb0NBLmNydDAMBgNVHRMBAf8EAjAAMIIBAwYKKwYBBAHWeQIEAgSB9ASB8QDv
AHUA7ku9t3XOYLrhQmkfq+GeZqMPfl+wctiDAMR7iXqo/csAAAFqxGMTnwAABAMA
RjBEAiAz13zKEoyqd4e/96SK/fxfjl7uR+xhfoDZeyA1BvtfOwIgTY+8nJMGekv8
leIVdW6AGh7oqH31CIGTAbNJJWzaSFYAdgCHdb/nWXz4jEOZX73zbv9WjUdWNv9K
tWDBtOr/XqCDDwAAAWrEYxTCAAAEAwBHMEUCIQDlWm7+limbRiurcqUwXav3NSmx
x/aMnolLbh6+f+b1XAIgQfinHwLw6pDr4R9UkndUsX8QFF4GXS3/IwRR8HCp+pIw
CgYIKoZIzj0EAwMDaAAwZQIwHg8JmjRtcq+OgV0vVmdVBPqehi1sQJ9PZ+51CG+Z
0GOu+2HwS/fyLRViwSc/MZoVAjEA7NgbgpPN4OIsZn2XjMGxemtVxGFS6ZR+1364
EEeHB9vhZAEjQSePAfjR9aAGhXRa
-----END CERTIFICATE-----`

const selfSigned = `-----BEGIN CERTIFICATE-----
MIIC/DCCAeSgAwIBAgIRAK0SWRVmi67xU3z0gkgY+PkwDQYJKoZIhvcNAQELBQAw
EjEQMA4GA1UEChMHQWNtZSBDbzAeFw0xNjA4MTkxNjMzNDdaFw0xNzA4MTkxNjMz
NDdaMBIxEDAOBgNVBAoTB0FjbWUgQ28wggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAw
ggEKAoIBAQDWkm1kdCwxyKEt6OTmZitkmLGH8cQu9z7rUdrhW8lWNm4kh2SuaUWP
pscBjda5iqg51aoKuWJR2rw6ElDne+X5eit2FT8zJgAU8v39lMFjbaVZfS9TFOYF
w0Tk0Luo/PyKJpZnwhsP++iiGQiteJbndy8aLKmJ2MpLfpDGIgxEIyNb5dgoDi0D
WReDCpE6K9WDYqvKVGnQ2Jvqqra6Gfx0tFkuqJxQuqA8aUOlPHcCH4KBZdNEoXdY
YL3E4dCAh0YiDs80wNZx4cHqEM3L8gTEFqW2Tn1TSuPZO6gjJ9QPsuUZVjaMZuuO
NVxqLGujZkDzARhC3fBpptMuaAfi20+BAgMBAAGjTTBLMA4GA1UdDwEB/wQEAwIF
oDATBgNVHSUEDDAKBggrBgEFBQcDATAMBgNVHRMBAf8EAjAAMBYGA1UdEQQPMA2C
C2Zvby5leGFtcGxlMA0GCSqGSIb3DQEBCwUAA4IBAQBPvvfnDhsHWt+/cfwdAVim
4EDn+hYOMkTQwU0pouYIvY8QXYkZ8MBxpBtBMK4JhFU+ewSWoBAEH2dCCvx/BDxN
UGTSJHMbsvJHcFvdmsvvRxOqQ/cJz7behx0cfoeHMwcs0/vWv8ms5wHesb5Ek7L0
pl01FCBGTcncVqr6RK1r4fTpeCCfRIERD+YRJz8TtPH6ydesfLL8jIV40H8NiDfG
vRAvOtNiKtPzFeQVdbRPOskC4rcHyPeiDAMAMixeLi63+CFty4da3r5lRezeedCE
cw3ESZzThBwWqvPOtJdpXdm+r57pDW8qD+/0lY8wfImMNkQAyCUCLg/1Lxt/hrBj
-----END CERTIFICATE-----`

const issuerSubjectMatchRoot = `-----BEGIN CERTIFICATE-----
MIICIDCCAYmgAwIBAgIIAj5CwoHlWuYwDQYJKoZIhvcNAQELBQAwIzEPMA0GA1UE
ChMGR29sYW5nMRAwDgYDVQQDEwdSb290IGNhMB4XDTE1MDEwMTAwMDAwMFoXDTI1
MDEwMTAwMDAwMFowIzEPMA0GA1UEChMGR29sYW5nMRAwDgYDVQQDEwdSb290IGNh
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDpDn8RDOZa5oaDcPZRBy4CeBH1
siSSOO4mYgLHlPE+oXdqwI/VImi2XeJM2uCFETXCknJJjYG0iJdrt/yyRFvZTQZw
+QzGj+mz36NqhGxDWb6dstB2m8PX+plZw7jl81MDvUnWs8yiQ/6twgu5AbhWKZQD
JKcNKCEpqa6UW0r5nwIDAQABo10wWzAOBgNVHQ8BAf8EBAMCAgQwHQYDVR0lBBYw
FAYIKwYBBQUHAwEGCCsGAQUFBwMCMA8GA1UdEwEB/wQFMAMBAf8wGQYDVR0OBBIE
EEA31wH7QC+4HH5UBCeMWQEwDQYJKoZIhvcNAQELBQADgYEAb4TfSeCZ1HFmHTKG
VsvqWmsOAGrRWm4fBiMH/8vRGnTkJEMLqiqgc3Ulgry/P6n4SIis7TqUOw3TiMhn
RGEz33Fsxa/tFoy/gvlJu+MqB1M2NyV33pGkdwl/b7KRWMQFieqO+uE7Ge/49pS3
eyfm5ITdK/WT9TzYhsU4AVZcn20=
-----END CERTIFICATE-----`

const issuerSubjectMatchLeaf = `-----BEGIN CERTIFICATE-----
MIICODCCAaGgAwIBAgIJAOjwnT/iW+qmMA0GCSqGSIb3DQEBCwUAMCMxDzANBgNV
BAoTBkdvbGFuZzEQMA4GA1UEAxMHUm9vdCBDQTAeFw0xNTAxMDEwMDAwMDBaFw0y
NTAxMDEwMDAwMDBaMCAxDzANBgNVBAoTBkdvbGFuZzENMAsGA1UEAxMETGVhZjCB
nzANBgkqhkiG9w0BAQEFAAOBjQAwgYkCgYEA20Z9ky4SJwZIvAYoIat+xLaiXf4e
UkWIejZHpQgNkkJbwoHAvpd5mED7T20U/SsTi8KlLmfY1Ame1iI4t0oLdHMrwjTx
0ZPlltl0e/NYn2xhPMCwQdTZKyskI3dbHDu9dV3OIFTPoWOHHR4kxPMdGlCLqrYU
Q+2Xp3Vi9BTIUtcCAwEAAaN3MHUwDgYDVR0PAQH/BAQDAgWgMB0GA1UdJQQWMBQG
CCsGAQUFBwMBBggrBgEFBQcDAjAMBgNVHRMBAf8EAjAAMBkGA1UdDgQSBBCfkRYf
Q0M+SabebbaA159gMBsGA1UdIwQUMBKAEEA31wH7QC+4HH5UBCeMWQEwDQYJKoZI
hvcNAQELBQADgYEAjYYF2on1HcUWFEG5NIcrXDiZ49laW3pb3gtcCEUJbxydMV8I
ynqjmdqDCyK+TwI1kU5dXDe/iSJYfTB20i/QoO53nnfA1hnr7KBjNWqAm4AagN5k
vEA4PCJprUYmoj3q9MKSSRYDlq5kIbl87mSRR4GqtAwJKxIasvOvULOxziQ=
-----END CERTIFICATE-----`

const x509v1TestRoot = `-----BEGIN CERTIFICATE-----
MIICIDCCAYmgAwIBAgIIAj5CwoHlWuYwDQYJKoZIhvcNAQELBQAwIzEPMA0GA1UE
ChMGR29sYW5nMRAwDgYDVQQDEwdSb290IENBMB4XDTE1MDEwMTAwMDAwMFoXDTI1
MDEwMTAwMDAwMFowIzEPMA0GA1UEChMGR29sYW5nMRAwDgYDVQQDEwdSb290IENB
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDpDn8RDOZa5oaDcPZRBy4CeBH1
siSSOO4mYgLHlPE+oXdqwI/VImi2XeJM2uCFETXCknJJjYG0iJdrt/yyRFvZTQZw
+QzGj+mz36NqhGxDWb6dstB2m8PX+plZw7jl81MDvUnWs8yiQ/6twgu5AbhWKZQD
JKcNKCEpqa6UW0r5nwIDAQABo10wWzAOBgNVHQ8BAf8EBAMCAgQwHQYDVR0lBBYw
FAYIKwYBBQUHAwEGCCsGAQUFBwMCMA8GA1UdEwEB/wQFMAMBAf8wGQYDVR0OBBIE
EEA31wH7QC+4HH5UBCeMWQEwDQYJKoZIhvcNAQELBQADgYEAcIwqeNUpQr9cOcYm
YjpGpYkQ6b248xijCK7zI+lOeWN89zfSXn1AvfsC9pSdTMeDklWktbF/Ad0IN8Md
h2NtN34ard0hEfHc8qW8mkXdsysVmq6cPvFYaHz+dBtkHuHDoy8YQnC0zdN/WyYB
/1JmacUUofl+HusHuLkDxmadogI=
-----END CERTIFICATE-----`

const x509v1TestIntermediate = `-----BEGIN CERTIFICATE-----
MIIByjCCATMCCQCCdEMsT8ykqTANBgkqhkiG9w0BAQsFADAjMQ8wDQYDVQQKEwZH
b2xhbmcxEDAOBgNVBAMTB1Jvb3QgQ0EwHhcNMTUwMTAxMDAwMDAwWhcNMjUwMTAx
MDAwMDAwWjAwMQ8wDQYDVQQKEwZHb2xhbmcxHTAbBgNVBAMTFFguNTA5djEgaW50
ZXJtZWRpYXRlMIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDJ2QyniAOT+5YL
jeinEBJr3NsC/Q2QJ/VKmgvp+xRxuKTHJiVmxVijmp0vWg8AWfkmuE4p3hXQbbqM
k5yxrk1n60ONhim2L4VXriEvCE7X2OXhTmBls5Ufr7aqIgPMikwjScCXwz8E8qI8
UxyAhnjeJwMYBU8TuwBImSd4LBHoQQIDAQABMA0GCSqGSIb3DQEBCwUAA4GBAIab
DRG6FbF9kL9jb/TDHkbVBk+sl/Pxi4/XjuFyIALlARgAkeZcPmL5tNW1ImHkwsHR
zWE77kJDibzd141u21ZbLsKvEdUJXjla43bdyMmEqf5VGpC3D4sFt3QVH7lGeRur
x5Wlq1u3YDL/j6s1nU2dQ3ySB/oP7J+vQ9V4QeM+
-----END CERTIFICATE-----`

const x509v1TestLeaf = `-----BEGIN CERTIFICATE-----
MIICMzCCAZygAwIBAgIJAPo99mqJJrpJMA0GCSqGSIb3DQEBCwUAMDAxDzANBgNV
BAoTBkdvbGFuZzEdMBsGA1UEAxMUWC41MDl2MSBpbnRlcm1lZGlhdGUwHhcNMTUw
MTAxMDAwMDAwWhcNMjUwMTAxMDAwMDAwWjArMQ8wDQYDVQQKEwZHb2xhbmcxGDAW
BgNVBAMTD2Zvby5leGFtcGxlLmNvbTCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkC
gYEApUh60Z+a5/oKJxG//Dn8CihSo2CJHNIIO3zEJZ1EeNSMZCynaIR6D3IPZEIR
+RG2oGt+f5EEukAPYxwasp6VeZEezoQWJ+97nPCT6DpwLlWp3i2MF8piK2R9vxkG
Z5n0+HzYk1VM8epIrZFUXSMGTX8w1y041PX/yYLxbdEifdcCAwEAAaNaMFgwDgYD
VR0PAQH/BAQDAgWgMB0GA1UdJQQWMBQGCCsGAQUFBwMBBggrBgEFBQcDAjAMBgNV
HRMBAf8EAjAAMBkGA1UdDgQSBBBFozXe0SnzAmjy+1U6M/cvMA0GCSqGSIb3DQEB
CwUAA4GBADYzYUvaToO/ucBskPdqXV16AaakIhhSENswYVSl97/sODaxsjishKq9
5R7siu+JnIFotA7IbBe633p75xEnLN88X626N/XRFG9iScLzpj0o0PWXBUiB+fxL
/jt8qszOXCv2vYdUTPNuPqufXLWMoirpuXrr1liJDmedCcAHepY/
-----END CERTIFICATE-----`

const ignoreCNWithSANRoot = `-----BEGIN CERTIFICATE-----
MIIDPzCCAiegAwIBAgIIJkzCwkNrPHMwDQYJKoZIhvcNAQELBQAwMDEQMA4GA1UE
ChMHVEVTVElORzEcMBoGA1UEAxMTKipUZXN0aW5nKiogUm9vdCBDQTAeFw0xNTAx
MDEwMDAwMDBaFw0yNTAxMDEwMDAwMDBaMDAxEDAOBgNVBAoTB1RFU1RJTkcxHDAa
BgNVBAMTEyoqVGVzdGluZyoqIFJvb3QgQ0EwggEiMA0GCSqGSIb3DQEBAQUAA4IB
DwAwggEKAoIBAQC4YAf5YqlXGcikvbMWtVrNICt+V/NNWljwfvSKdg4Inm7k6BwW
P6y4Y+n4qSYIWNU4iRkdpajufzctxQCO6ty13iw3qVktzcC5XBIiS6ymiRhhDgnY
VQqyakVGw9MxrPwdRZVlssUv3Hmy6tU+v5Ok31SLY5z3wKgYWvSyYs0b8bKNU8kf
2FmSHnBN16lxGdjhe3ji58F/zFMr0ds+HakrLIvVdFcQFAnQopM8FTHpoWNNzGU3
KaiO0jBbMFkd6uVjVnuRJ+xjuiqi/NWwiwQA+CEr9HKzGkxOF8nAsHamdmO1wW+w
OsCrC0qWQ/f5NTOVATTJe0vj88OMTvo3071VAgMBAAGjXTBbMA4GA1UdDwEB/wQE
AwICpDAdBgNVHSUEFjAUBggrBgEFBQcDAQYIKwYBBQUHAwIwDwYDVR0TAQH/BAUw
AwEB/zAZBgNVHQ4EEgQQQDfXAftAL7gcflQEJ4xZATANBgkqhkiG9w0BAQsFAAOC
AQEAGOn3XjxHyHbXLKrRmpwV447B7iNBXR5VlhwOgt1kWaHDL2+8f/9/h0HMkB6j
fC+/yyuYVqYuOeavqMGVrh33D2ODuTQcFlOx5lXukP46j3j+Lm0jjZ1qNX7vlP8I
VlUXERhbelkw8O4oikakwIY9GE8syuSgYf+VeBW/lvuAZQrdnPfabxe05Tre6RXy
nJHMB1q07YHpbwIkcV/lfCE9pig2nPXTLwYZz9cl46Ul5RCpPUi+IKURo3x8y0FU
aSLjI/Ya0zwUARMmyZ3RRGCyhIarPb20mKSaMf1/Nb23pS3k1QgmZhk5pAnXYsWu
BJ6bvwEAasFiLGP6Zbdmxb2hIA==
-----END CERTIFICATE-----`

const ignoreCNWithSANLeaf = `-----BEGIN CERTIFICATE-----
MIIDaTCCAlGgAwIBAgIJAONakvRTxgJhMA0GCSqGSIb3DQEBCwUAMDAxEDAOBgNV
BAoTB1RFU1RJTkcxHDAaBgNVBAMTEyoqVGVzdGluZyoqIFJvb3QgQ0EwHhcNMTUw
MTAxMDAwMDAwWhcNMjUwMTAxMDAwMDAwWjAsMRAwDgYDVQQKEwdURVNUSU5HMRgw
FgYDVQQDEw9mb28uZXhhbXBsZS5jb20wggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAw
ggEKAoIBAQDBqskp89V/JMIBBqcauKSOVLcMyIE/t0jgSWVrsI4sksBTabLsfMdS
ui2n+dHQ1dRBuw3o4g4fPrWwS3nMnV3pZUHEn2TPi5N1xkjTaxObXgKIY2GKmFP3
rJ9vYqHT6mT4K93kCHoRcmJWWySc7S3JAOhTcdB4G+tIdQJN63E+XRYQQfNrn5HZ
hxQoOzaguHFx+ZGSD4Ntk6BSZz5NfjqCYqYxe+iCpTpEEYhIpi8joSPSmkTMTxBW
S1W2gXbYNQ9KjNkGM6FnQsUJrSPMrWs4v3UB/U88N5LkZeF41SqD9ySFGwbGajFV
nyzj12+4K4D8BLhlOc0Eo/F/8GwOwvmxAgMBAAGjgYkwgYYwDgYDVR0PAQH/BAQD
AgWgMB0GA1UdJQQWMBQGCCsGAQUFBwMBBggrBgEFBQcDAjAMBgNVHRMBAf8EAjAA
MBkGA1UdDgQSBBCjeab27q+5pV43jBGANOJ1MBsGA1UdIwQUMBKAEEA31wH7QC+4
HH5UBCeMWQEwDwYDVR0RBAgwBocEfwAAATANBgkqhkiG9w0BAQsFAAOCAQEAGZfZ
ErTVxxpIg64s22mQpXSk/72THVQsfsKHzlXmztM0CJzH8ccoN67ZqKxJCfdiE/FI
Emb6BVV4cGPeIKpcxaM2dwX/Y+Y0JaxpQJvqLxs+EByRL0gPP3shgg86WWCjYLxv
AgOn862d/JXGDrC9vIlQ/DDQcyL5g0JV5UjG2G9TUigbnrXxBw7BoWK6wmoSaHnR
sZKEHSs3RUJvm7qqpA9Yfzm9jg+i9j32zh1xFacghAOmFRFXa9eCVeigZ/KK2mEY
j2kBQyvnyKsXHLAKUoUOpd6t/1PHrfXnGj+HmzZNloJ/BZ1kiWb4eLvMljoLGkZn
xZbqP3Krgjj4XNaXjg==
-----END CERTIFICATE-----`

const excludedNamesLeaf = `-----BEGIN CERTIFICATE-----
MIID4DCCAsigAwIBAgIHDUSFtJknhzANBgkqhkiG9w0BAQsFADCBnjELMAkGA1UE
BhMCVVMxEzARBgNVBAgMCkNhbGlmb3JuaWExEjAQBgNVBAcMCUxvcyBHYXRvczEU
MBIGA1UECgwLTmV0ZmxpeCBJbmMxLTArBgNVBAsMJFBsYXRmb3JtIFNlY3VyaXR5
ICgzNzM0NTE1NTYyODA2Mzk3KTEhMB8GA1UEAwwYSW50ZXJtZWRpYXRlIENBIGZv
ciAzMzkyMB4XDTE3MDIwODIxMTUwNFoXDTE4MDIwODIwMjQ1OFowgZAxCzAJBgNV
BAYTAlVTMRMwEQYDVQQIDApDYWxpZm9ybmlhMRIwEAYDVQQHDAlMb3MgR2F0b3Mx
FDASBgNVBAoMC05ldGZsaXggSW5jMS0wKwYDVQQLDCRQbGF0Zm9ybSBTZWN1cml0
eSAoMzczNDUxNTc0ODUwMjY5NikxEzARBgNVBAMMCjE3Mi4xNi4wLjEwggEiMA0G
CSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCZ0oP1bMv6bOeqcKbzinnGpNOpenhA
zdFFsgea62znWsH3Wg4+1Md8uPCqlaQIsaJQKZHc50eKD3bg0Io7c6kxHkBQr1b8
Q7cGeK3CjdqG3NwS/aizzrLKOwL693hFwwy7JY7GGCvogbhyQRKn6iV0U9zMm7bu
/9pQVV/wx8u01u2uAlLttjyQ5LJkxo5t8cATFVqxdN5J9eY//VSDiTwXnlpQITBP
/Ow+zYuZ3kFlzH3CtCOhOEvNG3Ar1NvP3Icq35PlHV+Eki4otnKfixwByoiGpqCB
UEIY04VrZJjwBxk08y/3jY2B3VLYGgi+rryyCxIqkB7UpSNPMMWSG4UpAgMBAAGj
LzAtMAwGA1UdEwEB/wQCMAAwHQYDVR0RBBYwFIIMYmVuZGVyLmxvY2FshwSsEAAB
MA0GCSqGSIb3DQEBCwUAA4IBAQCLW3JO8L7LKByjzj2RciPjCGH5XF87Wd20gYLq
sNKcFwCIeyZhnQy5aZ164a5G9AIk2HLvH6HevBFPhA9Ivmyv/wYEfnPd1VcFkpgP
hDt8MCFJ8eSjCyKdtZh1MPMLrLVymmJV+Rc9JUUYM9TIeERkpl0rskcO1YGewkYt
qKlWE+0S16+pzsWvKn831uylqwIb8ANBPsCX4aM4muFBHavSWAHgRO+P+yXVw8Q+
VQDnMHUe5PbZd1/+1KKVs1K/CkBCtoHNHp1d/JT+2zUQJphwja9CcgfFdVhSnHL4
oEEOFtqVMIuQfR2isi08qW/JGOHc4sFoLYB8hvdaxKWSE19A
-----END CERTIFICATE-----`

const excludedNamesIntermediate = `-----BEGIN CERTIFICATE-----
MIIDzTCCArWgAwIBAgIHDUSFqYeczDANBgkqhkiG9w0BAQsFADCBmTELMAkGA1UE
BhMCVVMxEzARBgNVBAgMCkNhbGlmb3JuaWExEjAQBgNVBAcMCUxvcyBHYXRvczEU
MBIGA1UECgwLTmV0ZmxpeCBJbmMxLTArBgNVBAsMJFBsYXRmb3JtIFNlY3VyaXR5
ICgzNzM0NTE1NDc5MDY0NjAyKTEcMBoGA1UEAwwTTG9jYWwgUm9vdCBmb3IgMzM5
MjAeFw0xNzAyMDgyMTE1MDRaFw0xODAyMDgyMDI0NThaMIGeMQswCQYDVQQGEwJV
UzETMBEGA1UECAwKQ2FsaWZvcm5pYTESMBAGA1UEBwwJTG9zIEdhdG9zMRQwEgYD
VQQKDAtOZXRmbGl4IEluYzEtMCsGA1UECwwkUGxhdGZvcm0gU2VjdXJpdHkgKDM3
MzQ1MTU1NjI4MDYzOTcpMSEwHwYDVQQDDBhJbnRlcm1lZGlhdGUgQ0EgZm9yIDMz
OTIwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCOyEs6tJ/t9emQTvlx
3FS7uJSou5rKkuqVxZdIuYQ+B2ZviBYUnMRT9bXDB0nsVdKZdp0hdchdiwNXDG/I
CiWu48jkcv/BdynVyayOT+0pOJSYLaPYpzBx1Pb9M5651ct9GSbj6Tz0ChVonoIE
1AIZ0kkebucZRRFHd0xbAKVRKyUzPN6HJ7WfgyauUp7RmlC35wTmrmARrFohQLlL
7oICy+hIQePMy9x1LSFTbPxZ5AUUXVC3eUACU3vLClF/Xs8XGHebZpUXCdMQjOGS
nq1eFguFHR1poSB8uSmmLqm4vqUH9CDhEgiBAC8yekJ8//kZQ7lUEqZj3YxVbk+Y
E4H5AgMBAAGjEzARMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQELBQADggEB
ADxrnmNX5gWChgX9K5fYwhFDj5ofxZXAKVQk+WjmkwMcmCx3dtWSm++Wdksj/ZlA
V1cLW3ohWv1/OAZuOlw7sLf98aJpX+UUmIYYQxDubq+4/q7VA7HzEf2k/i/oN1NI
JgtrhpPcZ/LMO6k7DYx0qlfYq8pTSfd6MI4LnWKgLc+JSPJJjmvspgio2ZFcnYr7
A264BwLo6v1Mos1o1JUvFFcp4GANlw0XFiWh7JXYRl8WmS5DoouUC+aNJ3lmyF6z
LbIjZCSfgZnk/LK1KU1j91FI2bc2ULYZvAC1PAg8/zvIgxn6YM2Q7ZsdEgWw0FpS
zMBX1/lk4wkFckeUIlkD55Y=
-----END CERTIFICATE-----`

const excludedNamesRoot = `-----BEGIN CERTIFICATE-----
MIIEGTCCAwGgAwIBAgIHDUSFpInn/zANBgkqhkiG9w0BAQsFADCBozELMAkGA1UE
BhMCVVMxEzARBgNVBAgMCkNhbGlmb3JuaWExEjAQBgNVBAcMCUxvcyBHYXRvczEU
MBIGA1UECgwLTmV0ZmxpeCBJbmMxLTArBgNVBAsMJFBsYXRmb3JtIFNlY3VyaXR5
ICgzNzMxNTA5NDM3NDYyNDg1KTEmMCQGA1UEAwwdTmFtZSBDb25zdHJhaW50cyBU
ZXN0IFJvb3QgQ0EwHhcNMTcwMjA4MjExNTA0WhcNMTgwMjA4MjAyNDU4WjCBmTEL
MAkGA1UEBhMCVVMxEzARBgNVBAgMCkNhbGlmb3JuaWExEjAQBgNVBAcMCUxvcyBH
YXRvczEUMBIGA1UECgwLTmV0ZmxpeCBJbmMxLTArBgNVBAsMJFBsYXRmb3JtIFNl
Y3VyaXR5ICgzNzM0NTE1NDc5MDY0NjAyKTEcMBoGA1UEAwwTTG9jYWwgUm9vdCBm
b3IgMzM5MjCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAJymcnX29ekc
7+MLyr8QuAzoHWznmGdDd2sITwWRjM89/21cdlHCGKSpULUNdFp9HDLWvYECtxt+
8TuzKiQz7qAerzGUT1zI5McIjHy0e/i4xIkfiBiNeTCuB/N9QRbZlcfM80ErkaA4
gCAFK8qZAcWkHIl6e+KaQFMPLKk9kckgAnVDHEJe8oLNCogCJ15558b65g05p9eb
5Lg+E98hoPRTQaDwlz3CZPfTTA2EiEZInSi8qzodFCbTpJUVTbiVUH/JtVjlibbb
smdcx5PORK+8ZJkhLEh54AjaWOX4tB/7Tkk8stg2VBmrIARt/j4UVj7cTrIWU3bV
m8TwHJG+YgsCAwEAAaNaMFgwDwYDVR0TAQH/BAUwAwEB/zBFBgNVHR4EPjA8oBww
CocICgEAAP//AAAwDoIMYmVuZGVyLmxvY2FsoRwwCocICgEAAP//AAAwDoIMYmVu
ZGVyLmxvY2FsMA0GCSqGSIb3DQEBCwUAA4IBAQAMjbheffPxtSKSv9NySW+8qmHs
n7Mb5GGyCFu+cMZSoSaabstbml+zHEFJvWz6/1E95K4F8jKhAcu/CwDf4IZrSD2+
Hee0DolVSQhZpnHgPyj7ZATz48e3aJaQPUlhCEOh0wwF4Y0N4FV0t7R6woLylYRZ
yU1yRHUqUYpN0DWFpsPbBqgM6uUAVO2ayBFhPgWUaqkmSbZ/Nq7isGvknaTmcIwT
6mOAFN0qFb4RGzfGJW7x6z7KCULS7qVDp6fU3tRoScHFEgRubks6jzQ1W5ooSm4o
+NQCZDd5eFeU8PpNX7rgaYE4GPq+EEmLVCBYmdctr8QVdqJ//8Xu3+1phjDy
-----END CERTIFICATE-----`

const invalidCNRoot = `-----BEGIN CERTIFICATE-----
MIIBFjCBvgIJAIsu4r+jb70UMAoGCCqGSM49BAMCMBQxEjAQBgNVBAsMCVRlc3Qg
cm9vdDAeFw0xODA3MTExODMyMzVaFw0yODA3MDgxODMyMzVaMBQxEjAQBgNVBAsM
CVRlc3Qgcm9vdDBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABF6oDgMg0LV6YhPj
QXaPXYCc2cIyCdqp0ROUksRz0pOLTc5iY2nraUheRUD1vRRneq7GeXOVNn7uXONg
oCGMjNwwCgYIKoZIzj0EAwIDRwAwRAIgDSiwgIn8g1lpruYH0QD1GYeoWVunfmrI
XzZZl0eW/ugCICgOfXeZ2GGy3wIC0352BaC3a8r5AAb2XSGNe+e9wNN6
-----END CERTIFICATE-----`

const validCNWithoutSAN = `-----BEGIN CERTIFICATE-----
MIIBJzCBzwIUB7q8t9mrDAL+UB1OFaMN5BEWFKQwCgYIKoZIzj0EAwIwFDESMBAG
A1UECwwJVGVzdCByb290MB4XDTE4MDcxMTE4NDcyNFoXDTI4MDcwODE4NDcyNFow
GjEYMBYGA1UEAwwPZm9vLmV4YW1wbGUuY29tMFkwEwYHKoZIzj0CAQYIKoZIzj0D
AQcDQgAEp6Z8IjOnR38Iky1fYTUu2kVndvKXcxiwARJKGtW3b0E8uwVp9AZd/+sr
p4ULTPdFToFAeqnGHbu62bkms8pQkDAKBggqhkjOPQQDAgNHADBEAiBTbNe3WWFR
cqUYo0sNUuoV+tCTMDJUS+0PWIW4qBqCOwIgFHdLDn5PCk9kJpfc0O2qZx03hdq0
h7olHCpY9yMRiz0=
-----END CERTIFICATE-----`

const rootWithoutSKID = `-----BEGIN CERTIFICATE-----
MIIBbzCCARSgAwIBAgIQeCkq3C8SOX/JM5PqYTl9cDAKBggqhkjOPQQDAjASMRAw
DgYDVQQKEwdBY21lIENvMB4XDTE5MDIwNDIyNTYzNFoXDTI5MDIwMTIyNTYzNFow
EjEQMA4GA1UEChMHQWNtZSBDbzBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABISm
jGlTr4dLOWT+BCTm2PzWRjk1DpLcSAh+Al8eB1Nc2eBWxYIH9qPirfatvqBOA4c5
ZwycRpFoaw6O+EmXnVujTDBKMA4GA1UdDwEB/wQEAwICpDATBgNVHSUEDDAKBggr
BgEFBQcDATAPBgNVHRMBAf8EBTADAQH/MBIGA1UdEQQLMAmCB2V4YW1wbGUwCgYI
KoZIzj0EAwIDSQAwRgIhAMaBYWFCjTfn0MNyQ0QXvYT/iIFompkIqzw6wB7qjLrA
AiEA3sn65V7G4tsjZEOpN0Jykn9uiTjqniqn/S/qmv8gIec=
-----END CERTIFICATE-----`

const leafWithAKID = `-----BEGIN CERTIFICATE-----
MIIBjTCCATSgAwIBAgIRAPCKYvADhKLPaWOtcTu2XYwwCgYIKoZIzj0EAwIwEjEQ
MA4GA1UEChMHQWNtZSBDbzAeFw0xOTAyMDQyMzA2NTJaFw0yOTAyMDEyMzA2NTJa
MBMxETAPBgNVBAoTCEFjbWUgTExDMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE
Wk5N+/8X97YT6ClFNIE5/4yc2YwKn921l0wrIJEcT2u+Uydm7EqtCJNtZjYMAnBd
Acp/wynpTwC6tBTsxcM0s6NqMGgwDgYDVR0PAQH/BAQDAgWgMBMGA1UdJQQMMAoG
CCsGAQUFBwMBMAwGA1UdEwEB/wQCMAAwHwYDVR0jBBgwFoAUwitfkXg0JglCjW9R
ssWvTAveakIwEgYDVR0RBAswCYIHZXhhbXBsZTAKBggqhkjOPQQDAgNHADBEAiBk
4LpWiWPOIl5PIhX9PDVkmjpre5oyoH/3aYwG8ABYuAIgCeSfbYueOOG2AdXuMqSU
ZZMqeJS7JldLx91sPUArY5A=
-----END CERTIFICATE-----`

const rootMatchingSKIDMismatchingSubject = `-----BEGIN CERTIFICATE-----
MIIBQjCB6aADAgECAgEAMAoGCCqGSM49BAMCMBExDzANBgNVBAMTBlJvb3QgQTAe
Fw0wOTExMTAyMzAwMDBaFw0xOTExMDgyMzAwMDBaMBExDzANBgNVBAMTBlJvb3Qg
QTBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABPK4p1uXq2aAeDtKDHIokg2rTcPM
2gq3N9Y96wiW6/7puBK1+INEW//cO9x6FpzkcsHw/TriAqy4sck/iDAvf9WjMjAw
MA8GA1UdJQQIMAYGBFUdJQAwDwYDVR0TAQH/BAUwAwEB/zAMBgNVHQ4EBQQDAQID
MAoGCCqGSM49BAMCA0gAMEUCIQDgtAp7iVHxMnKxZPaLQPC+Tv2r7+DJc88k2SKH
MPs/wQIgFjjNvBoQEl7vSHTcRGCCcFMdlN4l0Dqc9YwGa9fyrQs=
-----END CERTIFICATE-----`

const rootMismatchingSKIDMatchingSubject = `-----BEGIN CERTIFICATE-----
MIIBNDCB26ADAgECAgEAMAoGCCqGSM49BAMCMBExDzANBgNVBAMTBlJvb3QgQjAe
Fw0wOTExMTAyMzAwMDBaFw0xOTExMDgyMzAwMDBaMBExDzANBgNVBAMTBlJvb3Qg
QjBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABI1YRFcIlkWzm9BdEVrIsEQJ2dT6
qiW8/WV9GoIhmDtX9SEDHospc0Cgm+TeD2QYW2iMrS5mvNe4GSw0Jezg/bOjJDAi
MA8GA1UdJQQIMAYGBFUdJQAwDwYDVR0TAQH/BAUwAwEB/zAKBggqhkjOPQQDAgNI
ADBFAiEAukWOiuellx8bugRiwCS5XQ6IOJ1SZcjuZxj76WojwxkCIHqa71qNw8FM
DtA5yoL9M2pDFF6ovFWnaCe+KlzSwAW/
-----END CERTIFICATE-----`

const leafMatchingAKIDMatchingIssuer = `-----BEGIN CERTIFICATE-----
MIIBNTCB26ADAgECAgEAMAoGCCqGSM49BAMCMBExDzANBgNVBAMTBlJvb3QgQjAe
Fw0wOTExMTAyMzAwMDBaFw0xOTExMDgyMzAwMDBaMA8xDTALBgNVBAMTBExlYWYw
WTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAASNWERXCJZFs5vQXRFayLBECdnU+qol
vP1lfRqCIZg7V/UhAx6LKXNAoJvk3g9kGFtojK0uZrzXuBksNCXs4P2zoyYwJDAO
BgNVHSMEBzAFgAMBAgMwEgYDVR0RBAswCYIHZXhhbXBsZTAKBggqhkjOPQQDAgNJ
ADBGAiEAnV9XV7a4h0nfJB8pWv+pBUXRlRFA2uZz3mXEpee8NYACIQCWa+wL70GL
ePBQCV1F9sE2q4ZrnsT9TZoNrSe/bMDjzA==
-----END CERTIFICATE-----`

var unknownAuthorityErrorTests = []struct {
	name     string
	cert     string
	expected string
}{
	{"self-signed, cn", selfSignedWithCommonName, "x509: certificate signed by unknown authority (possibly because of \"empty\" while trying to verify candidate authority certificate \"test\")"},
	{"self-signed, no cn, org", selfSignedNoCommonNameWithOrgName, "x509: certificate signed by unknown authority (possibly because of \"empty\" while trying to verify candidate authority certificate \"ca\")"},
	{"self-signed, no cn, no org", selfSignedNoCommonNameNoOrgName, "x509: certificate signed by unknown authority (possibly because of \"empty\" while trying to verify candidate authority certificate \"serial:0\")"},
}

func TestUnknownAuthorityError(t *testing.T) {
	for i, tt := range unknownAuthorityErrorTests {
		t.Run(tt.name, func(t *testing.T) {
			der, _ := pem.Decode([]byte(tt.cert))
			if der == nil {
				t.Fatalf("#%d: Unable to decode PEM block", i)
			}
			c, err := ParseCertificate(der.Bytes)
			if err != nil {
				t.Fatalf("#%d: Unable to parse certificate -> %v", i, err)
			}
			uae := &UnknownAuthorityError{
				Cert:     c,
				hintErr:  fmt.Errorf("empty"),
				hintCert: c,
			}
			actual := uae.Error()
			if actual != tt.expected {
				t.Errorf("#%d: UnknownAuthorityError.Error() response invalid actual: %s expected: %s", i, actual, tt.expected)
			}
		})
	}
}

var nameConstraintTests = []struct {
	constraint, domain string
	expectError        bool
	shouldMatch        bool
}{
	{"", "anything.com", false, true},
	{"example.com", "example.com", false, true},
	{"example.com.", "example.com", true, false},
	{"example.com", "example.com.", true, false},
	{"example.com", "ExAmPle.coM", false, true},
	{"example.com", "exampl1.com", false, false},
	{"example.com", "www.ExAmPle.coM", false, true},
	{"example.com", "sub.www.ExAmPle.coM", false, true},
	{"example.com", "notexample.com", false, false},
	{".example.com", "example.com", false, false},
	{".example.com", "www.example.com", false, true},
	{".example.com", "www..example.com", true, false},
}

func TestNameConstraints(t *testing.T) {
	for i, test := range nameConstraintTests {
		result, err := matchDomainConstraint(test.domain, test.constraint)

		if err != nil && !test.expectError {
			t.Errorf("unexpected error for test #%d: domain=%s, constraint=%s, err=%s", i, test.domain, test.constraint, err)
			continue
		}

		if err == nil && test.expectError {
			t.Errorf("unexpected success for test #%d: domain=%s, constraint=%s", i, test.domain, test.constraint)
			continue
		}

		if result != test.shouldMatch {
			t.Errorf("unexpected result for test #%d: domain=%s, constraint=%s, result=%t", i, test.domain, test.constraint, result)
		}
	}
}

const selfSignedWithCommonName = `-----BEGIN CERTIFICATE-----
MIIDCjCCAfKgAwIBAgIBADANBgkqhkiG9w0BAQsFADAaMQswCQYDVQQKEwJjYTEL
MAkGA1UEAxMCY2EwHhcNMTYwODI4MTcwOTE4WhcNMjEwODI3MTcwOTE4WjAcMQsw
CQYDVQQKEwJjYTENMAsGA1UEAxMEdGVzdDCCASIwDQYJKoZIhvcNAQEBBQADggEP
ADCCAQoCggEBAOH55PfRsbvmcabfLLko1w/yuapY/hk13Cgmc3WE/Z1ZStxGiVxY
gQVH9n4W/TbUsrep/TmcC4MV7xEm5252ArcgaH6BeQ4QOTFj/6Jx0RT7U/ix+79x
8RRysf7OlzNpGIctwZEM7i/G+0ZfqX9ULxL/EW9tppSxMX1jlXZQarnU7BERL5cH
+G2jcbU9H28FXYishqpVYE9L7xrXMm61BAwvGKB0jcVW6JdhoAOSfQbbgp7JjIlq
czXqUsv1UdORO/horIoJptynTvuARjZzyWatya6as7wyOgEBllE6BjPK9zpn+lp3
tQ8dwKVqm/qBPhIrVqYG/Ec7pIv8mJfYabMCAwEAAaNZMFcwDgYDVR0PAQH/BAQD
AgOoMB0GA1UdJQQWMBQGCCsGAQUFBwMCBggrBgEFBQcDATAMBgNVHRMBAf8EAjAA
MAoGA1UdDgQDBAEAMAwGA1UdIwQFMAOAAQAwDQYJKoZIhvcNAQELBQADggEBAAAM
XMFphzq4S5FBcRdB2fRrmcoz+jEROBWvIH/1QUJeBEBz3ZqBaJYfBtQTvqCA5Rjw
dxyIwVd1W3q3aSulM0tO62UCU6L6YeeY/eq8FmpD7nMJo7kCrXUUAMjxbYvS3zkT
v/NErK6SgWnkQiPJBZNX1Q9+aSbLT/sbaCTdbWqcGNRuLGJkmqfIyoxRt0Hhpqsx
jP5cBaVl50t4qoCuVIE9cOucnxYXnI7X5HpXWvu8Pfxo4SwVjb1az8Fk5s8ZnxGe
fPB6Q3L/pKBe0SEe5GywpwtokPLB3lAygcuHbxp/1FlQ1NQZqq+vgXRIla26bNJf
IuYkJwt6w+LH/9HZgf8=
-----END CERTIFICATE-----`
const selfSignedNoCommonNameWithOrgName = `-----BEGIN CERTIFICATE-----
MIIC+zCCAeOgAwIBAgIBADANBgkqhkiG9w0BAQsFADAaMQswCQYDVQQKEwJjYTEL
MAkGA1UEAxMCY2EwHhcNMTYwODI4MTgxMzQ4WhcNMjEwODI3MTgxMzQ4WjANMQsw
CQYDVQQKEwJjYTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAL5EjrUa
7EtOMxWiIgTzp2FlQvncPsG329O3l3uNGnbigb8TmNMw2M8UhoDjd84pnU5RAfqd
8t5TJyw/ybnIKBN131Q2xX+gPQ0dFyMvcO+i1CUgCxmYZomKVA2MXO1RD1hLTYGS
gOVjc3no3MBwd8uVQp0NStqJ1QvLtNG4Uy+B28qe+ZFGGbjGqx8/CU4A8Szlpf7/
xAZR8w5qFUUlpA2LQYeHHJ5fQVXw7kyL1diNrKNi0G3qcY0IrBh++hT+hnEEXyXu
g8a0Ux18hoE8D6rAr34rCZl6AWfqW5wjwm+N5Ns2ugr9U4N8uCKJYMPHb2CtdubU
46IzVucpTfGLdaMCAwEAAaNZMFcwDgYDVR0PAQH/BAQDAgOoMB0GA1UdJQQWMBQG
CCsGAQUFBwMCBggrBgEFBQcDATAMBgNVHRMBAf8EAjAAMAoGA1UdDgQDBAEAMAwG
A1UdIwQFMAOAAQAwDQYJKoZIhvcNAQELBQADggEBAEn5SgVpJ3zjsdzPqK7Qd/sB
bYd1qtPHlrszjhbHBg35C6mDgKhcv4o6N+fuC+FojZb8lIxWzJtvT9pQbfy/V6u3
wOb816Hm71uiP89sioIOKCvSAstj/p9doKDOUaKOcZBTw0PS2m9eja8bnleZzBvK
rD8cNkHf74v98KvBhcwBlDifVzmkWzMG6TL1EkRXUyLKiWgoTUFSkCDV927oXXMR
DKnszq+AVw+K8hbeV2A7GqT7YfeqOAvSbatTDnDtKOPmlCnQui8A149VgZzXv7eU
29ssJSqjUPyp58dlV6ZuynxPho1QVZUOQgnJToXIQ3/5vIvJRXy52GJCs4/Gh/w=
-----END CERTIFICATE-----`
const selfSignedNoCommonNameNoOrgName = `-----BEGIN CERTIFICATE-----
MIIC7jCCAdagAwIBAgIBADANBgkqhkiG9w0BAQsFADAaMQswCQYDVQQKEwJjYTEL
MAkGA1UEAxMCY2EwHhcNMTYwODI4MTgxOTQ1WhcNMjEwODI3MTgxOTQ1WjAAMIIB
IjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAp3E+Jl6DpgzogHUW/i/AAcCM
fnNJLOamNVKFGmmxhb4XTHxRaWoTzrlsyzIMS0WzivvJeZVe6mWbvuP2kZanKgIz
35YXRTR9HbqkNTMuvnpUESzWxbGWE2jmt2+a/Jnz89FS4WIYRhF7nI2z8PvZOfrI
2gETTT2tEpoF2S4soaYfm0DBeT8K0/rogAaf+oeUS6V+v3miRcAooJgpNJGu9kqm
S0xKPn1RCFVjpiRd6YNS0xZirjYQIBMFBvoSoHjaOdgJptNRBprYPOxVJ/ItzGf0
kPmzPFCx2tKfxV9HLYBPgxi+fP3IIx8aIYuJn8yReWtYEMYU11hDPeAFN5Gm+wID
AQABo1kwVzAOBgNVHQ8BAf8EBAMCA6gwHQYDVR0lBBYwFAYIKwYBBQUHAwIGCCsG
AQUFBwMBMAwGA1UdEwEB/wQCMAAwCgYDVR0OBAMEAQAwDAYDVR0jBAUwA4ABADAN
BgkqhkiG9w0BAQsFAAOCAQEATZVOFeiCpPM5QysToLv+8k7Rjoqt6L5IxMUJGEpq
4ENmldmwkhEKr9VnYEJY3njydnnTm97d9vOfnLj9nA9wMBODeOO3KL2uJR2oDnmM
9z1NSe2aQKnyBb++DM3ZdikpHn/xEpGV19pYKFQVn35x3lpPh2XijqRDO/erKemb
w67CoNRb81dy+4Q1lGpA8ORoLWh5fIq2t2eNGc4qB8vlTIKiESzAwu7u3sRfuWQi
4R+gnfLd37FWflMHwztFbVTuNtPOljCX0LN7KcuoXYlr05RhQrmoN7fQHsrZMNLs
8FVjHdKKu+uPstwd04Uy4BR/H2y1yerN9j/L6ZkMl98iiA==
-----END CERTIFICATE-----`

const criticalExtRoot = `-----BEGIN CERTIFICATE-----
MIIBqzCCAVGgAwIBAgIJAJ+mI/85cXApMAoGCCqGSM49BAMCMB0xDDAKBgNVBAoT
A09yZzENMAsGA1UEAxMEUm9vdDAeFw0xNTAxMDEwMDAwMDBaFw0yNTAxMDEwMDAw
MDBaMB0xDDAKBgNVBAoTA09yZzENMAsGA1UEAxMEUm9vdDBZMBMGByqGSM49AgEG
CCqGSM49AwEHA0IABJGp9joiG2QSQA+1FczEDAsWo84rFiP3GTL+n+ugcS6TyNib
gzMsdbJgVi+a33y0SzLZxB+YvU3/4KTk8yKLC+2jejB4MA4GA1UdDwEB/wQEAwIC
BDAdBgNVHSUEFjAUBggrBgEFBQcDAQYIKwYBBQUHAwIwDwYDVR0TAQH/BAUwAwEB
/zAZBgNVHQ4EEgQQQDfXAftAL7gcflQEJ4xZATAbBgNVHSMEFDASgBBAN9cB+0Av
uBx+VAQnjFkBMAoGCCqGSM49BAMCA0gAMEUCIFeSV00fABFceWR52K+CfIgOHotY
FizzGiLB47hGwjMuAiEA8e0um2Kr8FPQ4wmFKaTRKHMaZizCGl3m+RG5QsE1KWo=
-----END CERTIFICATE-----`

const criticalExtIntermediate = `-----BEGIN CERTIFICATE-----
MIIBszCCAVmgAwIBAgIJAL2kcGZKpzVqMAoGCCqGSM49BAMCMB0xDDAKBgNVBAoT
A09yZzENMAsGA1UEAxMEUm9vdDAeFw0xNTAxMDEwMDAwMDBaFw0yNTAxMDEwMDAw
MDBaMCUxDDAKBgNVBAoTA09yZzEVMBMGA1UEAxMMSW50ZXJtZWRpYXRlMFkwEwYH
KoZIzj0CAQYIKoZIzj0DAQcDQgAESqVq92iPEq01cL4o99WiXDc5GZjpjNlzMS1n
rk8oHcVDp4tQRRQG3F4A6dF1rn/L923ha3b0fhDLlAvXZB+7EKN6MHgwDgYDVR0P
AQH/BAQDAgIEMB0GA1UdJQQWMBQGCCsGAQUFBwMBBggrBgEFBQcDAjAPBgNVHRMB
Af8EBTADAQH/MBkGA1UdDgQSBBCMGmiotXbbXVd7H40UsgajMBsGA1UdIwQUMBKA
EEA31wH7QC+4HH5UBCeMWQEwCgYIKoZIzj0EAwIDSAAwRQIhAOhhNRb6KV7h3wbE
cdap8bojzvUcPD78fbsQPCNw1jPxAiBOeAJhlTwpKn9KHpeJphYSzydj9NqcS26Y
xXbdbm27KQ==
-----END CERTIFICATE-----`

const criticalExtLeafWithExt = `-----BEGIN CERTIFICATE-----
MIIBxTCCAWugAwIBAgIJAJZAUtw5ccb1MAoGCCqGSM49BAMCMCUxDDAKBgNVBAoT
A09yZzEVMBMGA1UEAxMMSW50ZXJtZWRpYXRlMB4XDTE1MDEwMTAwMDAwMFoXDTI1
MDEwMTAwMDAwMFowJDEMMAoGA1UEChMDT3JnMRQwEgYDVQQDEwtleGFtcGxlLmNv
bTBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABF3ABa2+B6gUyg6ayCaRQWYY/+No
6PceLqEavZNUeVNuz7bS74Toy8I7R3bGMkMgbKpLSPlPTroAATvebTXoBaijgYQw
gYEwDgYDVR0PAQH/BAQDAgWgMB0GA1UdJQQWMBQGCCsGAQUFBwMBBggrBgEFBQcD
AjAMBgNVHRMBAf8EAjAAMBkGA1UdDgQSBBBRNtBL2vq8nCV3qVp7ycxMMBsGA1Ud
IwQUMBKAEIwaaKi1dttdV3sfjRSyBqMwCgYDUQMEAQH/BAAwCgYIKoZIzj0EAwID
SAAwRQIgVjy8GBgZFiagexEuDLqtGjIRJQtBcf7lYgf6XFPH1h4CIQCT6nHhGo6E
I+crEm4P5q72AnA/Iy0m24l7OvLuXObAmg==
-----END CERTIFICATE-----`

const criticalExtIntermediateWithExt = `-----BEGIN CERTIFICATE-----
MIIB2TCCAX6gAwIBAgIIQD3NrSZtcUUwCgYIKoZIzj0EAwIwHTEMMAoGA1UEChMD
T3JnMQ0wCwYDVQQDEwRSb290MB4XDTE1MDEwMTAwMDAwMFoXDTI1MDEwMTAwMDAw
MFowPTEMMAoGA1UEChMDT3JnMS0wKwYDVQQDEyRJbnRlcm1lZGlhdGUgd2l0aCBD
cml0aWNhbCBFeHRlbnNpb24wWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAAQtnmzH
mcRm10bdDBnJE7xQEJ25cLCL5okuEphRR0Zneo6+nQZikoh+UBbtt5GV3Dms7LeP
oF5HOplYDCd8wi/wo4GHMIGEMA4GA1UdDwEB/wQEAwICBDAdBgNVHSUEFjAUBggr
BgEFBQcDAQYIKwYBBQUHAwIwDwYDVR0TAQH/BAUwAwEB/zAZBgNVHQ4EEgQQKxdv
UuQZ6sO3XvBsxgNZ3zAbBgNVHSMEFDASgBBAN9cB+0AvuBx+VAQnjFkBMAoGA1ED
BAEB/wQAMAoGCCqGSM49BAMCA0kAMEYCIQCQzTPd6XKex+OAPsKT/1DsoMsg8vcG
c2qZ4Q0apT/kvgIhAKu2TnNQMIUdcO0BYQIl+Uhxc78dc9h4lO+YJB47pHGx
-----END CERTIFICATE-----`

const criticalExtLeaf = `-----BEGIN CERTIFICATE-----
MIIBzzCCAXWgAwIBAgIJANoWFIlhCI9MMAoGCCqGSM49BAMCMD0xDDAKBgNVBAoT
A09yZzEtMCsGA1UEAxMkSW50ZXJtZWRpYXRlIHdpdGggQ3JpdGljYWwgRXh0ZW5z
aW9uMB4XDTE1MDEwMTAwMDAwMFoXDTI1MDEwMTAwMDAwMFowJDEMMAoGA1UEChMD
T3JnMRQwEgYDVQQDEwtleGFtcGxlLmNvbTBZMBMGByqGSM49AgEGCCqGSM49AwEH
A0IABG1Lfh8A0Ho2UvZN5H0+ONil9c8jwtC0y0xIZftyQE+Fwr9XwqG3rV2g4M1h
GnJa9lV9MPHg8+b85Hixm0ZSw7SjdzB1MA4GA1UdDwEB/wQEAwIFoDAdBgNVHSUE
FjAUBggrBgEFBQcDAQYIKwYBBQUHAwIwDAYDVR0TAQH/BAIwADAZBgNVHQ4EEgQQ
UNhY4JhezH9gQYqvDMWrWDAbBgNVHSMEFDASgBArF29S5Bnqw7de8GzGA1nfMAoG
CCqGSM49BAMCA0gAMEUCIQClA3d4tdrDu9Eb5ZBpgyC+fU1xTZB0dKQHz6M5fPZA
2AIgN96lM+CPGicwhN24uQI6flOsO3H0TJ5lNzBYLtnQtlc=
-----END CERTIFICATE-----`

func TestValidHostname(t *testing.T) {
	tests := []struct {
		host                     string
		validInput, validPattern bool
	}{
		{host: "example.com", validInput: true, validPattern: true},
		{host: "eXample123-.com", validInput: true, validPattern: true},
		{host: "-eXample123-.com"},
		{host: ""},
		{host: "."},
		{host: "example..com"},
		{host: ".example.com"},
		{host: "example.com.", validInput: true},
		{host: "*.example.com."},
		{host: "*.example.com", validPattern: true},
		{host: "*foo.example.com"},
		{host: "foo.*.example.com"},
		{host: "exa_mple.com", validInput: true, validPattern: true},
		{host: "foo,bar"},
		{host: "project-dev:us-central1:main"},
	}
	for _, tt := range tests {
		if got := validHostnamePattern(tt.host); got != tt.validPattern {
			t.Errorf("validHostnamePattern(%q) = %v, want %v", tt.host, got, tt.validPattern)
		}
		if got := validHostnameInput(tt.host); got != tt.validInput {
			t.Errorf("validHostnameInput(%q) = %v, want %v", tt.host, got, tt.validInput)
		}
	}
}

func generateCert(cn string, isCA bool, issuer *Certificate, issuerKey crypto.PrivateKey) (*Certificate, crypto.PrivateKey, error) {
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, nil, err
	}

	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, _ := rand.Int(rand.Reader, serialNumberLimit)

	template := &Certificate{
		SerialNumber: serialNumber,
		Subject:      pkix.Name{CommonName: cn},
		NotBefore:    time.Now().Add(-1 * time.Hour),
		NotAfter:     time.Now().Add(24 * time.Hour),

		KeyUsage:              KeyUsageKeyEncipherment | KeyUsageDigitalSignature | KeyUsageCertSign,
		ExtKeyUsage:           []ExtKeyUsage{ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  isCA,
	}
	if issuer == nil {
		issuer = template
		issuerKey = priv
	}

	derBytes, err := CreateCertificate(rand.Reader, template, issuer, priv.Public(), issuerKey)
	if err != nil {
		return nil, nil, err
	}
	cert, err := ParseCertificate(derBytes)
	if err != nil {
		return nil, nil, err
	}

	return cert, priv, nil
}

func TestPathologicalChain(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping generation of a long chain of certificates in short mode")
	}

	// Build a chain where all intermediates share the same subject, to hit the
	// path building worst behavior.
	roots, intermediates := NewCertPool(), NewCertPool()

	parent, parentKey, err := generateCert("Root CA", true, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	roots.AddCert(parent)

	for i := 1; i < 100; i++ {
		parent, parentKey, err = generateCert("Intermediate CA", true, parent, parentKey)
		if err != nil {
			t.Fatal(err)
		}
		intermediates.AddCert(parent)
	}

	leaf, _, err := generateCert("Leaf", false, parent, parentKey)
	if err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	_, err = leaf.Verify(VerifyOptions{
		Roots:         roots,
		Intermediates: intermediates,
	})
	t.Logf("verification took %v", time.Since(start))

	if err == nil || !strings.Contains(err.Error(), "signature check attempts limit") {
		t.Errorf("expected verification to fail with a signature checks limit error; got %v", err)
	}
}

func TestLongChain(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping generation of a long chain of certificates in short mode")
	}

	roots, intermediates := NewCertPool(), NewCertPool()

	parent, parentKey, err := generateCert("Root CA", true, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	roots.AddCert(parent)

	for i := 1; i < 15; i++ {
		name := fmt.Sprintf("Intermediate CA #%d", i)
		parent, parentKey, err = generateCert(name, true, parent, parentKey)
		if err != nil {
			t.Fatal(err)
		}
		intermediates.AddCert(parent)
	}

	leaf, _, err := generateCert("Leaf", false, parent, parentKey)
	if err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	if _, err := leaf.Verify(VerifyOptions{
		Roots:         roots,
		Intermediates: intermediates,
	}); err != nil {
		t.Error(err)
	}
	t.Logf("verification took %v", time.Since(start))
}

func TestSystemRootsError(t *testing.T) {
	if runtime.GOOS == "windows" || runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		t.Skip("Windows and darwin do not use (or support) systemRoots")
	}

	defer func(oldSystemRoots *CertPool) { systemRoots = oldSystemRoots }(systemRootsPool())

	opts := VerifyOptions{
		Intermediates: NewCertPool(),
		DNSName:       "www.google.com",
		CurrentTime:   time.Unix(1677615892, 0),
	}

	if ok := opts.Intermediates.AppendCertsFromPEM([]byte(gtsIntermediate)); !ok {
		t.Fatalf("failed to parse intermediate")
	}

	leaf, err := certificateFromPEM(googleLeaf)
	if err != nil {
		t.Fatalf("failed to parse leaf: %v", err)
	}

	systemRoots = nil

	_, err = leaf.Verify(opts)
	if _, ok := err.(SystemRootsError); !ok {
		t.Errorf("error was not SystemRootsError: %v", err)
	}
}

func TestSystemRootsErrorUnwrap(t *testing.T) {
	var err1 = errors.New("err1")
	err := SystemRootsError{Err: err1}
	if !errors.Is(err, err1) {
		t.Error("errors.Is failed, wanted success")
	}
}

func macosMajorVersion(t *testing.T) (int, error) {
	cmd := testenv.Command(t, "sw_vers", "-productVersion")
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
			return 0, fmt.Errorf("%v: %v\n%s", cmd, err, ee.Stderr)
		}
		return 0, fmt.Errorf("%v: %v", cmd, err)
	}
	before, _, ok := strings.Cut(string(out), ".")
	major, err := strconv.Atoi(before)
	if !ok || err != nil {
		return 0, fmt.Errorf("%v: unexpected output: %q", cmd, out)
	}

	return major, nil
}

func TestIssue51759(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("only affects darwin")
	}

	testenv.MustHaveExecPath(t, "sw_vers")
	if vers, err := macosMajorVersion(t); err != nil {
		if builder := testenv.Builder(); builder != "" {
			t.Fatalf("unable to determine macOS version: %s", err)
		} else {
			t.Skip("unable to determine macOS version")
		}
	} else if vers < 11 {
		t.Skip("behavior only enforced in macOS 11 and after")
	}

	// badCertData contains a cert that we parse as valid
	// but that macOS SecCertificateCreateWithData rejects.
	const badCertData = "0\x82\x01U0\x82\x01\a\xa0\x03\x02\x01\x02\x02\x01\x020\x05\x06\x03+ep0R1P0N\x06\x03U\x04\x03\x13Gderpkey8dc58100b2493614ee1692831a461f3f4dd3f9b3b088e244f887f81b4906ac260\x1e\x17\r220112235755Z\x17\r220313235755Z0R1P0N\x06\x03U\x04\x03\x13Gderpkey8dc58100b2493614ee1692831a461f3f4dd3f9b3b088e244f887f81b4906ac260*0\x05\x06\x03+ep\x03!\x00bA\xd8e\xadW\xcb\xefZ\x89\xb5\"\x1eR\x9d\xba\x0e:\x1042Q@\u007f\xbd\xfb{ks\x04\xd1£\x020\x000\x05\x06\x03+ep\x03A\x00[\xa7\x06y\x86(\x94\x97\x9eLwA\x00\x01x\xaa\xbc\xbd Ê]\n(΅!ف0\xf5\x9a%I\x19<\xffo\xf1\xeaaf@\xb1\xa7\xaf\xfd\xe9R\xc7\x0f\x8d&\xd5\xfc\x0f;Ϙ\x82\x84a\xbc\r"
	badCert, err := ParseCertificate([]byte(badCertData))
	if err != nil {
		t.Fatal(err)
	}

	t.Run("leaf", func(t *testing.T) {
		opts := VerifyOptions{}
		expectedErr := "invalid leaf certificate"
		_, err = badCert.Verify(opts)
		if err == nil || err.Error() != expectedErr {
			t.Fatalf("unexpected error: want %q, got %q", expectedErr, err)
		}
	})

	goodCert, err := certificateFromPEM(googleLeaf)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("intermediate", func(t *testing.T) {
		opts := VerifyOptions{
			Intermediates: NewCertPool(),
		}
		opts.Intermediates.AddCert(badCert)
		expectedErr := "SecCertificateCreateWithData: invalid certificate"
		_, err = goodCert.Verify(opts)
		if err == nil || err.Error() != expectedErr {
			t.Fatalf("unexpected error: want %q, got %q", expectedErr, err)
		}
	})
}

type trustGraphEdge struct {
	Issuer         string
	Subject        string
	Type           int
	MutateTemplate func(*Certificate)
	Constraint     func([]*Certificate) error
}

type rootDescription struct {
	Subject        string
	MutateTemplate func(*Certificate)
	Constraint     func([]*Certificate) error
}

type trustGraphDescription struct {
	Roots []rootDescription
	Leaf  string
	Graph []trustGraphEdge
}

func genCertEdge(t *testing.T, subject string, key crypto.Signer, mutateTmpl func(*Certificate), certType int, issuer *Certificate, signer crypto.Signer) *Certificate {
	t.Helper()

	serial, err := rand.Int(rand.Reader, big.NewInt(100))
	if err != nil {
		t.Fatalf("failed to generate test serial: %s", err)
	}
	tmpl := &Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{CommonName: subject},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
	}
	if certType == rootCertificate || certType == intermediateCertificate {
		tmpl.IsCA, tmpl.BasicConstraintsValid = true, true
		tmpl.KeyUsage = KeyUsageCertSign
	} else if certType == leafCertificate {
		tmpl.DNSNames = []string{"localhost"}
	}
	if mutateTmpl != nil {
		mutateTmpl(tmpl)
	}

	if certType == rootCertificate {
		issuer = tmpl
		signer = key
	}

	d, err := CreateCertificate(rand.Reader, tmpl, issuer, key.Public(), signer)
	if err != nil {
		t.Fatalf("failed to generate test cert: %s", err)
	}
	c, err := ParseCertificate(d)
	if err != nil {
		t.Fatalf("failed to parse test cert: %s", err)
	}
	return c
}

func buildTrustGraph(t *testing.T, d trustGraphDescription) (*CertPool, *CertPool, *Certificate) {
	t.Helper()

	certs := map[string]*Certificate{}
	keys := map[string]crypto.Signer{}
	rootPool := NewCertPool()
	for _, r := range d.Roots {
		k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		if err != nil {
			t.Fatalf("failed to generate test key: %s", err)
		}
		root := genCertEdge(t, r.Subject, k, r.MutateTemplate, rootCertificate, nil, nil)
		if r.Constraint != nil {
			rootPool.AddCertWithConstraint(root, r.Constraint)
		} else {
			rootPool.AddCert(root)
		}
		certs[r.Subject] = root
		keys[r.Subject] = k
	}

	intermediatePool := NewCertPool()
	var leaf *Certificate
	for _, e := range d.Graph {
		issuerCert, ok := certs[e.Issuer]
		if !ok {
			t.Fatalf("unknown issuer %s", e.Issuer)
		}
		issuerKey, ok := keys[e.Issuer]
		if !ok {
			t.Fatalf("unknown issuer %s", e.Issuer)
		}

		k, ok := keys[e.Subject]
		if !ok {
			var err error
			k, err = ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
			if err != nil {
				t.Fatalf("failed to generate test key: %s", err)
			}
			keys[e.Subject] = k
		}
		cert := genCertEdge(t, e.Subject, k, e.MutateTemplate, e.Type, issuerCert, issuerKey)
		certs[e.Subject] = cert
		if e.Subject == d.Leaf {
			leaf = cert
		} else {
			if e.Constraint != nil {
				intermediatePool.AddCertWithConstraint(cert, e.Constraint)
			} else {
				intermediatePool.AddCert(cert)
			}
		}
	}

	return rootPool, intermediatePool, leaf
}

func chainsToStrings(chains [][]*Certificate) []string {
	chainStrings := []string{}
	for _, chain := range chains {
		names := []string{}
		for _, c := range chain {
			names = append(names, c.Subject.String())
		}
		chainStrings = append(chainStrings, strings.Join(names, " -> "))
	}
	slices.Sort(chainStrings)
	return chainStrings
}

func TestPathBuilding(t *testing.T) {
	tests := []struct {
		name           string
		graph          trustGraphDescription
		expectedChains []string
		expectedErr    string
	}{
		{
			// Build the following graph from RFC 4158, figure 7 (note that in this graph edges represent
			// certificates where the parent is the issuer and the child is the subject.) For the certificate
			// C->B, use an unsupported ExtKeyUsage (in this case ExtKeyUsageCodeSigning) which invalidates
			// the path Trust Anchor -> C -> B -> EE. The remaining valid paths should be:
			//   * Trust Anchor -> A -> B -> EE
			//   * Trust Anchor -> C -> A -> B -> EE
			//
			//     +---------+
			//     |  Trust  |
			//     | Anchor  |
			//     +---------+
			//      |       |
			//      v       v
			//   +---+    +---+
			//   | A |<-->| C |
			//   +---+    +---+
			//    |         |
			//    |  +---+  |
			//    +->| B |<-+
			//       +---+
			//         |
			//         v
			//       +----+
			//       | EE |
			//       +----+
			name: "bad EKU",
			graph: trustGraphDescription{
				Roots: []rootDescription{{Subject: "root"}},
				Leaf:  "leaf",
				Graph: []trustGraphEdge{
					{
						Issuer:  "root",
						Subject: "inter a",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "root",
						Subject: "inter c",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter c",
						Subject: "inter a",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter a",
						Subject: "inter c",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter c",
						Subject: "inter b",
						Type:    intermediateCertificate,
						MutateTemplate: func(t *Certificate) {
							t.ExtKeyUsage = []ExtKeyUsage{ExtKeyUsageCodeSigning}
						},
					},
					{
						Issuer:  "inter a",
						Subject: "inter b",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter b",
						Subject: "leaf",
						Type:    leafCertificate,
					},
				},
			},
			expectedChains: []string{
				"CN=leaf -> CN=inter b -> CN=inter a -> CN=inter c -> CN=root",
				"CN=leaf -> CN=inter b -> CN=inter a -> CN=root",
			},
		},
		{
			// Build the following graph from RFC 4158, figure 7 (note that in this graph edges represent
			// certificates where the parent is the issuer and the child is the subject.) For the certificate
			// C->B, use a unconstrained SAN which invalidates the path Trust Anchor -> C -> B -> EE. The
			// remaining valid paths should be:
			//   * Trust Anchor -> A -> B -> EE
			//   * Trust Anchor -> C -> A -> B -> EE
			//
			//     +---------+
			//     |  Trust  |
			//     | Anchor  |
			//     +---------+
			//      |       |
			//      v       v
			//   +---+    +---+
			//   | A |<-->| C |
			//   +---+    +---+
			//    |         |
			//    |  +---+  |
			//    +->| B |<-+
			//       +---+
			//         |
			//         v
			//       +----+
			//       | EE |
			//       +----+
			name: "bad EKU",
			graph: trustGraphDescription{
				Roots: []rootDescription{{Subject: "root"}},
				Leaf:  "leaf",
				Graph: []trustGraphEdge{
					{
						Issuer:  "root",
						Subject: "inter a",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "root",
						Subject: "inter c",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter c",
						Subject: "inter a",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter a",
						Subject: "inter c",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter c",
						Subject: "inter b",
						Type:    intermediateCertificate,
						MutateTemplate: func(t *Certificate) {
							t.PermittedDNSDomains = []string{"good"}
							t.DNSNames = []string{"bad"}
						},
					},
					{
						Issuer:  "inter a",
						Subject: "inter b",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter b",
						Subject: "leaf",
						Type:    leafCertificate,
					},
				},
			},
			expectedChains: []string{
				"CN=leaf -> CN=inter b -> CN=inter a -> CN=inter c -> CN=root",
				"CN=leaf -> CN=inter b -> CN=inter a -> CN=root",
			},
		},
		{
			// Build the following graph, we should find both paths:
			//   * Trust Anchor -> A -> C -> EE
			//   * Trust Anchor -> A -> B -> C -> EE
			//
			//	       +---------+
			//	       |  Trust  |
			//	       | Anchor  |
			//	       +---------+
			//	            |
			//	            v
			//	          +---+
			//	          | A |
			//	          +---+
			//	           | |
			//	           | +----+
			//	           |      v
			//	           |    +---+
			//	           |    | B |
			//	           |    +---+
			//	           |      |
			//	           |  +---v
			//	           v  v
			//            +---+
			//            | C |
			//            +---+
			//              |
			//              v
			//            +----+
			//            | EE |
			//            +----+
			name: "all paths",
			graph: trustGraphDescription{
				Roots: []rootDescription{{Subject: "root"}},
				Leaf:  "leaf",
				Graph: []trustGraphEdge{
					{
						Issuer:  "root",
						Subject: "inter a",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter a",
						Subject: "inter b",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter a",
						Subject: "inter c",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter b",
						Subject: "inter c",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter c",
						Subject: "leaf",
						Type:    leafCertificate,
					},
				},
			},
			expectedChains: []string{
				"CN=leaf -> CN=inter c -> CN=inter a -> CN=root",
				"CN=leaf -> CN=inter c -> CN=inter b -> CN=inter a -> CN=root",
			},
		},
		{
			// Build the following graph, which contains a cross-signature loop
			// (A and C cross sign each other). Paths that include the A -> C -> A
			// (and vice versa) loop should be ignored, resulting in the paths:
			//   * Trust Anchor -> A -> B -> EE
			//   * Trust Anchor -> C -> B -> EE
			//   * Trust Anchor -> A -> C -> B -> EE
			//   * Trust Anchor -> C -> A -> B -> EE
			//
			//     +---------+
			//     |  Trust  |
			//     | Anchor  |
			//     +---------+
			//      |       |
			//      v       v
			//   +---+    +---+
			//   | A |<-->| C |
			//   +---+    +---+
			//    |         |
			//    |  +---+  |
			//    +->| B |<-+
			//       +---+
			//         |
			//         v
			//       +----+
			//       | EE |
			//       +----+
			name: "ignore cross-sig loops",
			graph: trustGraphDescription{
				Roots: []rootDescription{{Subject: "root"}},
				Leaf:  "leaf",
				Graph: []trustGraphEdge{
					{
						Issuer:  "root",
						Subject: "inter a",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "root",
						Subject: "inter c",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter c",
						Subject: "inter a",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter a",
						Subject: "inter c",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter c",
						Subject: "inter b",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter a",
						Subject: "inter b",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter b",
						Subject: "leaf",
						Type:    leafCertificate,
					},
				},
			},
			expectedChains: []string{
				"CN=leaf -> CN=inter b -> CN=inter a -> CN=inter c -> CN=root",
				"CN=leaf -> CN=inter b -> CN=inter a -> CN=root",
				"CN=leaf -> CN=inter b -> CN=inter c -> CN=inter a -> CN=root",
				"CN=leaf -> CN=inter b -> CN=inter c -> CN=root",
			},
		},
		{
			// Build a simple two node graph, where the leaf is directly issued from
			// the root and both certificates have matching subject and public key, but
			// the leaf has SANs.
			name: "leaf with same subject, key, as parent but with SAN",
			graph: trustGraphDescription{
				Roots: []rootDescription{{Subject: "root"}},
				Leaf:  "root",
				Graph: []trustGraphEdge{
					{
						Issuer:  "root",
						Subject: "root",
						Type:    leafCertificate,
						MutateTemplate: func(c *Certificate) {
							c.DNSNames = []string{"localhost"}
						},
					},
				},
			},
			expectedChains: []string{
				"CN=root -> CN=root",
			},
		},
		{
			// Build a basic graph with two paths from leaf to root, but the path passing
			// through C should be ignored, because it has invalid EKU nesting.
			name: "ignore invalid EKU path",
			graph: trustGraphDescription{
				Roots: []rootDescription{{Subject: "root"}},
				Leaf:  "leaf",
				Graph: []trustGraphEdge{
					{
						Issuer:  "root",
						Subject: "inter a",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "root",
						Subject: "inter c",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter c",
						Subject: "inter b",
						Type:    intermediateCertificate,
						MutateTemplate: func(t *Certificate) {
							t.ExtKeyUsage = []ExtKeyUsage{ExtKeyUsageCodeSigning}
						},
					},
					{
						Issuer:  "inter a",
						Subject: "inter b",
						Type:    intermediateCertificate,
						MutateTemplate: func(t *Certificate) {
							t.ExtKeyUsage = []ExtKeyUsage{ExtKeyUsageServerAuth}
						},
					},
					{
						Issuer:  "inter b",
						Subject: "leaf",
						Type:    leafCertificate,
						MutateTemplate: func(t *Certificate) {
							t.ExtKeyUsage = []ExtKeyUsage{ExtKeyUsageServerAuth}
						},
					},
				},
			},
			expectedChains: []string{
				"CN=leaf -> CN=inter b -> CN=inter a -> CN=root",
			},
		},
		{
			// A name constraint on the root should apply to any names that appear
			// on the intermediate, meaning there is no valid chain.
			name: "constrained root, invalid intermediate",
			graph: trustGraphDescription{
				Roots: []rootDescription{
					{
						Subject: "root",
						MutateTemplate: func(t *Certificate) {
							t.PermittedDNSDomains = []string{"example.com"}
						},
					},
				},
				Leaf: "leaf",
				Graph: []trustGraphEdge{
					{
						Issuer:  "root",
						Subject: "inter",
						Type:    intermediateCertificate,
						MutateTemplate: func(t *Certificate) {
							t.DNSNames = []string{"beep.com"}
						},
					},
					{
						Issuer:  "inter",
						Subject: "leaf",
						Type:    leafCertificate,
						MutateTemplate: func(t *Certificate) {
							t.DNSNames = []string{"www.example.com"}
						},
					},
				},
			},
			expectedErr: "x509: a root or intermediate certificate is not authorized to sign for this name: DNS name \"beep.com\" is not permitted by any constraint",
		},
		{
			// A name constraint on the intermediate does not apply to the intermediate
			// itself, so this is a valid chain.
			name: "constrained intermediate, non-matching SAN",
			graph: trustGraphDescription{
				Roots: []rootDescription{{Subject: "root"}},
				Leaf:  "leaf",
				Graph: []trustGraphEdge{
					{
						Issuer:  "root",
						Subject: "inter",
						Type:    intermediateCertificate,
						MutateTemplate: func(t *Certificate) {
							t.DNSNames = []string{"beep.com"}
							t.PermittedDNSDomains = []string{"example.com"}
						},
					},
					{
						Issuer:  "inter",
						Subject: "leaf",
						Type:    leafCertificate,
						MutateTemplate: func(t *Certificate) {
							t.DNSNames = []string{"www.example.com"}
						},
					},
				},
			},
			expectedChains: []string{"CN=leaf -> CN=inter -> CN=root"},
		},
		{
			// A code constraint on the root, applying to one of two intermediates in the graph, should
			// result in only one valid chain.
			name: "code constrained root, two paths, one valid",
			graph: trustGraphDescription{
				Roots: []rootDescription{{Subject: "root", Constraint: func(chain []*Certificate) error {
					for _, c := range chain {
						if c.Subject.CommonName == "inter a" {
							return errors.New("bad")
						}
					}
					return nil
				}}},
				Leaf: "leaf",
				Graph: []trustGraphEdge{
					{
						Issuer:  "root",
						Subject: "inter a",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "root",
						Subject: "inter b",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter a",
						Subject: "inter c",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter b",
						Subject: "inter c",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter c",
						Subject: "leaf",
						Type:    leafCertificate,
					},
				},
			},
			expectedChains: []string{"CN=leaf -> CN=inter c -> CN=inter b -> CN=root"},
		},
		{
			// A code constraint on the root, applying to the only path, should result in an error.
			name: "code constrained root, one invalid path",
			graph: trustGraphDescription{
				Roots: []rootDescription{{Subject: "root", Constraint: func(chain []*Certificate) error {
					for _, c := range chain {
						if c.Subject.CommonName == "leaf" {
							return errors.New("bad")
						}
					}
					return nil
				}}},
				Leaf: "leaf",
				Graph: []trustGraphEdge{
					{
						Issuer:  "root",
						Subject: "inter",
						Type:    intermediateCertificate,
					},
					{
						Issuer:  "inter",
						Subject: "leaf",
						Type:    leafCertificate,
					},
				},
			},
			expectedErr: "x509: certificate signed by unknown authority (possibly because of \"bad\" while trying to verify candidate authority certificate \"root\")",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			roots, intermediates, leaf := buildTrustGraph(t, tc.graph)
			chains, err := leaf.Verify(VerifyOptions{
				Roots:         roots,
				Intermediates: intermediates,
			})
			if err != nil && err.Error() != tc.expectedErr {
				t.Fatalf("unexpected error: got %q, want %q", err, tc.expectedErr)
			}
			if len(tc.expectedChains) == 0 {
				return
			}
			gotChains := chainsToStrings(chains)
			if !slices.Equal(gotChains, tc.expectedChains) {
				t.Errorf("unexpected chains returned:\ngot:\n\t%s\nwant:\n\t%s", strings.Join(gotChains, "\n\t"), strings.Join(tc.expectedChains, "\n\t"))
			}
		})
	}
}

func TestEKUEnforcement(t *testing.T) {
	type ekuDescs struct {
		EKUs    []ExtKeyUsage
		Unknown []asn1.ObjectIdentifier
	}
	tests := []struct {
		name       string
		root       ekuDescs
		inters     []ekuDescs
		leaf       ekuDescs
		verifyEKUs []ExtKeyUsage
		err        string
	}{
		{
			name:       "valid, full chain",
			root:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth}},
			inters:     []ekuDescs{ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth}}},
			leaf:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth}},
			verifyEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth},
		},
		{
			name:       "valid, only leaf has EKU",
			root:       ekuDescs{},
			inters:     []ekuDescs{ekuDescs{}},
			leaf:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth}},
			verifyEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth},
		},
		{
			name:       "invalid, serverAuth not nested",
			root:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageClientAuth}},
			inters:     []ekuDescs{ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth, ExtKeyUsageClientAuth}}},
			leaf:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth, ExtKeyUsageClientAuth}},
			verifyEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth},
			err:        "x509: certificate specifies an incompatible key usage",
		},
		{
			name:       "valid, two EKUs, one path",
			root:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth}},
			inters:     []ekuDescs{ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth, ExtKeyUsageClientAuth}}},
			leaf:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth, ExtKeyUsageClientAuth}},
			verifyEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth, ExtKeyUsageClientAuth},
		},
		{
			name: "invalid, ladder",
			root: ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth}},
			inters: []ekuDescs{
				ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth, ExtKeyUsageClientAuth}},
				ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageClientAuth}},
				ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth, ExtKeyUsageClientAuth}},
				ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth}},
			},
			leaf:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth}},
			verifyEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth, ExtKeyUsageClientAuth},
			err:        "x509: certificate specifies an incompatible key usage",
		},
		{
			name:       "valid, intermediate has no EKU",
			root:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth}},
			inters:     []ekuDescs{ekuDescs{}},
			leaf:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth}},
			verifyEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth},
		},
		{
			name:       "invalid, intermediate has no EKU and no nested path",
			root:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageClientAuth}},
			inters:     []ekuDescs{ekuDescs{}},
			leaf:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth}},
			verifyEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth, ExtKeyUsageClientAuth},
			err:        "x509: certificate specifies an incompatible key usage",
		},
		{
			name:       "invalid, intermediate has unknown EKU",
			root:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth}},
			inters:     []ekuDescs{ekuDescs{Unknown: []asn1.ObjectIdentifier{{1, 2, 3}}}},
			leaf:       ekuDescs{EKUs: []ExtKeyUsage{ExtKeyUsageServerAuth}},
			verifyEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth},
			err:        "x509: certificate specifies an incompatible key usage",
		},
	}

	k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("failed to generate test key: %s", err)
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rootPool := NewCertPool()
			root := genCertEdge(t, "root", k, func(c *Certificate) {
				c.ExtKeyUsage = tc.root.EKUs
				c.UnknownExtKeyUsage = tc.root.Unknown
			}, rootCertificate, nil, k)
			rootPool.AddCert(root)

			parent := root
			interPool := NewCertPool()
			for i, interEKUs := range tc.inters {
				inter := genCertEdge(t, fmt.Sprintf("inter %d", i), k, func(c *Certificate) {
					c.ExtKeyUsage = interEKUs.EKUs
					c.UnknownExtKeyUsage = interEKUs.Unknown
				}, intermediateCertificate, parent, k)
				interPool.AddCert(inter)
				parent = inter
			}

			leaf := genCertEdge(t, "leaf", k, func(c *Certificate) {
				c.ExtKeyUsage = tc.leaf.EKUs
				c.UnknownExtKeyUsage = tc.leaf.Unknown
			}, intermediateCertificate, parent, k)

			_, err := leaf.Verify(VerifyOptions{Roots: rootPool, Intermediates: interPool, KeyUsages: tc.verifyEKUs})
			if err == nil && tc.err != "" {
				t.Errorf("expected error")
			} else if err != nil && err.Error() != tc.err {
				t.Errorf("unexpected error: want %q, got %q", err.Error(), tc.err)
			}
		})
	}
}

func TestVerifyEKURootAsLeaf(t *testing.T) {
	k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("failed to generate key: %s", err)
	}

	for _, tc := range []struct {
		rootEKUs   []ExtKeyUsage
		verifyEKUs []ExtKeyUsage
		succeed    bool
	}{
		{
			verifyEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth},
			succeed:    true,
		},
		{
			rootEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth},
			succeed:  true,
		},
		{
			rootEKUs:   []ExtKeyUsage{ExtKeyUsageServerAuth},
			verifyEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth},
			succeed:    true,
		},
		{
			rootEKUs:   []ExtKeyUsage{ExtKeyUsageServerAuth},
			verifyEKUs: []ExtKeyUsage{ExtKeyUsageAny},
			succeed:    true,
		},
		{
			rootEKUs:   []ExtKeyUsage{ExtKeyUsageAny},
			verifyEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth},
			succeed:    true,
		},
		{
			rootEKUs:   []ExtKeyUsage{ExtKeyUsageClientAuth},
			verifyEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth},
			succeed:    false,
		},
	} {
		t.Run(fmt.Sprintf("root EKUs %#v, verify EKUs %#v", tc.rootEKUs, tc.verifyEKUs), func(t *testing.T) {
			tmpl := &Certificate{
				SerialNumber: big.NewInt(1),
				Subject:      pkix.Name{CommonName: "root"},
				NotBefore:    time.Now().Add(-time.Hour),
				NotAfter:     time.Now().Add(time.Hour),
				DNSNames:     []string{"localhost"},
				ExtKeyUsage:  tc.rootEKUs,
			}
			rootDER, err := CreateCertificate(rand.Reader, tmpl, tmpl, k.Public(), k)
			if err != nil {
				t.Fatalf("failed to create certificate: %s", err)
			}
			root, err := ParseCertificate(rootDER)
			if err != nil {
				t.Fatalf("failed to parse certificate: %s", err)
			}
			roots := NewCertPool()
			roots.AddCert(root)

			_, err = root.Verify(VerifyOptions{Roots: roots, KeyUsages: tc.verifyEKUs})
			if err == nil && !tc.succeed {
				t.Error("verification succeed")
			} else if err != nil && tc.succeed {
				t.Errorf("verification failed: %q", err)
			}
		})
	}

}

func TestVerifyNilPubKey(t *testing.T) {
	c := &Certificate{
		RawIssuer:      []byte{1, 2, 3},
		AuthorityKeyId: []byte{1, 2, 3},
	}
	opts := &VerifyOptions{}
	opts.Roots = NewCertPool()
	r := &Certificate{
		RawSubject:   []byte{1, 2, 3},
		SubjectKeyId: []byte{1, 2, 3},
	}
	opts.Roots.AddCert(r)

	_, err := c.buildChains([]*Certificate{r}, nil, opts)
	if _, ok := err.(UnknownAuthorityError); !ok {
		t.Fatalf("buildChains returned unexpected error, got: %v, want %v", err, UnknownAuthorityError{})
	}
}

func TestVerifyBareWildcard(t *testing.T) {
	k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("failed to generate key: %s", err)
	}
	tmpl := &Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "test"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		DNSNames:     []string{"*"},
	}
	cDER, err := CreateCertificate(rand.Reader, tmpl, tmpl, k.Public(), k)
	if err != nil {
		t.Fatalf("failed to create certificate: %s", err)
	}
	c, err := ParseCertificate(cDER)
	if err != nil {
		t.Fatalf("failed to parse certificate: %s", err)
	}

	if err := c.VerifyHostname("label"); err == nil {
		t.Fatalf("VerifyHostname unexpected success with bare wildcard SAN")
	}
}
