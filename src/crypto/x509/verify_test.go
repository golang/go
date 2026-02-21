// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"crypto"
	"crypto/dsa"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/pem"
	"errors"
	"fmt"
	"internal/testenv"
	"log"
	"math/big"
	"net"
	"os"
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
		name:        "TooManyDNS",
		leaf:        generatePEMCertWithRepeatSAN(1677615892, 200, "fake.dns"),
		roots:       []string{generatePEMCertWithRepeatSAN(1677615892, 200, "fake.dns")},
		currentTime: 1677615892,
		dnsName:     "www.example.com",
		systemSkip:  true, // does not chain to a system root

		errorCallback: expectHostnameError("certificate is valid for 200 names, but none matched"),
	},
	{
		name:        "TooManyIPs",
		leaf:        generatePEMCertWithRepeatSAN(1677615892, 150, "4.3.2.1"),
		roots:       []string{generatePEMCertWithRepeatSAN(1677615892, 150, "4.3.2.1")},
		currentTime: 1677615892,
		dnsName:     "1.2.3.4",
		systemSkip:  true, // does not chain to a system root

		errorCallback: expectHostnameError("certificate is valid for 150 IP SANs, but none matched"),
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
		currentTime:   1524771953,
		dnsName:       "udctest.ads.vt.edu",

		expectedChains: [][]string{
			{
				"udctest.ads.vt.edu",
				"Virginia Tech Global Qualified Server CA",
				"Trusted Root CA SHA256 G2",
				"GlobalSign",
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
	{
		// Test if permitted dirname constraint works.
		name:          "DirnameParse",
		leaf:          dirNameConstraintLeafCA_permitted_ok,
		intermediates: []string{dirNameConstraintSubCA_permitted_ok},
		roots:         []string{dirNameConstraintRootCA_permitted_ok},
		currentTime:   1600000000,
		systemSkip:    true,

		expectedChains: [][]string{
			{"Leaf", "SubCA", "RootCA"},
		},
	},
	{
		// Test if permitted RDN match dirname constraint.
		name:          "DirnamePermit",
		leaf:          dirNameConstraintLeafCA_permitted_dirname_multirdn,
		intermediates: []string{dirNameConstraintSubCA_permitted_dirname_multirdn},
		roots:         []string{dirNameConstraintRootCA_permitted_dirname_multirdn},
		currentTime:   1600000000,
		systemSkip:    true,

		expectedChains: [][]string{
			{"Leaf", "SubCA", "RootCA"},
		},
	},
	{
		// Test if RootCA dirname constraint violation is ignored.
		name:          "DirnamePermitNotValidateRoot",
		leaf:          dirNameConstraintLeafCA_notpermitted_rootca,
		intermediates: []string{dirNameConstraintSubCA_notpermitted_rootca},
		roots:         []string{dirNameConstraintRootCA_notpermitted_rootca},
		currentTime:   1600000000,
		systemSkip:    true,

		expectedChains: [][]string{
			{"Leaf", "SubCA", "RootCA"},
		},
	},
	{
		// Test if SubCA dirname constraint violation (ST missing) fails.
		name:          "DirNamePermitSubCAViolationMissing",
		leaf:          dirNameConstraintLeafCA_notpermitted_subca_missing,
		intermediates: []string{dirNameConstraintSubCA_notpermitted_subca_missing},
		roots:         []string{dirNameConstraintRootCA_notpermitted_subca_missing},
		currentTime:   1600000000,
		systemSkip:    true,

		errorCallback: expectNameConstraintsError,
	},
	{
		// Test if SubCA dirname constraint violation (ST changed) fails.
		name:          "DirNamePermitSubCAViolationDiffer",
		leaf:          dirNameConstraintLeafCA_notpermitted_subca_changed,
		intermediates: []string{dirNameConstraintSubCA_notpermitted_subca_changed},
		roots:         []string{dirNameConstraintRootCA_notpermitted_subca_changed},
		currentTime:   1600000000,
		systemSkip:    true,

		errorCallback: expectNameConstraintsError,
	},
	{
		// Test if leaf dirname constraint violation (ST missing) fails.
		name:          "DirNamePermitCertViolationMissing",
		leaf:          dirNameConstraintLeafCA_notpermitted_leaf_missing,
		intermediates: []string{dirNameConstraintSubCA_notpermitted_leaf_missing},
		roots:         []string{dirNameConstraintRootCA_notpermitted_leaf_missing},
		currentTime:   1600000000,
		systemSkip:    true,

		errorCallback: expectNameConstraintsError,
	},
	{
		// Test if leaf dirname constraint violation (ST changed) fails.
		name:          "DirNamePermitCertViolationDiffer",
		leaf:          dirNameConstraintLeafCA_notpermitted_leaf_changed,
		intermediates: []string{dirNameConstraintSubCA_notpermitted_leaf_changed},
		roots:         []string{dirNameConstraintRootCA_notpermitted_leaf_changed},
		currentTime:   1600000000,
		systemSkip:    true,

		errorCallback: expectNameConstraintsError,
	},
	{
		// Test if excluded dirname works.
		name:          "DirnameExclude",
		leaf:          dirNameConstraintLeafCA_excluded_ok,
		intermediates: []string{dirNameConstraintSubCA_excluded_ok},
		roots:         []string{dirNameConstraintRootCA_excluded_ok},
		currentTime:   1600000000,
		systemSkip:    true,

		expectedChains: [][]string{
			{"Leaf", "SubCA", "RootCA"},
		},
	},
	{
		// Test if RootCA using excluded dirname is ignored.
		name:          "DirnameExcludeNotValidateRoot",
		leaf:          dirNameConstraintLeafCA_excluded_rootca,
		intermediates: []string{dirNameConstraintSubCA_excluded_rootca},
		roots:         []string{dirNameConstraintRootCA_excluded_rootca},
		currentTime:   1600000000,
		systemSkip:    true,

		expectedChains: [][]string{
			{"Leaf", "SubCA", "RootCA"},
		},
	},
	{
		// Test if SubCA using excluded dirname fails.
		name:          "DirnameExcludeSubCA",
		leaf:          dirNameConstraintLeafCA_excluded_subca,
		intermediates: []string{dirNameConstraintSubCA_excluded_subca},
		roots:         []string{dirNameConstraintRootCA_excluded_subca},
		currentTime:   1600000000,
		systemSkip:    true,

		errorCallback: expectNameConstraintsError,
	},
	{
		// Test if leaf using excluded dirname fails.
		name:          "DirnameExcludeCert",
		leaf:          dirNameConstraintLeafCA_excluded_leaf,
		intermediates: []string{dirNameConstraintSubCA_excluded_leaf},
		roots:         []string{dirNameConstraintRootCA_excluded_leaf},
		currentTime:   1600000000,
		systemSkip:    true,

		errorCallback: expectNameConstraintsError,
	},
	{
		// Test if permitted and excluded dirname works together.
		name:          "DirnamePermitExclude",
		leaf:          dirNameConstraintLeafCA_permitted_excluded_OK,
		intermediates: []string{dirNameConstraintSubCA_permitted_excluded_OK},
		roots:         []string{dirNameConstraintRootCA_permitted_excluded_OK},
		currentTime:   1600000000,
		systemSkip:    true,

		expectedChains: [][]string{
			{"Leaf", "SubCA", "RootCA"},
		},
	},
	{
		// Test if RootCA using dirname both in excluded and permitted works.
		name:          "DirnamePermitExcludeRoot",
		leaf:          dirNameConstraintLeafCA_permitted_excluded_rootca,
		intermediates: []string{dirNameConstraintSubCA_permitted_excluded_rootca},
		roots:         []string{dirNameConstraintRootCA_permitted_excluded_rootca},
		currentTime:   1600000000,
		systemSkip:    true,

		expectedChains: [][]string{
			{"Leaf", "SubCA", "RootCA"},
		},
	},
	{
		// Test if SubCA using dirname both in excluded and permitted fails.
		name:          "DirnamePermitExcludeSubCA",
		leaf:          dirNameConstraintLeafCA_permitted_excluded_subca,
		intermediates: []string{dirNameConstraintSubCA_permitted_excluded_subca},
		roots:         []string{dirNameConstraintRootCA_permitted_excluded_subca},
		currentTime:   1600000000,
		systemSkip:    true,

		errorCallback: expectNameConstraintsError,
	},
	{
		// Test if leaf using dirname both in excluded and permitted fails.
		name:          "DirnamePermitExcludeCert",
		leaf:          dirNameConstraintLeafCA_permitted_excluded_leaf,
		intermediates: []string{dirNameConstraintSubCA_permitted_excluded_leaf},
		roots:         []string{dirNameConstraintRootCA_permitted_excluded_leaf},
		currentTime:   1600000000,
		systemSkip:    true,

		errorCallback: expectNameConstraintsError,
	},
	{
		// Test if SubCA can restrict a constraint.
		name:          "DirnamePermitRootExcludeSubCAOK",
		leaf:          dirNameConstraintLeafCA_subca_restr_ok,
		intermediates: []string{dirNameConstraintSubCA_subca_restr_ok},
		roots:         []string{dirNameConstraintRootCA_subca_restr_ok},
		currentTime:   1600000000,
		systemSkip:    true,

		expectedChains: [][]string{
			{"Leaf", "SubCA", "RootCA"},
		},
	},
	{
		// Test if SubCA can restrict a constraint.
		name:          "DirnamePermitRootExcludeSubCAFail",
		leaf:          dirNameConstraintLeafCA_subca_restr_fail,
		intermediates: []string{dirNameConstraintSubCA_subca_restr_fail},
		roots:         []string{dirNameConstraintRootCA_subca_restr_fail},
		currentTime:   1600000000,
		systemSkip:    true,

		errorCallback: expectNameConstraintsError,
	},
	{
		// Test if SubCA can relax a constraint.
		name:          "DirnameExcludeRootPermitSubCA",
		leaf:          dirNameConstraintLeafCA_subca_relax_fail,
		intermediates: []string{dirNameConstraintSubCA_subca_relax_fail},
		roots:         []string{dirNameConstraintRootCA_subca_relax_fail},
		currentTime:   1600000000,
		systemSkip:    true,

		errorCallback: expectNameConstraintsError,
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

func generatePEMCertWithRepeatSAN(currentTime int64, count int, san string) string {
	cert := Certificate{
		NotBefore: time.Unix(currentTime, 0),
		NotAfter:  time.Unix(currentTime, 0),
	}
	if ip := net.ParseIP(san); ip != nil {
		cert.IPAddresses = slices.Repeat([]net.IP{ip}, count)
	} else {
		cert.DNSNames = slices.Repeat([]string{san}, count)
	}
	privKey, err := rsa.GenerateKey(rand.Reader, 4096)
	if err != nil {
		log.Fatal(err)
	}
	certBytes, err := CreateCertificate(rand.Reader, &cert, &cert, &privKey.PublicKey, privKey)
	if err != nil {
		log.Fatal(err)
	}
	return string(pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: certBytes,
	}))
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
MIIG+jCCBOKgAwIBAgIQWj9gbtPPkZs65N6TKyutRjANBgkqhkiG9w0BAQsFADCB
yzELMAkGA1UEBhMCVVMxETAPBgNVBAgTCFZpcmdpbmlhMRMwEQYDVQQHEwpCbGFj
a3NidXJnMSMwIQYDVQQLExpHbG9iYWwgUXVhbGlmaWVkIFNlcnZlciBDQTE8MDoG
A1UEChMzVmlyZ2luaWEgUG9seXRlY2huaWMgSW5zdGl0dXRlIGFuZCBTdGF0ZSBV
bml2ZXJzaXR5MTEwLwYDVQQDEyhWaXJnaW5pYSBUZWNoIEdsb2JhbCBRdWFsaWZp
ZWQgU2VydmVyIENBMB4XDTE4MDQyNjE5NDU1M1oXDTE5MTIxMDAwMDAwMFowgZAx
CzAJBgNVBAYTAlVTMREwDwYDVQQIEwhWaXJnaW5pYTETMBEGA1UEBxMKQmxhY2tz
YnVyZzE8MDoGA1UEChMzVmlyZ2luaWEgUG9seXRlY2huaWMgSW5zdGl0dXRlIGFu
ZCBTdGF0ZSBVbml2ZXJzaXR5MRswGQYDVQQDExJ1ZGN0ZXN0LmFkcy52dC5lZHUw
ggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCcoVBeV3AzdSGMzRWH0tuM
VluEj+sq4r9PuLDBAdgjjHi4ED8npT2/fgOalswInXspRvFS+pkEwTrmeZ7HPzRJ
HUE5YlX5Nc6WI8ZXPVg5E6GyoMy6gNlALwqsIvDCvqxBMc39oG6yOuGmQXdF6s0N
BJMrXc4aPz60s4QMWNO2OHL0pmnZqE1TxYRBHUY/dk3cfsIepIDDuSxRsNE/P/MI
pxm/uVOyiLEnPmOMsL430SZ7nC8PxUMqya9ok6Zaf7k54g7JJXDjE96VMCjMszIv
Ud9qe1PbokTOxlG/4QW7Qm0dPbiJhTUuoBzVAxzlOOkSFdXqSYKjC9tFcbr8y+pT
AgMBAAGjggIRMIICDTCBtgYIKwYBBQUHAQEEgakwgaYwXwYIKwYBBQUHMAKGU2h0
dHA6Ly93d3cucGtpLnZ0LmVkdS9nbG9iYWxxdWFsaWZpZWRzZXJ2ZXIvY2FjZXJ0
L2dsb2JhbHF1YWxpZmllZHNlcnZlcl9zaGEyNTYuY3J0MEMGCCsGAQUFBzABhjdo
dHRwOi8vdnRjYS5wa2kudnQuZWR1OjgwODAvZWpiY2EvcHVibGljd2ViL3N0YXR1
cy9vY3NwMB0GA1UdDgQWBBSzDLXee0wbgXpVQxvBQCophQDZbTAMBgNVHRMBAf8E
AjAAMB8GA1UdIwQYMBaAFLxiYCfV4zVIF+lLq0Vq0Miod3GMMGoGA1UdIARjMGEw
DgYMKwYBBAG0aAUCAgIBMA4GDCsGAQQBtGgFAgIBATA/BgwrBgEEAbRoBQICAwEw
LzAtBggrBgEFBQcCARYhaHR0cDovL3d3dy5wa2kudnQuZWR1L2dsb2JhbC9jcHMv
MEoGA1UdHwRDMEEwP6A9oDuGOWh0dHA6Ly93d3cucGtpLnZ0LmVkdS9nbG9iYWxx
dWFsaWZpZWRzZXJ2ZXIvY3JsL2NhY3JsLmNybDAOBgNVHQ8BAf8EBAMCBeAwHQYD
VR0lBBYwFAYIKwYBBQUHAwIGCCsGAQUFBwMBMB0GA1UdEQQWMBSCEnVkY3Rlc3Qu
YWRzLnZ0LmVkdTANBgkqhkiG9w0BAQsFAAOCAgEAD79kuyZbwQJCSBOVq9lA0lj4
juHM7RMBfp2GuWvhk5F90OMKQCNdITva3oq4uQzt013TtwposYXq/d0Jobk6RHxj
OJzRZVvEPsXLvKm8oLhz7/qgI8gcVeJFR9WgdNhjN1upn++EnABHUdDR77fgixuH
FFwNC0WSZ6G0+WgYV7MKD4jYWh1DXEaJtQCN763IaWGxgvQaLUwS423xgwsx+8rw
hCRYns5u8myTbUlEu2b+GYimiogtDFMT01A7y88vKl9g+3bx42dJHQNNmSzmYPfs
IljtQbVwJIyNL/rwjoz7BTk8e9WY0qUK7ZYh+oGK8kla8yfPKtkvOJV29KdFKzTm
42kNm6cH+U5gGwEEg+Xj66Q2yFH5J9kAoBazTepgQ/13wwTY0mU9PtKVBtMH5Y/u
MoNVZz6p7vWWRrY5qSXIaW9qyF3bZnmPEHHYTplWsyAyh8blGlqPnpayDflPiQF/
9y37kax5yhT0zPZW1ZwIZ5hDTO7pu5i83bYh3pzhvJNHtv74Nn/SX1dTZrWBi/HG
OSWK3CLz8qAEBe72XGoBjBzuk9VQxg6k52qjxCyYf7CBSQpTZhsNMf0gzu+JNATc
b+XaOqJT6uI/RfqAJVe16ZeXZIFZgQlzIwRS9vobq9fqTIpH/QxqgXROGqAlbBVp
/ByH6FEe6+oH1UCklhg=
-----END CERTIFICATE-----`

var nameConstraintsIntermediate1 = `-----BEGIN CERTIFICATE-----
MIIHVTCCBj2gAwIBAgINAecHzcaPEeFvu7X4TTANBgkqhkiG9w0BAQsFADBjMQsw
CQYDVQQGEwJCRTEVMBMGA1UECxMMVHJ1c3RlZCBSb290MRkwFwYDVQQKExBHbG9i
YWxTaWduIG52LXNhMSIwIAYDVQQDExlUcnVzdGVkIFJvb3QgQ0EgU0hBMjU2IEcy
MB4XDTE3MTIwNjAwMDAwMFoXDTIyMTIwNjAwMDAwMFowgcsxCzAJBgNVBAYTAlVT
MREwDwYDVQQIEwhWaXJnaW5pYTETMBEGA1UEBxMKQmxhY2tzYnVyZzEjMCEGA1UE
CxMaR2xvYmFsIFF1YWxpZmllZCBTZXJ2ZXIgQ0ExPDA6BgNVBAoTM1Zpcmdpbmlh
IFBvbHl0ZWNobmljIEluc3RpdHV0ZSBhbmQgU3RhdGUgVW5pdmVyc2l0eTExMC8G
A1UEAxMoVmlyZ2luaWEgVGVjaCBHbG9iYWwgUXVhbGlmaWVkIFNlcnZlciBDQTCC
AiIwDQYJKoZIhvcNAQEBBQADggIPADCCAgoCggIBALgIZhEaptBWADBqdJ45ueFG
zMXaGHnzNxoxR1fQIaaRQNdCg4cw3A4dWKMeEgYLtsp65ai3Xfw62Qaus0+KJ3Rh
gV+rihqK81NUzkls78fJlADVDI4fCTlothsrE1CTOMiy97jKHai5mVTiWxmcxpmj
v7fm5Nhc+uHgh2hIz6npryq495mD51ZrUTIaqAQN6Pw/VHfAmR524vgriTOjtp1t
4lA9pXGWjF/vkhAKFFheOQSQ00rngo2wHgCqMla64UTN0oz70AsCYNZ3jDLx0kOP
0YmMR3Ih91VA63kLqPXA0R6yxmmhhxLZ5bcyAy1SLjr1N302MIxLM/pSy6aquEnb
ELhzqyp9yGgRyGJay96QH7c4RJY6gtcoPDbldDcHI9nXngdAL4DrZkJ9OkDkJLyq
G66WZTF5q4EIs6yMdrywz0x7QP+OXPJrjYpbeFs6tGZCFnWPFfmHCRJF8/unofYr
heq+9J7Jx3U55S/k57NXbAM1RAJOuMTlfn9Etf9Dpoac9poI4Liav6rBoUQk3N3J
WqnVHNx/NdCyJ1/6UbKMJUZsStAVglsi6lVPo289HHOE4f7iwl3SyekizVOp01wU
in3ycnbZB/rXmZbwapSxTTSBf0EIOr9i4EGfnnhCAVA9U5uLrI5OEB69IY8PNX00
71s3Z2a2fio5c8m3JkdrAgMBAAGjggKdMIICmTAOBgNVHQ8BAf8EBAMCAQYwHQYD
VR0lBBYwFAYIKwYBBQUHAwEGCCsGAQUFBwMCMBIGA1UdEwEB/wQIMAYBAf8CAQAw
HQYDVR0OBBYEFLxiYCfV4zVIF+lLq0Vq0Miod3GMMB8GA1UdIwQYMBaAFMhjmwhp
VMKYyNnN4zO3UF74yQGbMIGNBggrBgEFBQcBAQSBgDB+MDcGCCsGAQUFBzABhito
dHRwOi8vb2NzcDIuZ2xvYmFsc2lnbi5jb20vdHJ1c3Ryb290c2hhMmcyMEMGCCsG
AQUFBzAChjdodHRwOi8vc2VjdXJlLmdsb2JhbHNpZ24uY29tL2NhY2VydC90cnVz
dHJvb3RzaGEyZzIuY3J0MIHyBgNVHR4EgeowgeeggbIwCIEGdnQuZWR1MAmCB2Jl
di5uZXQwCoIIdmNvbS5lZHUwCIIGdnQuZWR1MAyCCnZ0Y2dpdC5jb20wd6R1MHMx
CzAJBgNVBAYTAlVTMREwDwYDVQQIEwhWaXJnaW5pYTETMBEGA1UEBxMKQmxhY2tz
YnVyZzE8MDoGA1UEChMzVmlyZ2luaWEgUG9seXRlY2huaWMgSW5zdGl0dXRlIGFu
ZCBTdGF0ZSBVbml2ZXJzaXR5oTAwCocIAAAAAAAAAAAwIocgAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAwQQYDVR0fBDowODA2oDSgMoYwaHR0cDovL2Ny
bC5nbG9iYWxzaWduLmNvbS9ncy90cnVzdHJvb3RzaGEyZzIuY3JsMEwGA1UdIARF
MEMwQQYJKwYBBAGgMgE8MDQwMgYIKwYBBQUHAgEWJmh0dHBzOi8vd3d3Lmdsb2Jh
bHNpZ24uY29tL3JlcG9zaXRvcnkvMA0GCSqGSIb3DQEBCwUAA4IBAQArHocpEKTv
DW1Hw0USj60KN96aLJXTLm05s0LbjloeTePtDFtuisrbE85A0IhCwxdIl/VsQMZB
7mQZBEmLzR+NK1/Luvs7C6WTmkqrE8H7D73dSOab5fMZIXS91V/aEtEQGpJMhwi1
svd9TiiQrVkagrraeRWmTTz9BtUA3CeujuW2tShxF1ew4Q4prYw97EsE4HnKDJtu
RtyTqKsuh/rRvKMmgUdEPZbVI23yzUKhi/mTbyml/35x/f6f5p7OYIKcQ/34sts8
xoW9dfkWBQKAXCstXat3WJVilGXBFub6GoVZdnxTDipyMZhUT/vzXq2bPphjcdR5
YGbmwyYmChfa
-----END CERTIFICATE-----`

var nameConstraintsIntermediate2 = `-----BEGIN CERTIFICATE-----
MIIEXDCCA0SgAwIBAgILBAAAAAABNumCOV0wDQYJKoZIhvcNAQELBQAwTDEgMB4G
A1UECxMXR2xvYmFsU2lnbiBSb290IENBIC0gUjMxEzARBgNVBAoTCkdsb2JhbFNp
Z24xEzARBgNVBAMTCkdsb2JhbFNpZ24wHhcNMTIwNDI1MTEwMDAwWhcNMjcwNDI1
MTEwMDAwWjBjMQswCQYDVQQGEwJCRTEVMBMGA1UECxMMVHJ1c3RlZCBSb290MRkw
FwYDVQQKExBHbG9iYWxTaWduIG52LXNhMSIwIAYDVQQDExlUcnVzdGVkIFJvb3Qg
Q0EgU0hBMjU2IEcyMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAz80+
/Q2PAhLuYwe04YTLBLGKr1/JScHtDvAY5E94GjGxCbSR1/1VhL880UPJyN85tddO
oxZPgtIyZixDvvK+CgpT5webyBBbqK/ap7aoByghAJ7X520XZMRwKA6cEWa6tjCL
WH1zscxQxGzgtV50rn2ux2SapoCPxMpM4+tpEVwWJf3KP3NT+jd9GRaXWgNei5JK
Quo9l+cZkSeuoWijvaer5hcLCufPywMMQd0r6XXIM/l7g9DjMaE24d+fa2bWxQXC
8WT/PZ+D1KUEkdtn/ixADqsoiIibGn7M84EE9/NLjbzPrwROlBUJFz6cuw+II0rZ
8OFFeZ/OkHHYZq2h9wIDAQABo4IBJjCCASIwDgYDVR0PAQH/BAQDAgEGMA8GA1Ud
EwEB/wQFMAMBAf8wRwYDVR0gBEAwPjA8BgRVHSAAMDQwMgYIKwYBBQUHAgEWJmh0
dHBzOi8vd3d3Lmdsb2JhbHNpZ24uY29tL3JlcG9zaXRvcnkvMB0GA1UdDgQWBBTI
Y5sIaVTCmMjZzeMzt1Be+MkBmzA2BgNVHR8ELzAtMCugKaAnhiVodHRwOi8vY3Js
Lmdsb2JhbHNpZ24ubmV0L3Jvb3QtcjMuY3JsMD4GCCsGAQUFBwEBBDIwMDAuBggr
BgEFBQcwAYYiaHR0cDovL29jc3AyLmdsb2JhbHNpZ24uY29tL3Jvb3RyMzAfBgNV
HSMEGDAWgBSP8Et/qC5FJK5NUPpjmove4t0bvDANBgkqhkiG9w0BAQsFAAOCAQEA
XzbLwBjJiY6j3WEcxD3eVnsIY4pY3bl6660tgpxCuLVx4o1xyiVkS/BcQFD7GIoX
FBRrf5HibO1uSEOw0QZoRwlsio1VPg1PRaccG5C1sB51l/TL1XH5zldZBCnRYrrF
qCPorxi0xoRogj8kqkS2xyzYLElhx9X7jIzfZ8dC4mgOeoCtVvwM9xvmef3n6Vyb
7/hl3w/zWwKxWyKJNaF7tScD5nvtLUzyBpr++aztiyJ1WliWcS6W+V2gKg9rxEC/
rc2yJS70DvfkPiEnBJ2x2AHZV3yKTALUqurkV705JledqUT9I5frAwYNXZ8pNzde
n+DIcSIo7yKy6MX9czbFWQ==
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_permitted_ok = `-----BEGIN CERTIFICATE-----
MIIDvjCCAqagAwIBAgIUCf6ZyZVyoojtih3/xWDxu9ThBcQwDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDQ3WhcNNDAwNjEyMjM0
NDQ3WjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBAOiv42C7m4dqfMrONXUkc1qJ4b35ecKB50QGHf/G33QGTvX3ZPgJcsbY
DEHZ3avrvPoIVntktNYFJ8OrDZ2HWNdECMuPELLZsFkWCRoXSg/924pO35M9GsbQ
8k0JLrQYQ00Wpl8X/CYeUJ/Y+M6Op9Y8U8zSp6qpTV/hfuSixeiVE6NsuIjLY+DW
H+7UlgapR5fO3UuLEowaCaY6YC9FCljYrQ9z8LfFXL8g8g1s/7kzJJMfqtrag/op
Km2IB9cRCBQoAnsxyq0DmSiq5nKmD1A+f0OI5v+xbCtzBiwpX8aBN0vEl3wt9WAU
JwzfZGjc9/7CCAt1cLLgSbMm8XIT4l0CAwEAAaOBrzCBrDCBlQYDVR0eAQH/BIGK
MIGHoIGEMDOkMTAvMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQww
CgYDVQQKDANPcmcwTaRLMEkxEzARBgoJkiaJk/IsZAEZFgNjb20xFzAVBgoJkiaJ
k/IsZAEZFgdleGFtcGxlMRkwFwYKCZImiZPyLGQBGRYJcGVybWl0dGVkMBIGA1Ud
EwEB/wQIMAYBAf8CAQEwDQYJKoZIhvcNAQELBQADggEBAHvkDdR6ktMaz2/I8bOo
o0e0TPgEdt3u4SxUhK48Kunm3Rg93Z6/Off57hv/M4X3UkCsQeUhQ556EbNLrwpG
sUfMAdtfpZAIgPK75Mr/aDRvGf9grdt7s0jzO96rboFKpUFKpP3TFgaVDDuse92I
3KEh6N5aryTumz8W/qddbw/CcetXnucKQw4Hq3o+uPD8Neu8sWJjWeUXPYs2r5Lu
VCjkeTG/iMovWwHTHqSdLSs+SQP7RB35W573qDP02u1iwiS+hUXc7uO2/dujF4pS
NI4IEgWvKtPKPIUdFwPaNn4ycOvgFdYZTrWDMdq4cpgAD2Z945VA5K6KtUSW3xNn
2VA=
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_permitted_ok = `-----BEGIN CERTIFICATE-----
MIIDDTCCAfWgAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTcyMzQ0NDdaFw00MDAzMDQyMzQ0NDdaMD8xCzAJBgNVBAYTAkZP
MRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3Vi
Q0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQC7oC2drjJaDbo7QnCy
8tbyVZm6QeKZ/V/i7RSbdnf0RHNDM85J2i2qBrT5n2uRoV3D0/Ln8z/VgO/AqFhI
B3PNcyrTn2Cjf9ZLQJFuo8MQlouQgwRgai8tRWNS0IUJFjQeG49+JEm5tPIHLEoC
Hu6WfOnWTQ5otS/crdxEiMZ40UBAL/INw2XnlttY+H+2eyHrmzP0OvPASVkekZgs
7CtMer11GIAAS9zfKq67fpXTUuAZoMFYjEYRr+/qfY0rMwnC63YhfpOx3ob2KLPl
IMntwc95rHuXPYA1Fwss0YfWM2pPsjRtegYWpslhS1XZLWRQlr0uGj1qyiT3Ltcv
W1elAgMBAAGjEzARMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQENBQADggEB
AJkTjOcfHRnS9sjVu4MjUrKAR5g6QLwr0ae46hEjLBNR0GYdl6E3hv0UwCKHBG7Z
XnF69GH5NOpIcaJqeInpCabOi2tMDoX9IjcvLBIYLtEzltrR/hjM7N6kNYRLhY6U
voqQIUx0pSMIiCyg5ic8nfD04NR27dwxgu2aqx23oZw22FM1rRfvcfGWlRCK/gRv
Y/VE0jmGSRb3luBd9AEca0M9jSZ1pYo80Sq7tx3Dh0oNuxYQuOeUhwSb54tuR7xE
tD6UJw2/yzpUDIkFtnlohEC2IdxiOq8RHJkzzjnpBMsKeTgERIh8Vtwt7eNZ+uJ4
Ac5kF0XcLUKZljv7MoFWPIs=
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_permitted_ok = `-----BEGIN CERTIFICATE-----
MIIDFzCCAf+gAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxNzIzNDQ0N1oXDTM5MTEyNTIzNDQ0N1owPjELMAkGA1UEBhMCRk8x
EjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQ0wCwYDVQQDDARMZWFm
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAsUWA857XFQVL7tASKR8y
iUVy8eXPjk/Zl5D3Vb/5BeIoQ5zqLimyjkxjLaKTWnEl64byTKyVvACwzaCQaBs9
2SobUDCFYhw6iPfND32jmuONSdtLvJYuxVDpp+wr9bArEx4+b9C7n+rrFszgrcns
wMOVNZeZXomekLqO+WL9CLt4HHmXQcmUStl2/VqN4XwKMHWivECyUL5Y4RRAxkfb
rjmJoNtqiiswNzhalHod5UUvJADul5xCbY7NPn4q1/SzLwX/2WaOVfSuYzQdNabc
NgXc8AFfxNfRMJvmLPOcPMocFRMItUMBqpxyWRFDLKxnIe+ymgsvAWQLe9ixGz0O
mwIDAQABox8wHTAbBgNVHREEFDASghBsZWFmLmV4YW1wbGUuY29tMA0GCSqGSIb3
DQEBDQUAA4IBAQBML4KdOb+vZt9Xemb+i2fOZjiAlZLPa3zz7HHt7b/zSB9MDiRo
XNLT/zf+xD76rMGo6wCDDs9lqUyveBun5siXYAObsG/pOB/AJ+w4GTCm/rCngbOG
PhRWhLumCdGt4NQ2uZweNbDnGKUtjlFbG1KUabrV2X2eM0SnZnh8hq1qFr7SE/pF
Ro8HCf1ctqe0RG7mmqGWx5ilH9oMSG6Pc60iEkRB7hnBkTft7r2XIDCR4mVGJamj
JYn+C43+lBSFyAIkCHRbAOoxZo0WZC49Mahm8aNwsKtXlCCP6u/ESpGEGvMMsnIJ
MLp5X4M72jmFYuRgCMNfCGiVEJqvJmeismjX
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_permitted_dirname_multirdn = `-----BEGIN CERTIFICATE-----
MIIDvjCCAqagAwIBAgIUOjzggjKgLVBb637t75xq0PhCUXUwDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDQ4WhcNNDAwNjEyMjM0
NDQ4WjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBALqeJ5JPi92OJZw3omUnJw6IGrAK8KwfrCeDVsxII5Jg1AATmeEVd/KY
2vSTdU8eEx1odC7icatEKfN5TnSR4MFGPfQ74zIenD9IuZi8OjdiRGMS65ZvSZhR
gD88aKELIvwZzwBCgNd9i8RvzRodHhnmmDJ8MQW5yTnwu261jqODCDuXJidllVPB
pZy9O2rQ8bQ1hyl/1ZZxTHgKTsFj4fdpjW8oq2QgEBIiaVpdhl8iMnAMmiAlF9s4
x/i5cGdqui4wpUYBQfCvTY2a+hG0PQaBDX3fGNe0kTHTJQwI1GsvMc1Rpnyzs+6P
BlD5oBffhngAIvyTrfDhAAfF0bouQD0CAwEAAaOBrzCBrDCBlQYDVR0eAQH/BIGK
MIGHoIGEMDOkMTAvMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQww
CgYDVQQKDANPcmcwTaRLMEkxEzARBgoJkiaJk/IsZAEZFgNjb20xFzAVBgoJkiaJ
k/IsZAEZFgdleGFtcGxlMRkwFwYKCZImiZPyLGQBGRYJcGVybWl0dGVkMBIGA1Ud
EwEB/wQIMAYBAf8CAQEwDQYJKoZIhvcNAQELBQADggEBAJiBUa1P4ccHeAtdYkqw
ZeogyfSydfR9P0aSsuhVfZBU65gcsaNfhxGtQ2WW+BXFNK8YH3vp2rif+A0j9RG1
KhvEUEOjDAQEkRa4uuUnYMC/vjeV7QDeEsHOZF01Bp5HkkBDgu5GANkW0ea/h4m6
GnGQ9UW/RudJVAu00Swr1TBAk6wyAQmRxUwqpEA26EXzt2yfKTHlrDIzqjoB5hWG
Jh7FqhYSWF4dlnj4uBGrtklbfKrUAVzYcRe1rW1lMivOFHGMO07WYDfSrGE17KTt
QsDsLN28WZhx1u9YoluR5iLGF+fyBMsZCoIu/E7i0WIIR0O9T6rFpsZjzMyokTZ+
1y8=
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_permitted_dirname_multirdn = `-----BEGIN CERTIFICATE-----
MIIDDTCCAfWgAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTcyMzQ0NDhaFw00MDAzMDQyMzQ0NDhaMD8xCzAJBgNVBAYTAkZP
MRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3Vi
Q0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDK4DcFqHSbi+eMsEPC
OZgA0WiwStLMFw3MwZ9mp1gJDo6BAVav3osAomOoxB1A8ffAX0UOQZfeGdlXP2GN
S+0EYFOSNxzL5nwszygDbSnHOkwlSBlFNLvDi8qQ9YesSFTGt0LBeoz0k/VYP3y0
sMGolcZB2quVn/F2GXGMmNsZCKyEbEmHoAp6a15lXvBLVEVVSbpBKEdi4jy6840q
8Y75sBCD3N4q3uUukCdAFdlgD5zidn3d3+YJu/j/OJ7d5/Pk/R5O+qVVzwFn4SAO
hZZDTa6eG4dIDkkcdbaE9Ys6ep1oP/lRgBe9QD3TzkrQb/RIDXRJziBp2akd+LFf
xYwrAgMBAAGjEzARMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQENBQADggEB
ALFbPzdbzNuCI/qs34rV5K1nR7X/Lq85Gt0DbSKbcJV1DC/d9HTFl9Mi1G2AU1To
YIuG52QZm7534heGvllsn4W1yDEwZCfcEwFJqNZGp6PNM+8+G5+7JKyvuKLeWtWF
I5/oBeX/qzudIHvx97p/LRQ56E2fp11lExmMRIkYWVUPQTdylnOXZQ61A7Xyn3xv
bgXaf/nBC/zBnQ1atsVYtHpyt8JnidYxnsldVXoOqe3sVFq19zIrLamEJErA4xYS
b/dTNMuHrCUNHnqsRaiPgvMFLF+wr+O7R95tJRLFZReb9xUU/BaYjabLN3ViOw9N
7SKwkY3laFzYiSc0Dr4EhWE=
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_permitted_dirname_multirdn = `-----BEGIN CERTIFICATE-----
MIIDMTCCAhmgAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxNzIzNDQ0OFoXDTM5MTEyNTIzNDQ0OFowWDETMBEGCgmSJomT8ixk
ARkWA2NvbTEXMBUGCgmSJomT8ixkARkWB2V4YW1wbGUxGTAXBgoJkiaJk/IsZAEZ
FglwZXJtaXR0ZWQxDTALBgNVBAMMBExlYWYwggEiMA0GCSqGSIb3DQEBAQUAA4IB
DwAwggEKAoIBAQC+d487XDxy19YHRhebr6vwACM/quLTIht/WdoKcrXUAwWOlrXP
6cK6ekr8JWJsBXeI4GpFHv+ZLmxadYCWdZmSf5r2glUNETGoRLrFwyW1riGJigDK
L0Yuv25avRD+7VkDfs/pkHStaWRQ0CNbLEsypNujSen8GMqw3JSIdehNTDlErDpl
OY1MNTpYOOe65GC481FuxdpwVfoUShSEXC5FCF0bImBdKgbkU5cBeRUOgzIp/ANe
t/cCOeP6x9GqH0xbt9pnNfv1hHndIGITuzmzZZ8KX+gzRVly6bQ4K2HDlzgL8cNv
0pw0MIMUH9x5UtRhyvDtY2qir6fjukYQVlEpAgMBAAGjHzAdMBsGA1UdEQQUMBKC
EGxlYWYuZXhhbXBsZS5jb20wDQYJKoZIhvcNAQENBQADggEBAIBpybozqH+AtD0a
v89pZrnOIKSekdRBqgL2uwMuak99aNa4tqx1X4qMej9cbiQjYBBCCNitIuAmQTyd
a0MCvcoiCCVaiygqduLyLJV9FKPhOIh7/dUAKamPSdnGxNdg2Pb/B8EyZaIyz/RK
4i1i0fNFQdstYrcT/sGr8UR7k7q7xsYiT5aIGUAQZFunq9HrjuzUVarm24xDXMG7
9fcW79lhxK9OhFQZHLdaSzcwvkNc4j2aB/b8Yhoth955wROXF/2QGLfDjgvtgOsA
OiJOtt3RIoQtV08UClUfOdUTQjOJr9140qtvd1WkeKeftYeHtHpDfQx+oyFTexn8
RKfy9P8=
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_notpermitted_rootca = `-----BEGIN CERTIFICATE-----
MIIDsjCCApqgAwIBAgIUcczaknDNG0WPrt+RCoq8UxE1KU0wDQYJKoZIhvcNAQEL
BQAwOjELMAkGA1UEBhMCRk8xDDAKBgNVBAgMA0FueTEMMAoGA1UECgwDT3JnMQ8w
DQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDQ4WhcNNDAwNjEyMjM0NDQ4WjA6
MQswCQYDVQQGEwJGTzEMMAoGA1UECAwDQW55MQwwCgYDVQQKDANPcmcxDzANBgNV
BAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAMIVw3Am
qybiYS6ejJG9+Cte6DvlSXJeq4mYQk0LH3sg7kMCQXSPjMyZ7v6sshrd+GjYKnGm
87NpzFUp0EK1nJmuslVW5ETtRxHL3GpCQo0qtoEXpumJCmSI8agnqluFY4YqxVyp
K8vx2BA72YcQNCe7wQGHMgqKolGjKjUTAEHFVzixaJgjvHSSdh9yHpUOBFoWvWt/
10nZabzr0IQX9o2DGhBuqo2WPji7HOZYfU21g8EXRNybCf6KPnKH7uCiqPOFu+oJ
xA26L8VygGIvniWUsZdFH8ymuxRnsPXymZMoeHy4bIflU8HlmbCS4WbjmEYBxWj0
5v2T0oFqEVxCIEECAwEAAaOBrzCBrDCBlQYDVR0eAQH/BIGKMIGHoIGEMDOkMTAv
MQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcw
TaRLMEkxEzARBgoJkiaJk/IsZAEZFgNjb20xFzAVBgoJkiaJk/IsZAEZFgdleGFt
cGxlMRkwFwYKCZImiZPyLGQBGRYJcGVybWl0dGVkMBIGA1UdEwEB/wQIMAYBAf8C
AQEwDQYJKoZIhvcNAQELBQADggEBAB970E+C4RQKUd2G388Hqs2WelqQHUNpVvFC
DAD3T9+pfnndW9pALYF8naS41+lYHN3HN73JCZo2N54MUakNRGNZtSIhGuWMdM3p
Ndz5CYw0z1rXEcWcvhqChF21eXn2MbJeC6hT2W8TS8KK2pnG89DGwQmfDBRp3nn3
Rj2RGX6O81e11v81oEGWHhJF5d5LUsDQEr7V55oB9JMjJ8+XsQpWe029q1p17HMR
86JqgEqGqxhp9h1mhtL2/8skvTV+s+hyHOxvYFQGS3dr6MVLYZjZj1TcA6k2xCW8
yYFxpQAy/orCUWbGXSUew2y7c+3tUpkvnbLupZeiiqyKqw53RvQ=
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_notpermitted_rootca = `-----BEGIN CERTIFICATE-----
MIIDBzCCAe+gAwIBAgIBATANBgkqhkiG9w0BAQ0FADA6MQswCQYDVQQGEwJGTzEM
MAoGA1UECAwDQW55MQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RDQTAeFw0y
MDA2MTcyMzQ0NDhaFw00MDAzMDQyMzQ0NDhaMD8xCzAJBgNVBAYTAkZPMRIwEAYD
VQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3ViQ0EwggEi
MA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQC0Q8r0nlb5uB4FdUCy8ywl27dM
6vhK9hyjibiD4MPrmd8z7HVadVEqRNrmVYCEwh/2yLE7fXS66I0vd9qJNkZlaBYt
o/kkVJ4FCgNWAq5D/tR8/N03mYqYmdo38sCfFbbfH4TH6NWSClYl2lm1MjgdwLmy
z5CmRBIbTPR4VsK/G5Hl6uh8kboQPqR9L3JsEqRPlxB0AOpjVPDS7AjLpE8rfEDf
bfMVe1j1CdSmYan3/oYvSsg2HX6JZ4//Ka2dhw5shrkLhTvKsi7ZKMWyZKn5he+F
yFx2s/DhdrEGo1mC4cMiS0iIGIkFsP/0DrT7V6QKnn1bpuPoQ/59jQrPKDADAgMB
AAGjEzARMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQENBQADggEBAEvP+l3T
aoy9i+1OpXTFD2OJSZp4qbuLZsyz8OWJfh7OR72SmQmuujCRmaZiAUhnCm+Hye2J
enTsgOg+ivJ1s2LmLheo/lP49uRIXQXD7EAnTlC7DUxlHmaqaD4zoIXv50bjpK95
nUHnChXtG8ohjbLNU9Vpoau1knDjRt6I8HISvhyZK4xl+RUZ3yGdA/fHEJoy/01e
+eXfnO/6DS43okpQVYK5fSc7pLesPGvekGLHaLdyAwPvAzC+bhzjL5g1gaSdYbD8
AzLdi8PHLbxHJIBZPxYMhwUjdz6lRH0SCsTWFrWJERwPphNHxI//Y3Gxm5M+8kVS
D0H3uSJwOfS8JD0=
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_notpermitted_rootca = `-----BEGIN CERTIFICATE-----
MIIDFzCCAf+gAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxNzIzNDQ0OFoXDTM5MTEyNTIzNDQ0OFowPjELMAkGA1UEBhMCRk8x
EjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQ0wCwYDVQQDDARMZWFm
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzjY4bMjA/7wPMoHMQv6Z
tUKa9t/WXAm1LJMHSQshcsJqWYKjxNE+zUVVX2tJKxL1fp0KYjhbGmFafAcsZi/w
Mjj43W6jO1weLk66c1cXfnqflIemb/y73pJNnOOp+l3qdP98SsxnFthCdouSMNS1
H2jdOCf0vPEdVw5M8rOypzo+dzUQtJycF60Iq08/mRZSE4OOUsp05CavIDWSe1jL
r8d2ELWt5oI6P4yYOyec9zZ+QdgtaZI24besam9OemHjmlY1n4OQD5gXbur819Zb
CCdQ3GtY7iQbq0ehAyJwFbov++Om2RbnDe5+ZLnrYcSAs+Ev5X+94pzWNXk9VtvE
2QIDAQABox8wHTAbBgNVHREEFDASghBsZWFmLmV4YW1wbGUuY29tMA0GCSqGSIb3
DQEBDQUAA4IBAQB1XZX7ZXRJM4YUUXUZE5yx3ej3tV8/4SF7oXFZuhZ1QKz+To3t
yCPkqA0eNwUcxl+LQ4uWjVm/gT8HxXKL5arK7bjlD+sACh1d/Kpbz0JkfSNa55Da
gn42Z5HZLnIzSxA3FyQDjwZxoMquH6Hl21JO8RlHKSDzacEwzv5SPD8eKhv2IhyD
O/dTsn9gVLZ9bCoBVEXFlg9L5H+rmw3TY29ZofrP8VOrm3xT2dl1wQCjIJNkQDsy
DSGd1959IulS0/VHFR3AD3zLurzDqjqREYZ5oGwZYFEpUtA/PCuIykLR9kYEK6t8
nkCrY2/fNGhFGB3Vk9UD01u3u/fRQfCsFyf5
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_notpermitted_subca_missing = `-----BEGIN CERTIFICATE-----
MIIDvjCCAqagAwIBAgIUOZnUJpd1fjyoR4/IO5WCiAi+PvowDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDQ4WhcNNDAwNjEyMjM0
NDQ4WjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBAMDFX/Pn3LdxQxA7BNsnRFyO3o5WEyxyaDEIhbGoFm5aRnBe3jXaJNNg
KT9SoNHUXVqfm0y0d3jzZVmM5IjP56ZyL5N9Jo7Qev6TB4WbxO4od7AmIgG8e+wv
awLV223Yop0P/rhPG8cU4UYm4gWGHCi9gkyCGzKrNkz6csA5TTWOZ90VNn6MQzt4
IdENCVrREfr7kMyzF1tZV3CO+xdMyV/5JKrkIVUEGQzIyYsdp0PYE3RsdRiBGY0n
dWvqiyjhEc/IiChv0PP+rZltlHhXgFlyTdV0mVKIMPVk3376kpwLYcMU/Vk2MZ2Y
PQzThsvIZGtmRg6i6tr8l5WzfbgK2EsCAwEAAaOBrzCBrDCBlQYDVR0eAQH/BIGK
MIGHoIGEMDOkMTAvMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQww
CgYDVQQKDANPcmcwTaRLMEkxEzARBgoJkiaJk/IsZAEZFgNjb20xFzAVBgoJkiaJ
k/IsZAEZFgdleGFtcGxlMRkwFwYKCZImiZPyLGQBGRYJcGVybWl0dGVkMBIGA1Ud
EwEB/wQIMAYBAf8CAQEwDQYJKoZIhvcNAQELBQADggEBAFFyzGvkIC4DBbQMCPbc
H46MlrLUxkN6B5cautRtDEHQ8ArCHQtGvUNNGKVRkr6dgdpJ1JJ+YPISFFGg0Hkm
Q1ampZLBW2SmzQM3eU0fdUIzNtCdLz5Uxqhz7/gpHOUXyEqpKJ3KezsbGp2USGXN
Anw1ZK0dwtDDYo/CLBQB/aRpT4CewoAVI+g2fti8HgZjaw+Bu9hB7FW1+BAliMSB
nS1qHGyMwTwWAh9F4eJNvDoBsvklrIgQYmAKtbAqbYxnIscp3vSHIMyj4k1+AJR3
XeJ5XxDbEO0ukwpWyRfwZnXvYRlETgeK5V2Qm974U3wO8sZtjOiYjR4Sc7uo7de7
QHI=
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_notpermitted_subca_missing = `-----BEGIN CERTIFICATE-----
MIIC/zCCAeegAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTcyMzQ0NDlaFw00MDAzMDQyMzQ0NDlaMDExCzAJBgNVBAYTAkZP
MRIwEAYDVQQIDAlQZXJtaXR0ZWQxDjAMBgNVBAMMBVN1YkNBMIIBIjANBgkqhkiG
9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzv5q7cB8jpa1WAziOzX8MCWZ4L95gwpdB8Vq
7Rg/JXoALy0BWRPf+vwqXhsytXfTevLhsHNNkIu7ZWC5EZNkWJfShAlUXyy0aa/T
Qn3ALQ7uqwZvKTxmXXle6dLI55npcvV3+26TmzYwdSVaxO9oAY0RVl5xbQWHvLf0
dhSWvYllyiDEkJ4brrE9Los7eTPM6SaMU8OLqLJH3bxicfgWzZ9I6mfKSRgF6MCs
BwS5u3BRbJ7L33m+OxkGKnKoFJh4LsSoxEBfYdGkavkhSuluTnavbN9FJGqt+4W1
9c10wHABSAKXsPfIARVin3IRWIukyTFJt5nXJP8KMPliiQL7KQIDAQABoxMwETAP
BgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBDQUAA4IBAQAIK4/rwCYokHaEG2fM
oYUfIMS/jFnpU7AT0Q+9Fgm4EazAdvC39car42jjHJbw1YF2AqQXFFG3kpRG1Lvl
q0po31/WHtOMGRcDilt8Qr3etMa2j7wKeUY38WvoeV2xa8KuVmNCSTYrFXr8MFmY
cfA3GSydibasgpd6jEn4jooj6J0kABLHk5j2F14CcRvuMQ4QFoi7vDOAK5sm0Ff+
zd1o00Rs3IGvlWUnNIa2vmij6y/8iHbCWCKw0vC01ZqD8RDD6520mUqPx/LLhaIn
H0koBpvbZ1SuPoN7pKpiKl40SY92vb2ESLozm4Ew/tmFiIcOgwia8MU9HXq8pXiD
MO+Y
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_notpermitted_subca_missing = `-----BEGIN CERTIFICATE-----
MIIDCTCCAfGgAwIBAgIBATANBgkqhkiG9w0BAQ0FADAxMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQ4wDAYDVQQDDAVTdWJDQTAeFw0yMDA2MTcyMzQ0
NDlaFw0zOTExMjUyMzQ0NDlaMD4xCzAJBgNVBAYTAkZPMRIwEAYDVQQIDAlQZXJt
aXR0ZWQxDDAKBgNVBAoMA09yZzENMAsGA1UEAwwETGVhZjCCASIwDQYJKoZIhvcN
AQEBBQADggEPADCCAQoCggEBANcDUQdUZPpfHo2MJNmI/wQ//tuAlcEjqrAIJJa4
b0KcYmE68QxaN7imErm/03WPSY3wUmjtA0BJXMhidtDKy2TETXAvRSKjXovrXcft
rmt0sFl6fud1saSraZHp4x7OmFPYwjV5fTIM+a3eF+ncrp/mhsXonAEXq8AB0Tx+
tqZUYxQcXeeRgQWt9HF/ZZQ7UTjXqmDaumhPaaeWANlvjm30uy8HCVrMOAn93fgN
dTEHGjVPuIB/UyMz9KfTpr7WN3fyHIxlmuFEUwZo2InRK3XVMAZODJ3RDmIx/2jM
4BFs7rcYCV+wQsvCPQrxOqK5rS2gkZuAdeK/5fCiAEBM9UUCAwEAAaMfMB0wGwYD
VR0RBBQwEoIQbGVhZi5leGFtcGxlLmNvbTANBgkqhkiG9w0BAQ0FAAOCAQEAStJA
zg4RBCHex5AX10P+fqAF1P1cSgtOOrpviFR4FXTrM46wuJZiSlbt3E/p+zEivnoW
SeVulXG0he3umVn2rO/9cBhczGCbQCkxZygVujKzkW8zqy4GN2lZQOZc3NWNGK03
IMuwij/zE8JSK3xMELfW5BEKPut87lSWOD4ezCnrsFlGGOmlKG8NhLHB3P+l9vmi
FND4NmH2766rTB2Q1fGaDK6vWfB4S/QmotR1NMdRusfgu/kjSr0ImJWbXHqtfTxg
0rFJdsil+AFy0AiW/4/f4EdDESd4pbKdwGONGNEeZiHVbKCICDewlQAR5sH0aHou
PCo5OTfZrymZRtEKzA==
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_notpermitted_subca_changed = `-----BEGIN CERTIFICATE-----
MIIDvjCCAqagAwIBAgIUe3SmR56/ASYLO6+vXN9pwIwj7w0wDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDQ5WhcNNDAwNjEyMjM0
NDQ5WjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBAK9U8xt1FHV1W8Q2cD4ewmmoAtHTmNeH76sqN5X/5OaE9iM0NXePTzNx
OuIo0tFQ0L5e3PTPMEf9ayIA1i+gULKjhbYAw8NI55/olGGPRMpeJpE5PSJFg/DQ
eX4QwAIPTmH6xiSPVsc89VqCDbtunCtbWwQRtJ1Nws1tLJCpeCtcFVEiBg0bAze6
9QFqrmoc7Vb94JWHgxQEHFafdZI+EJQZ9KFIn0aypgX1aVp7h248OvmxAt0jm7al
2Pg7wHLQhdXxZWfETrV3IFhFPoy6dBin90ZB98Aw+gss1JWJYvdkodmkZMoIxCoE
6axLDhyogOMYiYp9NOO6uvA67JcV3aUCAwEAAaOBrzCBrDCBlQYDVR0eAQH/BIGK
MIGHoIGEMDOkMTAvMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQww
CgYDVQQKDANPcmcwTaRLMEkxEzARBgoJkiaJk/IsZAEZFgNjb20xFzAVBgoJkiaJ
k/IsZAEZFgdleGFtcGxlMRkwFwYKCZImiZPyLGQBGRYJcGVybWl0dGVkMBIGA1Ud
EwEB/wQIMAYBAf8CAQEwDQYJKoZIhvcNAQELBQADggEBAJAYXOz00pd5rX3eJGdm
/OQclUUoAibHJZ2KUEBpqOJ1Noha3t7ei9TIl1ZR88KkYJtoVFi/2sOvhHE1+TJ/
lpSjcqCcLEMELtGvcNyOq4dVS5Eo3IOLrFxUYTBxIAFZZrZj7gWtAXZeQur5i94r
SoJmUqB4Ry0wNvImEEkhr9nA+wsYxDG1zgWtmPxKZs0rHkZWOZpXpZYaXQ5U9aQp
/g3Q1eJJ9OFn/vavd7ek26/Embo3TB//FdPJKCNsBCGUCSDUj/ZsgCpHZKOlOsB+
z3wvPaMeSBlWViG0IXEout+ePmHUDdJFyA3wzR2cb11Oi7IlzL07H2uqosmcZ0nj
DIk=
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_notpermitted_subca_changed = `-----BEGIN CERTIFICATE-----
MIIDBzCCAe+gAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTcyMzQ0NDlaFw00MDAzMDQyMzQ0NDlaMDkxCzAJBgNVBAYTAkZP
MQwwCgYDVQQIDANBbnkxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3ViQ0EwggEi
MA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDUxe+fE/HMM6A/cxvPl6/cckCv
ky/DLaY8rGkoo/cwZDwEq3hBvcNmhr0cVWyGcGNvK3aiXxDoopFm9Xn/A2ntE6ks
A3crk7q2TgzD0BGsojQ0qXtjg2a+aDH+Cjj8Tay2U1r9E8+Ey7xharaDgI9XpnIw
VPHn2aD2l7zAALEl87hqt3DNskeVktPYHPRS+/HcitblC7sLRILJV4JlJ1UIZ3Xl
4u8+loyCqYXe4VgYBKixxB5Pp5loqiNO52IL4vL2RcZ97KTliaIVF6VolcRqSkmF
CZMJci0UPV/i52Ft2RLCPz4EKNOAUHbuCyNj4aJ4Z1hlmO1o7FaqoQG8yY7RAgMB
AAGjEzARMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQENBQADggEBAFOQV5H+
YKpPSwNMe58CihZ4tjfO7NeIC/gt3LPBdrPkRiELuMuBODSAOxEYK+jrKm/Ohtrz
DqGBx0rFDwmrc7wvDKji3ATvUj76Tx3c3yak5ePEXvnYUUKBZfsnTrcgQbVMb640
K4ESolprjjVeHDEgVa8nKWk/mk8JGxGGYD06jUaoGSf2M2nGzdR69SVf7FG+fAJd
QR5YpIN5Mp1Vy1L7hevOVUMwR84ldG171Xsaj+toVL4YraTXhzJTfdXg5ABIdOKr
kAFCjNJfU4VuaRJqN+4gWOmSEuGbLJTmMkT1UR6I9addCYiSQsIHqfycgiZHkASf
sIgrpXtSzaoe4pc=
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_notpermitted_subca_changed = `-----BEGIN CERTIFICATE-----
MIIDETCCAfmgAwIBAgIBATANBgkqhkiG9w0BAQ0FADA5MQswCQYDVQQGEwJGTzEM
MAoGA1UECAwDQW55MQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNBMB4XDTIw
MDYxNzIzNDQ0OVoXDTM5MTEyNTIzNDQ0OVowPjELMAkGA1UEBhMCRk8xEjAQBgNV
BAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQ0wCwYDVQQDDARMZWFmMIIBIjAN
BgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1Zm8kWAEpVojRFlPuJzFqX789YLv
5xLbS1D8/jz4WOKRixFQk4AuE8sv6VxTlMSv8G9XX9iGLcoPF0CnTBKaCsOGXSV9
qxcW+6Qiof29mapgo0KKgaNN7qHLK5TGhpjcxrZnbSYELl/irUMInNYpeInss6cB
h64eEOiunCF3hTDt+ySAfajky2tFRNu6AZw9f71MFIud5lSNGraqSeve1Uh+KE84
QY1EbOTTeZmCXkweBFYYSaCUFfM1Ro0K5wVrKSndInDtGNbvPhtchKxhQC7So30J
5IORjjOxpfzNXDz/F/ITL9h+Ge7zK006wT1W4R1jZVl+2QdNBUTqBoRpOQIDAQAB
ox8wHTAbBgNVHREEFDASghBsZWFmLmV4YW1wbGUuY29tMA0GCSqGSIb3DQEBDQUA
A4IBAQDNuKbGKcn5n++1DAzkelygR/jGf9s0JksNNUxIBFkPlpcOM9nCJhuEVftm
vL+xtLbIAfFc6NsTxZPuYReMoqmYbZljxKRvNKCYSIp1SpZ0expFpE7lGcEiNH/b
52nzryhE9MGvEiCVSM3k6Xn2ClDgInaIqpYa3+NBAcoITjy4AfX52XLmEJD2SE88
x0yGuGWN+kH0hp+lLbMHD9nkNZ9vnup4GlQDPjocCRCyW1Yqr/x4808hYuWh4vR5
gNthif+VowJxbv6o8+eeQFM/MoYcJS29Fv17MMQ44tv07vW4FD5GCPOwPdQPJ6qC
Hr2KWB6mW0pVQZq9hjxa1PZjq/AR
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_notpermitted_leaf_missing = `-----BEGIN CERTIFICATE-----
MIIDvjCCAqagAwIBAgIUBCnFa8P+O6piDTKh6BPes9iE7vAwDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDQ5WhcNNDAwNjEyMjM0
NDQ5WjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBALWTGlyJkZ/3qR1O+FviuVNzRO/II0qwTqIbS58MK6/UqHKR9lBT2Zlz
aAj4yGeEtZppGbvO8Pw+yWZMMk/mPa97tbdzkUC/elBbUg7UyBcPoMDT6pHngl7l
ar3ubde+K/RFWu1aqcZlKr/jmaKvUhEAgsqMT73MV8XRbDaUr9DhldHQe7gi2Hu2
rW5FoYwdiwZmmF+jmWNmcc/ZoZa0A+Pdusr+XeC+k3P5Jrn37iCQRJ/Q4BJxrv0y
V8Q7X/eExPdhdFJPZTh9gemK2ZNeUmqxrxrVIy/Vt58SECVpS67MPb6u4I5Wr+We
J1H3ND7cgz4dHi+WWRE2QUubZwirwU8CAwEAAaOBrzCBrDCBlQYDVR0eAQH/BIGK
MIGHoIGEMDOkMTAvMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQww
CgYDVQQKDANPcmcwTaRLMEkxEzARBgoJkiaJk/IsZAEZFgNjb20xFzAVBgoJkiaJ
k/IsZAEZFgdleGFtcGxlMRkwFwYKCZImiZPyLGQBGRYJcGVybWl0dGVkMBIGA1Ud
EwEB/wQIMAYBAf8CAQEwDQYJKoZIhvcNAQELBQADggEBAGs2ToR8fiUXW6Ud7aNP
z9fEO8eNSgV9yCZpT0/NGuHp+hcjaXY56xo3NIwVEq/LerGnDLWBokXkfFAInjSi
5GrIBmZR7rVqxfqOg50odUgD52vpVyjqsVEIdP8qjp6XtgPMp0QAaaOIjj/L6V7k
Rpp9v312sUvch5wCWSfyW5k5WRQV/SmUCthIuyGhw6REc+ZOxxm8m9Ahw6J3OViY
1GWZUrG1Qh6iWqrGQZOtFd5iqXrRqCSIyY7UBT2/cE7jrzoTjKTxNcimUawC3Ix0
iRuOlCYUBqao80167vIJcWqu1oms+A9u6D49Ja6XrYxysf7NaK5axOFqRmFHAYEQ
otc=
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_notpermitted_leaf_missing = `-----BEGIN CERTIFICATE-----
MIIDDTCCAfWgAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTcyMzQ0NDlaFw00MDAzMDQyMzQ0NDlaMD8xCzAJBgNVBAYTAkZP
MRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3Vi
Q0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCo6wfyS4TIHXcBkclL
yv/YHbVacWYWj7apY+G25l4DGyro0YTAF7goQfSLFHKFnWK+q52dIbsIZ7MnAi2C
Jfd25bGttHsz8dhT7DcMJlIDLoWOcd4e4KYzVF+CiVpsznVzH44jBwHjBm2qV2KR
u/u2qir161zKunsCZd/AgGpHxNcJ5EHhQwW/jynr4qA9+fweuw6qa1S/b/mxxd8G
8c1SzGK3pM06M+CR/pOjnzObFtsv77TPGTu89/901gdBLax7rId4wCIYLwIr0/DB
TIrJvqBTBkrqxIn3rfDfQ3WKUa2iZFw/laCCMC6bF7yR2Aq9lwoDf07xYzHDeJ1S
Wke9AgMBAAGjEzARMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQENBQADggEB
AB8rIgpfLED9OAyK0hYjPe4BC7wvXFB/KFV0lMaKgwJOsKFahOYroyw1m93nXGux
vd8LMhBsAbLoDJAR8FAW8yhgrOCdI0Klv9F7F96OJAZ67J0wn85HS4jwGolZRgvT
VDfglwp1sqH7U+h4EuK4mEFrCB/cXb4AUJpB62zBcR8lDZQpQBZeURhAmwBR5nRO
gajZxBMhXvjjEmc1k3aPqTCFq7sAmhLpS4DWSOIoAwkeo85EHi3dPZ0kJopVxESE
nVQJtIM1vfa0up0dsK8c286orsaN+r7XFqngCY1q52xL27LiexZp0wiGqW9oi1Zd
oOncdhkvpHousxdP9DT340w=
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_notpermitted_leaf_missing = `-----BEGIN CERTIFICATE-----
MIIDCTCCAfGgAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxNzIzNDQ1MFoXDTM5MTEyNTIzNDQ1MFowMDELMAkGA1UEBhMCRk8x
EjAQBgNVBAgMCVBlcm1pdHRlZDENMAsGA1UEAwwETGVhZjCCASIwDQYJKoZIhvcN
AQEBBQADggEPADCCAQoCggEBAKt5B4e2kjEKi70Fxqkv+g2KBvIiVE9Fg1D4sdpv
F/OOEIQ/yCNAl5GCinl24TGu/9pjMJSXmHRKYNXtPhjTCAWOhh7MRPXOcVQ9Sa1N
4uTQFQooz3gHtIWVeUzRAvl5SRDy9IN0claHdS7VOWUPrkV658/eV7BbMjPlQxrx
m/meUtj26B0BFQYOCGpRk/YAyS3aKlMICz0RrZqDmJH2zll60gm5t9hnzgI9+AXG
sTcSAFZLWadYnSEeLWT/HFgI+F9/RSQRb3NL2eGIf96EF//ab6Bwkkf7IxT0n2am
yxj+oRseQCkGU9Hn8KDVM4eDcZHKANAbLyZj08vj88N73UkCAwEAAaMfMB0wGwYD
VR0RBBQwEoIQbGVhZi5leGFtcGxlLmNvbTANBgkqhkiG9w0BAQ0FAAOCAQEAmMm8
LSdEa+8HR0/tOuevc/bp8PfepiME20MBzkp9/CJeqBiryu4vSzuzd6i6rLGw3VrE
JWox9Ju4VcaptcXwv05CINrvrzFbi93UMmUGGTz634AeLOAQSk8nWwmo84qjbvAh
sXB2Bi2aZVSooh9h2+d6Zm916uWfkZR2iHwNHXzpWPfq4BZM48YQh6mY1SCNHiL7
IMaeHUrp6CLli7xjHYRSjYldiYDIeycN3SkbfVvMp4bWCBZ4qdR4RtcXIul6I4wQ
N7TANKK0oa2RMS4nEKPQeRogDf5Vt9L9R+OpUV8deyb1KPywFRJ78nlzGfdsiL13
jVrfHBkSphYjljc/1Q==
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_notpermitted_leaf_changed = `-----BEGIN CERTIFICATE-----
MIIDvjCCAqagAwIBAgIUeir4p8XxdPWLdvbhVDafSTVBzD4wDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDUwWhcNNDAwNjEyMjM0
NDUwWjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBAKh2BrO04S0u3JUh+yCsYkgU8M+6sCx8L7UYUVjyzcWqcAi6hWm1NaJv
zbvZdHopY1BoPbktqaP03Il1CySV4TV8V8RudKBnNYeOLXlc8OUlHd5FH+me0Y9I
wH5Jv2adh1MO5IVssUVDIqDkX4p7Gs2UzAU32G9V+iH6+1QFqOj/F/uICbSQNY6y
SE/tOf9inW5x2nxdOEJ5YmiSqQ+nUGRgq0+5kSWooXFzHhfVSYmiuuO7aPZOnmW2
Rk0iskheHnLwL3F8LGSx96ToRFl6hk/EJI2CE0UR31PdTGYmqBKV3C76TSibawMh
/DNxQ+BPwm91xDVo5fW88/zUsUGODQUCAwEAAaOBrzCBrDCBlQYDVR0eAQH/BIGK
MIGHoIGEMDOkMTAvMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQww
CgYDVQQKDANPcmcwTaRLMEkxEzARBgoJkiaJk/IsZAEZFgNjb20xFzAVBgoJkiaJ
k/IsZAEZFgdleGFtcGxlMRkwFwYKCZImiZPyLGQBGRYJcGVybWl0dGVkMBIGA1Ud
EwEB/wQIMAYBAf8CAQEwDQYJKoZIhvcNAQELBQADggEBAEdtnkb9IftTDxSqxulq
EmnRS8MVya1zt+HNwNiMU8jvoS07M7ccoIi3GR+8VToWTRYtlhsLY31iiM65ZD+t
zTa2SAO74BAub1pgRH9s9mITlAJGSTCBPZunW2/bCul3vau+MKZLB1r0mSLAObS6
5Ydj1bSQWCs//OCunNAvQH7SoSLsttTYKRdzaOMhjzLGLJTyIHDB2HwWHZzCifDt
naZTFds9VaoDxXBsnFNZiSY2I042DZp4ftQom3JAlu2IdnpMVWLGqCZouzpCI4EP
qw3Su6ekSUz6KbZJbzlsrO6xCWmQ3RIiL1Lul2sCMznnyMh3UATKfKS2xRh+sLMn
DQs=
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_notpermitted_leaf_changed = `-----BEGIN CERTIFICATE-----
MIIDDTCCAfWgAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTcyMzQ0NTBaFw00MDAzMDQyMzQ0NTBaMD8xCzAJBgNVBAYTAkZP
MRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3Vi
Q0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCmQdQG2xqxMOOmtdF7
hZrmFADZRi0dmEGf+ggNVkTRnEwMbvYn/Yux7DcFYamUP/ULddff/ij1P/N6iJEm
o9mR4IntBmSUpSKhTYCQFxlMR5HcEkTTVOC+BaNi/tOxkVC4aWvrEIM0Hp+EiYt/
a6ZnwpLGI9Jwel6IAf+vMFQaPdoDYJrAv1OUhxeysfPWT6UuZiNLW0/uYnyKBfWE
d8cHjUqJs+6gQkGK/HVan7L9MgpzGj7uhRunVFjS8H7QBUIDH9a24zHZYt4ST0r0
rcJ7iDHTL2EXavjBEXKNDCSW2WRS6hPVJdFU/3P7EGp5xUZF2QsBztOq5wgPvSNL
e7wZAgMBAAGjEzARMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQENBQADggEB
AIxpJeku+FuklM4wh7hpyMLJCgeoOHKtVSSV3vRGG0Y9PI/p+gnnz/IBA0pnCYiq
XHtqVp9hk6/DoKbWRq7/lXiYUdsHClhaXMKjZoXMdLPuhYSh4mbEgL6zkdtgvEVs
RQhgmYWhb5ddkiXTOfEdhscjSC+pSizzTqUq7S/donMI04drVe3ePRW1WLEqsXDq
GK5vLiXNAMcZfO7LF0cwRtAv8ZHwJSW137MFJumZV3MqSYF/6kBrvi69Yc60xwdN
x0qqkHPFKVful9kdk0tAsQb+q0pBWSSQ6g0IkwSnfvONWyNWwuY6QULryJDPrljq
Y7Xbi8UDMMLE91YWZsh2Bj0=
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_notpermitted_leaf_changed = `-----BEGIN CERTIFICATE-----
MIIDETCCAfmgAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxNzIzNDQ1MFoXDTM5MTEyNTIzNDQ1MFowODELMAkGA1UEBhMCRk8x
DDAKBgNVBAgMA0FueTEMMAoGA1UECgwDT3JnMQ0wCwYDVQQDDARMZWFmMIIBIjAN
BgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAnax+6OUje5TLEe56CnceXS2Y4tav
EJbvTSpunOMYQlytHXvSR6P/fDc5sWL3AYeI1218aFJRHaHbCSg/Bu60RRmpZTuC
oHfat3dxygRwI4kAuDEege0ACWYQ/iFNqSug5QYdghAhuevO1nT5AcwhTVA88665
tjZhzTslIszlsgmol3Tc1bUH/SwXCWsUghjzLv0G914JwFQNQddrf1+/wOfVB6l8
O+9SpO/Y6hxQ/QOlAFn5/alZe9QSX4YQHIOZuedSEkLvLDiLOx05oi5FHmzVlIsd
nzFz+dID9fUWFwdy+B9aBzpXzKe+3bo5aN6kBzNLG0mGwH7aKEUOKM0S4QIDAQAB
ox8wHTAbBgNVHREEFDASghBsZWFmLmV4YW1wbGUuY29tMA0GCSqGSIb3DQEBDQUA
A4IBAQAh6LZx+VMHMlrvTzE33v9kaq9RpWIgyRGb8otk3kziyg10gRAqtHSkXaan
01sY71+jt6HVgv/um4qaaYsVyO2FWx/FTQf5xaCMpKZE7xVeck7QSKebq2u9jnBx
tzRF4SL5mcb6bX9FeCQTjWxZBj5/3HkWRnEOc/Wva0c257zQK0jEjla4EHesyoh0
kmLZzjqoEy7RyEDrXfirH8Ej+wUTSr6wheaMolRp3WAxkeK7bot4o8m7hD/dRxci
hL0up/65eS71M8VWIp4l99YdMtT3DkPXO4JARETrCIKq0BCESxRgoREWAYyQ19YV
x6P2q0VOe8PnGyy5ufPbbvWroK6+
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_excluded_ok = `-----BEGIN CERTIFICATE-----
MIIDtzCCAp+gAwIBAgIUKmG31+ydG9YyJMivYp9ABzmIe9gwDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDUwWhcNNDAwNjEyMjM0
NDUwWjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBALLjJ9XNAfJpuRDGAA0u0KNkQ++wZywbPH/EL9zDvjXPlTGTE7JqESv8
A9QMGtkfELsvoVMsP8npuICdEmSLzkgWmS+Dypng2Pj29/3fnxE0cII9WCh6pRg/
WUtfEjF4IShHWuPIlvSg+ucF8peSCm0CenVDWqd+uJ5msXdE0XyyuVUAfycMGzsX
fkiw6q91pEb5Lle+tmBXL8zT+JPvbLDPeHrkU/0+jPwX/xC+ziDHrfh9Hetddcd+
6xJRr72PX5cuQ9lcQvxnSQUTc6PJHascWBkklQxdOvy2blIFTKk8fSZqa6sJP5Qd
cnp1mcvinlpjZ3lgzMtu7TYkX0AnSP8CAwEAAaOBqDCBpTCBjgYDVR0eAQH/BIGD
MIGAoX4wMKQuMCwxCzAJBgNVBAYTAkZPMQ8wDQYDVQQIDAZEZW5pZWQxDDAKBgNV
BAoMA09yZzBKpEgwRjETMBEGCgmSJomT8ixkARkWA2NvbTEXMBUGCgmSJomT8ixk
ARkWB2V4YW1wbGUxFjAUBgoJkiaJk/IsZAEZFgZkZW5pZWQwEgYDVR0TAQH/BAgw
BgEB/wIBATANBgkqhkiG9w0BAQsFAAOCAQEAOtJkL4GXHbzJsOeLduGb8m99G14+
7ldlQ7zN8/MlkLx1q29ZF3xIWqgRug/mdIdPDkM+E7kmMESwXg832Zbmn4T2DrW1
ZWiAot3TPsI9P3uzHz21gFU+exruc+uNNwoD0rbvV5KEqKg/O6KU/n/1i0wybir1
Hp2HvEUZYKFWfHqbqbfyN+kEUUp0NNWPqoARAcuEr4p5YiFRMSrIu37NfWM2AHWd
fR6it0pc5ynH1UN49J6qwaCEut6pyY/fdzLgHiILdAYcR8fWTLViN+iiSQccUkUR
AeOkf65mEY6s4ul1VUyH+lD+ignUoRs/uIV073pZYxKnD/nAlQZj6o6abA==
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_excluded_ok = `-----BEGIN CERTIFICATE-----
MIIDDTCCAfWgAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTcyMzQ0NTBaFw00MDAzMDQyMzQ0NTBaMD8xCzAJBgNVBAYTAkZP
MRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3Vi
Q0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDC1wndgO+pMuo9f/a9
xO1yLvJjrq6GWAumFUDP7bV8n2njsSlPjQ/1fNS4LC3V5UCBcwC1a+CgWHd9SrpK
nw0iWRdxQekrbKelwzTihTo2eXgj5wJbEbE7QzrR3jFig0KTgavc5c+jWKLEDG1i
cmCNGC7MTwtwNgt2SmyxAeNa+EDOs1KY/mXCsh/tXLXbhehhtQxCITRfftrAje2q
EVjHyhI0WnMt4q2rf3buRoC087ufyk7G2rDzcghjSn6E0zrFn3HMOm+/7VL0OROr
9g5mDKTAbIOUjiqRYPkeYIjU+M/jyIw9Wn5xSvbcVIgqg3224VES9knrUnZT/EHC
Cg5BAgMBAAGjEzARMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQENBQADggEB
AKeSM3cbWwPyH09xlKBUrJDdLlVVJIKSchwx9IJk9/2DL3K5/ku8q0GJMIIY8YqS
4Bzi92yiEf/jizZeIrfK/rBIN3jxmx9cGdt0fVq3ZudOEW62ZdK3UsqnINGgv9UE
eCcAySWRZi9qkXDTui/7q7V2FCEJseqwgOI8N9TKxTnwTVyyCi/lyUYz7jOWU/JZ
QUG5HmMBe99z+yaF3JTa0AbeaTcj5urdklL6aOeTcaHEBoUPsTsLQlAYjI/I1XIm
KJlrxfY0DddCLr+CyJVCthLDxNdI70yDz9VNk38amJBoDaCMmQEeYwCZ6jk1d1Cl
bZ1UyIZnvOxXQQQB6+jRcfY=
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_excluded_ok = `-----BEGIN CERTIFICATE-----
MIIDFzCCAf+gAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxNzIzNDQ1MVoXDTM5MTEyNTIzNDQ1MVowPjELMAkGA1UEBhMCRk8x
EjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQ0wCwYDVQQDDARMZWFm
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAva5lDCQyJbOmhGswkMw0
P2GJ9AFxE50BysndmYtslER78rM61UYdVeFKpq/j2oreriCZLbrbrOuMRubGo4Sh
8uoUvxRahVE9aFhKIGtXirV/LNnKjGjvZ0M4XuDfnvheCP+SZsVU36l4LvJ7Dz98
UznLZvx1gUTZsgU69rxd03X/MOh5dTzDzmu01U4U3bhYd8LtCLELfxQVJfh43Rd6
nj1Y3sFmib0J6WI+V0cHKenx+fYjetO5TiJ9Qw3O1hLzDt7EUd1gPgle9jzwD+8n
MpxBdGTyzChAXoBVDgfBV8bxFVzDGyV40+dism8gXdmq3T9GHw/kt1vemWMfnUxZ
gQIDAQABox8wHTAbBgNVHREEFDASghBsZWFmLmV4YW1wbGUuY29tMA0GCSqGSIb3
DQEBDQUAA4IBAQCwu0rnwQkDndNxQcjgsShz9QDvnDur8H2zy3m9aJLNaFQCnTDx
ZeBEQNeanU7vM53MtdreWQXk9fRZ3kLCP/HxLBcBzu5KDUX55Au3Gtk5/MREXGyV
JksApOfU9sPxaDuhjeBHETkumM1T2CfMO/bzaA1D1zcwjJE9hrVmSmySm6WalcFx
QxCSAIZvWgQIZS8zYf9xTs6oKNFoZLWZZikASUpzRhqx87iLQztOiqO5yVLyNUuf
a59aVQbF7j2plSFoiODXoOF+QKmvD6ATx2OmmnCrM0N2aM8dVm0dLyxsxwUcw9mC
/5ZIq3EF09A56PdGEoaNU0J2bgj/nxQkosIv
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_excluded_rootca = `-----BEGIN CERTIFICATE-----
MIIDsTCCApmgAwIBAgIUciX5Nbh7s6Idfc5QHaSHAKvFqDYwDQYJKoZIhvcNAQEL
BQAwPTELMAkGA1UEBhMCRk8xDzANBgNVBAgMBkRlbmllZDEMMAoGA1UECgwDT3Jn
MQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDUxWhcNNDAwNjEyMjM0NDUx
WjA9MQswCQYDVQQGEwJGTzEPMA0GA1UECAwGRGVuaWVkMQwwCgYDVQQKDANPcmcx
DzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEB
AMZvcJrRcSl9BxPt4VpAZuXv9AyTWqqbED952LGjng/dKKf3Pjc+v1ySX2/62iiy
TUWuXWOesVZ5tECooT7JZyD0P/67QDgYqjtDYDiJ1SBmaiQ87/w+6hq9B9loyaiK
KniXov0IFHme/yvlUbpa458kXGriV5/oCMlBvnYb1TEZyMcBqyXEluw6CCAJ2FJx
izQubA2nGtg80meLiYoI9OAtDe6ERyb/HRJP0qVlEXQCrDDCC+PK7hMbgNgquFnm
UpLusoNEuacZMQW7gB/Fr0ZWK1HS10pnsf9GYdfR0xU/p6tqBGabcK7HrH7C0pb8
OZLvxdW+HW5L/M34YkutZCkCAwEAAaOBqDCBpTCBjgYDVR0eAQH/BIGDMIGAoX4w
MKQuMCwxCzAJBgNVBAYTAkZPMQ8wDQYDVQQIDAZEZW5pZWQxDDAKBgNVBAoMA09y
ZzBKpEgwRjETMBEGCgmSJomT8ixkARkWA2NvbTEXMBUGCgmSJomT8ixkARkWB2V4
YW1wbGUxFjAUBgoJkiaJk/IsZAEZFgZkZW5pZWQwEgYDVR0TAQH/BAgwBgEB/wIB
ATANBgkqhkiG9w0BAQsFAAOCAQEAgL7XZUy5U7fzDz3O9bTQcbAxNJ56ky01jJvy
Kj8Wuo+Ot04qO1V7kWF8jB39cwlw9RP2bptUTH6KGS6hJ2LOoYP8WMhOTkHAq/Np
ttNcjWce4KFNs8512URo/8FtQZ4fiFyhKfxLuWS3EX1I8B1OlJVZBsqtCH8rglQM
E0r4HWs5rhBvmD3vgxrFQsdGAN8Tconc769FxaHynySgK6bpsOHe8YWtnO8yCaha
Q68/4dmE3FcafVJT896d9N3w61GvQqsa8jbP2uUMWi3dg6YBqsRU1bh3CvBrYpYl
K0KZZLQ9bTbqH1ksZ9Iod1YeyhCpTd1kf3bkAHDNylqPus2vbw==
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_excluded_rootca = `-----BEGIN CERTIFICATE-----
MIIDCjCCAfKgAwIBAgIBATANBgkqhkiG9w0BAQ0FADA9MQswCQYDVQQGEwJGTzEP
MA0GA1UECAwGRGVuaWVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RDQTAe
Fw0yMDA2MTcyMzQ0NTFaFw00MDAzMDQyMzQ0NTFaMD8xCzAJBgNVBAYTAkZPMRIw
EAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3ViQ0Ew
ggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQClm2QGf5fZf3Ky+iXjaghW
qJp6L6tDMbMg9yAvA9M60xD6A+ND/1jSlKzPC+95m5oqdvCUXoM7riGCknzuUx77
CE8Usk2bojmEz06mpH46upuLjpQ1tK9sIYODupaT0wx1gj+HRn1p5OuGLFn08MRy
TORY0TdrcGTH5rSlmKrqxjD0fTBUJZh35u5FN6tOTMf3M8/ggOOkksQWW2FdK5eQ
2JcfUxFipyldsqIl9QdFGvybGzaD3MbqTJz0tKX9AA/PWqqC5yP30aS+c6Nt9wBw
bCBu86L6t8nyxGGzY+DwfVUOi6he0Nv48XF647ozDGzO7iyRHulWvfxIQv9P+8Xd
AgMBAAGjEzARMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQENBQADggEBAHgw
O4aO2+WfJWOnOKTTEE7Z4hQkVcrR9Aw30eX8CmNXrCtRzNEvJhhooGl7HX+DzHFZ
BZiUpCG2J5rT4qM3eiNYU15EXZcK7jr2hJeUo6GwR97QXJDqrtVz3ijc1yj3/bFX
kgZ+poFTmK8uktXa940pwuMrVUyyc1UVsSRDyujbogAGJIf6WQDFB14BghDMYvup
TM4PoJcj1AC9wClecTPOSRFX98Ld+Q6LI7sSYS7NwYl9DBZfzvjvoJ7Y81j/uSyY
brVMI3izH0/oOxVDcv/J8T66hfXjxKhaawnWJgz/4Vt95mn0tpSOOuV0nw9dGrIQ
8VfPrcqVUWGjuXmdylI=
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_excluded_rootca = `-----BEGIN CERTIFICATE-----
MIIDFzCCAf+gAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxNzIzNDQ1MVoXDTM5MTEyNTIzNDQ1MVowPjELMAkGA1UEBhMCRk8x
EjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQ0wCwYDVQQDDARMZWFm
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAyfIZLyzQT+alQzukkpIX
p2E7wBLQ+ss91sKjZ6OWj+NbWLkaUNyAN76kyPxPgint9p49xjqZY9GCUnGtvWYb
cYZQ1fj3crxt3Rn3qXMDPHUUE3HXJYXBl6wE8paIivqdqoqLdHHgNmfyXURxHuOM
NKcHpBbR7eZPmXVMikz2rD5jesRDZJUYCyRxD0OfeVYaUOX5mNZX10MvNs4WcVoC
TB3/WDPk+PjJKAserOIAi5xJNFaxpILo+iQ+s2+T+ewJA20rro0oLykNKKhvzVnJ
Mrn6gnHavxIoPsFkLUoRxF5vyQ/UJ2+DWsaKKqBDE01LCVjvpd4Z0OVTyU1f42zS
6QIDAQABox8wHTAbBgNVHREEFDASghBsZWFmLmV4YW1wbGUuY29tMA0GCSqGSIb3
DQEBDQUAA4IBAQAzKKVdTpFjgEmpZy43PFErJVIhLn8iz/a6jAKz80wlVsXhdjtC
4GXPP01KIfzJws1odZ0S/nKVuYZhFCbAudyTcZ+hfG7DOMHXEVjF49QmLh6Rl57z
KZaopWWlkNHtB3jQhdX6I+JeS3hQ9h3ZHg/lp1kxA5FIJx4iUdfEEcHP+VPprsax
NNkJhyMPXhEmzZkhBVwAriAsmaoPAhw0i8p706GoyPEXTTMkozckX846Zo1VKJpG
ko3gpLlcY/eyBDBN6wdoRWatA5hOaMrsArlOPwFnoIhWs5PI+drUiBTGEpoFgRAT
HCBEKADPXWJ3oEDPlvIN+Q6VsTHMskcjFWVe
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_excluded_subca = `-----BEGIN CERTIFICATE-----
MIIDtzCCAp+gAwIBAgIUFsSJpy7YvFGg4UPXR5J7xRF5ur8wDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDUxWhcNNDAwNjEyMjM0
NDUxWjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBAMNsQN73YR/WaK4LNvwHZ4saV602YIRW86DObnWz52OtxO+jnoysFN1H
T7YuyZziDGgN8Cbo4n3ERb3QuhAbaPUdeMZGeikLp9eb6043O6Bs9WDwspXlsUXx
/ngyBbeNB4DOqKyEwU5QAqWjpUbylpMXjNAaEQOU02BIFkIxCD/O0s4oZN6DFskc
3rwy8IYZduCeVNXVswdim2afo0tuhU0FNyrDBgAcCwNeyUKurCnVD4QF+XP8q1aC
LyoieXHjTOg7l4Ksa+VOyTMH0d9qzRrOcOQm78DPHefsAtBLfVaNWHMsUYWz0S0z
8BOFDzew2dGQa/d4DbjG75S9DkbvwC0CAwEAAaOBqDCBpTCBjgYDVR0eAQH/BIGD
MIGAoX4wMKQuMCwxCzAJBgNVBAYTAkZPMQ8wDQYDVQQIDAZEZW5pZWQxDDAKBgNV
BAoMA09yZzBKpEgwRjETMBEGCgmSJomT8ixkARkWA2NvbTEXMBUGCgmSJomT8ixk
ARkWB2V4YW1wbGUxFjAUBgoJkiaJk/IsZAEZFgZkZW5pZWQwEgYDVR0TAQH/BAgw
BgEB/wIBATANBgkqhkiG9w0BAQsFAAOCAQEAcUxARhOit2At4mUA5hrKmljssLwP
995QU6+645W0J7gxv487aawalithORk7zE8wm4BAB7mz4S6R8UdfNNY39pLEfh7h
JiTH5HUYqfolmT0GUTzvUtBCE8fkIf+IX5bVxPgNrEIAT6euJScpJkXlqr5ZuYts
EHSlKvgxBDbddgjs6N6OyXQSG6Po/4PpcgzhxKy8w2qfO3S4j1G5YEWwQBRFa5VJ
c4DNSz7ydRtkga4VDsIV0mbK0mjZeJDu+nb0V/rsyryvkZj9qSiH3M9y/3gfvFWD
GnifjahHwYe/xN4eNSgSxgqYx20TH4nthaFQgcDzIpqSmJfOHHCMNcg2JA==
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_excluded_subca = `-----BEGIN CERTIFICATE-----
MIIDCjCCAfKgAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTcyMzQ0NTFaFw00MDAzMDQyMzQ0NTFaMDwxCzAJBgNVBAYTAkZP
MQ8wDQYDVQQIDAZEZW5pZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3ViQ0Ew
ggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDUc2Muxik2k7dasG1BBksQ
5M5EPCc101nS6RZgD+LfpQl9/DD4qHZF8iMdGcaDg531wVfchlpejpsE9ghHsHrS
YHi4sfw2Puc1sMoJ19WDdOmAA3uWzfLUGp0T0jx9vifcElKgmczvUucAaHPLSK/O
MUafIABY0zOatVOBdB7XzfctQQQ2y5VgegLCDSPP5YXbzxernczSlF32q6pDPO83
Z8AfQmosRLsFL6w0iMtfcdeXzUuEh0n+mv32Gbg7zNORTzQK5PtO58LSyJoNO3Gn
gopjYInUYRd6dnI6QXfaS7vE0cVgnzXVRgtKoroIEbVs+XNrtqX3X0bg7pXSsCIn
AgMBAAGjEzARMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQENBQADggEBAB9E
GDcUX7KxUsIYfJQYnNIQIcFp9lcYIv8H5Xt1o5ZhNRYAOaRLTsdjQaB/nhiygwZS
eij+V935kAmcl0V0UWhb+W0a18kmPSWH+AHXElMfLXKzYabb5YA3EgiiuGV4uxBD
b3Wch++73P4m6AjApDaYG2MOFvoZuvmPNi5jGA+fGsxMe1gBoKmg1Z9QnGPQF1yB
/M9YJAL/qafrbHY7IyliCLMXgfMXRjXVCHSYwLuPfkXH5QIx4Ihks7NVVziNH/Kd
AqXUU8j8kU654PNggTJ23aar3JuRK0r1XOchG7z3YvUFK2O5Eov/Y7Z6FEjs4oK3
IwfESSS3nPVZijufvbU=
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_excluded_subca = `-----BEGIN CERTIFICATE-----
MIIDFDCCAfygAwIBAgIBATANBgkqhkiG9w0BAQ0FADA8MQswCQYDVQQGEwJGTzEP
MA0GA1UECAwGRGVuaWVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNBMB4X
DTIwMDYxNzIzNDQ1MVoXDTM5MTEyNTIzNDQ1MVowPjELMAkGA1UEBhMCRk8xEjAQ
BgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQ0wCwYDVQQDDARMZWFmMIIB
IjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA0jyOWMwIUsUif4UqnIkA4JJF
mN781Pn1cnOMTmXuBmmxb1CUJEmj45ZgaAWuYH5fokrH4CB7hNYQN6yDfhnZDfjJ
dFo/6d359qRYaUxnj8OuHx1HtqVDVeF/BiOXbBCBUwEhkpPg9yEBensgY12peeCU
ESOEjOffQDMsJXK7Ewavdbz1Zggy5hkL3fS1ToLd+QSsYCWNNdUSIhYhtwt+ebVO
BdwANOPPpFjtd4+TWf/4Hr/hXy6hv+XUFRrAxUiiKcwPQW/hcHQY6P/91gyRYPOv
9rREex3hjP9+czq/dDO6gwpGrBIeG0N+ubi1j7Qm2VGxpxGOQ90bpUqv7SsJ1QID
AQABox8wHTAbBgNVHREEFDASghBsZWFmLmV4YW1wbGUuY29tMA0GCSqGSIb3DQEB
DQUAA4IBAQB7njnf8EvV6Ks5jr8Jp2rLkKnKkiVHy0jo3tb2zlpqBHSoE0AjB7BO
Y9u0gN41YTSBDDEt4XYNB8zdgcykF7CjtkyjA5XZFrd746mtV4B14obUI9flUOGD
LQSvzsfQ8ZQXI3pODwKw5h4c4Oamd0PWxDH9cdOJ6gYQSYLvGTJ3Lw7mtLA8/FHz
iud7q4A7IzWqc4ddz1fgidA84PdEQuueIxe+f+dIEg0xbmx1WDE1M+DODNoJv1lO
0WJmtih1vXrMXZmRYrX+C6hpwe8eEUZ8I+ji166BE77c8WuVOVv8Xt9/VgbbfcV5
Yks9b1uKfoT2OFIZpWz2Koelx2QY3hXV
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_excluded_leaf = `-----BEGIN CERTIFICATE-----
MIIDtzCCAp+gAwIBAgIUER/ZnG2p8SujBgP67sNKnIsmpIMwDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDUxWhcNNDAwNjEyMjM0
NDUxWjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBAMouJboGH9bOAMsx/PX4D2tCay0+eWFrsN16XIoXy7E2ARjFzzEmKZPD
WaaMhFlNKP2qO/6pD5g2PY8I27nnyqmzXoz26NJ7e8H7fT7wmbrEcLHT4ish+X3E
NKxKkTG3SFNoAlCaTZvf6QhSMDcAsG5tEV1MotVo0xwrhITzbrOwejnn018OZJ21
CPK7mFPqCx6MF4gfzzGG2gTbNU9K4AjL0ykUAKRBs0oZ/Q/Qe9Iid6VW8A1+qUZT
lJ5WjxYgHyaeW0dlNAHhrnpo8bY0nZpvIoMisGJKSbyvp/HO1Om3mX3PfUjfi51u
pS5GUKQ/a/rvCu6kKeK3UU6U9eI1TfsCAwEAAaOBqDCBpTCBjgYDVR0eAQH/BIGD
MIGAoX4wMKQuMCwxCzAJBgNVBAYTAkZPMQ8wDQYDVQQIDAZEZW5pZWQxDDAKBgNV
BAoMA09yZzBKpEgwRjETMBEGCgmSJomT8ixkARkWA2NvbTEXMBUGCgmSJomT8ixk
ARkWB2V4YW1wbGUxFjAUBgoJkiaJk/IsZAEZFgZkZW5pZWQwEgYDVR0TAQH/BAgw
BgEB/wIBATANBgkqhkiG9w0BAQsFAAOCAQEASdeIWexo2e4JTIwEpa+Eq2LZL1+M
MO14ddKsVs4xXi/t0W2XCf3CA9OCSeLXQmPipS7y3VJ9XGjUwMIKJyuwpXTFQ4g8
D7t0eNfBlROIFSqDGhit0W0A6h8sTUKG+xfBhLCDmp3Y77Ma8ZzdTStCRyuP9Phf
5J7qXnQR3mJYxcU3rZ5aI5pJ90K7pQTHzvqqgKdLdvPxXuo6hATXxrO7LQGbZY6m
4LD0bZc/PnAYkz1XKL8qwDy+/J+00DaaAqCkKeaT/DwikI8YKCN8V6VsXLmiXFO+
y4ms6r+xEIRrWn6Bu9qdXN+R93uuIupzGVn5MorKpXOIIpg5FnfJmhREfw==
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_excluded_leaf = `-----BEGIN CERTIFICATE-----
MIIDDTCCAfWgAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTcyMzQ0NTJaFw00MDAzMDQyMzQ0NTJaMD8xCzAJBgNVBAYTAkZP
MRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3Vi
Q0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCkt+0vVGmZja1qEIPG
38tPtiKup2eZPRBwj6gwyZQgBrLn+/U5UcDiO1sfoNpCE9KHGxl2eiUZGirsahSA
gkS9Qs+zawQ9GPqh0btZ0KfzIzMYOjiD18rPSlqq/LJ5PILLt8Z3uYaeJYM9YOXI
biiGaAN7dPaht2iW92l+VIbHgEjEWpHU2ds2kEJmm848w35hoPsHuWxRYaLPr2va
XGAoiUSbuxNclXRTIm30xIosmGTPeBxrdJ5YH/uiVgfvaNFiQkbjNrEWt4b1DVwJ
mjB6o4XyX5mWI6Nu3ik05FXe7gYAfekjV+UeH0JcI/CW9q69gsHfx4suphA6KRSp
vidPAgMBAAGjEzARMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQENBQADggEB
AKEgKNLBlpbRGbIV9e6e3ma9lEn6x7QpHfFLXGajDznH6Yl/p4NKjqlQu0eQECzZ
em6cHze6sI2UNXPbHnfvDHSytdR5HPJB5NlWQ+ChTHw9aTGiscdYpQx8FmGQNLh/
ksdP5xaopCNcqEI1MdhjDzfy2yTjIpxe228nZPpJ6uufWhXogyOLRiES/flztuHQ
qn7HXTMRf8PL5L/XLK/W+g51j8lToIphAts1wWACBag+LVBexFBf9yWIm25A0V7C
Dzpqf3WexKIUoPeE+Aopuxp+GyCEBWOV4XNyTtQFuQRGPBm9yb9FoV1YRHJ9EQoM
7nnDEO5bB3baJR44wI7Y9ZI=
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_excluded_leaf = `-----BEGIN CERTIFICATE-----
MIIDFDCCAfygAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxNzIzNDQ1MloXDTM5MTEyNTIzNDQ1MlowOzELMAkGA1UEBhMCRk8x
DzANBgNVBAgMBkRlbmllZDEMMAoGA1UECgwDT3JnMQ0wCwYDVQQDDARMZWFmMIIB
IjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwU4n8x5VIRYEoAvc6aUtNMLZ
RLCScHDt6xrh0irM9jeMCKZeIlm2rfwj9NMMy6amHYXcIFfPWnFKQVMIvDicNS3Y
4bTW4JeaXqyIRcTexwXP3tTJriHCwwPx/pCtBXplMEB+3xQ2kWJMLmrQ5+nSGo72
EbnSZPb69AhYFW9R0MqqsP5jNEfJDkTHubiEY1ceZSXujRkntCsw203e75I066Rr
quA5a/OW8kPHXqwBJOr0UopE6bxisFdbbSSQB3FOtB+qlxXnt78ruE8bJA3shrAQ
70hcWvo03xh0UKSRjtnd+dXntY0ayOLBkG3eKXIUT+bUVXgx7RAm0FERFJBNAQID
AQABox8wHTAbBgNVHREEFDASghBsZWFmLmV4YW1wbGUuY29tMA0GCSqGSIb3DQEB
DQUAA4IBAQBMm5kHxsfwI5u3QoPdrYgbkUtejeJqrU3FpWGyn/covE82DabM8znR
ZFXCol+vDKPFUQM0E4b9Z08cJyCnj88GokhanBWio4hkkRMN5KoolsrfPCRy2CSw
cnbocsytyUj9sGVKedZCjwS7X4M+I+eTm+U2xTjF57BzsHqPBegX+Ofrj1SjxRLH
xBHNPCbcUJ+cvOOrRAf55p2mqrC7Q6fgClKC+fXAOtoqz/ZxZaD0PKbZlQCNWAkv
O9ec3/SAIOo6VB2dq99ipEeLOMDzFgIMAzqMcoLWJzyoeLk+QiMSfX0QyUP/6J2F
vYRU3G452BEcPAgk1NW0pvVP5aJDh3Ep
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_permitted_excluded_OK = `-----BEGIN CERTIFICATE-----
MIIEBjCCAu6gAwIBAgIUHHkAQg5TrY3k+UsAX2bN+Tmcn9QwDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDUyWhcNNDAwNjEyMjM0
NDUyWjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBAPD++dF1edxTqT0NoB33VdRMJBvl2MNvs1dDpxN1GDzGmqyHIrOZZbjo
k7xQhGoWBsqFS9Erz1o8li70l4ZC2yG7oQPTVqMGWMGza07GVK/jN4fT+qwTCT3+
P7X7i/D8UahwvnELlt6GxnNIpSY/vdQ4A8Cm9Xj5WpZ7hbsR6Bo5S8Zh7zzl0SAR
pkZ0ImtC//9+VbYh4LjHndQhGtyKBmL42J1yl4Adll2NvbNTVS3GU1CL8d3DVay3
IQ9T9adR8MMH6JcwZqrMoqOTRYarMxe+PyhRHAQJ+bWjunidInrKDpMltWSIbB56
LIXf0bYghMm8l5K1anG4LnLT7bnpLB8CAwEAAaOB9zCB9DCB3QYDVR0eAQH/BIHS
MIHPoIGEMDOkMTAvMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQww
CgYDVQQKDANPcmcwTaRLMEkxEzARBgoJkiaJk/IsZAEZFgNjb20xFzAVBgoJkiaJ
k/IsZAEZFgdleGFtcGxlMRkwFwYKCZImiZPyLGQBGRYJcGVybWl0dGVkoUYwRKRC
MEAxCzAJBgNVBAYTAkZPMRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09y
ZzEPMA0GA1UECwwGRGVuaWVkMBIGA1UdEwEB/wQIMAYBAf8CAQEwDQYJKoZIhvcN
AQELBQADggEBACBb5Z5bO8kRXwBZ5X45CCibEzXFIlCGT/ltx4pwYwjizu22zHOS
bh90opvy22DYS3CPdh5ymw399MW6w9eBGEKMF9vMAaO4ilM+Q2M1LpLtXfQ/42Pl
I++5UAQX30NdbO8yipt0gX4Iu+GkbTnrw5yk/+t8i0z69BEMhfazj7m/u4sv0ut/
HGHX2fwQvOiBRkj49MlBlIWV0DZL5ElpvQHj7RdQSvn4NW47jr33sOkRaupCpI5b
tflSTzc2EAMFuZdCGAoXFOMUqij3YiHWSU6UYPkQBli8uD1vYPSViap+WT8AeiFv
OeTzR443icXmzlf4wxl9jsTHGfRADhvLciY=
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_permitted_excluded_OK = `-----BEGIN CERTIFICATE-----
MIIDDTCCAfWgAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTcyMzQ0NTJaFw00MDAzMDQyMzQ0NTJaMD8xCzAJBgNVBAYTAkZP
MRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3Vi
Q0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCY5bIR8o7qsn4csuqR
7aeg266455zodK93EMNRUA5MMVa2dSTJYigNvkgrxJZtcftG6Wee6Ggnvd6AO8wC
pigLog7VEsH4eMU65BEZ4qafTKoRZm/N4I2JcApaL6DswSdvTjyPS6BNzFYvgnoc
cd/ohhqpIDTxlMQRC5tLWY3JDdN7M3azAdVW3wdaS7ED9jzgBCoM/r95VXSlsBss
4N2lLQ0Kiws1znbRsIv1YbMMTwcSAoGuCyQ7oRHvXicx3Uvuqmmkqvvx4XNfHyKd
C3MkxSynxeYBMnEOqVgsHd0UfxY/eiWUvpNQAUJErp3kQ8a7JWve8/R7OMJnr78u
mwITAgMBAAGjEzARMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQENBQADggEB
AHuotykffPrlv2U5tRH3ld+05d4Hg+O9hehwRpIFl/8bbCE1ODLMM59RilDgvp5m
hdsS5R/Ml3PyuuLvYG3X97MjWs7GgKrg/ujBFYR+NboiAgdmwxmJ8VofoHwnkoJH
Hkdz7BLuP/fxZ1by11I+VqIYzcGhGq2RvUwLv2njF4+hllPuZ2omHbbIFI6poDfS
ra0KukiyYcxxZkusKvURYnHwA1v7hTbGwnK9nSQnLdpd0PXvGj5L5x+OHTExywzf
hZ5Vr0ypm3R7sRWY//+CdfHYrRzlyGV8noM+O1ChM+WVTW/GN/TFfirH2HP40Npr
AGqLZQf2YoJszdfGQx13tX4=
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_permitted_excluded_OK = `-----BEGIN CERTIFICATE-----
MIIDFzCCAf+gAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxNzIzNDQ1MloXDTM5MTEyNTIzNDQ1MlowPjELMAkGA1UEBhMCRk8x
EjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQ0wCwYDVQQDDARMZWFm
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxvij2deQVA9eQoL7ub3M
KVMOyeab9ykDgwwGv8QUaEbhd7ceWRK2JTl52G/H2Fvjs5dL3Qs2xQg+A6J/cYdF
FwmJAcueD9UBXf6FjLmMVpkWHnRWb88lNCaiTkICxgBClCL1gXjdJmADc8JYMYMU
Oe6wRryR7hU9pAfvv/b5Af+PX9yZCVU0W+sV33E4RLsxXh00WwvSATAZrKJAyNH8
d3mAisKLPxiGoe8ScRM79QXseCmN2DyU+haH0gJQ53/P7E8bjyE93V1uzcTZ1Y4g
qZJn2U8OddcfXlaQrtXKaTK7M7ew7IK/0BDtEKVHuq00Rw06MAlJC3ZtFkVz7b7J
uQIDAQABox8wHTAbBgNVHREEFDASghBsZWFmLmV4YW1wbGUuY29tMA0GCSqGSIb3
DQEBDQUAA4IBAQBECiF98p9q6xsjb7tVFm3M7o2CcZz41rfI5uqNygL0NhJ/UAKL
uMRJAX8uZryAY43eDyENV8pG6E9eD1igicfOhHM6OOfYpVsKp4mGqolsJ+4LY+Xf
OBK6YCb7E0XxKm3Jd+BwbTZzAQREHAszs2DExtRnKSqVadK5iibuBiTqj7z2uQs2
IagL8eosXur1YlAofPsOcYY2F3pGLuUOarrGhG6pG3Zk4hIrRMSityydRUtPP5s0
BrkIuCAbQfUNNvZTTZHe3lGx0NS6dKZEifv0sNa9r7LE3P//3/ygMrdySnJdIWmQ
Fos0rNtOdCQiGF9A0GEe8+x8XvSQIduzO/bP
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_permitted_excluded_rootca = `-----BEGIN CERTIFICATE-----
MIIEKDCCAxCgAwIBAgIUN/OCI30gBFPmRoocQis4QA5OEpkwDQYJKoZIhvcNAQEL
BQAwUTELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQLDAZEZW5pZWQxDzANBgNVBAMMBlJvb3RDQTAeFw0yMDA2MTcy
MzQ0NTJaFw00MDA2MTIyMzQ0NTJaMFExCzAJBgNVBAYTAkZPMRIwEAYDVQQIDAlQ
ZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEPMA0GA1UECwwGRGVuaWVkMQ8wDQYDVQQD
DAZSb290Q0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDJalqWEi1E
6bWCCprS19c9IFSnlJdap0XAf1tMATraZ+PA9Tln59hXgCeerfOj5juSK7cVmBF2
WADibScC8gAa//QxOsrUzIR9HEmxDU0dk2Ky6+5dsGureyH6KN0ODJb0igXjo24V
t6pLyWugb/VHUhe3bSPO9tp41UiUe27RDuTKQs43DstM/i5HpAcZ6C7UGKE4JWNR
5/09qQdVWyDPllVC1kHP8lmFJMiHFpJQLu97jFKimIlsJRi0FzhEZ4bstSdhvv9g
JbR6Lf6vCl4KZx0SbAb4fgwoPMhxhshiIbwQg1R76kQCkH8n8k26+p4piN7Mk/yk
WWuZv+T9l6YxAgMBAAGjgfcwgfQwgd0GA1UdHgEB/wSB0jCBz6CBhDAzpDEwLzEL
MAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnME2k
SzBJMRMwEQYKCZImiZPyLGQBGRYDY29tMRcwFQYKCZImiZPyLGQBGRYHZXhhbXBs
ZTEZMBcGCgmSJomT8ixkARkWCXBlcm1pdHRlZKFGMESkQjBAMQswCQYDVQQGEwJG
TzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAsMBkRl
bmllZDASBgNVHRMBAf8ECDAGAQH/AgEBMA0GCSqGSIb3DQEBCwUAA4IBAQClCGBJ
fpzdwvHOA6J7g0AI9Dfd2pgVKlnlpg5el+atQsBobOgP4qLbvMku6xV9pDpGqh9d
tVhTyOzlEU3VKU3CRXA3m7VYlJsQ/KSjaAgxrw/d5xMlTVE8nH9LXGFvysCaZPMN
X3vW3NLGpagZF7NOLsq0QdlUsBcebLLN8ylpBprhdzu8gJimsfi2kqK62bLfd/fn
z7T02A6mz5+Wsc94SkYaGlxRkzUeOzTa9OobWRPgwWOY0vxLcmuEOtXDAQEU8FKa
oMeBUF7ioKlpFwWgDBcvqACu4NypB21fal/udhPDf54s0UMk6SDgAVSHCtA+fkYO
yvfJQWyYi1FqKLYa
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_permitted_excluded_rootca = `-----BEGIN CERTIFICATE-----
MIIDHjCCAgagAwIBAgIBATANBgkqhkiG9w0BAQ0FADBRMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAsMBkRlbmll
ZDEPMA0GA1UEAwwGUm9vdENBMB4XDTIwMDYxNzIzNDQ1MloXDTQwMDMwNDIzNDQ1
MlowPzELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ4wDAYDVQQDDAVTdWJDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoC
ggEBAMsVBEbP1/GskjZsqqLA8gh3e3EOTK//OiATz41UNK3VI1e5Rkc8AO8M5ZMj
SmRdxp04eXPP6MDOjKKOj4rQNH2UdGXi/nsMeYguvaQNamiUoipcsxrkNiYaiAR4
W7CpilV3g0O3X8nvTnN7FjVZ0JAoJeG/I/yEjRIZpbLZK6H52dsV0lic5MPNvEYu
Ev0TWQms4cqzDnzKg0/XKwjFvAX7kv6z1uvvip4JvDdlmYr6UKP2wZ7HBbMnOzyj
ZViF/gCePufpkkFmKYC7rNtqVnVVt+xqVPX6uq5jyGueEybZCuCgPlnbmKPqVWOW
clkMVl/3B4qLM4icdFEt+L3E6PsCAwEAAaMTMBEwDwYDVR0TAQH/BAUwAwEB/zAN
BgkqhkiG9w0BAQ0FAAOCAQEASDjFeGGAGFvvSgGjUzWfSURoqqKc1NzokXYBcvKs
XWLKXufRPeZWgEb87+QlSIQsd+punSEpPIlCWxGz/5iXS99YDU4enWgpZBB9mKnb
Y1MZWOTEYRAptuQ11Aw74do991unNyxFhUNBmNP2Vd8VP9ezLiKSoSxsRKStcVlv
OERLF9MILFmTQh/F2mV0nm4EbTsPhdkmdP7fG9hKKiVKyVQ+7VNo02APfibz2Bdb
zANfoW8iBqSqOEfaiQ87ZlNYgR9HX+0wQRotEEo2nRaj8zWc7q4cf2CrpSpnwMzf
vTEH4jCyq1Grw731lAAjvC6dLgdu4StYXMm5VX58GJq3Ng==
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_permitted_excluded_rootca = `-----BEGIN CERTIFICATE-----
MIIDFzCCAf+gAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxNzIzNDQ1M1oXDTM5MTEyNTIzNDQ1M1owPjELMAkGA1UEBhMCRk8x
EjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQ0wCwYDVQQDDARMZWFm
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAu5OyPa7p78CQX2ZwiT01
fhIgQGqTBfmQWQSQKBvYySCNTFnVBeeNxdYcHJRPlemYTJwva0S/z4zsJx2snpXW
ukB2CARUQxWrvT4soJ6f0R9UKFj21vefanvZn9P3zERzFji1QpuvYelqzNtalS/d
7rQmaNkqH4vgLFe2rl+mZeQBjHsFd4wlMU0PBNTEO+RScpY2jjcbBTBq1YMRLeFd
6FplGeWlSQCmYZD4HCBzzn/BEE0xRHNYB4ez39byPR7p0+AszRSdz0K7+OF/aTha
XChq7fMCk8Zh+4plKRvOh3LCCsc+cjHxK8CPHAU9XArM9r4QG4qmXamsLeYmZPxk
5QIDAQABox8wHTAbBgNVHREEFDASghBsZWFmLmV4YW1wbGUuY29tMA0GCSqGSIb3
DQEBDQUAA4IBAQBNiY3fPOBhrO5sB6m2Nd+UvbZjHnnSF5Jr3J5a8+/vIvFk/VVa
SsKVMxVgeQCcFsl8XSdJ0tGAlUKDpHUysUxsORCeqGMERlMTVBOEYnxf/zSo11Ib
MSRoRg61HdFLrXOW7w+DC5gvml8sp5qtD6C49WuppkcMk9eG7MwquwPyy00RR+Rh
d8gZeAg9/S3pHJIlD9knOsJdjXSnSsI9Y/f4l6/DWix45LZSsgNTOY6kNShiymq6
r0LNUsrf4EXHfT09ue0G7wM+jFNKce9/jVODd32RlauwoELCmZDn/NNE1fmgM94B
x3TmGdDmchE4ohuAxsd3QS/eP3fzCjSVUxWz
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_permitted_excluded_subca = `-----BEGIN CERTIFICATE-----
MIIEBjCCAu6gAwIBAgIUZoGLxoJB/t62La/s4Lz3rPlJeqUwDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDUzWhcNNDAwNjEyMjM0
NDUzWjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBANe1PwY2H1aiQ/I4x895MemeNJ6FSesXTwuVXAnfs1bq+bKnqQD7CLzO
edaBtmMZA7YRG/GFzUKcsBQpaHoGUqLOQ89Uz+JtjVzeE5ytqoUUfd7YxEphmNZV
Jh/yNqyoOoLQWs5ip5rNKPZgENTt2EiPZUrUFsuBs/drbla8ZJoF8w17wLhneanF
TlydJvnzUtqhtXV/qclMIbqi+CjGyIHtaQORR41nXu8JIEjzlZEUgarLZ4nAqU2q
3NhBJ03AaklcibAAXEy5OxYF9jV2+1gkkA0xFMB0Rt0JyhiC3E6QH/m+PjkCrKhM
tnw/mtrLAAsAaIP4t4a7HvXViWrckK8CAwEAAaOB9zCB9DCB3QYDVR0eAQH/BIHS
MIHPoIGEMDOkMTAvMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQww
CgYDVQQKDANPcmcwTaRLMEkxEzARBgoJkiaJk/IsZAEZFgNjb20xFzAVBgoJkiaJ
k/IsZAEZFgdleGFtcGxlMRkwFwYKCZImiZPyLGQBGRYJcGVybWl0dGVkoUYwRKRC
MEAxCzAJBgNVBAYTAkZPMRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09y
ZzEPMA0GA1UECwwGRGVuaWVkMBIGA1UdEwEB/wQIMAYBAf8CAQEwDQYJKoZIhvcN
AQELBQADggEBAE6Qfh/vmiCdpYXtQe9IlFZeH4429HPgGJ2rEo42kBt7BlDffH2+
8hfMW8TDMrIhnQqNLfOCr1wY5xybnXdURcoB4apjN1weFxsvAqJCAOXKHD/z1EHB
gyxwfNsUbzTfsMgX4ofa7OAKxhFDy3MVFnFQZsh2AWOwwr0t/abmw8fbeQzzyu+/
r0yFyPEc3nnX7SYVVoWyHf+yZg4OSDcHe7kazuHoGfbPEjLFsAdIORRVSzt3dSta
6/uhtKinfChOPdyk8yQB2uD2sE/UW1NLwQjm1r87eea38M31+UCAEhRqJXy3Gijj
+ntI1SF2rRUD0AKt08pKnhbcoPtC4ghZFw8=
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_permitted_excluded_subca = `-----BEGIN CERTIFICATE-----
MIIDHjCCAgagAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTcyMzQ0NTNaFw00MDAzMDQyMzQ0NTNaMFAxCzAJBgNVBAYTAkZP
MRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEPMA0GA1UECwwGRGVu
aWVkMQ4wDAYDVQQDDAVTdWJDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoC
ggEBAM3MbMOK0Hg6jMpQInol99Qo81sG4zTkDbLUwusvpurLLLCrzOsidf2Pxrv+
nrCw+mLfCKE1h953FQnE25C84ZGIInLDhVhOYWApzPxft+qOEBYA9UQm36gRrg+I
+1vERD0MPVzEGo4X1XkcWahlYoNLE9yvIGLNM2wTi+tcaFOnqclFWlVMpHlpaOO5
9UGsPZcqBO3EDP4pFGBc/c8NPvIayqnWt3xVie0CweYR9R42TR6wa5zBYbXN6QBX
qdJSVz6CyBT9dLQpAMAVGErEKLI09FtdAkjfkrIpJLOovzmOmXKgGW23B1shPc69
GQ/amkJ7rSv9xf8Fexe+hx08risCAwEAAaMTMBEwDwYDVR0TAQH/BAUwAwEB/zAN
BgkqhkiG9w0BAQ0FAAOCAQEAYpQNA3tDt8wjI01RaJuJD2qj+mGY3Awm6rhGEuPD
FbrR6Tx0BmuVXBFgBc0HkKvx2VL0MIzJ40sekl/4RPtHm1x7TIR0U/doqG1npEPk
V+YIQE2lwjhsJNH+yxYqQKhPd0aHrI2JMbeURuJURMlqXJ2P0wM/no9z4+7r2fmk
Voe4Yek22geOiyBo2Y1Gl7CyvPMuVJOBuzjssu8vciVYLaoEaQO49w++nfcq/8mr
ToDTThwne5AESDXi7ioyIMM7N+8SLlAB0xu0ElpgEqAvTZzYWDDu1rNKnGFghYgA
+GVcVI2E9k/eYD+wMR/5aCn+311HRUh7HorJhpr82JFElg==
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_permitted_excluded_subca = `-----BEGIN CERTIFICATE-----
MIIDKDCCAhCgAwIBAgIBATANBgkqhkiG9w0BAQ0FADBQMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAsMBkRlbmll
ZDEOMAwGA1UEAwwFU3ViQ0EwHhcNMjAwNjE3MjM0NDUzWhcNMzkxMTI1MjM0NDUz
WjA+MQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANP
cmcxDTALBgNVBAMMBExlYWYwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIB
AQDTMfU76XQ0fRzY2vuNEcB8FMO4Z1el2DoQHpp8FmvrCzd22PDU7KmnrPI9GJbD
oNZi644yKGS0MsoAyPvXwz+DxvxeI8vEz+VMSHWYR3bCyBcnTXurjDQpjXvlmmCK
nljgYrg8JAw1kirpGYbAV+QEOU+mdxdS/99Ux/Gn8WT2XwkGdhbZq53R6v4t9t2e
4lO45sS+uVadhOfKyKERtQj9n87gxjHw0E4LmxkxMAkWllYWPIVFpcVM9XwvkM+1
QLm+5alXt9h7UG0WcsC7fuP2eoBmZ9f9isT1cj7+S8cckZ3NsY2AoEMm3Gueza5l
HxEJY0VB9rJNGww6bILt1fcvAgMBAAGjHzAdMBsGA1UdEQQUMBKCEGxlYWYuZXhh
bXBsZS5jb20wDQYJKoZIhvcNAQENBQADggEBAESBbtTpdEbrJPPKrc+FVCcC+nPD
EMT6FQgnlZvplz0AuMlZvQ8LeSaS2onDy6BwOrFiZF2tLOz3X14WS/L9p0fMZn9H
Xxei1iR1rOlZaV+wUaLlpKFjbdrs2Y/6+586ikymnCLETnev66JVrcDth67Q7cqO
588kio1lWkbhaaIzkB8U/vEEzXWZX/b5OmZtp+F407DPK67Ubu37v9FZ5h/8M7t2
MRKP0OEWOOCk25WCMWpQrWj4WSjcukUykuG8H/B1XN9fij4vE48UGORxY+DpzR81
u82w8+dHO2+vc5bazKvO4kIK1l5bApcpEZdx+7G27dN3fN1IFEHh7q3Vu94=
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_permitted_excluded_leaf = `-----BEGIN CERTIFICATE-----
MIIEBjCCAu6gAwIBAgIUZ/AwWIImSJMUCRlk+v3NcAx9wpIwDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE3MjM0NDUzWhcNNDAwNjEyMjM0
NDUzWjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBAL6ZyXjgtDCmhyY+Bylm+hwfcR6Mmh1xZUV/zv/3pmejlPuqktdIjFKg
Ja3w9bTrgct+U/ugD014b4zNHiwQlWGzDL/zynB1/STJFWzNKwtYBclQhTqZX3fs
e4c6d2bnbe6BXghRQ/qxiknmAIJIfI4DhcLZTcQ3nQe09P1oxJQM4+MMWR9c8tnN
3jeqrVUUDpatVL2VLuaoYDStvxyeiKcJ9psZ5aFv5hALbx414PGKkiyOnZnHqmuB
92wzHv6PGihdViPmV3RbnQEqpcYNKA/uuLZr11rfxRfvwu5HsTDaps7BTTUseT/1
jucBG86Y7qZWHAUGLXhgXGj8W+oDHOECAwEAAaOB9zCB9DCB3QYDVR0eAQH/BIHS
MIHPoIGEMDOkMTAvMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQww
CgYDVQQKDANPcmcwTaRLMEkxEzARBgoJkiaJk/IsZAEZFgNjb20xFzAVBgoJkiaJ
k/IsZAEZFgdleGFtcGxlMRkwFwYKCZImiZPyLGQBGRYJcGVybWl0dGVkoUYwRKRC
MEAxCzAJBgNVBAYTAkZPMRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09y
ZzEPMA0GA1UECwwGRGVuaWVkMBIGA1UdEwEB/wQIMAYBAf8CAQEwDQYJKoZIhvcN
AQELBQADggEBAIvfjB5gfRUo1IXnzQwbYCkONLZUHT4hlQTtuEYZbRlt0JKdgmmg
TGqyzdtfsp2I2ph0DfVEaLRT86zRsl/w7oXx42JSWiAfcF+vE+2pVVT7mjMKEm4k
A3GAt5h5Ob+5mMj1rBgAxIPa74K2qVWVQGReIJkmkLUYiMwRHgvdSIgE/1ysuu7d
2GrE94s98AGRmLtHVVAYC3+aMf2Pg3/0gcPBG+LKw9ZPbjKFTxPOJy5Uoc+r/UFl
61rhY9wqsjfiaGA7nPzZPeFcwlBWL1DRW5MMOp2xl3epDkluo6MW5aNFunAsF7IG
48G6IU60NnEUfhVD15f34sIjmYu57vQU6j4=
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_permitted_excluded_leaf = `-----BEGIN CERTIFICATE-----
MIIDDTCCAfWgAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTcyMzQ0NTNaFw00MDAzMDQyMzQ0NTNaMD8xCzAJBgNVBAYTAkZP
MRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3Vi
Q0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCk66g519/o7Z1ZqIz8
3ZC5794Vhe3pP2LgixfaFslNnSaL+8Jw0mQzVPNS7KWjK2wuc3a4dKWfAEff++Bj
BKCfNiZhRo4B7F1cycSS2h0H2Hs/CiB2FjECdVGwE8vaPJ3I+IllyHD0pNu2W3kZ
Dq24HHeIiPaViSaGAYRwXm3AamZxaWrUyfmf1TQcNZaU3BgyFerE2Oq2yMN5FifG
NDd63kjXA3tLHiihBOFHnZ+SsOvldeGO4HgGYrHh05qomBX2t4zkoJu25lMN5KSX
6y8k7yuRpYA95LYcj7+rOEdtcxugBX+2xDVH+to9SmV1Tbksc6AlKuULKMaQs731
8NhnAgMBAAGjEzARMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQENBQADggEB
AGbNb6n7zNuV59aQlg2h6/BxfEakRqRCIuZugjmTGpJyQrGs0srZW+TZ26pTm50J
8jpTje0IeKiR5tm9TiHLjb/vCP70Ax4gyl5GxxsWL68uURhdr11E5FtUEz53VJGD
qJAzUlIS03Dxt2iri3UmpwHmOJ+2fQeoyUOYxCLXxNHOcYKC5M+c0KpHyve/sPp4
hqv3Bbh7hOVMyO7Pp1VdxUUdXRI1uCIC5SvME9+/PCYRcPhURNIp220IyMhGQAsj
t6hPoqywJJMtdvEEJF2U/cnQkcEfG6K4vMo9UNXSSsSsVf5LkvMUOApNj5Ix6QdH
6vMbY3mqAKmbZIG69Xn88VQ=
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_permitted_excluded_leaf = `-----BEGIN CERTIFICATE-----
MIIDKDCCAhCgAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxNzIzNDQ1NFoXDTM5MTEyNTIzNDQ1NFowTzELMAkGA1UEBhMCRk8x
EjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQ8wDQYDVQQLDAZEZW5p
ZWQxDTALBgNVBAMMBExlYWYwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIB
AQC/8TMEC7sZFol4Mq3lGOfrSsn0wKJ2pXfCSOH9fddBtbyev+SDMRO1kpgvUIbF
i76hFhh0TGxWv3gqHf4ducn2SaY6txFJ2KnkRsFh3a3OjJQolh3cM2tgH4dLKr3b
Ig9AkicN2cnE25LMA1k5cUOgioDUSF1slvj5vKPZLEPSJROYjkcqYjQoz40OOPcg
gzhX6gdnDJIIQ0fuhPxTB4XqUDLQaAAJAFXJlEqbppJ4UUIEAcScRjohl5ev+xNE
5oGoJAgf2t5fnRnDJ4Xc9Gf4UdntgO3L6+9Lb7UR3xqfYVrqb5uxXkNSJTLaHsSo
Sigzs5Z2yjNTf3C6jUbvpGtnAgMBAAGjHzAdMBsGA1UdEQQUMBKCEGxlYWYuZXhh
bXBsZS5jb20wDQYJKoZIhvcNAQENBQADggEBAAKx/WzBwH+Ct+OFLfhubUCBN7+8
LSJykUzzhxwHsvz1Wznl8HtnaveZfsT4Jy2TVSYtq9Uw//QSkbE7T5GNgUVgJX8F
wy4N/r3oe5FhHM9+3Gt/Y2emk57TGwYKNH3CkLLV49HU9ZAVs4zduTnIh5GEq5ME
fe/nED3uMLG8hjoJcbkFrPM93tDAUvECFj3CT6T3Tkkgayl5ne6ze6SpkptLnDz9
QqNkHVN1AabOnotzBLhoQ377aY+IulXI5nTPLCqOsoc/25sseQ3MUcz8gPGDdad5
LTr5PR1yDHhapT53uuEphwmgx5/LGSRpS17xVBcN49dBcRDH+7Y9jaJx4UI=
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_subca_restr_ok = `-----BEGIN CERTIFICATE-----
MIIDvjCCAqagAwIBAgIUdgdVSerl539uQkUf2PkVF8xeZY8wDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE4MDEyODIzWhcNNDAwNjEzMDEy
ODIzWjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBAOr1t17VOPHNQOZ1Cz9WGmryTmqP0DFOYpEapxoJ0mZtAfZ286Y649vb
Fu9TKaGy8cPkwOfDfFlyh68kfMi2SiJxU7b0ETyBdkI01fRlbUgzq043etDkPh+L
5Q3EGwd9nhQHwzPUKZFgk8x56wYW7lWcRV6wEiqpUegZYXdvuCzKaCMy8v3lFP1V
hPDWjTS/eudfUbsFgjZjtwy/DjzeMT7EmbmNafTRV0xJuc1+sjr5omSXLafH5zkZ
BqAbqbBpUEPuHncgAKUpjIpwkzJZlJ/WjzZIBCH0mQMoMUiOMEs2FqGvSpOCcYHd
sEtSBeJvy96j+dNsJkiB0V8ns4rEO6cCAwEAAaOBrzCBrDCBlQYDVR0eAQH/BIGK
MIGHoIGEMDOkMTAvMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQww
CgYDVQQKDANPcmcwTaRLMEkxEzARBgoJkiaJk/IsZAEZFgNjb20xFzAVBgoJkiaJ
k/IsZAEZFgdleGFtcGxlMRkwFwYKCZImiZPyLGQBGRYJcGVybWl0dGVkMBIGA1Ud
EwEB/wQIMAYBAf8CAQEwDQYJKoZIhvcNAQELBQADggEBAJi4InxaMxsNPnj9z+GI
1O3W6oofqQTm44+jYiwS6IGAw9fSp44HHbRigT1g2hGgz1T1FHJ1X2GbjakxQ25W
IGr0QDDMA5zYFjnX6NsNEpGcbg4HxAyaoC8HVhgI9xLNkHGZ2Qy/TJw0zcY2BlDZ
h42tTHc4pWzxyzx3o+nSxddqfJG0+xrOA9Y5NmSLOg2sRsdEETlOC8Zng+wZW8Ei
M/muDwyV0KB62lyPzPjxBEno7dmlMr69lHmVPAvArM89vqA2jpB3v8YugLjuQ/Pg
KlS8Rxg8FH00vlrWObdwWIc7p8cQIf5H8QRHCqURdbtkZ4XtG79I7Qk5wauzmZfN
f+s=
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_subca_restr_ok = `-----BEGIN CERTIFICATE-----
MIIDmTCCAoGgAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTgwMTI4MjNaFw00MDAzMDUwMTI4MjNaMD8xCzAJBgNVBAYTAkZP
MRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3Vi
Q0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCyG4FreI0nED+bgVY6
piX7DborHXEM77PNpdjLriZIyTYVWCgUmj0F72KRHWbkVtYrxVhy0JjfbJztvWoJ
RoXvvqi7RcEGa2ak0HPIPvV3iePqvsn8PTWC5tyrI46gMsIMJ0vbqjiTE4JgdipN
F/E4S89N5NudvrStA7Gob4z6AeIvmoHr3Gypy87i+in0xiHmIO99T3A1vb+SOwOy
IzTPXRbF4tp07DO3q9u8JpPys1CPlgxs9pnrBA2unXNv0grM98HXW/OulgVO/T2H
NHvkFt6GjhZOTyiFK6ROnmn82iveP2BqPTzZLVi0Pwwk8sC7YMipkc5iQ/u8FT4d
6hvhAgMBAAGjgZ4wgZswgYcGA1UdHgEB/wR9MHugeTBBpD8wPTELMAkGA1UEBhMC
Rk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQwwCgYDVQQLDANP
dVgwNKQyMDAxCzAJBgNVBAYTAkZPMRIwEAYDVQQIDAlQZXJtaXR0ZWQxDTALBgNV
BAoMBE9yZzIwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQ0FAAOCAQEAQC4m
/2AhJb6C2RzWtaOjk9OF+Y6rCgJq6zswwXy17apmNxG2c7ib6TU5SOf2A+IFAhaN
ZxhWXzVqq9H+qOT0HuhKdBcTf4SUncvQ4mlgo3fKZfgcVDZ4NC01/JfD8G768lVT
R1PJQhpR0SqrhUPhMTw7+3Dm0kHIO819t31AwAoFWH+QLOB6/w0bswaTMdOgVHGc
/2d9X4ZJfE5ufyOrf9EFxCQCFCSKOUzpzHxIDRgd2pBvvAS/+KO65tYsHO84Hc3v
cIWtYq48e2jFA7kWzKmosKTlLqwBOYqNQ0A5yrkQRhFzl4i1BwfQ4OMEJ3Rf/JIW
R2b9G1qpy36jqZQzxQ==
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_subca_restr_ok = `-----BEGIN CERTIFICATE-----
MIIDJTCCAg2gAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxODAxMjgyM1oXDTM5MTEyNjAxMjgyM1owTDELMAkGA1UEBhMCRk8x
EjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQwwCgYDVQQLDANPdVgx
DTALBgNVBAMMBExlYWYwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDh
w6nC02HVa2rOD/66Nzv7e+JNNePz7WkDUD9sA8Czjgly9gB94XCAmW38tovgAkcx
WQNaYermjaFVefWmicyjJDpdEsjrKbL83c8shp9a7a4LFlVSroZwjO7W8ioclPVH
dsb6OUq2t/l+roKa70TjSJv7ZdZI3//pWlnvhgCCFpXh4eIGxRTRw7xIMaj8Naw5
e5kdeT5hZ1BmLxwhZE0yFMaxVMfCAYqbT/MNmD7a3UFknRkafMfpGU/FXs44F723
jTywu8aattRiq/9NO0+NfrrSP0ALGQBXzsJMsGXFl0JBvNgXysbBGdMuvNwOSCa/
iBHoBXn7wKEWWDdruDHLAgMBAAGjHzAdMBsGA1UdEQQUMBKCEGxlYWYuZXhhbXBs
ZS5jb20wDQYJKoZIhvcNAQENBQADggEBADkGGgwWLTMQ+P7wMMTE32AC940rBHGM
yxckjRKCN0w2O+eu5vx7MjAMKb/dLT+KSKliPeRaU3vLaetXuIsuCnqKkdAW0V8e
hMjPuj1ud40Un5bdRmWhLjr+bok+Grcqb1HwRX82kzqkxY2wQRNrYwplJ2NBsr9W
+yBHxe0mcsKCMOuz52lXBwZ+dmOBzotcxRCQqB2WXPkvHPxVdbywLcT5lyC3DBMB
z6ITKAjZl9Zv/pM720649GpqW4nJJzCvnPGA6nhKmLhbr/QaXq/cwJ5C1IObNVW/
U+IxRmdHsRiGIDgL4rt37u5VA9QucSsfHeboePYjMU4nM4f6g7ixtGQ=
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_subca_restr_fail = `-----BEGIN CERTIFICATE-----
MIIDvjCCAqagAwIBAgIUY49WkHoq+tF4DzsOBI/lYhJjFDgwDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE4MDEyODIzWhcNNDAwNjEzMDEy
ODIzWjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBAM5I3sr7WI+CdKd0KpTH8PsIdt+SX0dtl5hoojYtaUdn7HMAwRDZbvVH
jy+/pvehrRN2FTtUngy+4qJaQsnFT1mBlKXhabLnU0vXHfkjVSmCYsC01KNjMeeN
Ilskp4jXkp38HK9tGH8VpWLiaICmhxsqQCeEDGBMKfjTm2WKrK3GbDwQ8ump7fQx
JSSpt5t8H1D01oXjvsZpefWjuhl2u18cNJ1irvmTmRw4aTEI8c5R4cFK+BNmbab6
fkRsvZNmdBQw1Q8G4zXGqrSx2xN7BmY58hZqrAQA+dVxNSitc8dElCZ6D1r/et0v
S8qZ+GbSAB2Fy8aF8dULy1HU2ZHy4tUCAwEAAaOBrzCBrDCBlQYDVR0eAQH/BIGK
MIGHoIGEMDOkMTAvMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQww
CgYDVQQKDANPcmcwTaRLMEkxEzARBgoJkiaJk/IsZAEZFgNjb20xFzAVBgoJkiaJ
k/IsZAEZFgdleGFtcGxlMRkwFwYKCZImiZPyLGQBGRYJcGVybWl0dGVkMBIGA1Ud
EwEB/wQIMAYBAf8CAQEwDQYJKoZIhvcNAQELBQADggEBAMtx9Ky+BRDma0CfMaSE
sN/Dac0F1qZj02FXKrgVRC/ifoKVYOjAuH1RveSjpDMnUYQ6qEnWhiXDuZ87C0a7
KTQuckGZfTo97hozkS6HHSVAwZi3P6G4DoQsqd+BXTYngYON2xGPgigKD71zdZG0
04QJEr3BAyOCft8kIzF3bsP3saxW/IdEQGYSR7WwbuP2XZvg3ZYIgUNpOxZH/Pi0
8MeQjm9PNLFeuzSAKLnioMhO41csEPdkB6ExS0DYfrbswwe16IT1wIPkRaMJRZN8
zGKlqTKP9Ft3jkSD5Q0H1noKAmZceIPUQEBoypCRKS5GYgjli3vPMpfneBSoQjf3
3Y8=
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_subca_restr_fail = `-----BEGIN CERTIFICATE-----
MIIDmTCCAoGgAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTgwMTI4MjNaFw00MDAzMDUwMTI4MjNaMD8xCzAJBgNVBAYTAkZP
MRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3Vi
Q0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDsKBQgO6RZ1N//V+59
+xhsPPO57gdzwlb0p4ifA6MNpg2p74XjNSoYm/TTQ2xs4Gyxj0Gp+ml2XUTMp6Oo
x7e9Z6MIJD1j6KFf2kapfH9mi6Tg79sa8pHSTEDKRFyNc8uNTwi4KwrQgT7cJLcp
hB/jfv2BKRVap8x/IkOI+Wk4lruVQhUAeRVEXElYlrQms8E93v/XHBkLbh1v07ah
Syf5gGtJ1YaC9eSkbGLXHCmUw4AlzyK4nH6lOPEfQme5EkhUOtWjsghkvCUmkMl3
yuk52/O1XK6D78eWA2MqjwFZ9oOKlSzSJiB56fpNqT2FYRQdgatBRYLNLLaIDpul
N1EbAgMBAAGjgZ4wgZswgYcGA1UdHgEB/wR9MHugeTBBpD8wPTELMAkGA1UEBhMC
Rk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQwwCgYDVQQLDANP
dVgwNKQyMDAxCzAJBgNVBAYTAkZPMRIwEAYDVQQIDAlQZXJtaXR0ZWQxDTALBgNV
BAoMBE9yZzIwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQ0FAAOCAQEAtS9k
V7Yv/c858v8kBdaGWSd3r14iMOuQznCOE3XTPrS1C0hinvKjWcx5iWIEJtgpJDy6
nX7YtR1RPtdyKkWG9dvY4AtDRN4zEOXdLR3UthDPlLSaZ3QIXUKw61SBtcHJfa+J
HcEC+ICjxxwsCcTFBcY576tE/OcM7e4kqgLkbz6Hzd0l9N5UYWNfQT7e6EbfSDhn
4WmPxIAKsE1Csn95hVymC7NGqFM+GOUu8KW3XCiMMcj091a4x+BA8Ks1+rQqoMyU
T6H2DccsggD+frCgtKgxOStRaVh/mOYGQjbbQyfg7/57bMTE312Q4D737P+6kMsm
7lqOF86/aYqAoBLGIg==
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_subca_restr_fail = `-----BEGIN CERTIFICATE-----
MIIDFzCCAf+gAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxODAxMjgyM1oXDTM5MTEyNjAxMjgyM1owPjELMAkGA1UEBhMCRk8x
EjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQ0wCwYDVQQDDARMZWFm
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAyQkri+DYu7Oc9jOp9yTb
enYM7jtJoWOJd41H7ExriVqAwMaPaFvIIzGdoi/iOEp1mRjLLmeypCjvcsmz2G8b
ocsskkHzKrLSvQ6qEY85i0QNz4mnWKrpeYcLVWfdcMkm5I+YC0ex7RoouLULiSb/
+c3CKEwPHkFCXxLOimzIcfirjWmqTxxmeW3IM6Yi/cToptXG8nCUVQ3re0oMWmxp
rt7r8BYbZskA93s2EFMBbQBAzhwBBFxIQwfikhoA5J1khIWndqiLTkW0aizHdzOm
ez4MoLzWApyXV666CYWdsDNXOFHdnp7ok7VF+D+OvXSODd6g8M7YlEuhLuPPlQPS
HQIDAQABox8wHTAbBgNVHREEFDASghBsZWFmLmV4YW1wbGUuY29tMA0GCSqGSIb3
DQEBDQUAA4IBAQCbMXhx+QLNsYhq/xXo2TzSdlRTW64oYNmgBtPAwqW0qY5cyDnz
/X/LLQFGae2mFFgtO21/uYLs3pbuThJj4kVBSwG7fm9cEib4GLpO9Gf+DJyNel1J
SQL80YYAa8yW9lQThTptFxB98piBwIxIO8sW+P60+zh/QNORXBoDfLXvgwGs7P77
Xv9FXWKOpdUUn2MIdCX25XdTP2W/avztwrAhoo3izC8uhK+mqVEixQGIdKx1o2Vq
v0IJQk/MARF9wXWtkX6UghGSii6OA95pEyv940JHYAZ6uZvvg97pN8pGJfO89m6h
TJExl0p1rxT7TbGrrRClMg9dXSXh6F6uew9U
-----END CERTIFICATE-----`

const dirNameConstraintRootCA_subca_relax_fail = `-----BEGIN CERTIFICATE-----
MIIDvjCCAqagAwIBAgIUVJsYgTK/J+ER7U2Qpo20qAokMzswDQYJKoZIhvcNAQEL
BQAwQDELMAkGA1UEBhMCRk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwD
T3JnMQ8wDQYDVQQDDAZSb290Q0EwHhcNMjAwNjE4MDEyODI0WhcNNDAwNjEzMDEy
ODI0WjBAMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQK
DANPcmcxDzANBgNVBAMMBlJvb3RDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBAMJDPxG4LHZnM4fK8hSgFxVFplpkwbyqDIoYJPi1GgJ+PCCvjGvFqSQw
FRu0bRXOim5x78kx2/abg7kKVU7P0izBNR3axFsfUPdcSDR8kwAwaC80inmAibMP
z62OrM/rpSykRct1tVFwScYWfX8TT0LlWr21bkO3TekEle3DhW2Aj9eRSG4oOqrs
D8i7cMlLMSTq+2VwRiXXhchax3H7T23O33wKs/8aDFiuhwqHAa6YTFDBkOFmL60Y
i55a8sTV7hgQfnvsuExOIYQKvUu06iBcMywTTpE/ZrX2X53L8U5UoUvXogSNi4Nf
k/vGWhYiYl6Gby2FlC4uw1hOT4hRU18CAwEAAaOBrzCBrDCBlQYDVR0eAQH/BIGK
MIGHoIGEMDOkMTAvMQswCQYDVQQGEwJGTzESMBAGA1UECAwJUGVybWl0dGVkMQww
CgYDVQQKDANPcmcwTaRLMEkxEzARBgoJkiaJk/IsZAEZFgNjb20xFzAVBgoJkiaJ
k/IsZAEZFgdleGFtcGxlMRkwFwYKCZImiZPyLGQBGRYJcGVybWl0dGVkMBIGA1Ud
EwEB/wQIMAYBAf8CAQEwDQYJKoZIhvcNAQELBQADggEBAFsmKPJMhx7CQ8rhtRO1
sV4ybRu4Ghe2kE3HH84Q9RonZ5oDIAo7xNBWBgiGTjNK3jdtalNsmuErmGHLSPHh
u+nHmJWcyN+ebq9AGXwZ02YHeql0EXbgt7mp6+yYd8kPRr4mtzeWhU/qeR1nXo8W
rxRdtEt7jgeEXiD4vnGQgbCtokn2563RVMRaHW4jL8OuSIQvk8kirUBFi1nTsx33
sGs64n9SFFl8xBFSbt0UPuJKM0twV7b2P1bZgp9NoBO7TaInmk3Kp8+F/x+O3ee+
c5LdSWd1i4gT+71mRmuGLZ0GQ9ZRVS+8tNlMRQ09Sl3pjF4gPWtjRtTGBB5ZWqjR
YKY=
-----END CERTIFICATE-----`
const dirNameConstraintSubCA_subca_relax_fail = `-----BEGIN CERTIFICATE-----
MIIDmTCCAoGgAwIBAgIBATANBgkqhkiG9w0BAQ0FADBAMQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDzANBgNVBAMMBlJvb3RD
QTAeFw0yMDA2MTgwMTI4MjRaFw00MDAzMDUwMTI4MjRaMD8xCzAJBgNVBAYTAkZP
MRIwEAYDVQQIDAlQZXJtaXR0ZWQxDDAKBgNVBAoMA09yZzEOMAwGA1UEAwwFU3Vi
Q0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCxlKpMxkI/BCGHzinE
JhFp91Me94aNmhLNL5ollDc+S4AYEDhpHGPP5ilqt5+RL906lQtD0bG6qIJ7PiKd
ipya+428BbPDSDJIYQx14NBZ/IU5ZIudH2NGY3J4WW27/wn2v0busWt6ZS1+IYuB
B3mRrDyZjYB2hQlmkAEQdi9+cU5u8T7jb5RAbu6cK0b9zfPGy5lP1MGmowDRgsQE
jIuS6LXK2BXjyd9RatJjv9woCZuQ1qR+YZY5OsQLCm5JFQZZ5/QHezthokaePBAN
0kTHLEm80l1SJKk/x6AuhM2E2xsfqorSu3Vxx81uio9lLkx5PRZrGjEdN2660i0q
wUjpAgMBAAGjgZ4wgZswgYcGA1UdHgEB/wR9MHugeTBBpD8wPTELMAkGA1UEBhMC
Rk8xEjAQBgNVBAgMCVBlcm1pdHRlZDEMMAoGA1UECgwDT3JnMQwwCgYDVQQLDANP
dVgwNKQyMDAxCzAJBgNVBAYTAkZPMRIwEAYDVQQIDAlQZXJtaXR0ZWQxDTALBgNV
BAoMBE9yZzIwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQ0FAAOCAQEAjrsR
rdwDO99BnpAIWseZR1B+W7QX4PrM9Eoh9G9YuUVzILXTBAoq2X0YSXp1+etkWMd8
3Al9PiGoEfhCIPToczfAmgQtIJowfFI0wnwB5GLce1wDQEgihRkM9ful4DZQ+gdW
ToJrJD2v7gMJAX8Y3ByOm3NKoHzlyiyVgo7pJhNjIZStpqbRXKw8rS9FXAM5M6zG
ht+e6bmFTKuRTxyK1W51r0vdsV6O0R+1VH8mj5XXFmTZ9EU0ExNjXl3LOU+mjrKM
uxjzuJT6t2dGwOYEzZH3mVV/CfpIlXDbmVSsJRkh0s+ssXY3ZFN4o1yBYWx9Z/Nq
PN0V1hB67ZfEK3OOyA==
-----END CERTIFICATE-----`
const dirNameConstraintLeafCA_subca_relax_fail = `-----BEGIN CERTIFICATE-----
MIIDGDCCAgCgAwIBAgIBATANBgkqhkiG9w0BAQ0FADA/MQswCQYDVQQGEwJGTzES
MBAGA1UECAwJUGVybWl0dGVkMQwwCgYDVQQKDANPcmcxDjAMBgNVBAMMBVN1YkNB
MB4XDTIwMDYxODAxMjgyNFoXDTM5MTEyNjAxMjgyNFowPzELMAkGA1UEBhMCRk8x
EjAQBgNVBAgMCVBlcm1pdHRlZDENMAsGA1UECgwET3JnMjENMAsGA1UEAwwETGVh
ZjCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAM+0S7NZaxtLZIrg6s8N
a2JzHBYHizdqlaL0oowf6Er7c483/VzYwKJ2AB61R3Sxr3/6SQSmyjiXZW5rVWJp
WJH5MKka+LKi/fWvDWPHFc4bM0xGZ7F0wowewvvnbOGmRr9HFTqXs6O4p1MWe5Qy
5jah/HEHaRCP1QMxa2GVgMdNijMDdmvicF149yB7FY1mqbBXA0OED8fTvW/oP4ns
aX5HiY38sM0gXtx2t+HiThIB5utVhefAnNCKFmjLIpiMpAxa+JQcgdVz2PkLBAVk
473idSFVsbTL0A0Ney1kyoCI0yU7eK/I5TgCBbNnGMbJy7xdzwexepzrJQGd3PQi
6KUCAwEAAaMfMB0wGwYDVR0RBBQwEoIQbGVhZi5leGFtcGxlLmNvbTANBgkqhkiG
9w0BAQ0FAAOCAQEATYyFhjlHa+EdWdVyE4hRYEJUeoSL1CLWSNsnRqlSRUj6f47B
Z6ppHFCIaOKS1HWTRmxlR8ZZacGesdrISec21CJkfMjQe8ha4Og9mwTxYBvfkAz0
n7czYC97N7b4+wTwwGREclO4QphcgUwW1FOU5yPHAGLmKNvJpXEKN+TDya08Gmam
OQMzV3zvbJlDICnYjwgIbdlTjDvdOhGmiSF/MU5F2FXXN9kotcRQdJQL6lPIB84/
ixyGmqlNXtjKU9ADJIrNIFajvXTl4ln2Jrfu42hKCpDsvT07GLYSTfinHQjeTiUR
G1IZnzmYGDbyUgD74HsLWxQoS2zIEuwb/R2mKQ==
-----END CERTIFICATE-----`

var globalSignRoot = `-----BEGIN CERTIFICATE-----
MIIDXzCCAkegAwIBAgILBAAAAAABIVhTCKIwDQYJKoZIhvcNAQELBQAwTDEgMB4G
A1UECxMXR2xvYmFsU2lnbiBSb290IENBIC0gUjMxEzARBgNVBAoTCkdsb2JhbFNp
Z24xEzARBgNVBAMTCkdsb2JhbFNpZ24wHhcNMDkwMzE4MTAwMDAwWhcNMjkwMzE4
MTAwMDAwWjBMMSAwHgYDVQQLExdHbG9iYWxTaWduIFJvb3QgQ0EgLSBSMzETMBEG
A1UEChMKR2xvYmFsU2lnbjETMBEGA1UEAxMKR2xvYmFsU2lnbjCCASIwDQYJKoZI
hvcNAQEBBQADggEPADCCAQoCggEBAMwldpB5BngiFvXAg7aEyiie/QV2EcWtiHL8
RgJDx7KKnQRfJMsuS+FggkbhUqsMgUdwbN1k0ev1LKMPgj0MK66X17YUhhB5uzsT
gHeMCOFJ0mpiLx9e+pZo34knlTifBtc+ycsmWQ1z3rDI6SYOgxXG71uL0gRgykmm
KPZpO/bLyCiR5Z2KYVc3rHQU3HTgOu5yLy6c+9C7v/U9AOEGM+iCK65TpjoWc4zd
QQ4gOsC0p6Hpsk+QLjJg6VfLuQSSaGjlOCZgdbKfd/+RFO+uIEn8rUAVSNECMWEZ
XriX7613t2Saer9fwRPvm2L7DWzgVGkWqQPabumDk3F2xmmFghcCAwEAAaNCMEAw
DgYDVR0PAQH/BAQDAgEGMA8GA1UdEwEB/wQFMAMBAf8wHQYDVR0OBBYEFI/wS3+o
LkUkrk1Q+mOai97i3Ru8MA0GCSqGSIb3DQEBCwUAA4IBAQBLQNvAUKr+yAzv95ZU
RUm7lgAJQayzE4aGKAczymvmdLm6AC2upArT9fHxD4q/c2dKg8dEe3jgr25sbwMp
jjM5RcOO5LlXbKr8EpbsU8Yt5CRsuZRj+9xTaGdWPoO4zzUhw8lo/s7awlOqzJCK
6fBdRoyV3XpYKBovHd7NADdBj+1EbddTKJd+82cEHhXXipa0095MJ6RMG3NzdvQX
mcIfeg7jLQitChws/zyrVQ4PkX4268NXSb7hLi18YIvDQVETI53O9zJrlAGomecs
Mx86OyXShkDOOyyGeMlhLxS67ttVb9+E7gUJTb0o2HLO02JQZR7rkpeDMdmztcpH
WD9f
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
				t.Errorf("unexpected error: got %q, want %q", err.Error(), tc.err)
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

func TestPoliciesValid(t *testing.T) {
	// These test cases, the comments, and the certificates they rely on, are
	// stolen from BoringSSL [0]. We skip the tests which involve certificate
	// parsing as part of the verification process. Those tests are in
	// TestParsePolicies.
	//
	// [0] https://boringssl.googlesource.com/boringssl/+/264f4f7a958af6c4ccb04662e302a99dfa7c5b85/crypto/x509/x509_test.cc#5913

	testOID1 := mustNewOIDFromInts([]uint64{1, 2, 840, 113554, 4, 1, 72585, 2, 1})
	testOID2 := mustNewOIDFromInts([]uint64{1, 2, 840, 113554, 4, 1, 72585, 2, 2})
	testOID3 := mustNewOIDFromInts([]uint64{1, 2, 840, 113554, 4, 1, 72585, 2, 3})
	testOID4 := mustNewOIDFromInts([]uint64{1, 2, 840, 113554, 4, 1, 72585, 2, 4})
	testOID5 := mustNewOIDFromInts([]uint64{1, 2, 840, 113554, 4, 1, 72585, 2, 5})

	loadTestCert := func(t *testing.T, path string) *Certificate {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		p, _ := pem.Decode(b)
		c, err := ParseCertificate(p.Bytes)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}

	root := loadTestCert(t, "testdata/policy_root.pem")
	root_cross_inhibit_mapping := loadTestCert(t, "testdata/policy_root_cross_inhibit_mapping.pem")
	root2 := loadTestCert(t, "testdata/policy_root2.pem")
	intermediate := loadTestCert(t, "testdata/policy_intermediate.pem")
	intermediate_any := loadTestCert(t, "testdata/policy_intermediate_any.pem")
	intermediate_mapped := loadTestCert(t, "testdata/policy_intermediate_mapped.pem")
	intermediate_mapped_any := loadTestCert(t, "testdata/policy_intermediate_mapped_any.pem")
	intermediate_mapped_oid3 := loadTestCert(t, "testdata/policy_intermediate_mapped_oid3.pem")
	intermediate_require := loadTestCert(t, "testdata/policy_intermediate_require.pem")
	intermediate_require1 := loadTestCert(t, "testdata/policy_intermediate_require1.pem")
	intermediate_require2 := loadTestCert(t, "testdata/policy_intermediate_require2.pem")
	intermediate_require_no_policies := loadTestCert(t, "testdata/policy_intermediate_require_no_policies.pem")
	leaf := loadTestCert(t, "testdata/policy_leaf.pem")
	leaf_any := loadTestCert(t, "testdata/policy_leaf_any.pem")
	leaf_none := loadTestCert(t, "testdata/policy_leaf_none.pem")
	leaf_oid1 := loadTestCert(t, "testdata/policy_leaf_oid1.pem")
	leaf_oid2 := loadTestCert(t, "testdata/policy_leaf_oid2.pem")
	leaf_oid3 := loadTestCert(t, "testdata/policy_leaf_oid3.pem")
	leaf_oid4 := loadTestCert(t, "testdata/policy_leaf_oid4.pem")
	leaf_oid5 := loadTestCert(t, "testdata/policy_leaf_oid5.pem")
	leaf_require := loadTestCert(t, "testdata/policy_leaf_require.pem")
	leaf_require1 := loadTestCert(t, "testdata/policy_leaf_require1.pem")

	type testCase struct {
		chain                 []*Certificate
		policies              []OID
		requireExplicitPolicy bool
		inhibitPolicyMapping  bool
		inhibitAnyPolicy      bool
		valid                 bool
	}

	tests := []testCase{
		// The chain is good for |oid1| and |oid2|, but not |oid3|.
		{
			chain:                 []*Certificate{leaf, intermediate, root},
			requireExplicitPolicy: true,
			valid:                 true,
		},
		{
			chain:                 []*Certificate{leaf, intermediate, root},
			policies:              []OID{testOID1},
			requireExplicitPolicy: true,
			valid:                 true,
		},
		{
			chain:                 []*Certificate{leaf, intermediate, root},
			policies:              []OID{testOID2},
			requireExplicitPolicy: true,
			valid:                 true,
		},
		{
			chain:                 []*Certificate{leaf, intermediate, root},
			policies:              []OID{testOID3},
			requireExplicitPolicy: true,
			valid:                 false,
		},
		{
			chain:                 []*Certificate{leaf, intermediate, root},
			policies:              []OID{testOID1, testOID2},
			requireExplicitPolicy: true,
			valid:                 true,
		},
		{
			chain:                 []*Certificate{leaf, intermediate, root},
			policies:              []OID{testOID1, testOID3},
			requireExplicitPolicy: true,
			valid:                 true,
		},
		// Without |X509_V_FLAG_EXPLICIT_POLICY|, the policy tree is built and
		// intersected with user-specified policies, but it is not required to result
		// in any valid policies.
		{
			chain:    []*Certificate{leaf, intermediate, root},
			policies: []OID{testOID1},
			valid:    true,
		},
		{
			chain:    []*Certificate{leaf, intermediate, root},
			policies: []OID{testOID3},
			valid:    true,
		},
		// However, a CA with policy constraints can require an explicit policy.
		{
			chain:    []*Certificate{leaf, intermediate_require, root},
			policies: []OID{testOID1},
			valid:    true,
		},
		{
			chain:    []*Certificate{leaf, intermediate_require, root},
			policies: []OID{testOID3},
			valid:    false,
		},
		// requireExplicitPolicy applies even if the application does not configure a
		// user-initial-policy-set. If the validation results in no policies, the
		// chain is invalid.
		{
			chain:                 []*Certificate{leaf_none, intermediate_require, root},
			requireExplicitPolicy: true,
			valid:                 false,
		},
		// A leaf can also set requireExplicitPolicy.
		{
			chain: []*Certificate{leaf_require, intermediate, root},
			valid: true,
		},
		{
			chain:    []*Certificate{leaf_require, intermediate, root},
			policies: []OID{testOID1},
			valid:    true,
		},
		{
			chain:    []*Certificate{leaf_require, intermediate, root},
			policies: []OID{testOID3},
			valid:    false,
		},
		// requireExplicitPolicy is a count of certificates to skip. If the value is
		// not zero by the end of the chain, it doesn't count.
		{
			chain:    []*Certificate{leaf, intermediate_require1, root},
			policies: []OID{testOID3},
			valid:    false,
		},
		{
			chain:    []*Certificate{leaf, intermediate_require2, root},
			policies: []OID{testOID3},
			valid:    true,
		},
		{
			chain:    []*Certificate{leaf_require1, intermediate, root},
			policies: []OID{testOID3},
			valid:    true,
		},
		// If multiple certificates specify the constraint, the more constrained value
		// wins.
		{
			chain:    []*Certificate{leaf_require1, intermediate_require1, root},
			policies: []OID{testOID3},
			valid:    false,
		},
		{
			chain:    []*Certificate{leaf_require, intermediate_require2, root},
			policies: []OID{testOID3},
			valid:    false,
		},
		// An intermediate that requires an explicit policy, but then specifies no
		// policies should fail verification as a result.
		{
			chain:    []*Certificate{leaf, intermediate_require_no_policies, root},
			policies: []OID{testOID1},
			valid:    false,
		},
		// A constrained intermediate's policy extension has a duplicate policy, which
		// is invalid.
		// {
		// 	chain:    []*Certificate{leaf, intermediate_require_duplicate, root},
		// 	policies: []OID{testOID1},
		// 	valid:    false,
		// },
		// The leaf asserts anyPolicy, but the intermediate does not. The resulting
		// valid policies are the intersection.
		{
			chain:                 []*Certificate{leaf_any, intermediate, root},
			policies:              []OID{testOID1},
			requireExplicitPolicy: true,
			valid:                 true,
		},
		{
			chain:                 []*Certificate{leaf_any, intermediate, root},
			policies:              []OID{testOID3},
			requireExplicitPolicy: true,
			valid:                 false,
		},
		// The intermediate asserts anyPolicy, but the leaf does not. The resulting
		// valid policies are the intersection.
		{
			chain:                 []*Certificate{leaf, intermediate_any, root},
			policies:              []OID{testOID1},
			requireExplicitPolicy: true,
			valid:                 true,
		},
		{
			chain:                 []*Certificate{leaf, intermediate_any, root},
			policies:              []OID{testOID3},
			requireExplicitPolicy: true,
			valid:                 false,
		},
		// Both assert anyPolicy. All policies are valid.
		{
			chain:                 []*Certificate{leaf_any, intermediate_any, root},
			policies:              []OID{testOID1},
			requireExplicitPolicy: true,
			valid:                 true,
		},
		{
			chain:                 []*Certificate{leaf_any, intermediate_any, root},
			policies:              []OID{testOID3},
			requireExplicitPolicy: true,
			valid:                 true,
		},
		// With just a trust anchor, policy checking silently succeeds.
		{
			chain:                 []*Certificate{root},
			policies:              []OID{testOID1},
			requireExplicitPolicy: true,
			valid:                 true,
		},
		// Although |intermediate_mapped_oid3| contains many mappings, it only accepts
		// OID3. Nodes should not be created for the other mappings.
		{
			chain:                 []*Certificate{leaf_oid1, intermediate_mapped_oid3, root},
			policies:              []OID{testOID3},
			requireExplicitPolicy: true,
			valid:                 true,
		},
		{
			chain:                 []*Certificate{leaf_oid4, intermediate_mapped_oid3, root},
			policies:              []OID{testOID4},
			requireExplicitPolicy: true,
			valid:                 false,
		},
		// Policy mapping can be inhibited, either by the caller or a certificate in
		// the chain, in which case mapped policies are unassertable (apart from some
		// anyPolicy edge cases).
		{
			chain:                 []*Certificate{leaf_oid1, intermediate_mapped_oid3, root},
			policies:              []OID{testOID3},
			requireExplicitPolicy: true,
			inhibitPolicyMapping:  true,
			valid:                 false,
		},
		{
			chain:                 []*Certificate{leaf_oid1, intermediate_mapped_oid3, root_cross_inhibit_mapping, root2},
			policies:              []OID{testOID3},
			requireExplicitPolicy: true,
			valid:                 false,
		},
	}

	for _, useAny := range []bool{false, true} {
		var intermediate *Certificate
		if useAny {
			intermediate = intermediate_mapped_any
		} else {
			intermediate = intermediate_mapped
		}
		extraTests := []testCase{
			// OID3 is mapped to {OID1, OID2}, which means OID1 and OID2 (or both) are
			// acceptable for OID3.
			{
				chain:                 []*Certificate{leaf, intermediate, root},
				policies:              []OID{testOID3},
				requireExplicitPolicy: true,
				valid:                 true,
			},
			{
				chain:                 []*Certificate{leaf_oid1, intermediate, root},
				policies:              []OID{testOID3},
				requireExplicitPolicy: true,
				valid:                 true,
			},
			{
				chain:                 []*Certificate{leaf_oid2, intermediate, root},
				policies:              []OID{testOID3},
				requireExplicitPolicy: true,
				valid:                 true,
			},
			// If the intermediate's policies were anyPolicy, OID3 at the leaf, despite
			// being mapped, is still acceptable as OID3 at the root. Despite the OID3
			// having expected_policy_set = {OID1, OID2}, it can match the anyPolicy
			// node instead.
			//
			// If the intermediate's policies listed OIDs explicitly, OID3 at the leaf
			// is not acceptable as OID3 at the root. OID3 has expected_polciy_set =
			// {OID1, OID2} and no other node allows OID3.
			{
				chain:                 []*Certificate{leaf_oid3, intermediate, root},
				policies:              []OID{testOID3},
				requireExplicitPolicy: true,
				valid:                 useAny,
			},
			// If the intermediate's policies were anyPolicy, OID1 at the leaf is no
			// longer acceptable as OID1 at the root because policies only match
			// anyPolicy when they match no other policy.
			//
			// If the intermediate's policies listed OIDs explicitly, OID1 at the leaf
			// is acceptable as OID1 at the root because it will match both OID1 and
			// OID3 (mapped) policies.
			{
				chain:                 []*Certificate{leaf_oid1, intermediate, root},
				policies:              []OID{testOID1},
				requireExplicitPolicy: true,
				valid:                 !useAny,
			},
			// All pairs of OID4 and OID5 are mapped together, so either can stand for
			// the other.
			{
				chain:                 []*Certificate{leaf_oid4, intermediate, root},
				policies:              []OID{testOID4},
				requireExplicitPolicy: true,
				valid:                 true,
			},
			{
				chain:                 []*Certificate{leaf_oid4, intermediate, root},
				policies:              []OID{testOID5},
				requireExplicitPolicy: true,
				valid:                 true,
			},
			{
				chain:                 []*Certificate{leaf_oid5, intermediate, root},
				policies:              []OID{testOID4},
				requireExplicitPolicy: true,
				valid:                 true,
			},
			{
				chain:                 []*Certificate{leaf_oid5, intermediate, root},
				policies:              []OID{testOID5},
				requireExplicitPolicy: true,
				valid:                 true,
			},
			{
				chain:                 []*Certificate{leaf_oid4, intermediate, root},
				policies:              []OID{testOID4, testOID5},
				requireExplicitPolicy: true,
				valid:                 true,
			},
		}
		tests = append(tests, extraTests...)
	}

	for i, tc := range tests {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			valid := policiesValid(tc.chain, VerifyOptions{
				CertificatePolicies:   tc.policies,
				requireExplicitPolicy: tc.requireExplicitPolicy,
				inhibitPolicyMapping:  tc.inhibitPolicyMapping,
				inhibitAnyPolicy:      tc.inhibitAnyPolicy,
			})
			if valid != tc.valid {
				t.Errorf("policiesValid: got %t, want %t", valid, tc.valid)
			}
		})
	}
}

func TestInvalidPolicyWithAnyKeyUsage(t *testing.T) {
	loadTestCert := func(t *testing.T, path string) *Certificate {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		p, _ := pem.Decode(b)
		c, err := ParseCertificate(p.Bytes)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}

	testOID3 := mustNewOIDFromInts([]uint64{1, 2, 840, 113554, 4, 1, 72585, 2, 3})
	root, intermediate, leaf := loadTestCert(t, "testdata/policy_root.pem"), loadTestCert(t, "testdata/policy_intermediate_require.pem"), loadTestCert(t, "testdata/policy_leaf.pem")

	expectedErr := "x509: no valid chains built: 1 candidate chains with invalid policies"

	roots, intermediates := NewCertPool(), NewCertPool()
	roots.AddCert(root)
	intermediates.AddCert(intermediate)

	_, err := leaf.Verify(VerifyOptions{
		Roots:               roots,
		Intermediates:       intermediates,
		KeyUsages:           []ExtKeyUsage{ExtKeyUsageAny},
		CertificatePolicies: []OID{testOID3},
	})
	if err == nil {
		t.Fatal("unexpected success, invalid policy shouldn't be bypassed by passing VerifyOptions.KeyUsages with ExtKeyUsageAny")
	} else if err.Error() != expectedErr {
		t.Fatalf("unexpected error, got %q, want %q", err, expectedErr)
	}
}

func TestCertificateChainSignedByECDSA(t *testing.T) {
	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	root := &Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "X"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(365 * 24 * time.Hour),
		IsCA:                  true,
		KeyUsage:              KeyUsageCertSign | KeyUsageCRLSign,
		BasicConstraintsValid: true,
	}
	caDER, err := CreateCertificate(rand.Reader, root, root, &caKey.PublicKey, caKey)
	if err != nil {
		t.Fatal(err)
	}
	root, err = ParseCertificate(caDER)
	if err != nil {
		t.Fatal(err)
	}

	leafKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	leaf := &Certificate{
		SerialNumber:          big.NewInt(42),
		Subject:               pkix.Name{CommonName: "leaf"},
		NotBefore:             time.Now().Add(-10 * time.Minute),
		NotAfter:              time.Now().Add(24 * time.Hour),
		KeyUsage:              KeyUsageDigitalSignature,
		ExtKeyUsage:           []ExtKeyUsage{ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}
	leafDER, err := CreateCertificate(rand.Reader, leaf, root, &leafKey.PublicKey, caKey)
	if err != nil {
		t.Fatal(err)
	}
	leaf, err = ParseCertificate(leafDER)
	if err != nil {
		t.Fatal(err)
	}

	inter, err := ParseCertificate(dsaSelfSignedCNX(t))
	if err != nil {
		t.Fatal(err)
	}

	inters := NewCertPool()
	inters.AddCert(root)
	inters.AddCert(inter)

	wantErr := "certificate signed by unknown authority"
	_, err = leaf.Verify(VerifyOptions{Intermediates: inters, Roots: NewCertPool()})
	if !strings.Contains(err.Error(), wantErr) {
		t.Errorf("got %v, want %q", err, wantErr)
	}
}

// dsaSelfSignedCNX produces DER-encoded
// certificate with the properties:
//
//	Subject=Issuer=CN=X
//	DSA SPKI
//	Matching inner/outer signature OIDs
//	Dummy ECDSA signature
func dsaSelfSignedCNX(t *testing.T) []byte {
	t.Helper()
	var params dsa.Parameters
	if err := dsa.GenerateParameters(&params, rand.Reader, dsa.L1024N160); err != nil {
		t.Fatal(err)
	}

	var dsaPriv dsa.PrivateKey
	dsaPriv.Parameters = params
	if err := dsa.GenerateKey(&dsaPriv, rand.Reader); err != nil {
		t.Fatal(err)
	}
	dsaPub := &dsaPriv.PublicKey

	type dsaParams struct{ P, Q, G *big.Int }
	paramDER, err := asn1.Marshal(dsaParams{dsaPub.P, dsaPub.Q, dsaPub.G})
	if err != nil {
		t.Fatal(err)
	}
	yDER, err := asn1.Marshal(dsaPub.Y)
	if err != nil {
		t.Fatal(err)
	}

	spki := publicKeyInfo{
		Algorithm: pkix.AlgorithmIdentifier{
			Algorithm:  oidPublicKeyDSA,
			Parameters: asn1.RawValue{FullBytes: paramDER},
		},
		PublicKey: asn1.BitString{Bytes: yDER, BitLength: 8 * len(yDER)},
	}

	rdn := pkix.Name{CommonName: "X"}.ToRDNSequence()
	b, err := asn1.Marshal(rdn)
	if err != nil {
		t.Fatal(err)
	}
	rawName := asn1.RawValue{FullBytes: b}

	algoIdent := pkix.AlgorithmIdentifier{Algorithm: oidSignatureDSAWithSHA256}
	tbs := tbsCertificate{
		Version:            0,
		SerialNumber:       big.NewInt(1002),
		SignatureAlgorithm: algoIdent,
		Issuer:             rawName,
		Validity:           validity{NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(24 * time.Hour)},
		Subject:            rawName,
		PublicKey:          spki,
	}
	c := certificate{
		TBSCertificate:     tbs,
		SignatureAlgorithm: algoIdent,
		SignatureValue:     asn1.BitString{Bytes: []byte{0}, BitLength: 8},
	}
	dsaDER, err := asn1.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	return dsaDER
}
