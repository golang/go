// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"crypto/internal/cryptotest"
	"crypto/internal/cryptotest/x509limbo"
	"encoding/json"
	"encoding/pem"
	"flag"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

var limboCases = flag.String("limbo_cases", "", "comma-separated limbo case ids to run; if empty, all cases run")

// Instances where we do **not** produce an error, but the test corpus says
// we should have. The map value justifies each allow.
var allowedUnexpectedVerifications = map[string]string{
	// TODO(@cpu): triage and justify, or fix.
	"rfc5280::aki::leaf-missing-aki":                       "requires triage",
	"rfc5280::aki::intermediate-missing-aki":               "requires triage",
	"rfc5280::aki::cross-signed-root-missing-aki":          "requires triage",
	"rfc5280::eku::ee-eku-empty":                           "requires triage",
	"rfc5280::nc::permitted-dns-match-noncritical":         "requires triage",
	"rfc5280::nc::invalid-dnsname-leading-period":          "requires triage",
	"rfc5280::nc::not-allowed-in-ee-noncritical":           "requires triage",
	"rfc5280::nc::not-allowed-in-ee-critical":              "requires triage",
	"rfc5280::pc::ica-noncritical-pc":                      "requires triage",
	"rfc5280::san::noncritical-with-empty-subject":         "requires triage",
	"rfc5280::san::underscore-dns":                         "requires triage",
	"rfc5280::serial::too-long":                            "requires triage",
	"rfc5280::serial::zero":                                "requires triage",
	"rfc5280::serial::negative":                            "requires triage",
	"rfc5280::ski::root-missing-ski":                       "requires triage",
	"rfc5280::ski::intermediate-missing-ski":               "requires triage",
	"rfc5280::ca-empty-subject":                            "requires triage",
	"rfc5280::root-non-critical-basic-constraints":         "requires triage",
	"rfc5280::root-inconsistent-ca-extensions":             "requires triage",
	"rfc5280::leaf-ku-keycertsign":                         "requires triage",
	"webpki::aki::root-with-aki-missing-keyidentifier":     "requires triage",
	"webpki::aki::root-with-aki-authoritycertissuer":       "requires triage",
	"webpki::aki::root-with-aki-authoritycertserialnumber": "requires triage",
	"webpki::aki::root-with-aki-all-fields":                "requires triage",
	"webpki::aki::root-with-aki-ski-mismatch":              "requires triage",
	"webpki::eku::ee-anyeku":                               "requires triage",
	"webpki::eku::ee-critical-eku":                         "requires triage",
	"webpki::eku::ee-without-eku":                          "requires triage",
	"webpki::eku::root-has-eku":                            "requires triage",
	"webpki::san::public-suffix-wildcard-san":              "requires triage",
	"webpki::san::san-critical-with-nonempty-subject":      "requires triage",
	"webpki::forbidden-dsa-leaf":                           "requires triage",
	"webpki::forbidden-weak-rsa-key-in-root":               "requires triage",
	"webpki::forbidden-weak-rsa-in-leaf":                   "requires triage",
	"webpki::forbidden-rsa-not-divisable-by-8-in-root":     "requires triage",
	"webpki::forbidden-rsa-key-not-divisable-by-8-in-leaf": "requires triage",
	"webpki::ee-basicconstraints-ca":                       "requires triage",
	"webpki::ca-as-leaf":                                   "requires triage",

	// Our implementation handles these degenerate name constraint tests
	// without error. They are described as standards compliant but are
	// marked expected-reject upstream because quadratic implementations
	// hit a fixed DoS prevention limit. nc-dos-3 is not listed: it matches
	// the expected failure result, but due to the use of a subject CN
	// without SAN, not because of quadratic NC checking.
	"pathological::nc-dos-1": "standards compliant; upstream rejects due to quadratic DoS limit",
	"pathological::nc-dos-2": "standards compliant; upstream rejects due to quadratic DoS limit",

	// These webpki::cn::* cases test CABF BR 7.1.4.3 constraints on the
	// CN field. Go's x509 package intentionally ignores the legacy Common Name
	// (CN) field for hostname matching (see Certificate.VerifyHostname), so
	// verification succeeds via the well-formed SAN even when the CN is
	// non-conformant.
	"webpki::cn::case-mismatch":               "Go ignores legacy CN",
	"webpki::cn::ipv4-hex-mismatch":           "Go ignores legacy CN",
	"webpki::cn::ipv4-leading-zeros-mismatch": "Go ignores legacy CN",
	"webpki::cn::ipv6-non-rfc5952-mismatch":   "Go ignores legacy CN",
	"webpki::cn::ipv6-uncompressed-mismatch":  "Go ignores legacy CN",
	"webpki::cn::ipv6-uppercase-mismatch":     "Go ignores legacy CN",
	"webpki::cn::not-in-san":                  "Go ignores legacy CN",
	"webpki::cn::punycode-not-in-san":         "Go ignores legacy CN",
	"webpki::cn::utf8-vs-punycode-mismatch":   "Go ignores legacy CN",
}

// Instances where we produce an error, but the test corpus says we
// shouldn't have. The map value justifies each allow.
var allowedUnexpectedFailures = map[string]string{
	// TODO(@cpu): triage and justify, or fix.
	"pathlen::self-issued-certs-pathlen":     "requires triage",
	"rfc5280::nc::permitted-dn-match":        "requires triage",
	"rfc5280::nc::permitted-self-issued":     "requires triage",
	"rfc5280::nc::nc-forbids-othername-noop": "requires triage",
	"rfc5280::validity::notafter-fractional": "requires triage",
}

var extKeyUsagesMap = map[x509limbo.KnownEKUs]ExtKeyUsage{
	x509limbo.KnownEKUsAnyExtendedKeyUsage: ExtKeyUsageAny,
	x509limbo.KnownEKUsClientAuth:          ExtKeyUsageClientAuth,
	x509limbo.KnownEKUsCodeSigning:         ExtKeyUsageCodeSigning,
	x509limbo.KnownEKUsEmailProtection:     ExtKeyUsageEmailProtection,
	x509limbo.KnownEKUsOCSPSigning:         ExtKeyUsageOCSPSigning,
	x509limbo.KnownEKUsServerAuth:          ExtKeyUsageServerAuth,
	x509limbo.KnownEKUsTimeStamping:        ExtKeyUsageTimeStamping,
}

// Tests the x509 package using the test vectors from https://x509-limbo.com/
func TestX509Limbo(t *testing.T) {
	testenv.SkipIfShortAndSlow(t)

	limboDir := cryptotest.FetchModule(t, x509limbo.X509LimboModule, x509limbo.X509LimboVersion)

	limboJson, err := os.ReadFile(filepath.Join(limboDir, "limbo.json"))
	if err != nil {
		t.Fatalf("error reading limbo.json: %v", err)
	}

	var limbo x509limbo.Limbo
	if err := json.Unmarshal(limboJson, &limbo); err != nil {
		t.Fatalf("failed to unmarshal limbo.json: %v", err)
	}

	for _, tc := range limbo.Testcases {
		t.Run(tc.Id, func(t *testing.T) {
			t.Parallel()

			if *limboCases != "" && !slices.Contains(strings.Split(*limboCases, ","), tc.Id) {
				t.Skip("filtered out by -limbo_cases")
			}

			if slices.Contains(tc.Features, x509limbo.FeatureHasCrl) {
				t.Skipf("CRL revocation checking not supported")
			}

			if slices.Contains(tc.Features, x509limbo.FeatureMaxChainDepth) {
				t.Skipf("customizable max chain depth not supported")
			}

			if len(tc.SignatureAlgorithms) != 0 {
				// Note: there are no limbo.json test cases that specify signature
				// algorithms at this time, so this skip is largely a no-op.
				t.Skipf("signature algorithms are not customizable through the x509 interface")
			}

			if len(tc.KeyUsage) != 0 &&
				!slices.Contains(tc.KeyUsage, x509limbo.KeyUsageDigitalSignature) {
				// Note: there are no limbo.json test cases that specify key usages other
				// than digitalSignature at this time, so this skip is largely a no-op.
				t.Skipf("key usage checks other than Digital Signature are not supported")
			}

			// In the server validation context we may be given a single expected
			// peer name to use for our verify options.
			var verifyDnsName string
			if tc.ExpectedPeerName != nil && tc.ValidationKind == x509limbo.ValidationKindSERVER {
				switch tc.ExpectedPeerName.Kind {
				case x509limbo.PeerKindDNS:
					verifyDnsName = tc.ExpectedPeerName.Value
				case x509limbo.PeerKindIP:
					verifyDnsName = fmt.Sprintf("[%s]", tc.ExpectedPeerName.Value)
				default:
					t.Skipf("unsupported peer name kind: %v", tc.ExpectedPeerName.Kind)
				}
			}

			roots, intermediates := NewCertPool(), NewCertPool()
			for _, rootPem := range tc.TrustedCerts {
				roots.AppendCertsFromPEM([]byte(rootPem))
			}
			for _, intermediatePem := range tc.UntrustedIntermediates {
				intermediates.AppendCertsFromPEM([]byte(intermediatePem))
			}

			block, rest := pem.Decode([]byte(tc.PeerCertificate))
			if block == nil {
				t.Fatalf("unable to PEM decode peer certificate")
			} else if block.Type != "CERTIFICATE" {
				t.Fatalf("unexpected data, expected cert: %+#v", *block)
			} else if len(rest) > 0 {
				t.Fatalf("peer certificate has %d trailing bytes", len(rest))
			}

			peer, parseErr := ParseCertificate(block.Bytes)
			if parseErr != nil {
				if tc.ExpectedResult == x509limbo.ExpectedResultFAILURE {
					// The test expects failure and we detect an error at parse
					// time instead of verification time. Considered a pass.
					return
				}
				printChainDetails(t, tc, parseErr)
				t.Errorf("expected success, parsing peer certificate failed: %v", parseErr)
				return
			}

			validationTime := time.Now()
			if tc.ValidationTime != nil {
				vtStr, ok := tc.ValidationTime.(string)
				if !ok {
					t.Fatalf("validation time is not a string: %T %v", tc.ValidationTime, tc.ValidationTime)
				}
				parsed, err := time.Parse(time.RFC3339, vtStr)
				if err != nil {
					t.Fatalf("invalid validation time %q: %v", vtStr, err)
				}
				validationTime = parsed
			}

			var ekus []ExtKeyUsage
			for _, elem := range tc.ExtendedKeyUsage {
				eku, ok := extKeyUsagesMap[elem]
				if !ok {
					t.Skipf("unsupported extended key usage: %v", elem)
				}
				ekus = append(ekus, eku)
			}

			_, err := peer.Verify(VerifyOptions{
				DNSName:       verifyDnsName,
				Intermediates: intermediates,
				Roots:         roots,
				CurrentTime:   validationTime,
				KeyUsages:     ekus,
			})
			if err == nil && tc.ExpectedResult == x509limbo.ExpectedResultFAILURE {
				if _, allowed := allowedUnexpectedVerifications[tc.Id]; !allowed {
					printChainDetails(t, tc, nil)
					t.Errorf("expected failure, built chain without error")
				}
			} else if err != nil && tc.ExpectedResult == x509limbo.ExpectedResultSUCCESS {
				if _, allowed := allowedUnexpectedFailures[tc.Id]; !allowed {
					printChainDetails(t, tc, err)
					t.Errorf("expected success, built chain with error: %v", err)
				}
			}

			// In the client validation context we may be given multiple expected
			// peer names so we check these explicitly after path building.
			// The DNSName in our VerifyOpts will have been empty.
			if tc.ValidationKind == x509limbo.ValidationKindCLIENT {
				for _, name := range tc.ExpectedPeerNames {
					if name.Kind != x509limbo.PeerKindIP && name.Kind != x509limbo.PeerKindDNS {
						// We don't support verifying RFC8222 peer names.
						t.Skipf("unsupported peer name kind: %v", name.Kind)
					}
					err = peer.VerifyHostname(name.Value)
					// We don't check allowedUnexpectedVerifications or allowedUnexpectedFailures
					// here because there aren't any that apply to ValidationKindCLIENT
					// at this time.
					if err == nil && tc.ExpectedResult == x509limbo.ExpectedResultFAILURE {
						printChainDetails(t, tc, nil)
						t.Errorf("expected failure, built chain without error")
					} else if err != nil && tc.ExpectedResult == x509limbo.ExpectedResultSUCCESS {
						printChainDetails(t, tc, err)
						t.Errorf("expected success, built chain with error: %v", err)
					}
				}
			}
		})
	}
}

func printChainDetails(t *testing.T, tc x509limbo.Testcase, actualResult error) {
	t.Log("----")
	t.Logf("testcase: %q expected result: %v actual result: %v", tc.Id, tc.ExpectedResult, actualResult)
	t.Log("trust anchor PEM:")
	for _, root := range tc.TrustedCerts {
		t.Log(root)
	}
	t.Log("intermediates PEM:")
	for _, intermediate := range tc.UntrustedIntermediates {
		t.Log(intermediate)
	}
	t.Log("end entity PEM:")
	t.Log(tc.PeerCertificate)
	t.Log("----")
}
