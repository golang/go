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
	// These are instances where we should consider updating the implementation.
	"rfc5280::san::noncritical-with-empty-subject":    "TODO(#79741)",
	"webpki::san::san-critical-with-nonempty-subject": "TODO(#79741)",
	"rfc5280::nc::not-allowed-in-ee-noncritical":      "TODO(#79742)",
	"rfc5280::nc::not-allowed-in-ee-critical":         "TODO(#79742)",
	"rfc5280::eku::ee-eku-empty":                      "TODO(#79743)",
	"rfc5280::ca-empty-subject":                       "TODO(#79744)",

	// Underscores and other invalid characters are presently allowed after
	// tightening up the validation caused issues with real world certificates.
	"rfc5280::san::underscore-dns": "TODO(#75835)",

	// Go does not apply CABF key-strength policies.
	"webpki::forbidden-dsa-leaf":                           "Go doesn't enforce CABF key strength policies",
	"webpki::forbidden-weak-rsa-key-in-root":               "Go doesn't enforce CABF key strength policies",
	"webpki::forbidden-weak-rsa-in-leaf":                   "Go doesn't enforce CABF key strength policies",
	"webpki::forbidden-rsa-not-divisable-by-8-in-root":     "Go doesn't enforce CABF key strength policies",
	"webpki::forbidden-rsa-key-not-divisable-by-8-in-leaf": "Go doesn't enforce CABF key strength policies",

	// We don't want to take a public suffix data dependency, other heuristics
	// are incomplete and will interact badly with private PKIs.
	"webpki::san::public-suffix-wildcard-san": "Go doesn't include the PSL in its stdlib",

	// Trust anchors are implicitly considered issuers regardless of basic
	// constraints extension.
	"rfc5280::root-non-critical-basic-constraints": "Go only considers BC on intermediates",
	// Similarly, KeyUsage status flags are ignored by design. See Certificate.isValid
	// comment in body of implementation.
	"rfc5280::root-inconsistent-ca-extensions": "Go ignores KU, only considers BC on intermediates",
	"rfc5280::leaf-ku-keycertsign":             "Go ignores KU, only considers BC on intermediates",

	// Enforcing ee-basicconstraints-ca/ca-as-leaf may additionally break the
	// somewhat common practice of using a self-signed issuer as the sole leaf
	// certificate in a chain.
	"webpki::ee-basicconstraints-ca": "Go ignores KU",
	"webpki::ca-as-leaf":             "Go ignores KU",

	// Certificate.Verify documents that we allow a leading period for DNS
	// name constraints, similar to emails/URIs.
	"rfc5280::nc::invalid-dnsname-leading-period": "Go accepts leading period",

	// AKI is not load-bearing for validation. We only use it as a
	// parent-ordering hint in CertPool.findPotentialParents.
	"rfc5280::aki::cross-signed-root-missing-aki":          "Go only uses AKI for ordering hint, not a verification requirement",
	"rfc5280::aki::leaf-missing-aki":                       "Go only uses AKI for ordering hint, not a verification requirement",
	"webpki::aki::root-with-aki-missing-keyidentifier":     "Go does not enforce CABF requirement that root AKI contain a keyIdentifier field",
	"webpki::aki::root-with-aki-authoritycertissuer":       "Go does not enforce CABF prohibition on authorityCertIssuer in root AKI",
	"webpki::aki::root-with-aki-authoritycertserialnumber": "Go does not enforce CABF prohibition on authorityCertSerialNumber in root AKI",
	"webpki::aki::root-with-aki-all-fields":                "Go does not enforce CABF restrictions on AKI field composition in roots",
	"webpki::aki::root-with-aki-ski-mismatch":              "Go does not enforce CABF requirement that a self-signed root's AKI keyIdentifier match its SKI",

	// Enforcing criticality is of dubious value in these cases and likely bumps
	// into incorrect real world certificates. Additionally, no other verifiers
	// tested by x509-limbo upstream treat these as a failure condition.
	"webpki::eku::ee-critical-eku":                 "Go doesn't reject this extension when marked critical",
	"rfc5280::nc::permitted-dns-match-noncritical": "Go doesn't require this extension to be critical",
	"rfc5280::pc::ica-noncritical-pc":              "Go doesn't require this extension to be critical",

	// Serial parsing enforces no negatives, but doesn't enforce max length or
	// non-zero. Important roots have a serial of zero, and enforcing serial
	// length broke enough private PKIs that the enforcement change was reverted.
	"rfc5280::serial::too-long": "Causes significant breakage of real-world private PKIs",
	"rfc5280::serial::zero":     "RFC 5280 says certificate users SHOULD gracefully handle zero",

	// These are skipped based on CT analysis of affected certificates.
	// See https://github.com/golang/go/issues/65085#issuecomment-1932886623
	"rfc5280::ski::root-missing-ski":         "would break various trusted Verisign roots",
	"rfc5280::ski::intermediate-missing-ski": "would break various trusted intermediates",
	"rfc5280::aki::intermediate-missing-aki": "would break real world certificates",

	// Go enforces EKU as an application-level capability filter, not according
	// to CABF webpki policy where (for e.g.) anyExtendedKeyUsage is forbidden
	// on leaves.
	"webpki::eku::ee-anyeku":      "Go treats anyExtendedKeyUsage as overriding any other key usage.",
	"webpki::eku::ee-without-eku": "Go skips certs with no EKU when checking chain usage.",
	"webpki::eku::root-has-eku":   "Go allows a root to have an EKU as a downward constraint",

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
	// This looks like a small oversight in our implementation, and should be
	// fixed.
	"rfc5280::nc::permitted-self-issued": "TODO(#79746)",

	// The spec-conformant behavior weakens the security value of pathlen, and
	// has limited real-world impact on webpki certificates. Other
	// implementations like mozilla::pkix have reached a similar conclusion.
	// See https://bugzilla.mozilla.org/show_bug.cgi?id=926265 and
	// https://github.com/golang/go/issues/79745#issuecomment-4578179884
	"pathlen::self-issued-certs-pathlen": "Go prefers a stricter pathen implementation",

	// Limbo argues there are no OtherName GeneralName's in the chain being
	// validated, and so it should pass. We take a more conservative stance
	// backed by 5280 §4.2 that we have a critical extension we can't process,
	// and don't make a determination based on usage in verification.
	"rfc5280::nc::nc-forbids-othername-noop": "Go rejects critical NC with GeneralName types it doesn't implement",

	// Per the test's description there is "no clear 'winning' interpretation"
	// between second-granularity checks vs instantaneous. Changing our
	// behavior in this case seems low-priority.
	"rfc5280::validity::notafter-fractional": "Go uses instantaneous time comparisons",
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

			if slices.Contains(tc.Features, x509limbo.FeatureNameConstraintDn) {
				t.Skipf("name constraints for DirectoryNames are not supported")
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
