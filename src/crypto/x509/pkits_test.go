// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

var nistTestPolicies = map[string]OID{
	"anyPolicy":          anyPolicyOID,
	"NIST-test-policy-1": mustNewOIDFromInts([]uint64{2, 16, 840, 1, 101, 3, 2, 1, 48, 1}),
	"NIST-test-policy-2": mustNewOIDFromInts([]uint64{2, 16, 840, 1, 101, 3, 2, 1, 48, 2}),
	"NIST-test-policy-3": mustNewOIDFromInts([]uint64{2, 16, 840, 1, 101, 3, 2, 1, 48, 3}),
	"NIST-test-policy-6": mustNewOIDFromInts([]uint64{2, 16, 840, 1, 101, 3, 2, 1, 48, 6}),
}

func TestNISTPKITSPolicy(t *testing.T) {
	// This test runs a subset of the NIST PKI path validation test suite that
	// focuses of policy validation, rather than the entire suite. Since the
	// suite assumes you are only validating the path, rather than building
	// _and_ validating the path, we take the path as given and run
	// policiesValid on it.

	certDir := "testdata/nist-pkits/certs"

	var testcases []struct {
		Name                        string
		CertPath                    []string
		InitialPolicySet            []string
		InitialPolicyMappingInhibit bool
		InitialExplicitPolicy       bool
		InitialAnyPolicyInhibit     bool
		ShouldValidate              bool
		Skipped                     bool
	}
	b, err := os.ReadFile("testdata/nist-pkits/vectors.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &testcases); err != nil {
		t.Fatal(err)
	}

	policyTests := map[string]bool{
		"4.8.1 All Certificates Same Policy Test1 (Subpart 1)":     true,
		"4.8.1 All Certificates Same Policy Test1 (Subpart 2)":     true,
		"4.8.1 All Certificates Same Policy Test1 (Subpart 3)":     true,
		"4.8.1 All Certificates Same Policy Test1 (Subpart 4)":     true,
		"4.8.2 All Certificates No Policies Test2 (Subpart 1)":     true,
		"4.8.2 All Certificates No Policies Test2 (Subpart 2)":     true,
		"4.8.3 Different Policies Test3 (Subpart 1)":               true,
		"4.8.3 Different Policies Test3 (Subpart 2)":               true,
		"4.8.3 Different Policies Test3 (Subpart 3)":               true,
		"4.8.4 Different Policies Test4":                           true,
		"4.8.5 Different Policies Test5":                           true,
		"4.8.6 Overlapping Policies Test6 (Subpart 1)":             true,
		"4.8.6 Overlapping Policies Test6 (Subpart 2)":             true,
		"4.8.6 Overlapping Policies Test6 (Subpart 3)":             true,
		"4.8.7 Different Policies Test7":                           true,
		"4.8.8 Different Policies Test8":                           true,
		"4.8.9 Different Policies Test9":                           true,
		"4.8.10 All Certificates Same Policies Test10 (Subpart 1)": true,
		"4.8.10 All Certificates Same Policies Test10 (Subpart 2)": true,
		"4.8.10 All Certificates Same Policies Test10 (Subpart 3)": true,
		"4.8.11 All Certificates AnyPolicy Test11 (Subpart 1)":     true,
		"4.8.11 All Certificates AnyPolicy Test11 (Subpart 2)":     true,
		"4.8.12 Different Policies Test12":                         true,
		"4.8.13 All Certificates Same Policies Test13 (Subpart 1)": true,
		"4.8.13 All Certificates Same Policies Test13 (Subpart 2)": true,
		"4.8.13 All Certificates Same Policies Test13 (Subpart 3)": true,
		"4.8.14 AnyPolicy Test14 (Subpart 1)":                      true,
		"4.8.14 AnyPolicy Test14 (Subpart 2)":                      true,
		"4.8.15 User Notice Qualifier Test15":                      true,
		"4.8.16 User Notice Qualifier Test16":                      true,
		"4.8.17 User Notice Qualifier Test17":                      true,
		"4.8.18 User Notice Qualifier Test18 (Subpart 1)":          true,
		"4.8.18 User Notice Qualifier Test18 (Subpart 2)":          true,
		"4.8.19 User Notice Qualifier Test19":                      true,
		"4.8.20 CPS Pointer Qualifier Test20":                      true,
		"4.9.1 Valid RequireExplicitPolicy Test1":                  true,
		"4.9.2 Valid RequireExplicitPolicy Test2":                  true,
		"4.9.3 Invalid RequireExplicitPolicy Test3":                true,
		"4.9.4 Valid RequireExplicitPolicy Test4":                  true,
		"4.9.5 Invalid RequireExplicitPolicy Test5":                true,
		"4.9.6 Valid Self-Issued requireExplicitPolicy Test6":      true,
		"4.9.7 Invalid Self-Issued requireExplicitPolicy Test7":    true,
		"4.9.8 Invalid Self-Issued requireExplicitPolicy Test8":    true,
		"4.10.1.1 Valid Policy Mapping Test1 (Subpart 1)":          true,
		"4.10.1.2 Valid Policy Mapping Test1 (Subpart 2)":          true,
		"4.10.1.3 Valid Policy Mapping Test1 (Subpart 3)":          true,
		"4.10.2 Invalid Policy Mapping Test2 (Subpart 1)":          true,
		"4.10.2 Invalid Policy Mapping Test2 (Subpart 2)":          true,
		"4.10.3 Valid Policy Mapping Test3 (Subpart 1)":            true,
		"4.10.3 Valid Policy Mapping Test3 (Subpart 2)":            true,
		"4.10.4 Invalid Policy Mapping Test4":                      true,
		"4.10.5 Valid Policy Mapping Test5 (Subpart 1)":            true,
		"4.10.5 Valid Policy Mapping Test5 (Subpart 2)":            true,
		"4.10.6 Valid Policy Mapping Test6 (Subpart 1)":            true,
		"4.10.6 Valid Policy Mapping Test6 (Subpart 2)":            true,
		"4.10.7 Invalid Mapping From anyPolicy Test7":              true,
		"4.10.8 Invalid Mapping To anyPolicy Test8":                true,
		"4.10.9 Valid Policy Mapping Test9":                        true,
		"4.10.10 Invalid Policy Mapping Test10":                    true,
		"4.10.11 Valid Policy Mapping Test11":                      true,
		"4.10.12 Valid Policy Mapping Test12 (Subpart 1)":          true,
		"4.10.12 Valid Policy Mapping Test12 (Subpart 2)":          true,
		"4.10.13 Valid Policy Mapping Test13 (Subpart 1)":          true,
		"4.10.13 Valid Policy Mapping Test13 (Subpart 2)":          true,
		"4.10.13 Valid Policy Mapping Test13 (Subpart 3)":          true,
		"4.10.14 Valid Policy Mapping Test14":                      true,
		"4.11.1 Invalid inhibitPolicyMapping Test1":                true,
		"4.11.2 Valid inhibitPolicyMapping Test2":                  true,
		"4.11.3 Invalid inhibitPolicyMapping Test3":                true,
		"4.11.4 Valid inhibitPolicyMapping Test4":                  true,
		"4.11.5 Invalid inhibitPolicyMapping Test5":                true,
		"4.11.6 Invalid inhibitPolicyMapping Test6":                true,
		"4.11.7 Valid Self-Issued inhibitPolicyMapping Test7":      true,
		"4.11.8 Invalid Self-Issued inhibitPolicyMapping Test8":    true,
		"4.11.9 Invalid Self-Issued inhibitPolicyMapping Test9":    true,
		"4.11.10 Invalid Self-Issued inhibitPolicyMapping Test10":  true,
		"4.11.11 Invalid Self-Issued inhibitPolicyMapping Test11":  true,
		"4.12.1 Invalid inhibitAnyPolicy Test1":                    true,
		"4.12.2 Valid inhibitAnyPolicy Test2":                      true,
		"4.12.3 inhibitAnyPolicy Test3 (Subpart 1)":                true,
		"4.12.3 inhibitAnyPolicy Test3 (Subpart 2)":                true,
		"4.12.4 Invalid inhibitAnyPolicy Test4":                    true,
		"4.12.5 Invalid inhibitAnyPolicy Test5":                    true,
		"4.12.6 Invalid inhibitAnyPolicy Test6":                    true,
		"4.12.7 Valid Self-Issued inhibitAnyPolicy Test7":          true,
		"4.12.8 Invalid Self-Issued inhibitAnyPolicy Test8":        true,
		"4.12.9 Valid Self-Issued inhibitAnyPolicy Test9":          true,
		"4.12.10 Invalid Self-Issued inhibitAnyPolicy Test10":      true,
	}

	for _, tc := range testcases {
		if !policyTests[tc.Name] {
			continue
		}
		t.Run(tc.Name, func(t *testing.T) {
			var chain []*Certificate
			for _, c := range tc.CertPath {
				certDER, err := os.ReadFile(filepath.Join(certDir, c))
				if err != nil {
					t.Fatal(err)
				}
				cert, err := ParseCertificate(certDER)
				if err != nil {
					t.Fatal(err)
				}
				chain = append(chain, cert)
			}
			slices.Reverse(chain)

			var initialPolicies []OID
			for _, pstr := range tc.InitialPolicySet {
				policy, ok := nistTestPolicies[pstr]
				if !ok {
					t.Fatalf("unknown test policy: %s", pstr)
				}
				initialPolicies = append(initialPolicies, policy)
			}

			valid := policiesValid(chain, VerifyOptions{
				CertificatePolicies:   initialPolicies,
				inhibitPolicyMapping:  tc.InitialPolicyMappingInhibit,
				requireExplicitPolicy: tc.InitialExplicitPolicy,
				inhibitAnyPolicy:      tc.InitialAnyPolicyInhibit,
			})
			if !valid {
				if !tc.ShouldValidate {
					return
				}
				t.Fatalf("Failed to validate: %s", err)
			}
			if !tc.ShouldValidate {
				t.Fatal("Expected path validation to fail")
			}
		})
	}
}
