// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test uses Netflix's BetterTLS test suite to test the crypto/x509
// path building and name constraint validation.
//
// The test data in JSON form is around 31MB, so we fetch the BetterTLS
// go module and use it to generate the JSON data on-the-fly in a tmp dir.
//
// For more information, see:
// https://github.com/netflix/bettertls
// https://netflixtechblog.com/bettertls-c9915cd255c0

package x509

import (
	"crypto/internal/cryptotest"
	"encoding/base64"
	"encoding/json"
	"internal/testenv"
	"os"
	"path/filepath"
	"testing"
)

// TestBetterTLS runs the "pathbuilding" and "nameconstraints" suites of
// BetterTLS.
//
// The test cases in the pathbuilding suite are designed to test edge-cases
// for path building and validation. In particular, the ["chain of pain"][0]
// scenario where a validator treats path building as an operation with
// a single possible outcome, instead of many.
//
// The test cases in the nameconstraints suite are designed to test edge-cases
// for name constraint parsing and validation.
//
// [0]: https://medium.com/@sleevi_/path-building-vs-path-verifying-the-chain-of-pain-9fbab861d7d6
func TestBetterTLS(t *testing.T) {
	testenv.SkipIfShortAndSlow(t)

	data, roots := betterTLSTestData(t)

	for _, suite := range []string{"pathbuilding", "nameconstraints"} {
		t.Run(suite, func(t *testing.T) {
			runTestSuite(t, suite, &data, roots)
		})
	}
}

func runTestSuite(t *testing.T, suiteName string, data *betterTLS, roots *CertPool) {
	suite, exists := data.Suites[suiteName]
	if !exists {
		t.Fatalf("missing %s suite", suiteName)
	}

	t.Logf(
		"running %s test suite with %d test cases",
		suiteName, len(suite.TestCases))

	for _, tc := range suite.TestCases {
		t.Logf("testing %s test case %d", suiteName, tc.ID)

		certsDER, err := tc.Certs()
		if err != nil {
			t.Fatalf(
				"failed to decode certificates for test case %d: %v",
				tc.ID, err)
		}

		if len(certsDER) == 0 {
			t.Fatalf("test case %d has no certificates", tc.ID)
		}

		eeCert, err := ParseCertificate(certsDER[0])
		if err != nil {
			// Several constraint test cases contain invalid end-entity
			// certificate extensions that we reject ahead of verification
			// time. We consider this a pass and skip further processing.
			//
			// For example, a SAN with a uniformResourceIdentifier general name
			// containing the value `"http://foo.bar, DNS:test.localhost"`, or
			// an iPAddress general name of the wrong length.
			if suiteName == "nameconstraints" && tc.Expected == expectedReject {
				t.Logf(
					"skipping expected reject test case %d "+
						"- end entity certificate parse error: %v",
					tc.ID, err)
				continue
			}
			t.Fatalf(
				"failed to parse end entity certificate for test case %d: %v",
				tc.ID, err)
		}

		intermediates := NewCertPool()
		for i, certDER := range certsDER[1:] {
			cert, err := ParseCertificate(certDER)
			if err != nil {
				t.Fatalf(
					"failed to parse intermediate certificate %d for test case %d: %v",
					i+1, tc.ID, err)
			}
			intermediates.AddCert(cert)
		}

		_, err = eeCert.Verify(VerifyOptions{
			Roots:         roots,
			Intermediates: intermediates,
			DNSName:       tc.Hostname,
			KeyUsages:     []ExtKeyUsage{ExtKeyUsageServerAuth},
		})

		switch tc.Expected {
		case expectedAccept:
			if err != nil {
				t.Errorf(
					"test case %d failed: expected success, got error: %v",
					tc.ID, err)
			}
		case expectedReject:
			if err == nil {
				t.Errorf(
					"test case %d failed: expected failure, but verification succeeded",
					tc.ID)
			}
		default:
			t.Fatalf(
				"test case %d failed: unknown expected result: %s",
				tc.ID, tc.Expected)
		}
	}
}

func betterTLSTestData(t *testing.T) (betterTLS, *CertPool) {
	const (
		bettertlsModule  = "github.com/Netflix/bettertls"
		bettertlsVersion = "v0.0.0-20250909192348-e1e99e353074"
	)

	bettertlsDir := cryptotest.FetchModule(t, bettertlsModule, bettertlsVersion)

	tempDir := t.TempDir()
	testsJSONPath := filepath.Join(tempDir, "tests.json")

	cmd := testenv.Command(t, testenv.GoToolPath(t),
		"run", "./test-suites/cmd/bettertls",
		"export-tests",
		"--out", testsJSONPath)
	cmd.Dir = bettertlsDir

	t.Log("running bettertls export-tests command")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf(
			"failed to run bettertls export-tests: %v\nOutput: %s",
			err, output)
	}

	jsonData, err := os.ReadFile(testsJSONPath)
	if err != nil {
		t.Fatalf("failed to read exported tests.json: %v", err)
	}

	t.Logf("successfully loaded tests.json at %s", testsJSONPath)

	var data betterTLS
	if err := json.Unmarshal(jsonData, &data); err != nil {
		t.Fatalf("failed to unmarshal JSON data: %v", err)
	}

	t.Logf("testing betterTLS revision: %s", data.Revision)
	t.Logf("number of test suites: %d", len(data.Suites))

	rootDER, err := data.RootCert()
	if err != nil {
		t.Fatalf("failed to decode trust root: %v", err)
	}

	rootCert, err := ParseCertificate(rootDER)
	if err != nil {
		t.Fatalf("failed to parse trust root certificate: %v", err)
	}

	roots := NewCertPool()
	roots.AddCert(rootCert)

	return data, roots
}

type betterTLS struct {
	Revision string                    `json:"betterTlsRevision"`
	Root     string                    `json:"trustRoot"`
	Suites   map[string]betterTLSSuite `json:"suites"`
}

func (b *betterTLS) RootCert() ([]byte, error) {
	return base64.StdEncoding.DecodeString(b.Root)
}

type betterTLSSuite struct {
	TestCases []betterTLSTest `json:"testCases"`
}

type betterTLSTest struct {
	ID           uint32         `json:"id"`
	Certificates []string       `json:"certificates"`
	Hostname     string         `json:"hostname"`
	Expected     expectedResult `json:"expected"`
}

func (test *betterTLSTest) Certs() ([][]byte, error) {
	certs := make([][]byte, len(test.Certificates))
	for i, cert := range test.Certificates {
		decoded, err := base64.StdEncoding.DecodeString(cert)
		if err != nil {
			return nil, err
		}
		certs[i] = decoded
	}
	return certs, nil
}

type expectedResult string

const (
	expectedAccept expectedResult = "ACCEPT"
	expectedReject expectedResult = "REJECT"
)
