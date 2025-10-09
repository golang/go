// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"encoding/asn1"
	"encoding/pem"
	"os"
	"strings"
	"testing"

	cryptobyte_asn1 "golang.org/x/crypto/cryptobyte/asn1"
)

func TestParseASN1String(t *testing.T) {
	tests := []struct {
		name        string
		tag         cryptobyte_asn1.Tag
		value       []byte
		expected    string
		expectedErr string
	}{
		{
			name:     "T61String",
			tag:      cryptobyte_asn1.T61String,
			value:    []byte{80, 81, 82},
			expected: string("PQR"),
		},
		{
			name:     "PrintableString",
			tag:      cryptobyte_asn1.PrintableString,
			value:    []byte{80, 81, 82},
			expected: string("PQR"),
		},
		{
			name:        "PrintableString (invalid)",
			tag:         cryptobyte_asn1.PrintableString,
			value:       []byte{1, 2, 3},
			expectedErr: "invalid PrintableString",
		},
		{
			name:     "UTF8String",
			tag:      cryptobyte_asn1.UTF8String,
			value:    []byte{80, 81, 82},
			expected: string("PQR"),
		},
		{
			name:        "UTF8String (invalid)",
			tag:         cryptobyte_asn1.UTF8String,
			value:       []byte{255},
			expectedErr: "invalid UTF-8 string",
		},
		{
			name:     "BMPString",
			tag:      cryptobyte_asn1.Tag(asn1.TagBMPString),
			value:    []byte{80, 81},
			expected: string("偑"),
		},
		{
			name:        "BMPString (invalid length)",
			tag:         cryptobyte_asn1.Tag(asn1.TagBMPString),
			value:       []byte{255},
			expectedErr: "invalid BMPString",
		},
		{
			name:     "IA5String",
			tag:      cryptobyte_asn1.IA5String,
			value:    []byte{80, 81},
			expected: string("PQ"),
		},
		{
			name:        "IA5String (invalid)",
			tag:         cryptobyte_asn1.IA5String,
			value:       []byte{255},
			expectedErr: "invalid IA5String",
		},
		{
			name:     "NumericString",
			tag:      cryptobyte_asn1.Tag(asn1.TagNumericString),
			value:    []byte{49, 50},
			expected: string("12"),
		},
		{
			name:        "NumericString (invalid)",
			tag:         cryptobyte_asn1.Tag(asn1.TagNumericString),
			value:       []byte{80},
			expectedErr: "invalid NumericString",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out, err := parseASN1String(tc.tag, tc.value)
			if err != nil && err.Error() != tc.expectedErr {
				t.Fatalf("parseASN1String returned unexpected error: got %q, want %q", err, tc.expectedErr)
			} else if err == nil && tc.expectedErr != "" {
				t.Fatalf("parseASN1String didn't fail, expected: %s", tc.expectedErr)
			}
			if out != tc.expected {
				t.Fatalf("parseASN1String returned unexpected value: got %q, want %q", out, tc.expected)
			}
		})
	}
}

const policyPEM = `-----BEGIN CERTIFICATE-----
MIIGeDCCBWCgAwIBAgIUED9KQBi0ScBDoufB2mgAJ63G5uIwDQYJKoZIhvcNAQEL
BQAwVTELMAkGA1UEBhMCVVMxGDAWBgNVBAoTD1UuUy4gR292ZXJubWVudDENMAsG
A1UECxMERlBLSTEdMBsGA1UEAxMURmVkZXJhbCBCcmlkZ2UgQ0EgRzQwHhcNMjAx
MDIyMTcwNDE5WhcNMjMxMDIyMTcwNDE5WjCBgTELMAkGA1UEBhMCVVMxHTAbBgNV
BAoTFFN5bWFudGVjIENvcnBvcmF0aW9uMR8wHQYDVQQLExZTeW1hbnRlYyBUcnVz
dCBOZXR3b3JrMTIwMAYDVQQDEylTeW1hbnRlYyBDbGFzcyAzIFNTUCBJbnRlcm1l
ZGlhdGUgQ0EgLSBHMzCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAL2p
75cMpx86sS2aH4r+0o8r+m/KTrPrknWP0RA9Kp6sewAzkNa7BVwg0jOhyamiv1iP
Cns10usoH93nxYbXLWF54vOLRdYU/53KEPNmgkj2ipMaTLuaReBghNibikWSnAmy
S8RItaDMs8tdF2goKPI4xWiamNwqe92VC+pic2tq0Nva3Y4kvMDJjtyje3uduTtL
oyoaaHkrX7i7gE67psnMKj1THUtre1JV1ohl9+oOuyot4p3eSxVlrMWiiwb11bnk
CakecOz/mP2DHMGg6pZ/BeJ+ThaLUylAXECARIqHc9UwRPKC9BfLaCX4edIoeYiB
loRs4KdqLdg/I9eTwKkCAwEAAaOCAxEwggMNMB0GA1UdDgQWBBQ1Jn1QleGhwb0F
1cOdd0LHDBOWjDAfBgNVHSMEGDAWgBR58ABJ6393wl1BAmU0ipAjmx4HbzAOBgNV
HQ8BAf8EBAMCAQYwDwYDVR0TAQH/BAUwAwEB/zCBiAYDVR0gBIGAMH4wDAYKYIZI
AWUDAgEDAzAMBgpghkgBZQMCAQMMMAwGCmCGSAFlAwIBAw4wDAYKYIZIAWUDAgED
DzAMBgpghkgBZQMCAQMSMAwGCmCGSAFlAwIBAxMwDAYKYIZIAWUDAgEDFDAMBgpg
hkgBZQMCAQMlMAwGCmCGSAFlAwIBAyYwggESBgNVHSEEggEJMIIBBTAbBgpghkgB
ZQMCAQMDBg1ghkgBhvhFAQcXAwEGMBsGCmCGSAFlAwIBAwwGDWCGSAGG+EUBBxcD
AQcwGwYKYIZIAWUDAgEDDgYNYIZIAYb4RQEHFwMBDjAbBgpghkgBZQMCAQMPBg1g
hkgBhvhFAQcXAwEPMBsGCmCGSAFlAwIBAxIGDWCGSAGG+EUBBxcDARIwGwYKYIZI
AWUDAgEDEwYNYIZIAYb4RQEHFwMBETAbBgpghkgBZQMCAQMUBg1ghkgBhvhFAQcX
AwEUMBsGCmCGSAFlAwIBAyUGDWCGSAGG+EUBBxcDAQgwGwYKYIZIAWUDAgEDJgYN
YIZIAYb4RQEHFwMBJDBgBggrBgEFBQcBCwRUMFIwUAYIKwYBBQUHMAWGRGh0dHA6
Ly9zc3Atc2lhLnN5bWF1dGguY29tL1NUTlNTUC9DZXJ0c19Jc3N1ZWRfYnlfQ2xh
c3MzU1NQQ0EtRzMucDdjMA8GA1UdJAQIMAaAAQCBAQAwCgYDVR02BAMCAQAwUQYI
KwYBBQUHAQEERTBDMEEGCCsGAQUFBzAChjVodHRwOi8vcmVwby5mcGtpLmdvdi9i
cmlkZ2UvY2FDZXJ0c0lzc3VlZFRvZmJjYWc0LnA3YzA3BgNVHR8EMDAuMCygKqAo
hiZodHRwOi8vcmVwby5mcGtpLmdvdi9icmlkZ2UvZmJjYWc0LmNybDANBgkqhkiG
9w0BAQsFAAOCAQEAA751TycC1f/WTkHmedF9ZWxP58Jstmwvkyo8bKueJ0eF7LTG
BgQlzE2B9vke4sFhd4V+BdgOPGE1dsGzllYKCWg0BhkCBs5kIJ7F6Ay6G1TBuGU1
Ie8247GL+P9pcC5TVvXHC/62R2w3DuD/vAPLbYEbSQjobXlsqt8Kmtd6yK/jVuDV
BTZMdZmvoNtjemqmgcBXHsf0ctVm0m6tH5uYqyVxu8tfyUis6Cf303PHj+spWP1k
gc5PYnVF0ot7qAmNFENIpbKg3BdusBkF9rGxLaDSUBvSc7+s9iQz9d/iRuAebrYu
+eqUlJ2lsjS1U8qyPmlH+spfPNbAEQEsuP32Aw==
-----END CERTIFICATE-----
`

func TestPolicyParse(t *testing.T) {
	b, _ := pem.Decode([]byte(policyPEM))
	c, err := ParseCertificate(b.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Policies) != 9 {
		t.Errorf("unexpected number of policies: got %d, want %d", len(c.Policies), 9)
	}
	if len(c.PolicyMappings) != 9 {
		t.Errorf("unexpected number of policy mappings: got %d, want %d", len(c.PolicyMappings), 9)
	}
	if !c.RequireExplicitPolicyZero {
		t.Error("expected RequireExplicitPolicyZero to be set")
	}
	if !c.InhibitPolicyMappingZero {
		t.Error("expected InhibitPolicyMappingZero to be set")
	}
	if !c.InhibitAnyPolicyZero {
		t.Error("expected InhibitAnyPolicyZero to be set")
	}
}

func TestParsePolicies(t *testing.T) {
	for _, tc := range []string{
		"testdata/policy_leaf_duplicate.pem",
		"testdata/policy_leaf_invalid.pem",
	} {
		t.Run(tc, func(t *testing.T) {
			b, err := os.ReadFile(tc)
			if err != nil {
				t.Fatal(err)
			}
			p, _ := pem.Decode(b)
			_, err = ParseCertificate(p.Bytes)
			if err == nil {
				t.Error("parsing should've failed")
			}
		})
	}
}

func TestDomainNameValid(t *testing.T) {
	for _, tc := range []struct {
		name       string
		dnsName    string
		constraint bool
		valid      bool
	}{
		// TODO(#75835): these tests are for stricter name validation, which we
		// had to disable. Once we reenable these strict checks, behind a
		// GODEBUG, we should add them back in.
		// {"empty name, name", "", false, false},
		// {"254 char label, name", strings.Repeat("a.a", 84) + "aaa", false, false},
		// {"254 char label, constraint", strings.Repeat("a.a", 84) + "aaa", true, false},
		// {"253 char label, name", strings.Repeat("a.a", 84) + "aa", false, false},
		// {"253 char label, constraint", strings.Repeat("a.a", 84) + "aa", true, false},
		// {"64 char single label, name", strings.Repeat("a", 64), false, false},
		// {"64 char single label, constraint", strings.Repeat("a", 64), true, false},
		// {"64 char label, name", "a." + strings.Repeat("a", 64), false, false},
		// {"64 char label, constraint", "a." + strings.Repeat("a", 64), true, false},

		// TODO(#75835): these are the inverse of the tests above, they should be removed
		// once the strict checking is enabled.
		{"254 char label, name", strings.Repeat("a.a", 84) + "aaa", false, true},
		{"254 char label, constraint", strings.Repeat("a.a", 84) + "aaa", true, true},
		{"253 char label, name", strings.Repeat("a.a", 84) + "aa", false, true},
		{"253 char label, constraint", strings.Repeat("a.a", 84) + "aa", true, true},
		{"64 char single label, name", strings.Repeat("a", 64), false, true},
		{"64 char single label, constraint", strings.Repeat("a", 64), true, true},
		{"64 char label, name", "a." + strings.Repeat("a", 64), false, true},
		{"64 char label, constraint", "a." + strings.Repeat("a", 64), true, true},

		// Check we properly enforce properties of domain names.
		{"empty name, constraint", "", true, true},
		{"empty label, name", "a..a", false, false},
		{"empty label, constraint", "a..a", true, false},
		{"period, name", ".", false, false},
		{"period, constraint", ".", true, false}, // TODO(roland): not entirely clear if this is a valid constraint (require at least one label?)
		{"valid, name", "a.b.c", false, true},
		{"valid, constraint", "a.b.c", true, true},
		{"leading period, name", ".a.b.c", false, false},
		{"leading period, constraint", ".a.b.c", true, true},
		{"trailing period, name", "a.", false, false},
		{"trailing period, constraint", "a.", true, false},
		{"bare label, name", "a", false, true},
		{"bare label, constraint", "a", true, true},
		{"63 char single label, name", strings.Repeat("a", 63), false, true},
		{"63 char single label, constraint", strings.Repeat("a", 63), true, true},
		{"63 char label, name", "a." + strings.Repeat("a", 63), false, true},
		{"63 char label, constraint", "a." + strings.Repeat("a", 63), true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			valid := domainNameValid(tc.dnsName, tc.constraint)
			if tc.valid != valid {
				t.Errorf("domainNameValid(%q, %t) = %v; want %v", tc.dnsName, tc.constraint, !tc.valid, tc.valid)
			}
			// Also check that we enforce the same properties as domainToReverseLabels
			trimmedName := tc.dnsName
			if tc.constraint && len(trimmedName) > 1 && trimmedName[0] == '.' {
				trimmedName = trimmedName[1:]
			}
			_, revValid := domainToReverseLabels(trimmedName)
			if valid != revValid {
				t.Errorf("domainNameValid(%q, %t) = %t != domainToReverseLabels(%q) = %t", tc.dnsName, tc.constraint, valid, trimmedName, revValid)
			}
		})
	}
}

func TestRoundtripWeirdSANs(t *testing.T) {
	// TODO(#75835): check that certificates we create with CreateCertificate that have malformed SAN values
	// can be parsed by ParseCertificate. We should eventually restrict this, but for now we have to maintain
	// this property as people have been relying on it.
	k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	badNames := []string{
		"baredomain",
		"baredomain.",
		strings.Repeat("a", 255),
		strings.Repeat("a", 65) + ".com",
	}
	tmpl := &Certificate{
		EmailAddresses: badNames,
		DNSNames:       badNames,
	}
	b, err := CreateCertificate(rand.Reader, tmpl, tmpl, &k.PublicKey, k)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ParseCertificate(b)
	if err != nil {
		t.Fatalf("Couldn't roundtrip certificate: %v", err)
	}
}

func FuzzDomainNameValid(f *testing.F) {
	f.Fuzz(func(t *testing.T, data string) {
		domainNameValid(data, false)
		domainNameValid(data, true)
	})
}
