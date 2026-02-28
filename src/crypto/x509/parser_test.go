// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package x509

import (
	"encoding/asn1"
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
