// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package auth

import (
	"net/http"
	"reflect"
	"testing"
)

func TestParseUserAuth(t *testing.T) {
	data := `https://example.com

Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
Authorization: Basic jpvcGVuc2VzYW1lYWxhZGRpb

https://hello.com

Authorization: Basic GVuc2VzYW1lYWxhZGRpbjpvc
Authorization: Basic 1lYWxhZGRplW1lYWxhZGRpbs
Data: Test567

`
	// Build the expected header
	header1 := http.Header{
		"Authorization": []string{
			"Basic YWxhZGRpbjpvcGVuc2VzYW1l",
			"Basic jpvcGVuc2VzYW1lYWxhZGRpb",
		},
	}
	header2 := http.Header{
		"Authorization": []string{
			"Basic GVuc2VzYW1lYWxhZGRpbjpvc",
			"Basic 1lYWxhZGRplW1lYWxhZGRpbs",
		},
		"Data": []string{
			"Test567",
		},
	}
	credentials, err := parseUserAuth(data)
	if err != nil {
		t.Errorf("parseUserAuth(%s): %v", data, err)
	}
	gotHeader, ok := credentials["example.com"]
	if !ok || !reflect.DeepEqual(gotHeader, header1) {
		t.Errorf("parseUserAuth(%s):\nhave %q\nwant %q", data, gotHeader, header1)
	}
	gotHeader, ok = credentials["hello.com"]
	if !ok || !reflect.DeepEqual(gotHeader, header2) {
		t.Errorf("parseUserAuth(%s):\nhave %q\nwant %q", data, gotHeader, header2)
	}
}

func TestParseUserAuthInvalid(t *testing.T) {
	testCases := []string{
		// Missing new line after url.
		`https://example.com
Authorization: Basic AVuc2VzYW1lYWxhZGRpbjpvc

`,
		// Missing url.
		`Authorization: Basic AVuc2VzYW1lYWxhZGRpbjpvc

`,
		// Missing url.
		`https://example.com

Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
Authorization: Basic jpvcGVuc2VzYW1lYWxhZGRpb

Authorization: Basic GVuc2VzYW1lYWxhZGRpbjpvc
Authorization: Basic 1lYWxhZGRplW1lYWxhZGRpbs
Data: Test567

`,
		// Wrong order.
		`Authorization: Basic AVuc2VzYW1lYWxhZGRpbjpvc

https://example.com

`,
		// Missing new lines after URL.
		`https://example.com
`,
		// Missing new line after empty header.
		`https://example.com

`,
		// Missing new line between blocks.
		`https://example.com

Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
Authorization: Basic jpvcGVuc2VzYW1lYWxhZGRpb
https://hello.com

Authorization: Basic GVuc2VzYW1lYWxhZGRpbjpvc
Authorization: Basic 1lYWxhZGRplW1lYWxhZGRpbs
Data: Test567

`,
		// Continuation in URL line
		`https://example.com/
 Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
`,

		// Continuation in header line
		`https://example.com

Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
 Authorization: Basic jpvcGVuc2VzYW1lYWxhZGRpb
`,

		// Continuation in multiple header lines
		`https://example.com

Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
 Authorization: Basic jpvcGVuc2VzYW1lYWxhZGRpb
 Authorization: Basic dGhpc2lzYWxvbmdzdHJpbmc=
`,

		// Continuation with mixed spacing
		`https://example.com

Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
  Authorization: Basic jpvcGVuc2VzYW1lYWxhZGRpb
`,

		// Continuation with tab character
		`https://example.com

Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
        Authorization: Basic jpvcGVuc2VzYW1lYWxhZGRpb
`,
		// Continuation at the start of a block
		` https://example.com

Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
`,

		// Continuation after a blank line
		`https://example.com


Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
`,
	}
	for _, tc := range testCases {
		if credentials, err := parseUserAuth(tc); err == nil {
			t.Errorf("parseUserAuth(%s) should have failed, but got: %v", tc, credentials)
		}
	}
}

func TestParseUserAuthDuplicated(t *testing.T) {
	data := `https://example.com

Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
Authorization: Basic jpvcGVuc2VzYW1lYWxhZGRpb

https://example.com

Authorization: Basic GVuc2VzYW1lYWxhZGRpbjpvc
Authorization: Basic 1lYWxhZGRplW1lYWxhZGRpbs
Data: Test567

`
	// Build the expected header
	header := http.Header{
		"Authorization": []string{
			"Basic GVuc2VzYW1lYWxhZGRpbjpvc",
			"Basic 1lYWxhZGRplW1lYWxhZGRpbs",
		},
		"Data": []string{
			"Test567",
		},
	}
	credentials, err := parseUserAuth(data)
	if err != nil {
		t.Errorf("parseUserAuth(%s): %v", data, err)
	}
	gotHeader, ok := credentials["example.com"]
	if !ok || !reflect.DeepEqual(gotHeader, header) {
		t.Errorf("parseUserAuth(%s):\nhave %q\nwant %q", data, gotHeader, header)
	}
}

func TestParseUserAuthEmptyHeader(t *testing.T) {
	data := "https://example.com\n\n\n"
	// Build the expected header
	header := http.Header{}
	credentials, err := parseUserAuth(data)
	if err != nil {
		t.Errorf("parseUserAuth(%s): %v", data, err)
	}
	gotHeader, ok := credentials["example.com"]
	if !ok || !reflect.DeepEqual(gotHeader, header) {
		t.Errorf("parseUserAuth(%s):\nhave %q\nwant %q", data, gotHeader, header)
	}
}

func TestParseUserAuthEmpty(t *testing.T) {
	data := ``
	// Build the expected header
	credentials, err := parseUserAuth(data)
	if err != nil {
		t.Errorf("parseUserAuth(%s) should have succeeded", data)
	}
	if credentials == nil {
		t.Errorf("parseUserAuth(%s) should have returned a non-nil credential map, but got %v", data, credentials)
	}
}
