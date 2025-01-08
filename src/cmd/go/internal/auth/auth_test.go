// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package auth

import (
	"net/http"
	"reflect"
	"testing"
)

func TestCredentialCache(t *testing.T) {
	testCases := []netrcLine{
		{"api.github.com", "user", "pwd"},
		{"test.host", "user2", "pwd2"},
		{"oneline", "user3", "pwd3"},
		{"hasmacro.too", "user4", "pwd4"},
		{"hasmacro.too", "user5", "pwd5"},
	}
	for _, tc := range testCases {
		want := http.Request{Header: make(http.Header)}
		want.SetBasicAuth(tc.login, tc.password)
		storeCredential(tc.machine, want.Header)
		got := &http.Request{Header: make(http.Header)}
		ok := loadCredential(got, tc.machine)
		if !ok || !reflect.DeepEqual(got.Header, want.Header) {
			t.Errorf("loadCredential(%q):\nhave %q\nwant %q", tc.machine, got.Header, want.Header)
		}
	}

	// Having stored those credentials, we should be able to look up longer URLs too.
	extraCases := []netrcLine{
		{"https://api.github.com/foo", "user", "pwd"},
		{"https://api.github.com/foo/bar/baz", "user", "pwd"},
		{"https://example.com/abc", "", ""},
		{"https://example.com/?/../api.github.com/", "", ""},
		{"https://example.com/?/../api.github.com", "", ""},
		{"https://example.com/../api.github.com/", "", ""},
		{"https://example.com/../api.github.com", "", ""},
	}
	for _, tc := range extraCases {
		want := http.Request{Header: make(http.Header)}
		if tc.login != "" {
			want.SetBasicAuth(tc.login, tc.password)
		}
		got := &http.Request{Header: make(http.Header)}
		loadCredential(got, tc.machine)
		if !reflect.DeepEqual(got.Header, want.Header) {
			t.Errorf("loadCredential(%q):\nhave %q\nwant %q", tc.machine, got.Header, want.Header)
		}
	}
}

func TestCredentialCacheDelete(t *testing.T) {
	// Store a credential for api.github.com
	want := http.Request{Header: make(http.Header)}
	want.SetBasicAuth("user", "pwd")
	storeCredential("api.github.com", want.Header)
	got := &http.Request{Header: make(http.Header)}
	ok := loadCredential(got, "api.github.com")
	if !ok || !reflect.DeepEqual(got.Header, want.Header) {
		t.Errorf("parseNetrc:\nhave %q\nwant %q", got.Header, want.Header)
	}
	// Providing an empty header for api.github.com should clear credentials.
	want = http.Request{Header: make(http.Header)}
	storeCredential("api.github.com", want.Header)
	got = &http.Request{Header: make(http.Header)}
	ok = loadCredential(got, "api.github.com")
	if ok {
		t.Errorf("loadCredential:\nhave %q\nwant %q", got.Header, want.Header)
	}
}
