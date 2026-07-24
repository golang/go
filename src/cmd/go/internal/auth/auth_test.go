// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package auth

import (
	"cmd/go/internal/cfg"
	"cmd/go/internal/web/intercept"
	"cmd/internal/quoted"
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

func TestCredentialCacheTrailingSlash(t *testing.T) {
	// Store a credential for api.github.com/foo/bar
	want := http.Request{Header: make(http.Header)}
	want.SetBasicAuth("user", "pwd")
	storeCredential("api.github.com/foo", want.Header)
	got := &http.Request{Header: make(http.Header)}
	ok := loadCredential(got, "api.github.com/foo/bar")
	if !ok || !reflect.DeepEqual(got.Header, want.Header) {
		t.Errorf("parseNetrc:\nhave %q\nwant %q", got.Header, want.Header)
	}
	got2 := &http.Request{Header: make(http.Header)}
	ok = loadCredential(got2, "https://api.github.com/foo/bar/")
	if !ok || !reflect.DeepEqual(got2.Header, want.Header) {
		t.Errorf("parseNetrc:\nhave %q\nwant %q", got2.Header, want.Header)
	}
}

func TestParseClientCertificate(t *testing.T) {
	dir := t.TempDir()
	certFile := dir + "/client cert.pem"
	keyFile := dir + "/client key.pem"

	for _, test := range []struct {
		name  string
		words []string
		want  ClientCertificate
	}{
		{
			name:  "combined PEM",
			words: []string{"mtls", "https://REGISTRY.example.com./", certFile},
			want:  ClientCertificate{Origin: "https://registry.example.com:443", CertFile: certFile, KeyFile: certFile},
		},
		{
			name:  "separate key and port",
			words: []string{"mtls", "https://registry.example.com:8443", certFile, keyFile},
			want:  ClientCertificate{Origin: "https://registry.example.com:8443", CertFile: certFile, KeyFile: keyFile},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := parseClientCertificate(test.words)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, test.want) {
				t.Fatalf("parseClientCertificate() = %#v, want %#v", got, test.want)
			}
		})
	}

	for _, words := range [][]string{
		{"mtls"},
		{"mtls", "http://registry.example.com", certFile},
		{"mtls", "https://user@registry.example.com", certFile},
		{"mtls", "https://registry.example.com/path", certFile},
		{"mtls", "https://registry.example.com?query", certFile},
		{"mtls", "https://registry.example.com?", certFile},
		{"mtls", "https://\u0130.example.com", certFile},
		{"mtls", "https://registry.example.com", "relative-cert.pem"},
		{"mtls", "https://registry.example.com", certFile, "relative-key.pem"},
	} {
		if _, err := parseClientCertificate(words); err == nil {
			t.Errorf("parseClientCertificate(%q) succeeded, want error", words)
		}
	}
}

func TestClientCertificateForRequest(t *testing.T) {
	cert := ClientCertificate{
		Origin:   "https://registry.example.com:443",
		CertFile: "/client-cert.pem",
		KeyFile:  "/client-key.pem",
	}
	clientCertificateCache.Store(cert.Origin, cert)
	defer clientCertificateCache.Delete(cert.Origin)
	lookalikeCert := ClientCertificate{
		Origin:   "https://i.example:443",
		CertFile: "/lookalike-client-cert.pem",
		KeyFile:  "/lookalike-client-key.pem",
	}
	clientCertificateCache.Store(lookalikeCert.Origin, lookalikeCert)
	defer clientCertificateCache.Delete(lookalikeCert.Origin)

	for _, test := range []struct {
		name      string
		url       string
		host      string
		testHooks bool
		want      bool
	}{
		{name: "exact origin", url: "https://registry.example.com/module", want: true},
		{name: "case and trailing dot", url: "https://REGISTRY.example.com./module", want: true},
		{name: "explicit default port", url: "https://registry.example.com:443/module", want: true},
		{name: "different port", url: "https://registry.example.com:8443/module", want: false},
		{name: "subdomain", url: "https://sub.registry.example.com/module", want: false},
		{name: "different scheme", url: "http://registry.example.com/module", want: false},
		{name: "IDNA lookalike", url: "https://\u0130.example/module", want: false},
		{name: "intercepted logical Host", url: "https://127.0.0.1/module", host: "registry.example.com", testHooks: true, want: true},
		{name: "Host ignored without test hooks", url: "https://127.0.0.1/module", host: "registry.example.com", want: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if test.testHooks != intercept.TestHooksEnabled {
				defer func(saved bool) { intercept.TestHooksEnabled = saved }(intercept.TestHooksEnabled)
				intercept.TestHooksEnabled = test.testHooks
			}
			req, err := http.NewRequest("GET", test.url, nil)
			if err != nil {
				t.Fatal(err)
			}
			req.Host = test.host
			got, ok := ClientCertificateForRequest(req)
			if ok != test.want {
				t.Fatalf("ClientCertificateForRequest() found = %t, want %t", ok, test.want)
			}
			if ok && !reflect.DeepEqual(got, cert) {
				t.Fatalf("ClientCertificateForRequest() = %#v, want %#v", got, cert)
			}
		})
	}
}

func TestRunGoAuthMTLS(t *testing.T) {
	oldGOAUTH := cfg.GOAUTH
	defer func() { cfg.GOAUTH = oldGOAUTH }()

	dir := t.TempDir()
	first := dir + "/first client.pem"
	second := dir + "/second client.pem"
	origin := "https://registry.example.com:443"
	defer clientCertificateCache.Delete(origin)

	// The first GOAUTH method has priority when multiple methods configure
	// the same origin.
	firstCommand, err := quoted.Join([]string{"mtls", "https://registry.example.com", first})
	if err != nil {
		t.Fatal(err)
	}
	secondCommand, err := quoted.Join([]string{"mtls", "https://registry.example.com", second})
	if err != nil {
		t.Fatal(err)
	}
	cfg.GOAUTH = firstCommand + "; " + secondCommand
	runGoAuth(http.DefaultClient, nil, "")

	req, err := http.NewRequest("GET", "https://registry.example.com/module", nil)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := ClientCertificateForRequest(req)
	if !ok {
		t.Fatal("ClientCertificateForRequest did not find GOAUTH=mtls configuration")
	}
	want := ClientCertificate{Origin: origin, CertFile: first, KeyFile: first}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ClientCertificateForRequest() = %#v, want %#v", got, want)
	}

	req, err = http.NewRequest("GET", "https://registry.example.com/module", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !AddCredentials(http.DefaultClient, req, nil, "") {
		t.Error("AddCredentials did not report the matching client certificate")
	}
}
