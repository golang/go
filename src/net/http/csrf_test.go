// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// httptestNewRequest works around https://go.dev/issue/73151.
func httptestNewRequest(method, target string) *http.Request {
	req := httptest.NewRequest(method, target, nil)
	req.URL.Scheme = ""
	req.URL.Host = ""
	return req
}

var okHandler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
})

func TestCrossOriginProtectionSecFetchSite(t *testing.T) {
	protection := http.NewCrossOriginProtection()
	handler := protection.Handler(okHandler)

	tests := []struct {
		name           string
		method         string
		secFetchSite   string
		origin         string
		expectedStatus int
	}{
		{"same-origin allowed", "POST", "same-origin", "", http.StatusOK},
		{"none allowed", "POST", "none", "", http.StatusOK},
		{"cross-site blocked", "POST", "cross-site", "", http.StatusForbidden},
		{"same-site blocked", "POST", "same-site", "", http.StatusForbidden},

		{"no header with no origin", "POST", "", "", http.StatusOK},
		{"no header with matching origin", "POST", "", "https://example.com", http.StatusOK},
		{"no header with mismatched origin", "POST", "", "https://attacker.example", http.StatusForbidden},
		{"no header with null origin", "POST", "", "null", http.StatusForbidden},

		{"GET allowed", "GET", "cross-site", "", http.StatusOK},
		{"HEAD allowed", "HEAD", "cross-site", "", http.StatusOK},
		{"OPTIONS allowed", "OPTIONS", "cross-site", "", http.StatusOK},
		{"PUT blocked", "PUT", "cross-site", "", http.StatusForbidden},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptestNewRequest(tc.method, "https://example.com/")
			if tc.secFetchSite != "" {
				req.Header.Set("Sec-Fetch-Site", tc.secFetchSite)
			}
			if tc.origin != "" {
				req.Header.Set("Origin", tc.origin)
			}

			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)

			if w.Code != tc.expectedStatus {
				t.Errorf("got status %d, want %d", w.Code, tc.expectedStatus)
			}
		})
	}
}

func TestCrossOriginProtectionTrustedOriginBypass(t *testing.T) {
	protection := http.NewCrossOriginProtection()
	err := protection.AddTrustedOrigin("https://trusted.example")
	if err != nil {
		t.Fatalf("AddTrustedOrigin: %v", err)
	}
	handler := protection.Handler(okHandler)

	tests := []struct {
		name           string
		origin         string
		secFetchSite   string
		expectedStatus int
	}{
		{"trusted origin without sec-fetch-site", "https://trusted.example", "", http.StatusOK},
		{"trusted origin with cross-site", "https://trusted.example", "cross-site", http.StatusOK},
		{"untrusted origin without sec-fetch-site", "https://attacker.example", "", http.StatusForbidden},
		{"untrusted origin with cross-site", "https://attacker.example", "cross-site", http.StatusForbidden},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptestNewRequest("POST", "https://example.com/")
			req.Header.Set("Origin", tc.origin)
			if tc.secFetchSite != "" {
				req.Header.Set("Sec-Fetch-Site", tc.secFetchSite)
			}

			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)

			if w.Code != tc.expectedStatus {
				t.Errorf("got status %d, want %d", w.Code, tc.expectedStatus)
			}
		})
	}
}

func TestCrossOriginProtectionPatternBypass(t *testing.T) {
	protection := http.NewCrossOriginProtection()
	protection.AddInsecureBypassPattern("/bypass/")
	protection.AddInsecureBypassPattern("/only/{foo}")
	protection.AddInsecureBypassPattern("/no-trailing")
	protection.AddInsecureBypassPattern("/yes-trailing/")
	protection.AddInsecureBypassPattern("PUT /put-only/")
	protection.AddInsecureBypassPattern("GET /get-only/")
	protection.AddInsecureBypassPattern("POST /post-only/")
	handler := protection.Handler(okHandler)

	tests := []struct {
		name           string
		path           string
		secFetchSite   string
		expectedStatus int
	}{
		{"bypass path without sec-fetch-site", "/bypass/", "", http.StatusOK},
		{"bypass path with cross-site", "/bypass/", "cross-site", http.StatusOK},
		{"non-bypass path without sec-fetch-site", "/api/", "", http.StatusForbidden},
		{"non-bypass path with cross-site", "/api/", "cross-site", http.StatusForbidden},

		{"redirect to bypass path without ..", "/foo/../bypass/bar", "", http.StatusForbidden},
		{"redirect to bypass path with trailing slash", "/bypass", "", http.StatusForbidden},
		{"redirect to non-bypass path with ..", "/foo/../api/bar", "", http.StatusForbidden},
		{"redirect to non-bypass path with trailing slash", "/api", "", http.StatusForbidden},

		{"wildcard bypass", "/only/123", "", http.StatusOK},
		{"non-wildcard", "/only/123/foo", "", http.StatusForbidden},

		// https://go.dev/issue/75054
		{"no trailing slash exact match", "/no-trailing", "", http.StatusOK},
		{"no trailing slash with slash", "/no-trailing/", "", http.StatusForbidden},
		{"yes trailing slash exact match", "/yes-trailing/", "", http.StatusOK},
		{"yes trailing slash without slash", "/yes-trailing", "", http.StatusForbidden},

		{"method-specific hit", "/post-only/", "", http.StatusOK},
		{"method-specific miss (PUT)", "/put-only/", "", http.StatusForbidden},
		{"method-specific miss (GET)", "/get-only/", "", http.StatusForbidden},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptestNewRequest("POST", "https://example.com"+tc.path)
			req.Header.Set("Origin", "https://attacker.example")
			if tc.secFetchSite != "" {
				req.Header.Set("Sec-Fetch-Site", tc.secFetchSite)
			}

			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)

			if w.Code != tc.expectedStatus {
				t.Errorf("got status %d, want %d", w.Code, tc.expectedStatus)
			}
		})
	}
}

func TestCrossOriginProtectionSetDenyHandler(t *testing.T) {
	protection := http.NewCrossOriginProtection()

	handler := protection.Handler(okHandler)

	req := httptestNewRequest("POST", "https://example.com/")
	req.Header.Set("Sec-Fetch-Site", "cross-site")

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("got status %d, want %d", w.Code, http.StatusForbidden)
	}

	customErrHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTeapot)
		io.WriteString(w, "custom error")
	})
	protection.SetDenyHandler(customErrHandler)

	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusTeapot {
		t.Errorf("got status %d, want %d", w.Code, http.StatusTeapot)
	}

	if !strings.Contains(w.Body.String(), "custom error") {
		t.Errorf("expected custom error message, got: %q", w.Body.String())
	}

	req = httptestNewRequest("GET", "https://example.com/")

	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("got status %d, want %d", w.Code, http.StatusOK)
	}

	protection.SetDenyHandler(nil)

	req = httptestNewRequest("POST", "https://example.com/")
	req.Header.Set("Sec-Fetch-Site", "cross-site")

	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("got status %d, want %d", w.Code, http.StatusForbidden)
	}
}

func TestCrossOriginProtectionAddTrustedOriginErrors(t *testing.T) {
	protection := http.NewCrossOriginProtection()

	tests := []struct {
		name    string
		origin  string
		wantErr bool
	}{
		{"valid origin", "https://example.com", false},
		{"valid origin with port", "https://example.com:8080", false},
		{"http origin", "http://example.com", false},
		{"missing scheme", "example.com", true},
		{"missing host", "https://", true},
		{"trailing slash", "https://example.com/", true},
		{"with path", "https://example.com/path", true},
		{"with query", "https://example.com?query=value", true},
		{"with fragment", "https://example.com#fragment", true},
		{"invalid url", "https://ex ample.com", true},
		{"empty string", "", true},
		{"null", "null", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := protection.AddTrustedOrigin(tc.origin)
			if (err != nil) != tc.wantErr {
				t.Errorf("AddTrustedOrigin(%q) error = %v, wantErr %v", tc.origin, err, tc.wantErr)
			}
		})
	}
}

func TestCrossOriginProtectionAddingBypassesConcurrently(t *testing.T) {
	protection := http.NewCrossOriginProtection()
	handler := protection.Handler(okHandler)

	req := httptestNewRequest("POST", "https://example.com/")
	req.Header.Set("Origin", "https://concurrent.example")
	req.Header.Set("Sec-Fetch-Site", "cross-site")

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("got status %d, want %d", w.Code, http.StatusForbidden)
	}

	start := make(chan struct{})
	done := make(chan struct{})
	go func() {
		close(start)
		defer close(done)
		for range 10 {
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)
		}
	}()

	// Add bypasses while the requests are in flight.
	<-start
	protection.AddTrustedOrigin("https://concurrent.example")
	protection.AddInsecureBypassPattern("/foo/")
	<-done

	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("After concurrent bypass addition, got status %d, want %d", w.Code, http.StatusOK)
	}
}

func TestCrossOriginProtectionServer(t *testing.T) {
	protection := http.NewCrossOriginProtection()
	protection.AddTrustedOrigin("https://trusted.example")
	protection.AddInsecureBypassPattern("/bypass/")
	handler := protection.Handler(okHandler)

	ts := httptest.NewServer(handler)
	defer ts.Close()

	tests := []struct {
		name           string
		method         string
		url            string
		origin         string
		secFetchSite   string
		expectedStatus int
	}{
		{"cross-site", "POST", ts.URL, "https://attacker.example", "cross-site", http.StatusForbidden},
		{"same-origin", "POST", ts.URL, "", "same-origin", http.StatusOK},
		{"origin matches host", "POST", ts.URL, ts.URL, "", http.StatusOK},
		{"trusted origin", "POST", ts.URL, "https://trusted.example", "", http.StatusOK},
		{"untrusted origin", "POST", ts.URL, "https://attacker.example", "", http.StatusForbidden},
		{"bypass path", "POST", ts.URL + "/bypass/", "https://attacker.example", "", http.StatusOK},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req, err := http.NewRequest(tc.method, tc.url, nil)
			if err != nil {
				t.Fatalf("NewRequest: %v", err)
			}
			if tc.origin != "" {
				req.Header.Set("Origin", tc.origin)
			}
			if tc.secFetchSite != "" {
				req.Header.Set("Sec-Fetch-Site", tc.secFetchSite)
			}
			client := &http.Client{}
			resp, err := client.Do(req)
			if err != nil {
				t.Fatalf("Do: %v", err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != tc.expectedStatus {
				t.Errorf("got status %d, want %d", resp.StatusCode, tc.expectedStatus)
			}
		})
	}
}
