// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package web

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

func TestUserAgent(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(r.UserAgent()))
	}))
	defer ts.Close()

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal("parse httptest url:", err)
	}
	res, err := Get(Insecure, u)
	if err != nil {
		t.Error("http get:", err)
	}
	b, err := io.ReadAll(res.Body)
	if err != nil {
		t.Error("read response body:", err)
	}
	gotUserAgent := string(bytes.TrimSpace(b))
	if gotUserAgent != userAgent {
		t.Errorf("User-Agent: %s, want %s", gotUserAgent, userAgent)
	}
}

func TestGoGet1(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/redirect-a":
			goGet := r.URL.Query().Get("go-get")
			if goGet != "1" {
				t.Errorf("missing go-get=1 on initial request: %s", r.URL.String())
			}
			http.Redirect(w, r, "/redirect-b", http.StatusFound)
		case "/redirect-b":
			goGet := r.URL.Query().Get("go-get")
			if goGet != "1" {
				t.Errorf("missing go-get=1 on redirected request: %s", r.URL.String())
			}
			http.Redirect(w, r, "/finish?other=param", http.StatusFound)
		case "/finish":
			goGet := r.URL.Query().Get("go-get")
			if goGet != "1" {
				t.Errorf("missing go-get=1 on final request: %s", r.URL.String())
			}
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer ts.Close()

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal("parse httptest url:", err)
	}
	u.Path = "/redirect-a"
	u.RawQuery = "go-get=1"

	res, err := Get(Insecure, u)
	if err != nil {
		t.Errorf("http get: %v", err)
	} else if res.StatusCode != http.StatusOK {
		t.Errorf("http status != 200: %v", res.Status)
	}
}
