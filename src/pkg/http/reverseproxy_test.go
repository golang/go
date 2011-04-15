// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reverse proxy tests.

package http_test

import (
	. "http"
	"http/httptest"
	"io/ioutil"
	"testing"
)

func TestReverseProxy(t *testing.T) {
	const backendResponse = "I am the backend"
	const backendStatus = 404
	backend := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.Header.Get("X-Forwarded-For") == "" {
			t.Errorf("didn't get X-Forwarded-For header")
		}
		w.Header().Set("X-Foo", "bar")
		w.WriteHeader(backendStatus)
		w.Write([]byte(backendResponse))
	}))
	defer backend.Close()
	backendURL, err := ParseURL(backend.URL)
	if err != nil {
		t.Fatal(err)
	}
	proxyHandler := NewSingleHostReverseProxy(backendURL)
	frontend := httptest.NewServer(proxyHandler)
	defer frontend.Close()

	res, _, err := Get(frontend.URL)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if g, e := res.StatusCode, backendStatus; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	if g, e := res.Header.Get("X-Foo"), "bar"; g != e {
		t.Errorf("got X-Foo %q; expected %q", g, e)
	}
	bodyBytes, _ := ioutil.ReadAll(res.Body)
	if g, e := string(bodyBytes), backendResponse; g != e {
		t.Errorf("got body %q; expected %q", g, e)
	}
}
