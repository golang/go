// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package main

import (
	"net/http/httptest"
	"testing"
)

func TestTipRedirects(t *testing.T) {
	mux := newServeMux(&Proxy{builder: &godocBuilder{}})
	req := httptest.NewRequest("GET", "http://example.com/foo?bar=baz", nil)
	req.Header.Set("X-Forwarded-Proto", "http")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != 302 {
		t.Errorf("expected Code to be 302, got %d", w.Code)
	}
	want := "https://example.com/foo?bar=baz"
	if loc := w.Header().Get("Location"); loc != want {
		t.Errorf("Location header: got %s, want %s", loc, want)
	}
}
