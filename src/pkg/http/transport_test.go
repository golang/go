// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for transport.go

package http_test

import (
	"fmt"
	. "http"
	"http/httptest"
	"io/ioutil"
	"testing"
)

func TestTransportNilURL(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "Hi")
	}))
	defer ts.Close()

	req := new(Request)
	req.URL = nil // what we're actually testing
	req.Method = "GET"
	req.RawURL = ts.URL
	req.Proto = "HTTP/1.1"
	req.ProtoMajor = 1
	req.ProtoMinor = 1

	// TODO(bradfitz): test &transport{} and not DefaultTransport
	// once Transport is exported.
	res, err := DefaultTransport.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected RoundTrip error: %v", err)
	}
	body, err := ioutil.ReadAll(res.Body)
	if g, e := string(body), "Hi"; g != e {
		t.Fatalf("Expected response body of %q; got %q", e, g)
	}
}
