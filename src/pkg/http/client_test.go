// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for client.go

package http

import (
	"io/ioutil"
	"strings"
	"testing"
)

func TestClient(t *testing.T) {
	// TODO: add a proper test suite.  Current test merely verifies that
	// we can retrieve the Google robots.txt file.

	r, _, err := Get("http://www.google.com/robots.txt")
	var b []byte
	if err == nil {
		b, err = ioutil.ReadAll(r.Body)
		r.Body.Close()
	}
	if err != nil {
		t.Error(err)
	} else if s := string(b); !strings.HasPrefix(s, "User-agent:") {
		t.Errorf("Incorrect page body (did not begin with User-agent): %q", s)
	}
}

func TestClientHead(t *testing.T) {
	r, err := Head("http://www.google.com/robots.txt")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := r.Header["Last-Modified"]; !ok {
		t.Error("Last-Modified header not found.")
	}
}
