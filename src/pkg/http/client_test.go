// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for client.go

package http

import (
	"fmt";
	"http";
	"io";
	"strings";
	"testing";
)

func TestClient(t *testing.T) {
	// TODO: add a proper test suite.  Current test merely verifies that
	// we can retrieve the Google robots.txt file.

	r, url, err := Get("http://www.google.com/robots.txt");
	var b []byte;
	if err == nil {
		b, err = io.ReadAll(r.Body);
		r.Body.Close();
	}
	if err != nil {
		t.Error(err);
	} else if s := string(b); !strings.HasPrefix(s, "User-agent:") {
		t.Errorf("Incorrect page body (did not begin with User-agent): %q", s);
	}
}
