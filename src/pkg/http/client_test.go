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
	// we can retrieve the Google home page.
	
	r, url, err := Get("http://www.google.com");
	var b []byte;
	if err == nil {
		b, err = io.ReadAll(r.Body);
		r.Body.Close();
	}

	// TODO: io.ErrEOF check is needed because we're sometimes getting
	// this error when nothing is actually wrong.  rsc suspects a bug
	// in bufio.  Can remove the ErrEOF check once the bug is fixed
	// (expected to occur within a few weeks of this writing, 6/9/09).
	if err != nil && err != io.ErrEOF {
		t.Errorf("Error fetching URL: %v", err);
	} else {
		s := string(b);
		if (!strings.HasPrefix(s, "<html>")) {
			t.Errorf("Incorrect page body (did not begin with <html>): %q", s);
		}
	}
}
