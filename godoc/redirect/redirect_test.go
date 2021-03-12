// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package redirect

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

type redirectResult struct {
	status int
	path   string
}

func errorResult(status int) redirectResult {
	return redirectResult{status, ""}
}

func TestRedirects(t *testing.T) {
	var tests = map[string]redirectResult{
		"/foo": errorResult(404),
	}

	mux := http.NewServeMux()
	Register(mux)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	for path, want := range tests {
		if want.path != "" && want.path[0] == '/' {
			// All redirects are absolute.
			want.path = ts.URL + want.path
		}

		req, err := http.NewRequest("GET", ts.URL+path, nil)
		if err != nil {
			t.Errorf("(path: %q) unexpected error: %v", path, err)
			continue
		}

		resp, err := http.DefaultTransport.RoundTrip(req)
		if err != nil {
			t.Errorf("(path: %q) unexpected error: %v", path, err)
			continue
		}
		resp.Body.Close() // We only care about the headers, so close the body immediately.

		if resp.StatusCode != want.status {
			t.Errorf("(path: %q) got status %d, want %d", path, resp.StatusCode, want.status)
		}

		if want.status != 301 && want.status != 302 {
			// Not a redirect. Just check status.
			continue
		}

		out, _ := resp.Location()
		if got := out.String(); got != want.path {
			t.Errorf("(path: %q) got %s, want %s", path, got, want.path)
		}
	}
}
