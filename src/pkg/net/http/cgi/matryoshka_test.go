// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests a Go CGI program running under a Go CGI host process.
// Further, the two programs are the same binary, just checking
// their environment to figure out what mode to run in.

package cgi

import (
	"fmt"
	"net/http"
	"os"
	"testing"
)

// This test is a CGI host (testing host.go) that runs its own binary
// as a child process testing the other half of CGI (child.go).
func TestHostingOurselves(t *testing.T) {
	h := &Handler{
		Path: os.Args[0],
		Root: "/test.go",
		Args: []string{"-test.run=TestBeChildCGIProcess"},
	}
	expectedMap := map[string]string{
		"test":                  "Hello CGI-in-CGI",
		"param-a":               "b",
		"param-foo":             "bar",
		"env-GATEWAY_INTERFACE": "CGI/1.1",
		"env-HTTP_HOST":         "example.com",
		"env-PATH_INFO":         "",
		"env-QUERY_STRING":      "foo=bar&a=b",
		"env-REMOTE_ADDR":       "1.2.3.4",
		"env-REMOTE_HOST":       "1.2.3.4",
		"env-REQUEST_METHOD":    "GET",
		"env-REQUEST_URI":       "/test.go?foo=bar&a=b",
		"env-SCRIPT_FILENAME":   os.Args[0],
		"env-SCRIPT_NAME":       "/test.go",
		"env-SERVER_NAME":       "example.com",
		"env-SERVER_PORT":       "80",
		"env-SERVER_SOFTWARE":   "go",
	}
	replay := runCgiTest(t, h, "GET /test.go?foo=bar&a=b HTTP/1.0\nHost: example.com\n\n", expectedMap)

	if expected, got := "text/html; charset=utf-8", replay.Header().Get("Content-Type"); got != expected {
		t.Errorf("got a Content-Type of %q; expected %q", got, expected)
	}
	if expected, got := "X-Test-Value", replay.Header().Get("X-Test-Header"); got != expected {
		t.Errorf("got a X-Test-Header of %q; expected %q", got, expected)
	}
}

// Test that a child handler only writing headers works.
func TestChildOnlyHeaders(t *testing.T) {
	h := &Handler{
		Path: os.Args[0],
		Root: "/test.go",
		Args: []string{"-test.run=TestBeChildCGIProcess"},
	}
	expectedMap := map[string]string{
		"_body": "",
	}
	replay := runCgiTest(t, h, "GET /test.go?no-body=1 HTTP/1.0\nHost: example.com\n\n", expectedMap)
	if expected, got := "X-Test-Value", replay.Header().Get("X-Test-Header"); got != expected {
		t.Errorf("got a X-Test-Header of %q; expected %q", got, expected)
	}
}

// Note: not actually a test.
func TestBeChildCGIProcess(t *testing.T) {
	if os.Getenv("REQUEST_METHOD") == "" {
		// Not in a CGI environment; skipping test.
		return
	}
	Serve(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		rw.Header().Set("X-Test-Header", "X-Test-Value")
		req.ParseForm()
		if req.FormValue("no-body") == "1" {
			return
		}
		fmt.Fprintf(rw, "test=Hello CGI-in-CGI\n")
		for k, vv := range req.Form {
			for _, v := range vv {
				fmt.Fprintf(rw, "param-%s=%s\n", k, v)
			}
		}
		for _, kv := range os.Environ() {
			fmt.Fprintf(rw, "env-%s\n", kv)
		}
	}))
	os.Exit(0)
}
