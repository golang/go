// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests a Go CGI program running under a Go CGI host process.
// Further, the two programs are the same binary, just checking
// their environment to figure out what mode to run in.

package cgi

import (
	"bytes"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
)

// This test is a CGI host (testing host.go) that runs its own binary
// as a child process testing the other half of CGI (child.go).
func TestHostingOurselves(t *testing.T) {
	testenv.MustHaveExec(t)

	h := &Handler{
		Path: os.Args[0],
		Root: "/test.go",
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
		"env-REMOTE_PORT":       "1234",
		"env-REQUEST_METHOD":    "GET",
		"env-REQUEST_URI":       "/test.go?foo=bar&a=b",
		"env-SCRIPT_FILENAME":   os.Args[0],
		"env-SCRIPT_NAME":       "/test.go",
		"env-SERVER_NAME":       "example.com",
		"env-SERVER_PORT":       "80",
		"env-SERVER_SOFTWARE":   "go",
	}
	replay := runCgiTest(t, h, "GET /test.go?foo=bar&a=b HTTP/1.0\nHost: example.com\n\n", expectedMap)

	if expected, got := "text/plain; charset=utf-8", replay.Header().Get("Content-Type"); got != expected {
		t.Errorf("got a Content-Type of %q; expected %q", got, expected)
	}
	if expected, got := "X-Test-Value", replay.Header().Get("X-Test-Header"); got != expected {
		t.Errorf("got a X-Test-Header of %q; expected %q", got, expected)
	}
}

type customWriterRecorder struct {
	w io.Writer
	*httptest.ResponseRecorder
}

func (r *customWriterRecorder) Write(p []byte) (n int, err error) {
	return r.w.Write(p)
}

type limitWriter struct {
	w io.Writer
	n int
}

func (w *limitWriter) Write(p []byte) (n int, err error) {
	if len(p) > w.n {
		p = p[:w.n]
	}
	if len(p) > 0 {
		n, err = w.w.Write(p)
		w.n -= n
	}
	if w.n == 0 {
		err = errors.New("past write limit")
	}
	return
}

// If there's an error copying the child's output to the parent, test
// that we kill the child.
func TestKillChildAfterCopyError(t *testing.T) {
	testenv.MustHaveExec(t)

	h := &Handler{
		Path: os.Args[0],
		Root: "/test.go",
	}
	req, _ := http.NewRequest("GET", "http://example.com/test.go?write-forever=1", nil)
	rec := httptest.NewRecorder()
	var out bytes.Buffer
	const writeLen = 50 << 10
	rw := &customWriterRecorder{&limitWriter{&out, writeLen}, rec}

	h.ServeHTTP(rw, req)
	if out.Len() != writeLen || out.Bytes()[0] != 'a' {
		t.Errorf("unexpected output: %q", out.Bytes())
	}
}

// Test that a child handler writing only headers works.
// golang.org/issue/7196
func TestChildOnlyHeaders(t *testing.T) {
	testenv.MustHaveExec(t)

	h := &Handler{
		Path: os.Args[0],
		Root: "/test.go",
	}
	expectedMap := map[string]string{
		"_body": "",
	}
	replay := runCgiTest(t, h, "GET /test.go?no-body=1 HTTP/1.0\nHost: example.com\n\n", expectedMap)
	if expected, got := "X-Test-Value", replay.Header().Get("X-Test-Header"); got != expected {
		t.Errorf("got a X-Test-Header of %q; expected %q", got, expected)
	}
}

// Test that a child handler does not receive a nil Request Body.
// golang.org/issue/39190
func TestNilRequestBody(t *testing.T) {
	testenv.MustHaveExec(t)

	h := &Handler{
		Path: os.Args[0],
		Root: "/test.go",
	}
	expectedMap := map[string]string{
		"nil-request-body": "false",
	}
	_ = runCgiTest(t, h, "POST /test.go?nil-request-body=1 HTTP/1.0\nHost: example.com\n\n", expectedMap)
	_ = runCgiTest(t, h, "POST /test.go?nil-request-body=1 HTTP/1.0\nHost: example.com\nContent-Length: 0\n\n", expectedMap)
}

func TestChildContentType(t *testing.T) {
	testenv.MustHaveExec(t)

	h := &Handler{
		Path: os.Args[0],
		Root: "/test.go",
	}
	var tests = []struct {
		name   string
		body   string
		wantCT string
	}{
		{
			name:   "no body",
			wantCT: "text/plain; charset=utf-8",
		},
		{
			name:   "html",
			body:   "<html><head><title>test page</title></head><body>This is a body</body></html>",
			wantCT: "text/html; charset=utf-8",
		},
		{
			name:   "text",
			body:   strings.Repeat("gopher", 86),
			wantCT: "text/plain; charset=utf-8",
		},
		{
			name:   "jpg",
			body:   "\xFF\xD8\xFF" + strings.Repeat("B", 1024),
			wantCT: "image/jpeg",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			expectedMap := map[string]string{"_body": tt.body}
			req := fmt.Sprintf("GET /test.go?exact-body=%s HTTP/1.0\nHost: example.com\n\n", url.QueryEscape(tt.body))
			replay := runCgiTest(t, h, req, expectedMap)
			if got := replay.Header().Get("Content-Type"); got != tt.wantCT {
				t.Errorf("got a Content-Type of %q; expected it to start with %q", got, tt.wantCT)
			}
		})
	}
}

// golang.org/issue/7198
func Test500WithNoHeaders(t *testing.T)     { want500Test(t, "/immediate-disconnect") }
func Test500WithNoContentType(t *testing.T) { want500Test(t, "/no-content-type") }
func Test500WithEmptyHeaders(t *testing.T)  { want500Test(t, "/empty-headers") }

func want500Test(t *testing.T, path string) {
	h := &Handler{
		Path: os.Args[0],
		Root: "/test.go",
	}
	expectedMap := map[string]string{
		"_body": "",
	}
	replay := runCgiTest(t, h, "GET "+path+" HTTP/1.0\nHost: example.com\n\n", expectedMap)
	if replay.Code != 500 {
		t.Errorf("Got code %d; want 500", replay.Code)
	}
}
