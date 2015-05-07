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
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"testing"
	"time"
)

// iOS cannot fork, so we skip some tests
var iOS = runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64")

// This test is a CGI host (testing host.go) that runs its own binary
// as a child process testing the other half of CGI (child.go).
func TestHostingOurselves(t *testing.T) {
	if runtime.GOOS == "nacl" || iOS {
		t.Skipf("skipping on %s/%s", runtime.GOOS, runtime.GOARCH)
	}

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

	if expected, got := "text/html; charset=utf-8", replay.Header().Get("Content-Type"); got != expected {
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
	if runtime.GOOS == "nacl" || iOS {
		t.Skipf("skipping on %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	defer func() { testHookStartProcess = nil }()
	proc := make(chan *os.Process, 1)
	testHookStartProcess = func(p *os.Process) {
		proc <- p
	}

	h := &Handler{
		Path: os.Args[0],
		Root: "/test.go",
		Args: []string{"-test.run=TestBeChildCGIProcess"},
	}
	req, _ := http.NewRequest("GET", "http://example.com/test.cgi?write-forever=1", nil)
	rec := httptest.NewRecorder()
	var out bytes.Buffer
	const writeLen = 50 << 10
	rw := &customWriterRecorder{&limitWriter{&out, writeLen}, rec}

	donec := make(chan bool, 1)
	go func() {
		h.ServeHTTP(rw, req)
		donec <- true
	}()

	select {
	case <-donec:
		if out.Len() != writeLen || out.Bytes()[0] != 'a' {
			t.Errorf("unexpected output: %q", out.Bytes())
		}
	case <-time.After(5 * time.Second):
		t.Errorf("timeout. ServeHTTP hung and didn't kill the child process?")
		select {
		case p := <-proc:
			p.Kill()
			t.Logf("killed process")
		default:
			t.Logf("didn't kill process")
		}
	}
}

// Test that a child handler writing only headers works.
// golang.org/issue/7196
func TestChildOnlyHeaders(t *testing.T) {
	if runtime.GOOS == "nacl" || iOS {
		t.Skipf("skipping on %s/%s", runtime.GOOS, runtime.GOARCH)
	}

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

// golang.org/issue/7198
func Test500WithNoHeaders(t *testing.T)     { want500Test(t, "/immediate-disconnect") }
func Test500WithNoContentType(t *testing.T) { want500Test(t, "/no-content-type") }
func Test500WithEmptyHeaders(t *testing.T)  { want500Test(t, "/empty-headers") }

func want500Test(t *testing.T, path string) {
	h := &Handler{
		Path: os.Args[0],
		Root: "/test.go",
		Args: []string{"-test.run=TestBeChildCGIProcess"},
	}
	expectedMap := map[string]string{
		"_body": "",
	}
	replay := runCgiTest(t, h, "GET "+path+" HTTP/1.0\nHost: example.com\n\n", expectedMap)
	if replay.Code != 500 {
		t.Errorf("Got code %d; want 500", replay.Code)
	}
}

type neverEnding byte

func (b neverEnding) Read(p []byte) (n int, err error) {
	for i := range p {
		p[i] = byte(b)
	}
	return len(p), nil
}

// Note: not actually a test.
func TestBeChildCGIProcess(t *testing.T) {
	if os.Getenv("REQUEST_METHOD") == "" {
		// Not in a CGI environment; skipping test.
		return
	}
	switch os.Getenv("REQUEST_URI") {
	case "/immediate-disconnect":
		os.Exit(0)
	case "/no-content-type":
		fmt.Printf("Content-Length: 6\n\nHello\n")
		os.Exit(0)
	case "/empty-headers":
		fmt.Printf("\nHello")
		os.Exit(0)
	}
	Serve(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		rw.Header().Set("X-Test-Header", "X-Test-Value")
		req.ParseForm()
		if req.FormValue("no-body") == "1" {
			return
		}
		if req.FormValue("write-forever") == "1" {
			io.Copy(rw, neverEnding('a'))
			for {
				time.Sleep(5 * time.Second) // hang forever, until killed
			}
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
