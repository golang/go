// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for package cgi

package cgi

import (
	"bufio"
	"exec"
	"fmt"
	"http"
	"http/httptest"
	"os"
	"strings"
	"testing"
)

var cgiScriptWorks = canRun("./testdata/test.cgi")

func canRun(s string) bool {
	c, err := exec.Run(s, []string{s}, nil, ".", exec.DevNull, exec.DevNull, exec.DevNull)
	if err != nil {
		return false
	}
	w, err := c.Wait(0)
	if err != nil {
		return false
	}
	return w.Exited() && w.ExitStatus() == 0
}

func newRequest(httpreq string) *http.Request {
	buf := bufio.NewReader(strings.NewReader(httpreq))
	req, err := http.ReadRequest(buf)
	if err != nil {
		panic("cgi: bogus http request in test: " + httpreq)
	}
	req.RemoteAddr = "1.2.3.4"
	return req
}

func runCgiTest(t *testing.T, h *Handler, httpreq string, expectedMap map[string]string) *httptest.ResponseRecorder {
	rw := httptest.NewRecorder()
	req := newRequest(httpreq)
	h.ServeHTTP(rw, req)

	// Make a map to hold the test map that the CGI returns.
	m := make(map[string]string)
	linesRead := 0
readlines:
	for {
		line, err := rw.Body.ReadString('\n')
		switch {
		case err == os.EOF:
			break readlines
		case err != nil:
			t.Fatalf("unexpected error reading from CGI: %v", err)
		}
		linesRead++
		trimmedLine := strings.TrimRight(line, "\r\n")
		split := strings.Split(trimmedLine, "=", 2)
		if len(split) != 2 {
			t.Fatalf("Unexpected %d parts from invalid line number %v: %q; existing map=%v",
				len(split), linesRead, line, m)
		}
		m[split[0]] = split[1]
	}

	for key, expected := range expectedMap {
		if got := m[key]; got != expected {
			t.Errorf("for key %q got %q; expected %q", key, got, expected)
		}
	}
	return rw
}

func skipTest(t *testing.T) bool {
	if !cgiScriptWorks {
		// No Perl on Windows, needed by test.cgi
		// TODO: make the child process be Go, not Perl.
		t.Logf("Skipping test: test.cgi failed.")
		return true
	}
	return false
}


func TestCGIBasicGet(t *testing.T) {
	if skipTest(t) {
		return
	}
	h := &Handler{
		Path: "testdata/test.cgi",
		Root: "/test.cgi",
	}
	expectedMap := map[string]string{
		"test":                  "Hello CGI",
		"param-a":               "b",
		"param-foo":             "bar",
		"env-GATEWAY_INTERFACE": "CGI/1.1",
		"env-HTTP_HOST":         "example.com",
		"env-PATH_INFO":         "",
		"env-QUERY_STRING":      "foo=bar&a=b",
		"env-REMOTE_ADDR":       "1.2.3.4",
		"env-REMOTE_HOST":       "1.2.3.4",
		"env-REQUEST_METHOD":    "GET",
		"env-REQUEST_URI":       "/test.cgi?foo=bar&a=b",
		"env-SCRIPT_FILENAME":   "testdata/test.cgi",
		"env-SCRIPT_NAME":       "/test.cgi",
		"env-SERVER_NAME":       "example.com",
		"env-SERVER_PORT":       "80",
		"env-SERVER_SOFTWARE":   "go",
	}
	replay := runCgiTest(t, h, "GET /test.cgi?foo=bar&a=b HTTP/1.0\nHost: example.com\n\n", expectedMap)

	if expected, got := "text/html", replay.Header().Get("Content-Type"); got != expected {
		t.Errorf("got a Content-Type of %q; expected %q", got, expected)
	}
	if expected, got := "X-Test-Value", replay.Header().Get("X-Test-Header"); got != expected {
		t.Errorf("got a X-Test-Header of %q; expected %q", got, expected)
	}
}

func TestCGIBasicGetAbsPath(t *testing.T) {
	if skipTest(t) {
		return
	}
	pwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd error: %v", err)
	}
	h := &Handler{
		Path: pwd + "/testdata/test.cgi",
		Root: "/test.cgi",
	}
	expectedMap := map[string]string{
		"env-REQUEST_URI":     "/test.cgi?foo=bar&a=b",
		"env-SCRIPT_FILENAME": pwd + "/testdata/test.cgi",
		"env-SCRIPT_NAME":     "/test.cgi",
	}
	runCgiTest(t, h, "GET /test.cgi?foo=bar&a=b HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

func TestPathInfo(t *testing.T) {
	if skipTest(t) {
		return
	}
	h := &Handler{
		Path: "testdata/test.cgi",
		Root: "/test.cgi",
	}
	expectedMap := map[string]string{
		"param-a":             "b",
		"env-PATH_INFO":       "/extrapath",
		"env-QUERY_STRING":    "a=b",
		"env-REQUEST_URI":     "/test.cgi/extrapath?a=b",
		"env-SCRIPT_FILENAME": "testdata/test.cgi",
		"env-SCRIPT_NAME":     "/test.cgi",
	}
	runCgiTest(t, h, "GET /test.cgi/extrapath?a=b HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

func TestPathInfoDirRoot(t *testing.T) {
	if skipTest(t) {
		return
	}
	h := &Handler{
		Path: "testdata/test.cgi",
		Root: "/myscript/",
	}
	expectedMap := map[string]string{
		"env-PATH_INFO":       "bar",
		"env-QUERY_STRING":    "a=b",
		"env-REQUEST_URI":     "/myscript/bar?a=b",
		"env-SCRIPT_FILENAME": "testdata/test.cgi",
		"env-SCRIPT_NAME":     "/myscript/",
	}
	runCgiTest(t, h, "GET /myscript/bar?a=b HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

func TestDupHeaders(t *testing.T) {
	if skipTest(t) {
		return
	}
	h := &Handler{
		Path: "testdata/test.cgi",
	}
	expectedMap := map[string]string{
		"env-REQUEST_URI":     "/myscript/bar?a=b",
		"env-SCRIPT_FILENAME": "testdata/test.cgi",
		"env-HTTP_COOKIE":     "nom=NOM; yum=YUM",
		"env-HTTP_X_FOO":      "val1, val2",
	}
	runCgiTest(t, h, "GET /myscript/bar?a=b HTTP/1.0\n"+
		"Cookie: nom=NOM\n"+
		"Cookie: yum=YUM\n"+
		"X-Foo: val1\n"+
		"X-Foo: val2\n"+
		"Host: example.com\n\n",
		expectedMap)
}

func TestPathInfoNoRoot(t *testing.T) {
	if skipTest(t) {
		return
	}
	h := &Handler{
		Path: "testdata/test.cgi",
		Root: "",
	}
	expectedMap := map[string]string{
		"env-PATH_INFO":       "/bar",
		"env-QUERY_STRING":    "a=b",
		"env-REQUEST_URI":     "/bar?a=b",
		"env-SCRIPT_FILENAME": "testdata/test.cgi",
		"env-SCRIPT_NAME":     "/",
	}
	runCgiTest(t, h, "GET /bar?a=b HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

func TestCGIBasicPost(t *testing.T) {
	if skipTest(t) {
		return
	}
	postReq := `POST /test.cgi?a=b HTTP/1.0
Host: example.com
Content-Type: application/x-www-form-urlencoded
Content-Length: 15

postfoo=postbar`
	h := &Handler{
		Path: "testdata/test.cgi",
		Root: "/test.cgi",
	}
	expectedMap := map[string]string{
		"test":               "Hello CGI",
		"param-postfoo":      "postbar",
		"env-REQUEST_METHOD": "POST",
		"env-CONTENT_LENGTH": "15",
		"env-REQUEST_URI":    "/test.cgi?a=b",
	}
	runCgiTest(t, h, postReq, expectedMap)
}

func chunk(s string) string {
	return fmt.Sprintf("%x\r\n%s\r\n", len(s), s)
}

// The CGI spec doesn't allow chunked requests.
func TestCGIPostChunked(t *testing.T) {
	if skipTest(t) {
		return
	}
	postReq := `POST /test.cgi?a=b HTTP/1.1
Host: example.com
Content-Type: application/x-www-form-urlencoded
Transfer-Encoding: chunked

` + chunk("postfoo") + chunk("=") + chunk("postbar") + chunk("")

	h := &Handler{
		Path: "testdata/test.cgi",
		Root: "/test.cgi",
	}
	expectedMap := map[string]string{}
	resp := runCgiTest(t, h, postReq, expectedMap)
	if got, expected := resp.Code, http.StatusBadRequest; got != expected {
		t.Fatalf("Expected %v response code from chunked request body; got %d",
			expected, got)
	}
}

func TestRedirect(t *testing.T) {
	if skipTest(t) {
		return
	}
	h := &Handler{
		Path: "testdata/test.cgi",
		Root: "/test.cgi",
	}
	rec := runCgiTest(t, h, "GET /test.cgi?loc=http://foo.com/ HTTP/1.0\nHost: example.com\n\n", nil)
	if e, g := 302, rec.Code; e != g {
		t.Errorf("expected status code %d; got %d", e, g)
	}
	if e, g := "http://foo.com/", rec.Header().Get("Location"); e != g {
		t.Errorf("expected Location header of %q; got %q", e, g)
	}
}

func TestInternalRedirect(t *testing.T) {
	if skipTest(t) {
		return
	}
	baseHandler := http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		fmt.Fprintf(rw, "basepath=%s\n", req.URL.Path)
		fmt.Fprintf(rw, "remoteaddr=%s\n", req.RemoteAddr)
	})
	h := &Handler{
		Path:                "testdata/test.cgi",
		Root:                "/test.cgi",
		PathLocationHandler: baseHandler,
	}
	expectedMap := map[string]string{
		"basepath":   "/foo",
		"remoteaddr": "1.2.3.4",
	}
	runCgiTest(t, h, "GET /test.cgi?loc=/foo HTTP/1.0\nHost: example.com\n\n", expectedMap)
}
