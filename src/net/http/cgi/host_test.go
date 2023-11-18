// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for package cgi

package cgi

import (
	"bufio"
	"fmt"
	"internal/testenv"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestMain executes the test binary as the cgi server if
// SERVER_SOFTWARE is set, and runs the tests otherwise.
func TestMain(m *testing.M) {
	// SERVER_SOFTWARE swap variable is set when starting the cgi server.
	if os.Getenv("SERVER_SOFTWARE") != "" {
		cgiMain()
		os.Exit(0)
	}

	os.Exit(m.Run())
}

func newRequest(httpreq string) *http.Request {
	buf := bufio.NewReader(strings.NewReader(httpreq))
	req, err := http.ReadRequest(buf)
	if err != nil {
		panic("cgi: bogus http request in test: " + httpreq)
	}
	req.RemoteAddr = "1.2.3.4:1234"
	return req
}

func runCgiTest(t *testing.T, h *Handler,
	httpreq string,
	expectedMap map[string]string, checks ...func(reqInfo map[string]string)) *httptest.ResponseRecorder {
	rw := httptest.NewRecorder()
	req := newRequest(httpreq)
	h.ServeHTTP(rw, req)
	runResponseChecks(t, rw, expectedMap, checks...)
	return rw
}

func runResponseChecks(t *testing.T, rw *httptest.ResponseRecorder,
	expectedMap map[string]string, checks ...func(reqInfo map[string]string)) {
	// Make a map to hold the test map that the CGI returns.
	m := make(map[string]string)
	m["_body"] = rw.Body.String()
	linesRead := 0
readlines:
	for {
		line, err := rw.Body.ReadString('\n')
		switch {
		case err == io.EOF:
			break readlines
		case err != nil:
			t.Fatalf("unexpected error reading from CGI: %v", err)
		}
		linesRead++
		trimmedLine := strings.TrimRight(line, "\r\n")
		k, v, ok := strings.Cut(trimmedLine, "=")
		if !ok {
			t.Fatalf("Unexpected response from invalid line number %v: %q; existing map=%v",
				linesRead, line, m)
		}
		m[k] = v
	}

	for key, expected := range expectedMap {
		got := m[key]
		if key == "cwd" {
			// For Windows. golang.org/issue/4645.
			fi1, _ := os.Stat(got)
			fi2, _ := os.Stat(expected)
			if os.SameFile(fi1, fi2) {
				got = expected
			}
		}
		if got != expected {
			t.Errorf("for key %q got %q; expected %q", key, got, expected)
		}
	}
	for _, check := range checks {
		check(m)
	}
}

func TestCGIBasicGet(t *testing.T) {
	testenv.MustHaveExec(t)
	h := &Handler{
		Path: os.Args[0],
		Root: "/test.cgi",
	}
	expectedMap := map[string]string{
		"test":                  "Hello CGI",
		"param-a":               "b",
		"param-foo":             "bar",
		"env-GATEWAY_INTERFACE": "CGI/1.1",
		"env-HTTP_HOST":         "example.com:80",
		"env-PATH_INFO":         "",
		"env-QUERY_STRING":      "foo=bar&a=b",
		"env-REMOTE_ADDR":       "1.2.3.4",
		"env-REMOTE_HOST":       "1.2.3.4",
		"env-REMOTE_PORT":       "1234",
		"env-REQUEST_METHOD":    "GET",
		"env-REQUEST_URI":       "/test.cgi?foo=bar&a=b",
		"env-SCRIPT_FILENAME":   os.Args[0],
		"env-SCRIPT_NAME":       "/test.cgi",
		"env-SERVER_NAME":       "example.com",
		"env-SERVER_PORT":       "80",
		"env-SERVER_SOFTWARE":   "go",
	}
	replay := runCgiTest(t, h, "GET /test.cgi?foo=bar&a=b HTTP/1.0\nHost: example.com:80\n\n", expectedMap)

	if expected, got := "text/html", replay.Header().Get("Content-Type"); got != expected {
		t.Errorf("got a Content-Type of %q; expected %q", got, expected)
	}
	if expected, got := "X-Test-Value", replay.Header().Get("X-Test-Header"); got != expected {
		t.Errorf("got a X-Test-Header of %q; expected %q", got, expected)
	}
}

func TestCGIEnvIPv6(t *testing.T) {
	testenv.MustHaveExec(t)
	h := &Handler{
		Path: os.Args[0],
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
		"env-REMOTE_ADDR":       "2000::3000",
		"env-REMOTE_HOST":       "2000::3000",
		"env-REMOTE_PORT":       "12345",
		"env-REQUEST_METHOD":    "GET",
		"env-REQUEST_URI":       "/test.cgi?foo=bar&a=b",
		"env-SCRIPT_FILENAME":   os.Args[0],
		"env-SCRIPT_NAME":       "/test.cgi",
		"env-SERVER_NAME":       "example.com",
		"env-SERVER_PORT":       "80",
		"env-SERVER_SOFTWARE":   "go",
	}

	rw := httptest.NewRecorder()
	req := newRequest("GET /test.cgi?foo=bar&a=b HTTP/1.0\nHost: example.com\n\n")
	req.RemoteAddr = "[2000::3000]:12345"
	h.ServeHTTP(rw, req)
	runResponseChecks(t, rw, expectedMap)
}

func TestCGIBasicGetAbsPath(t *testing.T) {
	absPath, err := filepath.Abs(os.Args[0])
	if err != nil {
		t.Fatal(err)
	}
	testenv.MustHaveExec(t)
	h := &Handler{
		Path: absPath,
		Root: "/test.cgi",
	}
	expectedMap := map[string]string{
		"env-REQUEST_URI":     "/test.cgi?foo=bar&a=b",
		"env-SCRIPT_FILENAME": absPath,
		"env-SCRIPT_NAME":     "/test.cgi",
	}
	runCgiTest(t, h, "GET /test.cgi?foo=bar&a=b HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

func TestPathInfo(t *testing.T) {
	testenv.MustHaveExec(t)
	h := &Handler{
		Path: os.Args[0],
		Root: "/test.cgi",
	}
	expectedMap := map[string]string{
		"param-a":             "b",
		"env-PATH_INFO":       "/extrapath",
		"env-QUERY_STRING":    "a=b",
		"env-REQUEST_URI":     "/test.cgi/extrapath?a=b",
		"env-SCRIPT_FILENAME": os.Args[0],
		"env-SCRIPT_NAME":     "/test.cgi",
	}
	runCgiTest(t, h, "GET /test.cgi/extrapath?a=b HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

func TestPathInfoDirRoot(t *testing.T) {
	testenv.MustHaveExec(t)
	h := &Handler{
		Path: os.Args[0],
		Root: "/myscript//",
	}
	expectedMap := map[string]string{
		"env-PATH_INFO":       "/bar",
		"env-QUERY_STRING":    "a=b",
		"env-REQUEST_URI":     "/myscript/bar?a=b",
		"env-SCRIPT_FILENAME": os.Args[0],
		"env-SCRIPT_NAME":     "/myscript",
	}
	runCgiTest(t, h, "GET /myscript/bar?a=b HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

func TestDupHeaders(t *testing.T) {
	testenv.MustHaveExec(t)
	h := &Handler{
		Path: os.Args[0],
	}
	expectedMap := map[string]string{
		"env-REQUEST_URI":     "/myscript/bar?a=b",
		"env-SCRIPT_FILENAME": os.Args[0],
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

// Issue 16405: CGI+http.Transport differing uses of HTTP_PROXY.
// Verify we don't set the HTTP_PROXY environment variable.
// Hope nobody was depending on it. It's not a known header, though.
func TestDropProxyHeader(t *testing.T) {
	testenv.MustHaveExec(t)
	h := &Handler{
		Path: os.Args[0],
	}
	expectedMap := map[string]string{
		"env-REQUEST_URI":     "/myscript/bar?a=b",
		"env-SCRIPT_FILENAME": os.Args[0],
		"env-HTTP_X_FOO":      "a",
	}
	runCgiTest(t, h, "GET /myscript/bar?a=b HTTP/1.0\n"+
		"X-Foo: a\n"+
		"Proxy: should_be_stripped\n"+
		"Host: example.com\n\n",
		expectedMap,
		func(reqInfo map[string]string) {
			if v, ok := reqInfo["env-HTTP_PROXY"]; ok {
				t.Errorf("HTTP_PROXY = %q; should be absent", v)
			}
		})
}

func TestPathInfoNoRoot(t *testing.T) {
	testenv.MustHaveExec(t)
	h := &Handler{
		Path: os.Args[0],
		Root: "",
	}
	expectedMap := map[string]string{
		"env-PATH_INFO":       "/bar",
		"env-QUERY_STRING":    "a=b",
		"env-REQUEST_URI":     "/bar?a=b",
		"env-SCRIPT_FILENAME": os.Args[0],
		"env-SCRIPT_NAME":     "",
	}
	runCgiTest(t, h, "GET /bar?a=b HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

func TestCGIBasicPost(t *testing.T) {
	testenv.MustHaveExec(t)
	postReq := `POST /test.cgi?a=b HTTP/1.0
Host: example.com
Content-Type: application/x-www-form-urlencoded
Content-Length: 15

postfoo=postbar`
	h := &Handler{
		Path: os.Args[0],
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
	testenv.MustHaveExec(t)
	postReq := `POST /test.cgi?a=b HTTP/1.1
Host: example.com
Content-Type: application/x-www-form-urlencoded
Transfer-Encoding: chunked

` + chunk("postfoo") + chunk("=") + chunk("postbar") + chunk("")

	h := &Handler{
		Path: os.Args[0],
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
	testenv.MustHaveExec(t)
	h := &Handler{
		Path: os.Args[0],
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
	testenv.MustHaveExec(t)
	baseHandler := http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		fmt.Fprintf(rw, "basepath=%s\n", req.URL.Path)
		fmt.Fprintf(rw, "remoteaddr=%s\n", req.RemoteAddr)
	})
	h := &Handler{
		Path:                os.Args[0],
		Root:                "/test.cgi",
		PathLocationHandler: baseHandler,
	}
	expectedMap := map[string]string{
		"basepath":   "/foo",
		"remoteaddr": "1.2.3.4:1234",
	}
	runCgiTest(t, h, "GET /test.cgi?loc=/foo HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

// TestCopyError tests that we kill the process if there's an error copying
// its output. (for example, from the client having gone away)
func TestCopyError(t *testing.T) {
	testenv.MustHaveExec(t)
	if runtime.GOOS == "windows" {
		t.Skipf("skipping test on %q", runtime.GOOS)
	}
	h := &Handler{
		Path: os.Args[0],
		Root: "/test.cgi",
	}
	ts := httptest.NewServer(h)
	defer ts.Close()

	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequest("GET", "http://example.com/test.cgi?bigresponse=1", nil)
	err = req.Write(conn)
	if err != nil {
		t.Fatalf("Write: %v", err)
	}

	res, err := http.ReadResponse(bufio.NewReader(conn), req)
	if err != nil {
		t.Fatalf("ReadResponse: %v", err)
	}

	pidstr := res.Header.Get("X-CGI-Pid")
	if pidstr == "" {
		t.Fatalf("expected an X-CGI-Pid header in response")
	}
	pid, err := strconv.Atoi(pidstr)
	if err != nil {
		t.Fatalf("invalid X-CGI-Pid value")
	}

	var buf [5000]byte
	n, err := io.ReadFull(res.Body, buf[:])
	if err != nil {
		t.Fatalf("ReadFull: %d bytes, %v", n, err)
	}

	childRunning := func() bool {
		return isProcessRunning(pid)
	}

	if !childRunning() {
		t.Fatalf("pre-conn.Close, expected child to be running")
	}
	conn.Close()

	tries := 0
	for tries < 25 && childRunning() {
		time.Sleep(50 * time.Millisecond * time.Duration(tries))
		tries++
	}
	if childRunning() {
		t.Fatalf("post-conn.Close, expected child to be gone")
	}
}

func TestDir(t *testing.T) {
	testenv.MustHaveExec(t)
	cwd, _ := os.Getwd()
	h := &Handler{
		Path: os.Args[0],
		Root: "/test.cgi",
		Dir:  cwd,
	}
	expectedMap := map[string]string{
		"cwd": cwd,
	}
	runCgiTest(t, h, "GET /test.cgi HTTP/1.0\nHost: example.com\n\n", expectedMap)

	cwd, _ = os.Getwd()
	cwd, _ = filepath.Split(os.Args[0])
	h = &Handler{
		Path: os.Args[0],
		Root: "/test.cgi",
	}
	expectedMap = map[string]string{
		"cwd": cwd,
	}
	runCgiTest(t, h, "GET /test.cgi HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

func TestEnvOverride(t *testing.T) {
	testenv.MustHaveExec(t)
	cgifile, _ := filepath.Abs("testdata/test.cgi")

	cwd, _ := os.Getwd()
	h := &Handler{
		Path: os.Args[0],
		Root: "/test.cgi",
		Dir:  cwd,
		Env: []string{
			"SCRIPT_FILENAME=" + cgifile,
			"REQUEST_URI=/foo/bar",
			"PATH=/wibble"},
	}
	expectedMap := map[string]string{
		"cwd":                 cwd,
		"env-SCRIPT_FILENAME": cgifile,
		"env-REQUEST_URI":     "/foo/bar",
		"env-PATH":            "/wibble",
	}
	runCgiTest(t, h, "GET /test.cgi HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

func TestHandlerStderr(t *testing.T) {
	testenv.MustHaveExec(t)
	var stderr strings.Builder
	h := &Handler{
		Path:   os.Args[0],
		Root:   "/test.cgi",
		Stderr: &stderr,
	}

	rw := httptest.NewRecorder()
	req := newRequest("GET /test.cgi?writestderr=1 HTTP/1.0\nHost: example.com\n\n")
	h.ServeHTTP(rw, req)
	if got, want := stderr.String(), "Hello, stderr!\n"; got != want {
		t.Errorf("Stderr = %q; want %q", got, want)
	}
}

func TestRemoveLeadingDuplicates(t *testing.T) {
	tests := []struct {
		env  []string
		want []string
	}{
		{
			env:  []string{"a=b", "b=c", "a=b2"},
			want: []string{"b=c", "a=b2"},
		},
		{
			env:  []string{"a=b", "b=c", "d", "e=f"},
			want: []string{"a=b", "b=c", "d", "e=f"},
		},
	}
	for _, tt := range tests {
		got := removeLeadingDuplicates(tt.env)
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("removeLeadingDuplicates(%q) = %q; want %q", tt.env, got, tt.want)
		}
	}
}
