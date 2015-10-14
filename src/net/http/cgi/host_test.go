// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for package cgi

package cgi

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

func newRequest(httpreq string) *http.Request {
	buf := bufio.NewReader(strings.NewReader(httpreq))
	req, err := http.ReadRequest(buf)
	if err != nil {
		panic("cgi: bogus http request in test: " + httpreq)
	}
	req.RemoteAddr = "1.2.3.4:1234"
	return req
}

func runCgiTest(t *testing.T, h *Handler, httpreq string, expectedMap map[string]string) *httptest.ResponseRecorder {
	rw := httptest.NewRecorder()
	req := newRequest(httpreq)
	h.ServeHTTP(rw, req)
	runResponseChecks(t, rw, expectedMap)
	return rw
}

func runResponseChecks(t *testing.T, rw *httptest.ResponseRecorder, expectedMap map[string]string) {
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
		split := strings.SplitN(trimmedLine, "=", 2)
		if len(split) != 2 {
			t.Fatalf("Unexpected %d parts from invalid line number %v: %q; existing map=%v",
				len(split), linesRead, line, m)
		}
		m[split[0]] = split[1]
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
}

var cgiTested, cgiWorks bool

func check(t *testing.T) {
	if !cgiTested {
		cgiTested = true
		cgiWorks = exec.Command("./testdata/test.cgi").Run() == nil
	}
	if !cgiWorks {
		// No Perl on Windows, needed by test.cgi
		// TODO: make the child process be Go, not Perl.
		t.Skip("Skipping test: test.cgi failed.")
	}
}

func TestCGIBasicGet(t *testing.T) {
	check(t)
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
		"env-REMOTE_PORT":       "1234",
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

func TestCGIEnvIPv6(t *testing.T) {
	check(t)
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
		"env-REMOTE_ADDR":       "2000::3000",
		"env-REMOTE_HOST":       "2000::3000",
		"env-REMOTE_PORT":       "12345",
		"env-REQUEST_METHOD":    "GET",
		"env-REQUEST_URI":       "/test.cgi?foo=bar&a=b",
		"env-SCRIPT_FILENAME":   "testdata/test.cgi",
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
	check(t)
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
	check(t)
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
	check(t)
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
	check(t)
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
	check(t)
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
	check(t)
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
	check(t)
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
	check(t)
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
	check(t)
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
		"remoteaddr": "1.2.3.4:1234",
	}
	runCgiTest(t, h, "GET /test.cgi?loc=/foo HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

// TestCopyError tests that we kill the process if there's an error copying
// its output. (for example, from the client having gone away)
func TestCopyError(t *testing.T) {
	check(t)
	if runtime.GOOS == "windows" {
		t.Skipf("skipping test on %q", runtime.GOOS)
	}
	h := &Handler{
		Path: "testdata/test.cgi",
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
		return isProcessRunning(t, pid)
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

func TestDirUnix(t *testing.T) {
	check(t)
	if runtime.GOOS == "windows" {
		t.Skipf("skipping test on %q", runtime.GOOS)
	}
	cwd, _ := os.Getwd()
	h := &Handler{
		Path: "testdata/test.cgi",
		Root: "/test.cgi",
		Dir:  cwd,
	}
	expectedMap := map[string]string{
		"cwd": cwd,
	}
	runCgiTest(t, h, "GET /test.cgi HTTP/1.0\nHost: example.com\n\n", expectedMap)

	cwd, _ = os.Getwd()
	cwd = filepath.Join(cwd, "testdata")
	h = &Handler{
		Path: "testdata/test.cgi",
		Root: "/test.cgi",
	}
	expectedMap = map[string]string{
		"cwd": cwd,
	}
	runCgiTest(t, h, "GET /test.cgi HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

func TestDirWindows(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Skipping windows specific test.")
	}

	cgifile, _ := filepath.Abs("testdata/test.cgi")

	var perl string
	var err error
	perl, err = exec.LookPath("perl")
	if err != nil {
		t.Skip("Skipping test: perl not found.")
	}
	perl, _ = filepath.Abs(perl)

	cwd, _ := os.Getwd()
	h := &Handler{
		Path: perl,
		Root: "/test.cgi",
		Dir:  cwd,
		Args: []string{cgifile},
		Env:  []string{"SCRIPT_FILENAME=" + cgifile},
	}
	expectedMap := map[string]string{
		"cwd": cwd,
	}
	runCgiTest(t, h, "GET /test.cgi HTTP/1.0\nHost: example.com\n\n", expectedMap)

	// If not specify Dir on windows, working directory should be
	// base directory of perl.
	cwd, _ = filepath.Split(perl)
	if cwd != "" && cwd[len(cwd)-1] == filepath.Separator {
		cwd = cwd[:len(cwd)-1]
	}
	h = &Handler{
		Path: perl,
		Root: "/test.cgi",
		Args: []string{cgifile},
		Env:  []string{"SCRIPT_FILENAME=" + cgifile},
	}
	expectedMap = map[string]string{
		"cwd": cwd,
	}
	runCgiTest(t, h, "GET /test.cgi HTTP/1.0\nHost: example.com\n\n", expectedMap)
}

func TestEnvOverride(t *testing.T) {
	cgifile, _ := filepath.Abs("testdata/test.cgi")

	var perl string
	var err error
	perl, err = exec.LookPath("perl")
	if err != nil {
		t.Skipf("Skipping test: perl not found.")
	}
	perl, _ = filepath.Abs(perl)

	cwd, _ := os.Getwd()
	h := &Handler{
		Path: perl,
		Root: "/test.cgi",
		Dir:  cwd,
		Args: []string{cgifile},
		Env: []string{
			"SCRIPT_FILENAME=" + cgifile,
			"REQUEST_URI=/foo/bar",
			"PATH=/wibble"},
	}
	expectedMap := map[string]string{
		"cwd": cwd,
		"env-SCRIPT_FILENAME": cgifile,
		"env-REQUEST_URI":     "/foo/bar",
		"env-PATH":            "/wibble",
	}
	runCgiTest(t, h, "GET /test.cgi HTTP/1.0\nHost: example.com\n\n", expectedMap)
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
