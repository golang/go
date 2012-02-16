// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	. "net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"
)

const (
	testFile       = "testdata/file"
	testFileLength = 11
)

var ServeFileRangeTests = []struct {
	start, end int
	r          string
	code       int
}{
	{0, testFileLength, "", StatusOK},
	{0, 5, "0-4", StatusPartialContent},
	{2, testFileLength, "2-", StatusPartialContent},
	{testFileLength - 5, testFileLength, "-5", StatusPartialContent},
	{3, 8, "3-7", StatusPartialContent},
	{0, 0, "20-", StatusRequestedRangeNotSatisfiable},
}

func TestServeFile(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		ServeFile(w, r, "testdata/file")
	}))
	defer ts.Close()

	var err error

	file, err := ioutil.ReadFile(testFile)
	if err != nil {
		t.Fatal("reading file:", err)
	}

	// set up the Request (re-used for all tests)
	var req Request
	req.Header = make(Header)
	if req.URL, err = url.Parse(ts.URL); err != nil {
		t.Fatal("ParseURL:", err)
	}
	req.Method = "GET"

	// straight GET
	_, body := getBody(t, "straight get", req)
	if !equal(body, file) {
		t.Fatalf("body mismatch: got %q, want %q", body, file)
	}

	// Range tests
	for i, rt := range ServeFileRangeTests {
		req.Header.Set("Range", "bytes="+rt.r)
		if rt.r == "" {
			req.Header["Range"] = nil
		}
		r, body := getBody(t, fmt.Sprintf("test %d", i), req)
		if r.StatusCode != rt.code {
			t.Errorf("range=%q: StatusCode=%d, want %d", rt.r, r.StatusCode, rt.code)
		}
		if rt.code == StatusRequestedRangeNotSatisfiable {
			continue
		}
		h := fmt.Sprintf("bytes %d-%d/%d", rt.start, rt.end-1, testFileLength)
		if rt.r == "" {
			h = ""
		}
		cr := r.Header.Get("Content-Range")
		if cr != h {
			t.Errorf("header mismatch: range=%q: got %q, want %q", rt.r, cr, h)
		}
		if !equal(body, file[rt.start:rt.end]) {
			t.Errorf("body mismatch: range=%q: got %q, want %q", rt.r, body, file[rt.start:rt.end])
		}
	}
}

var fsRedirectTestData = []struct {
	original, redirect string
}{
	{"/test/index.html", "/test/"},
	{"/test/testdata", "/test/testdata/"},
	{"/test/testdata/file/", "/test/testdata/file"},
}

func TestFSRedirect(t *testing.T) {
	ts := httptest.NewServer(StripPrefix("/test", FileServer(Dir("."))))
	defer ts.Close()

	for _, data := range fsRedirectTestData {
		res, err := Get(ts.URL + data.original)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		if g, e := res.Request.URL.Path, data.redirect; g != e {
			t.Errorf("redirect from %s: got %s, want %s", data.original, g, e)
		}
	}
}

type testFileSystem struct {
	open func(name string) (File, error)
}

func (fs *testFileSystem) Open(name string) (File, error) {
	return fs.open(name)
}

func TestFileServerCleans(t *testing.T) {
	ch := make(chan string, 1)
	fs := FileServer(&testFileSystem{func(name string) (File, error) {
		ch <- name
		return nil, errors.New("file does not exist")
	}})
	tests := []struct {
		reqPath, openArg string
	}{
		{"/foo.txt", "/foo.txt"},
		{"//foo.txt", "/foo.txt"},
		{"/../foo.txt", "/foo.txt"},
	}
	req, _ := NewRequest("GET", "http://example.com", nil)
	for n, test := range tests {
		rec := httptest.NewRecorder()
		req.URL.Path = test.reqPath
		fs.ServeHTTP(rec, req)
		if got := <-ch; got != test.openArg {
			t.Errorf("test %d: got %q, want %q", n, got, test.openArg)
		}
	}
}

func TestFileServerImplicitLeadingSlash(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("TempDir: %v", err)
	}
	defer os.RemoveAll(tempDir)
	if err := ioutil.WriteFile(filepath.Join(tempDir, "foo.txt"), []byte("Hello world"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	ts := httptest.NewServer(StripPrefix("/bar/", FileServer(Dir(tempDir))))
	defer ts.Close()
	get := func(suffix string) string {
		res, err := Get(ts.URL + suffix)
		if err != nil {
			t.Fatalf("Get %s: %v", suffix, err)
		}
		b, err := ioutil.ReadAll(res.Body)
		if err != nil {
			t.Fatalf("ReadAll %s: %v", suffix, err)
		}
		return string(b)
	}
	if s := get("/bar/"); !strings.Contains(s, ">foo.txt<") {
		t.Logf("expected a directory listing with foo.txt, got %q", s)
	}
	if s := get("/bar/foo.txt"); s != "Hello world" {
		t.Logf("expected %q, got %q", "Hello world", s)
	}
}

func TestDirJoin(t *testing.T) {
	wfi, err := os.Stat("/etc/hosts")
	if err != nil {
		t.Logf("skipping test; no /etc/hosts file")
		return
	}
	test := func(d Dir, name string) {
		f, err := d.Open(name)
		if err != nil {
			t.Fatalf("open of %s: %v", name, err)
		}
		defer f.Close()
		gfi, err := f.Stat()
		if err != nil {
			t.Fatalf("stat of %s: %v", name, err)
		}
		if !os.SameFile(gfi, wfi) {
			t.Errorf("%s got different file", name)
		}
	}
	test(Dir("/etc/"), "/hosts")
	test(Dir("/etc/"), "hosts")
	test(Dir("/etc/"), "../../../../hosts")
	test(Dir("/etc"), "/hosts")
	test(Dir("/etc"), "hosts")
	test(Dir("/etc"), "../../../../hosts")

	// Not really directories, but since we use this trick in
	// ServeFile, test it:
	test(Dir("/etc/hosts"), "")
	test(Dir("/etc/hosts"), "/")
	test(Dir("/etc/hosts"), "../")
}

func TestEmptyDirOpenCWD(t *testing.T) {
	test := func(d Dir) {
		name := "fs_test.go"
		f, err := d.Open(name)
		if err != nil {
			t.Fatalf("open of %s: %v", name, err)
		}
		defer f.Close()
	}
	test(Dir(""))
	test(Dir("."))
	test(Dir("./"))
}

func TestServeFileContentType(t *testing.T) {
	const ctype = "icecream/chocolate"
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.FormValue("override") == "1" {
			w.Header().Set("Content-Type", ctype)
		}
		ServeFile(w, r, "testdata/file")
	}))
	defer ts.Close()
	get := func(override, want string) {
		resp, err := Get(ts.URL + "?override=" + override)
		if err != nil {
			t.Fatal(err)
		}
		if h := resp.Header.Get("Content-Type"); h != want {
			t.Errorf("Content-Type mismatch: got %q, want %q", h, want)
		}
	}
	get("0", "text/plain; charset=utf-8")
	get("1", ctype)
}

func TestServeFileMimeType(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		ServeFile(w, r, "testdata/style.css")
	}))
	defer ts.Close()
	resp, err := Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	want := "text/css; charset=utf-8"
	if h := resp.Header.Get("Content-Type"); h != want {
		t.Errorf("Content-Type mismatch: got %q, want %q", h, want)
	}
}

func TestServeFileFromCWD(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		ServeFile(w, r, "fs_test.go")
	}))
	defer ts.Close()
	r, err := Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if r.StatusCode != 200 {
		t.Fatalf("expected 200 OK, got %s", r.Status)
	}
}

func TestServeFileWithContentEncoding(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Encoding", "foo")
		ServeFile(w, r, "testdata/file")
	}))
	defer ts.Close()
	resp, err := Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if g, e := resp.ContentLength, int64(-1); g != e {
		t.Errorf("Content-Length mismatch: got %d, want %d", g, e)
	}
}

func TestServeIndexHtml(t *testing.T) {
	const want = "index.html says hello\n"
	ts := httptest.NewServer(FileServer(Dir(".")))
	defer ts.Close()

	for _, path := range []string{"/testdata/", "/testdata/index.html"} {
		res, err := Get(ts.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		b, err := ioutil.ReadAll(res.Body)
		if err != nil {
			t.Fatal("reading Body:", err)
		}
		if s := string(b); s != want {
			t.Errorf("for path %q got %q, want %q", path, s, want)
		}
		res.Body.Close()
	}
}

func TestServeContent(t *testing.T) {
	type req struct {
		name    string
		modtime time.Time
		content io.ReadSeeker
	}
	ch := make(chan req, 1)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		p := <-ch
		ServeContent(w, r, p.name, p.modtime, p.content)
	}))
	defer ts.Close()

	css, err := os.Open("testdata/style.css")
	if err != nil {
		t.Fatal(err)
	}
	defer css.Close()

	ch <- req{"style.css", time.Time{}, css}
	res, err := Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if g, e := res.Header.Get("Content-Type"), "text/css; charset=utf-8"; g != e {
		t.Errorf("style.css: content type = %q, want %q", g, e)
	}
	if g := res.Header.Get("Last-Modified"); g != "" {
		t.Errorf("want empty Last-Modified; got %q", g)
	}

	fi, err := css.Stat()
	if err != nil {
		t.Fatal(err)
	}
	ch <- req{"style.html", fi.ModTime(), css}
	res, err = Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if g, e := res.Header.Get("Content-Type"), "text/html; charset=utf-8"; g != e {
		t.Errorf("style.html: content type = %q, want %q", g, e)
	}
	if g := res.Header.Get("Last-Modified"); g == "" {
		t.Errorf("want non-empty last-modified")
	}
}

// verifies that sendfile is being used on Linux
func TestLinuxSendfile(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Logf("skipping; linux-only test")
		return
	}
	_, err := exec.LookPath("strace")
	if err != nil {
		t.Logf("skipping; strace not found in path")
		return
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	lnf, err := ln.(*net.TCPListener).File()
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	var buf bytes.Buffer
	child := exec.Command("strace", "-f", os.Args[0], "-test.run=TestLinuxSendfileChild")
	child.ExtraFiles = append(child.ExtraFiles, lnf)
	child.Env = append([]string{"GO_WANT_HELPER_PROCESS=1"}, os.Environ()...)
	child.Stdout = &buf
	child.Stderr = &buf
	err = child.Start()
	if err != nil {
		t.Logf("skipping; failed to start straced child: %v", err)
		return
	}

	res, err := Get(fmt.Sprintf("http://%s/", ln.Addr()))
	if err != nil {
		t.Fatalf("http client error: %v", err)
	}
	_, err = io.Copy(ioutil.Discard, res.Body)
	if err != nil {
		t.Fatalf("client body read error: %v", err)
	}
	res.Body.Close()

	// Force child to exit cleanly.
	Get(fmt.Sprintf("http://%s/quit", ln.Addr()))
	child.Wait()

	rx := regexp.MustCompile(`sendfile(64)?\(\d+,\s*\d+,\s*NULL,\s*\d+\)\s*=\s*\d+\s*\n`)
	rxResume := regexp.MustCompile(`<\.\.\. sendfile(64)? resumed> \)\s*=\s*\d+\s*\n`)
	out := buf.String()
	if !rx.MatchString(out) && !rxResume.MatchString(out) {
		t.Errorf("no sendfile system call found in:\n%s", out)
	}
}

func getBody(t *testing.T, testName string, req Request) (*Response, []byte) {
	r, err := DefaultClient.Do(&req)
	if err != nil {
		t.Fatalf("%s: for URL %q, send error: %v", testName, req.URL.String(), err)
	}
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("%s: for URL %q, reading body: %v", testName, req.URL.String(), err)
	}
	return r, b
}

// TestLinuxSendfileChild isn't a real test. It's used as a helper process
// for TestLinuxSendfile.
func TestLinuxSendfileChild(*testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	defer os.Exit(0)
	fd3 := os.NewFile(3, "ephemeral-port-listener")
	ln, err := net.FileListener(fd3)
	if err != nil {
		panic(err)
	}
	mux := NewServeMux()
	mux.Handle("/", FileServer(Dir("testdata")))
	mux.HandleFunc("/quit", func(ResponseWriter, *Request) {
		os.Exit(0)
	})
	s := &Server{Handler: mux}
	err = s.Serve(ln)
	if err != nil {
		panic(err)
	}
}

func equal(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
