// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"fmt"
	. "http"
	"http/httptest"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"url"
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
	_, body := getBody(t, req)
	if !equal(body, file) {
		t.Fatalf("body mismatch: got %q, want %q", body, file)
	}

	// Range tests
	for _, rt := range ServeFileRangeTests {
		req.Header.Set("Range", "bytes="+rt.r)
		if rt.r == "" {
			req.Header["Range"] = nil
		}
		r, body := getBody(t, req)
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
		return nil, os.ENOENT
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
		if gfi.Ino != wfi.Ino {
			t.Errorf("%s got different inode", name)
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

func TestServeFileContentType(t *testing.T) {
	const ctype = "icecream/chocolate"
	override := false
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		if override {
			w.Header().Set("Content-Type", ctype)
		}
		ServeFile(w, r, "testdata/file")
	}))
	defer ts.Close()
	get := func(want string) {
		resp, err := Get(ts.URL)
		if err != nil {
			t.Fatal(err)
		}
		if h := resp.Header.Get("Content-Type"); h != want {
			t.Errorf("Content-Type mismatch: got %q, want %q", h, want)
		}
	}
	get("text/plain; charset=utf-8")
	override = true
	get(ctype)
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
		defer res.Body.Close()
		b, err := ioutil.ReadAll(res.Body)
		if err != nil {
			t.Fatal("reading Body:", err)
		}
		if s := string(b); s != want {
			t.Errorf("for path %q got %q, want %q", path, s, want)
		}
	}
}

func getBody(t *testing.T, req Request) (*Response, []byte) {
	r, err := DefaultClient.Do(&req)
	if err != nil {
		t.Fatal(req.URL.String(), "send:", err)
	}
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		t.Fatal("reading Body:", err)
	}
	return r, b
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
